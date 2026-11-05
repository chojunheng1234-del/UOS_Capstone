#%%
import math
import time

import pandas as pd
import torch
import torch.nn as nn
#%%
def train_function(model, train_dataset, config, device):
    """KL warm-up + 검증 손실 기준 조기 종료. 가장 좋은 epoch의 가중치로 되돌려 반환한다.

    검증 데이터는 학습 데이터의 일부(val_frac)이며, 홀드아웃과는 별개다.
    """
    X_num, X_cat = train_dataset.X_num, train_dataset.X_cat
    g = torch.Generator().manual_seed(config["seed"])
    perm = torch.randperm(len(X_num), generator=g)
    n_val = int(len(perm) * config["val_frac"])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    steps = math.ceil(len(tr_idx) / config["batch_size"])
    warmup_steps = max(1, config["warmup"] * steps)

    best, best_state, patience, history = float("inf"), None, 0, []
    for epoch in range(config["epochs"]):
        # 학습
        model.train(); t0 = time.time(); total = 0.0
        shuffled = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for step, idx in enumerate(shuffled.split(config["batch_size"])):
            beta = min(1.0, (epoch * steps + step + 1) / warmup_steps)
            loss, _, _ = model(X_num[idx].to(device), X_cat[idx].to(device), beta)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(idx)

        # 검증 (β = 1)
        model.eval(); val = rec = kl = 0.0
        with torch.no_grad():
            for idx in val_idx.split(config["batch_size"]):
                l, r, k = model(X_num[idx].to(device), X_cat[idx].to(device))
                val += l.item() * len(idx); rec += r.item() * len(idx); kl += k.item() * len(idx)
        val, rec, kl = val / len(val_idx), rec / len(val_idx), kl / len(val_idx)

        history.append(dict(epoch=epoch + 1, train=total / len(tr_idx), val=val, rec=rec, kl=kl))
        print(f"[epoch {epoch + 1:3d}] train {history[-1]['train']:.3f} | val {val:.3f} "
              f"(rec {rec:.3f}, kl {kl:.3f}) | {time.time() - t0:.0f}s")

        if val < best - 1e-3:
            best, patience = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config["patience"]:
                print(f"조기 종료 (epoch {epoch + 1})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)
