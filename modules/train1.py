"""
모델 ①: Gaussian VAE 학습. TabSyn의 train1.py 구조를 따른다.

순수 VAE만 둔다. 손실은 표준 ELBO의 음수 (행별 합):
  재구성 = 0.5 × Σ(연속 오차²)  [분산 1의 Gaussian]  +  Σ 범주별 cross-entropy
  KL     = Σ KL(q(z|x) || N(0, I))  [디코더가 쓰는 토큰만, [CLS] 제외]
  손실   = 재구성 + KL   (β = 1 고정. TabSyn의 β 스케줄링·평균 손실은 쓰지 않는다)

학습 절차(옵티마이저, 검증 분할, 조기 종료)는 모든 비교 모델에서 같게 둔다.

Reference:
[1] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/main.py
"""
#%%
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
#%%
def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    """행별 평균 (재구성, KL, 범주 복원 정확도)"""
    mse_loss = 0.5 * (X_num - Recon_X_num).pow(2).sum(dim = 1).mean()
    ce_loss = 0
    acc = 0
    for idx, x_cat in enumerate(Recon_X_cat):
        ce_loss += F.cross_entropy(x_cat, X_cat[:, idx])
        acc += (x_cat.argmax(dim = -1) == X_cat[:, idx]).float().mean()
    acc /= (idx + 1)

    mu, logvar = mu_z[:, 1:], logvar_z[:, 1:]          # [CLS] 토큰은 디코더가 쓰지 않으므로 제외
    loss_kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim = (1, 2)).mean()
    return mse_loss, ce_loss, loss_kld, acc


def train_function(model, X_num, X_cat, config, device):
    """
    X_num:  연속 변수 (전처리 완료)     FloatTensor [N, 연속 변수 수]
    X_cat:  범주 번호 0 ~ k-1          LongTensor  [N, 범주 변수 수]
    config: seed, val_frac, lr, epochs, batch_size, patience
    반환:   검증 손실이 가장 낮은 epoch의 가중치를 불러온 model, epoch별 기록(DataFrame)
    """
    g = torch.Generator().manual_seed(config['seed'])
    perm = torch.randperm(len(X_num), generator = g)
    n_val = int(len(perm) * config['val_frac'])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])

    best_val_loss, best_state, patience, history = float('inf'), None, 0, []
    for epoch in range(config['epochs']):
        logs = {'loss': [], 'loss_mse': [], 'loss_ce': [], 'loss_kld': [], 'train_acc': []}
        model.train(); t0 = time.time()
        for idx in tr_idx[torch.randperm(len(tr_idx), generator = g)].split(config['batch_size']):
            batch_num, batch_cat = X_num[idx].to(device), X_cat[idx].to(device)

            optimizer.zero_grad()
            Recon_X_num, Recon_X_cat, mu_z, logvar_z = model(batch_num, batch_cat)
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z)
            loss = loss_mse + loss_ce + loss_kld
            loss.backward()
            optimizer.step()

            for key, v in zip(logs, [loss, loss_mse, loss_ce, loss_kld, train_acc]):
                logs[key].append(v.item())

        ''' Evaluation (검증 데이터) '''
        model.eval()
        val = {'val_mse': 0.0, 'val_ce': 0.0, 'val_kld': 0.0, 'val_acc': 0.0}
        with torch.no_grad():
            for idx in val_idx.split(config['batch_size']):
                batch_num, batch_cat = X_num[idx].to(device), X_cat[idx].to(device)
                Recon_X_num, Recon_X_cat, mu_z, logvar_z = model(batch_num, batch_cat)
                for key, v in zip(val, compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z)):
                    val[key] += v.item() * len(idx) / len(val_idx)
        val_loss = val['val_mse'] + val['val_ce'] + val['val_kld']

        history.append(dict(epoch = epoch + 1, **{k: np.mean(v) for k, v in logs.items()}, **val, val_loss = val_loss))
        print(f"[epoch {epoch + 1:03d}] loss {np.mean(logs['loss']):.4f} (mse {np.mean(logs['loss_mse']):.4f}, "
              f"ce {np.mean(logs['loss_ce']):.4f}, kld {np.mean(logs['loss_kld']):.4f}) | "
              f"val loss {val_loss:.4f}, acc {val['val_acc']:.4f} | {time.time() - t0:.0f}s")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, patience = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config['patience']:
                print(f"Early stopping (epoch {epoch + 1})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)
