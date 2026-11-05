"""
Stage 1 VAE 학습. TabSyn의 train1.py 구조를 따른다.
  손실 = MSE(연속) + CE(범주) + β × KL,  β는 max_beta에서 시작해 검증 손실이 정체하면 × lambda (min_beta까지)

원본과 다른 점
  - 검증 데이터: TabSyn은 test 분할을 쓰지만, 여기서는 학습 데이터의 일부(val_frac)를 쓴다 (홀드아웃은 평가 전용)
  - 검증 손실이 가장 낮은 epoch의 가중치를 저장하고, 오래 개선이 없으면 조기 종료한다

Reference:
[1] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/main.py
"""
#%%
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
#%%
def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
        x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss /= (idx + 1)
    acc /= total_num

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_function(model, X_num, X_cat, config, device):
    """
    X_num:  연속 변수 (전처리 완료, 표준화 등)      FloatTensor [N, 연속 변수 수]
    X_cat:  범주 번호 0 ~ k-1                      LongTensor  [N, 범주 변수 수]
    config: seed, val_frac, lr, epochs, batch_size, max_beta, min_beta, lambda, beta_patience, stop_patience
    반환:   가장 좋은 epoch의 가중치를 불러온 model, epoch별 기록(DataFrame)
    """
    beta = config['max_beta']

    g = torch.Generator().manual_seed(config['seed'])
    perm = torch.randperm(len(X_num), generator = g)
    n_val = int(len(perm) * config['val_frac'])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.95, patience = 10)

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
            loss = loss_mse + loss_ce + beta * loss_kld
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
        val_loss = val['val_mse'] + val['val_ce']
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, patience = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience % config['beta_patience'] == 0 and beta > config['min_beta']:
                beta = beta * config['lambda']                      # 정체하면 KL 가중치를 줄인다
            if patience >= config['stop_patience']:
                print(f"Early stopping (epoch {epoch + 1})")
                break

        history.append(dict(epoch = epoch + 1, beta = beta, **{k: np.mean(v) for k, v in logs.items()}, **val))
        print(f"[epoch {epoch + 1:03d}] loss {np.mean(logs['loss']):.4f} (mse {np.mean(logs['loss_mse']):.4f}, "
              f"ce {np.mean(logs['loss_ce']):.4f}, kld {np.mean(logs['loss_kld']):.4f}) | "
              f"val mse {val['val_mse']:.4f}, ce {val['val_ce']:.4f}, acc {val['val_acc']:.4f} | "
              f"beta {beta:.5f} | {time.time() - t0:.0f}s")

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)
