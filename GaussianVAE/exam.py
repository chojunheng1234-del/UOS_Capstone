#%%
""" 0. Reproducibility """
import torch
from modules.utils import set_random_seed

device = (torch.device("cuda:0") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
config = {
    "dataset": "amazon",
    "seed": 42,
    "data_dir": "./data",        # 원자료 위치 (없으면 자동으로 내려받음)
    "sample_frac": 1.0,          # 빠른 시험: 0.05
}
set_random_seed(config["seed"])
print("device =", device)
#%%
""" 1. Load the raw data """
import pandas as pd
import numpy as np

from datasets.raw_data import load_raw_data

raw = load_raw_data(config)
data, continuous_features, categorical_features, high_card_features, tier_bins, ClfTarget = raw
features = continuous_features + categorical_features
print(data.shape)
print(data.head())
# %%
""" 2. Preprocess the raw data """
from datasets.preprocess import build_datasets

config["holdout_frac"] = 0.2

# 학습 데이터로만 범주 번호·표준화 통계(info)를 만들고, 홀드아웃에 같은 규칙을 적용한다
train_dataset, holdout_dataset, info = build_datasets(config, raw)
print(f"학습 {len(train_dataset):,}행 | 홀드아웃 {len(holdout_dataset):,}행")
print("X_num:", tuple(train_dataset.X_num.shape), "| X_cat:", tuple(train_dataset.X_cat.shape))
for col, k in zip(categorical_features, info["categories"]):
    print(f"  {col}: 범주 {k:,}개")

#%%
""" 2-1. 원본 데이터와 비교 """
import matplotlib.pyplot as plt

def visualize_data(idx):
    col = features[idx]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if col in continuous_features:
        j = continuous_features.index(col)
        ax1.hist(train_dataset.raw_data[col], bins=30)
        ax2.hist(train_dataset.X_num[:, j].numpy(), bins=30)
    elif col in high_card_features:
        # 상품 ID는 값 자체를 그릴 수 없으므로 상품별 리뷰 수 분포(로그-로그)로 본다
        cnt = train_dataset.raw_data[col].value_counts().value_counts().sort_index()
        ax1.loglog(cnt.index, cnt.values, "o", ms=3)
        ax1.set_xlabel("# reviews per product"); ax1.set_ylabel("# products")
        j = categorical_features.index(col)
        ax2.hist(train_dataset.X_cat[:, j].numpy(), bins=100); ax2.set_yscale("log")
        ax2.set_xlabel("category code c (frequent → rare)")
    else:
        j = categorical_features.index(col)
        train_dataset.raw_data[col].value_counts().sort_index().plot.bar(ax=ax1)
        ax2.hist(train_dataset.X_cat[:, j].numpy(), bins=np.arange(info["categories"][j] + 1) - 0.5)
        ax2.set_xlabel("category code")
    ax1.set_title(f"Original: {col}")
    ax2.set_title(f"Preprocessed: {col}")
    plt.tight_layout()
    plt.show()

for idx in range(len(features)):
    visualize_data(idx)
# %%
""" 3. Variational Autoencoder (VAE) """
import math
import torch.nn as nn
import torch.nn.functional as F
from datasets.preprocess import decode

class VAE(nn.Module):
    """연속 + 범주 혼합형 VAE.
    - 범주는 임베딩으로 입력하고, softmax로 출력한다 (상품 10만 개를 숫자 하나로 넣을 수 없기 때문)
    - 연속은 그대로 입력하고, Gaussian으로 출력한다
    """
    def __init__(self, config, info, device):
        super(VAE, self).__init__()

        self.config = config
        self.device = device
        self.hidden_dim = config["hidden_dim"]
        self.beta = config["beta"]
        self.categories = info["categories"]
        self.n_cont = len(info["continuous_features"])

        """embedding (범주 번호 → 벡터, +1: 학습에 없는 범주 UNK)"""
        self.embeddings = nn.ModuleList(
            [nn.Embedding(k + 1, min(config["emb_dim"], k + 1)) for k in self.categories]).to(device)
        input_dim = self.n_cont + sum(e.embedding_dim for e in self.embeddings)

        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["latent_dim"] * 2),
        ).to(device)

        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(device)
        self.cont_head = nn.Linear(self.hidden_dim, self.n_cont).to(device)        # 연속: 평균
        self.cont_logsig = nn.Parameter(torch.zeros(self.n_cont, device=device))   # 연속: log 표준편차
        self.cat_heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, k) for k in self.categories]).to(device)   # 범주: 범주별 점수

    def get_posterior(self, x_num, x_cat):
        x = torch.cat([x_num] + [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)], dim=1)
        h = self.encoder(x)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar

    def encode(self, x_num, x_cat):
        mean, logvar = self.get_posterior(x_num, x_cat)
        noise = torch.randn(x_num.size(0), self.config["latent_dim"]).to(self.device)
        latent = mean + torch.exp(logvar / 2) * noise
        return mean, logvar, latent

    def decode(self, latent):
        h = self.decoder(latent)
        return self.cont_head(h), [head(h) for head in self.cat_heads]

    def forward(self, x_num, x_cat):
        mean, logvar, latent = self.encode(x_num, x_cat)
        xhat = self.decode(latent)
        return mean, logvar, latent, xhat

    def loss(self, x_num, x_cat, mean, logvar, xhat):
        cont_mean, cat_logits = xhat
        # 재구성: 연속은 Gaussian 음의 로그우도, 범주는 cross-entropy
        recon = (0.5 * ((x_num - cont_mean) / self.cont_logsig.exp()) ** 2
                 + self.cont_logsig + 0.5 * math.log(2 * math.pi)).sum(dim=1).mean()
        for j, logits in enumerate(cat_logits):
            recon = recon + F.cross_entropy(logits, x_cat[:, j])
        # KL(q(z|x) || N(0, I))
        kl = -0.5 * (1 + logvar - mean ** 2 - logvar.exp()).sum(dim=1).mean()
        return recon + self.beta * kl, recon, kl

    def generate_synthetic_data(self, n, info):
        steps = n // self.config["batch_size"] + 1
        X_num, X_cat = [], []
        with torch.no_grad():
            for _ in range(steps):
                noise = torch.randn(self.config["batch_size"], self.config["latent_dim"]).to(self.device)
                cont_mean, cat_logits = self.decode(noise)
                X_num.append((cont_mean + self.cont_logsig.exp() * torch.randn_like(cont_mean)).cpu())
                # 범주는 가장 큰 값(argmax)이 아니라 확률에 따라 뽑는다 → 희귀 범주도 생성됨
                X_cat.append(torch.stack(
                    [torch.multinomial(F.softmax(l, dim=-1), 1).squeeze(1) for l in cat_logits], dim=1).cpu())
        X_num = torch.cat(X_num)[:n].numpy()
        X_cat = torch.cat(X_cat)[:n].numpy()

        # 범주 번호 → 원래 값, 표준화 → 원척도
        syndata = decode(X_num, X_cat, info)
        return syndata[features]

config["batch_size"] = 512
config["hidden_dim"] = 256
config["latent_dim"] = 16
config["emb_dim"] = 32
config["beta"] = 1.0
model = VAE(config, info, device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# model.get_posterior(train_dataset.X_num[:2].to(device), train_dataset.X_cat[:2].to(device))
# %%
""" 4. Train """
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm
config["epochs"] = 10
config["lr"] = 1e-3

# 행이 56만 개라 한 행씩 꺼내면 느리다 → 배치 단위로 한 번에 꺼낸다
dataloader = DataLoader(
    train_dataset,
    sampler=BatchSampler(RandomSampler(train_dataset), batch_size=config["batch_size"], drop_last=False),
    batch_size=None,
)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

model.train()
for epoch in tqdm(range(config["epochs"]), desc="Training VAE"):
    epoch_loss = epoch_recon = epoch_kl = 0
    for x_num, x_cat in dataloader:
        x_num, x_cat = x_num.to(device), x_cat.to(device)

        optimizer.zero_grad()
        mean, logvar, latent, xhat = model(x_num, x_cat)
        loss, recon, kl = model.loss(x_num, x_cat, mean, logvar, xhat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item(); epoch_recon += recon.item(); epoch_kl += kl.item()

    print(f"Epoch {epoch:03d} | Loss: {epoch_loss / len(dataloader):.4f} "
          f"(recon {epoch_recon / len(dataloader):.4f}, KL {epoch_kl / len(dataloader):.4f})")
# %%
""" 5. Inference """
model.eval()
n = len(train_dataset)
syndata = model.generate_synthetic_data(n, info)
print(syndata.head())
print(syndata.shape)
# %%
""" 6. Visualize """
def visualize_synthetic(idx):
    col = features[idx]
    _, ax = plt.subplots(figsize=(5, 4))
    real = train_dataset.raw_data[col]
    if col in continuous_features:
        ax.hist(real, bins=30, alpha=0.5, label="Real", density=True)
        ax.hist(syndata[col], bins=30, alpha=0.5, label="Synthetic", density=True)
    elif col in high_card_features:
        # 상품별 등장 횟수를 많은 순으로 정렬해 비교 (긴 꼬리를 얼마나 재현하는지)
        ax.loglog(np.arange(1, real.nunique() + 1), real.value_counts().values, label="Real")
        ax.loglog(np.arange(1, syndata[col].nunique() + 1), syndata[col].value_counts().values, label="Synthetic")
        ax.set_xlabel("product rank"); ax.set_ylabel("# rows")
    else:
        pd.DataFrame({"Real": real.value_counts(normalize=True),
                      "Synthetic": syndata[col].value_counts(normalize=True)}).sort_index().plot.bar(ax=ax)
    ax.set_title(col)
    ax.legend()
    plt.tight_layout()
    plt.show()

for idx in range(len(features)):
    visualize_synthetic(idx)
#%%
""" 6-1. Evaluate """
from datasets.preprocess import encode
from evaluation.metrics import evaluate

def evaluate_synthetic(syndata, name):
    real = (train_dataset.X_num.numpy(), train_dataset.X_cat.numpy())
    # 기준선: 홀드아웃 (학습에 없는 상품 UNK 행은 제외)
    hold_num, hold_cat = holdout_dataset.X_num.numpy(), holdout_dataset.X_cat.numpy()
    known = (hold_cat < np.array(info["categories"])).all(axis=1)
    syn_num, syn_cat = encode(syndata, info)
    return pd.DataFrame({
        "real_holdout (기준)": evaluate(real, (hold_num[known], hold_cat[known]), info),
        name: evaluate(real, (syn_num.numpy(), syn_cat.numpy()), info),
    })

results = evaluate_synthetic(syndata, "VAE")
print(results.round(4).to_string())
# %%
""" 7. Beta-VAE (Scheduling Beta) """
model_beta = VAE(config, info, device)
optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=config["lr"])

config["epochs"] = 30
beta = config["beta"]
best_loss = float("inf")
patience = 0

model_beta.train()
for epoch in tqdm(range(config["epochs"]), desc="Training Beta-VAE"):
    epoch_loss = 0
    for x_num, x_cat in dataloader:
        x_num, x_cat = x_num.to(device), x_cat.to(device)

        optimizer_beta.zero_grad()
        mean, logvar, latent, xhat = model_beta(x_num, x_cat)
        loss, recon, kl = model_beta.loss(x_num, x_cat, mean, logvar, xhat)
        loss.backward()
        optimizer_beta.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience = 0
    else:
        patience += 1
        if patience >= 3:
            beta = beta * 0.7
            model_beta.beta = beta
            patience = 0
            print(f"Epoch {epoch:03d} | Beta decreased to {beta:.4f}")

    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Beta: {beta:.4f}")

model_beta.eval()
n = len(train_dataset)
syndata_beta = model_beta.generate_synthetic_data(n, info)
print(syndata_beta.head())
print(syndata_beta.shape)

results["Beta-VAE"] = evaluate_synthetic(syndata_beta, "Beta-VAE")["Beta-VAE"]
print(results.round(4).to_string())
# %%
