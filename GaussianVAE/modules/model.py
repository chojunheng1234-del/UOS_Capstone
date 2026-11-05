#%%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class MixedVAE(nn.Module):
    """연속 + 범주 혼합형 VAE의 공통 인코더·디코더.
    사전분포에 관한 부분(kl, sample_z)만 하위 클래스에서 정의한다.
    → GMVAE, 제안 모델도 이 클래스를 상속해 사전분포만 바꾼다 (SPT가 TabSyn에서 사전분포만 바꾼 것과 같은 방식).
    """
    def __init__(self, d_numerical, categories, d_latent=16, d_hidden=256, d_emb=32):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_latent = d_latent

        # 인코더: [연속 변수, 범주별 임베딩] → (μ, log σ²)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(k + 1, min(d_emb, k + 1)) for k in categories])     # +1: 학습에 없는 범주(UNK)
        d_in = d_numerical + sum(e.embedding_dim for e in self.cat_embeddings)
        self.encoder = nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(),
                                     nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.fc_mu = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        # 디코더: z → 변수별 출력
        self.decoder = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.SiLU(),
                                     nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.num_head = nn.Linear(d_hidden, d_numerical)                      # 연속: Gaussian 평균
        self.num_logsig = nn.Parameter(torch.zeros(d_numerical))              # 연속: 변수별 log σ (학습)
        self.cat_heads = nn.ModuleList([nn.Linear(d_hidden, k) for k in categories])   # 범주: softmax

    def encode(self, x_num, x_cat):
        x = torch.cat([x_num] + [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embeddings)], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        return self.num_head(h), [head(h) for head in self.cat_heads]

    def recon_nll(self, x_num, x_cat, num_mean, cat_logits):
        """재구성 음의 로그우도 (행별)"""
        nll = (0.5 * ((x_num - num_mean) / self.num_logsig.exp()) ** 2
               + self.num_logsig + 0.5 * math.log(2 * math.pi)).sum(dim=1)
        for j, logits in enumerate(cat_logits):
            target = x_cat[:, j].clamp(max=self.categories[j] - 1)            # UNK는 학습 데이터에 없음
            nll = nll + F.cross_entropy(logits, target, reduction="none")
        return nll

    def forward(self, x_num, x_cat, beta=1.0):
        """손실 = 재구성 NLL + β × KL. (전체 손실, 재구성, KL)을 반환한다."""
        mu, logvar = self.encode(x_num, x_cat)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)              # 재매개변수화
        rec = self.recon_nll(x_num, x_cat, *self.decode(z))
        kl = self.kl(mu, logvar)
        return (rec + beta * kl).mean(), rec.mean(), kl.mean()

    @torch.no_grad()
    def sample(self, n, batch_size=2048):
        """사전분포에서 z를 뽑아 생성한다. 범주는 argmax가 아니라 확률에 따라 샘플링한다 (희귀 범주 재현)."""
        nums, cats = [], []
        for i in range(0, n, batch_size):
            num_mean, cat_logits = self.decode(self.sample_z(min(batch_size, n - i)))
            nums.append(num_mean + self.num_logsig.exp() * torch.randn_like(num_mean))
            cats.append(torch.stack([torch.multinomial(F.softmax(l, dim=-1), 1).squeeze(1)
                                     for l in cat_logits], dim=1))
        return torch.cat(nums).cpu(), torch.cat(cats).cpu()


class GaussianVAE(MixedVAE):
    """모델 ①: 사전분포 p(z) = N(0, I)"""

    def kl(self, mu, logvar):
        return 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(dim=1)

    def sample_z(self, n):
        return torch.randn(n, self.d_latent, device=self.fc_mu.weight.device)
