"""
모델 ②: GMVAE. 모델 ①(model1.Model_VAE)과 인코더·디코더·학습 절차가 같고, 사전분포만 혼합정규로 바꾼다.

  p(z) = Σ_y π_y N(μ_y, diag σ_y²),   y = 1, ..., G        (z: [CLS]를 뺀 토큰들을 펼친 벡터)
  q(y|x): 인코더 출력의 [CLS] 토큰 → 선형층 → softmax     ([CLS]는 행 전체를 요약하는 자리)
  KL     = Σ_y q(y|x) KL(q(z|x) || N(μ_y, σ_y²)) + KL(q(y|x) || p(y))
  생성:   y ~ p(y) → z ~ N(μ_y, σ_y²) → 디코더

Reference:
[1] Dilokthanakul et al. (2016), Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model1 import Model_VAE
#%%
class GMVAE(Model_VAE):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head = 1, factor = 32, bias = True,
                 n_components = 50):
        super(GMVAE, self).__init__(num_layers, d_numerical, categories, d_token,
                                    n_head = n_head, factor = factor, bias = bias)
        d_z = self.n_latent_tokens * d_token                       # 펼친 z의 차원
        self.n_components = n_components

        self.fc_qy = nn.Linear(d_token, n_components)              # q(y|x)
        self.mu_y = nn.Parameter(torch.randn(n_components, d_z))   # 성분별 평균
        self.logvar_y = nn.Parameter(torch.zeros(n_components, d_z))   # 성분별 log 분산
        self.py_logits = nn.Parameter(torch.zeros(n_components))   # 성분 비율 p(y) (학습)

    def q_y(self, mu_z):
        """성분 소속확률 q(y|x)  [B, G]"""
        return F.softmax(self.fc_qy(mu_z[:, 0]), dim = -1)

    def kl(self, mu_z, logvar_z):
        """Σ_y q(y|x) KL(q(z|x) || N(μ_y, σ_y²)) + KL(q(y|x) || p(y)). 행별 값의 배치 평균"""
        qy = self.q_y(mu_z)
        mu, logvar = mu_z[:, 1:].flatten(1), logvar_z[:, 1:].flatten(1)      # [B, d_z]

        kl_zy = 0.5 * (self.logvar_y[None] - logvar[:, None] - 1
                       + (logvar.exp()[:, None] + (mu[:, None] - self.mu_y[None]) ** 2)
                       / self.logvar_y.exp()[None]).sum(dim = -1)            # [B, G]
        log_py = F.log_softmax(self.py_logits, dim = -1)
        kl_y = (qy * (torch.log(qy + 1e-10) - log_py[None])).sum(dim = 1)   # [B]

        return ((qy * kl_zy).sum(dim = 1) + kl_y).mean()

    def sample_z(self, n):
        """y ~ p(y) → z ~ N(μ_y, σ_y²)  [n, 토큰 수, d_token]"""
        y = torch.multinomial(F.softmax(self.py_logits, dim = -1), n, replacement = True)
        z = self.mu_y[y] + torch.exp(0.5 * self.logvar_y[y]) * torch.randn_like(self.mu_y[y])
        return z.view(n, self.n_latent_tokens, self.d_token)
