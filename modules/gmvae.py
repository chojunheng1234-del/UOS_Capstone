"""
모델 ②: GMVAE
  Encoder  x → Tokenizer (변수 하나 = 토큰 하나) → Transformer → (μ, log σ²)
  Decoder  z → Transformer → Reconstructor → 연속 변수 (μ, log σ²), 범주별 점수
  사전분포  p(z) = Σ_y π_y N(μ_y, diag σ_y²),  y = 1, ..., G   (z: [CLS]를 뺀 토큰들을 펼친 벡터)
  q(y|x)    인코더 출력의 [CLS] 토큰 → 선형층 → softmax         ([CLS]는 행 전체를 요약하는 자리)
  디코더    p(x|z): 연속 변수 N(μ(z), σ²(z)), 범주 변수 softmax
  생성      y ~ p(y) → z ~ N(μ_y, σ_y²) → x ~ p(x|z)

이 파일만 읽어도 모델 전체를 알 수 있도록 뼈대(Tokenizer ~ Decoder)를 gaussian_vae.py와 같은 코드로 둔다.
gaussian_vae.py와 다른 곳은 맨 아래 Model_GMVAE의 사전분포 부분(kl, sample_z)뿐이다.

Reference:
[1] Kingma & Welling (2014), Auto-Encoding Variational Bayes
[2] Dilokthanakul et al. (2016), Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders
[3] Gorishniy et al. (2021), Revisiting Deep Learning Models for Tabular Data (FT-Transformer: Tokenizer, Transformer)

코드 출처: Tokenizer, MultiheadAttention, Transformer, Reconstructor는 https://github.com/amazon-science/tabsyn (Apache-2.0)의 코드를 수정해 사용
"""
#%%
import math

import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
#%%
class Tokenizer(nn.Module):
    """[CLS] + 연속 변수 (값 × 가중치 벡터) + 범주 변수 (임베딩) → [B, 1 + 연속 수 + 범주 수, d_token]"""
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        d_bias = d_numerical + len(categories)
        category_offsets = torch.tensor([0] + categories[:-1], dtype=torch.long).cumsum(0)
        self.register_buffer('category_offsets', category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories), d_token)
        nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None

        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + len(self.category_offsets)

    def forward(self, x_num, x_cat):
        x_num = torch.cat(
            [torch.ones(len(x_num), 1, device=x_num.device)]  # [CLS]
            + [x_num],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        x = torch.cat(
            [x, self.category_embeddings(x_cat.long() + self.category_offsets[None])],
            dim=1,
        )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization = 'kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a / b, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x

class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout = 0.0,
        ffn_dropout = 0.0,
        residual_dropout = 0.0,
        prenormalization = True,
        initialization = 'kaiming',
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()

            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x

class Reconstructor(nn.Module):
    """디코더 출력 토큰 → 연속 변수의 평균·log σ² / 범주별 점수(logit)
    연속 변수는 Gaussian 디코더 p(x|z) = N(μ(z), σ²(z)) (Kingma & Welling, 2014)"""
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))           # 평균 μ
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.weight_logvar = nn.Parameter(Tensor(d_numerical, d_token))    # log σ²
        nn.init.xavier_uniform_(self.weight_logvar, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_logvar = torch.mul(h_num, self.weight_logvar.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_logvar, recon_x_cat


class Encoder(nn.Module):
    """인코더 q(z|x): 행 → 토큰 → Transformer → 토큰별 (μ, log σ²)  [B, 1 + 변수 수, d_token]"""
    def __init__(self, d_numerical, categories, num_layers, d_token, n_head = 1, factor = 32, bias = True):
        super(Encoder, self).__init__()

        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder_mu = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.encoder_logvar = Transformer(num_layers, d_token, n_head, d_token, factor)

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        mu_z = self.encoder_mu(x)
        logvar_z = self.encoder_logvar(x)

        return mu_z, logvar_z

class Decoder(nn.Module):
    """디코더 p(x|z): z → Transformer → 연속 변수의 (μ, log σ²), 범주별 점수(logit)"""
    def __init__(self, d_numerical, categories, num_layers, d_token, n_head = 1, factor = 32):
        super(Decoder, self).__init__()

        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def forward(self, z):
        h = self.decoder(z)
        recon_x_num, recon_x_logvar, recon_x_cat = self.Reconstructor(h)

        return recon_x_num, recon_x_logvar, recon_x_cat

class Model_GMVAE(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head = 1, factor = 32, bias = True,
                 n_components = 50):
        super(Model_GMVAE, self).__init__()

        self.encoder = Encoder(d_numerical, categories, num_layers, d_token, n_head = n_head, factor = factor, bias = bias)
        self.decoder = Decoder(d_numerical, categories, num_layers, d_token, n_head = n_head, factor = factor)
        self.d_token = d_token
        self.n_latent_tokens = d_numerical + len(categories)       # [CLS] 제외

        # 혼합정규 사전분포
        d_z = self.n_latent_tokens * d_token                       # 펼친 z의 차원
        self.n_components = n_components
        self.fc_qy = nn.Linear(d_token, n_components)              # q(y|x)
        self.mu_y = nn.Parameter(torch.randn(n_components, d_z))   # 성분별 평균
        self.logvar_y = nn.Parameter(torch.zeros(n_components, d_z))   # 성분별 log 분산
        self.py_logits = nn.Parameter(torch.zeros(n_components))   # 성분 비율 p(y) (학습)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        mu_z, logvar_z = self.encoder(x_num, x_cat)                 # 인코더 q(z|x)
        z = self.reparameterize(mu_z, logvar_z)
        recon_x_num, recon_x_logvar, recon_x_cat = self.decoder(z[:, 1:])   # 디코더 p(x|z). [CLS]는 쓰지 않는다

        return recon_x_num, recon_x_logvar, recon_x_cat, mu_z, logvar_z

    """ 사전분포 p(z): 혼합정규 """
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

    @torch.no_grad()
    def sample(self, n, batch_size = 2048):
        """z ~ p(z) → x ~ p(x|z): 연속은 N(μ, σ²)에서, 범주는 확률분포에서 샘플링"""
        nums, cats = [], []
        for i in range(0, n, batch_size):
            z = self.sample_z(min(batch_size, n - i))
            x_num, x_logvar, x_cat = self.decoder(z)
            nums.append((x_num + torch.exp(0.5 * x_logvar) * torch.randn_like(x_num)).cpu())
            # 범주는 디코더가 준 확률분포 p(x|z)에서 뽑는다 (argmax를 쓰면 희귀 범주가 생성되지 않음)
            cats.append(torch.stack([torch.multinomial(F.softmax(l, dim = -1), 1).squeeze(1)
                                     for l in x_cat], dim = 1).cpu())
        return torch.cat(nums), torch.cat(cats)
