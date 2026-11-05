"""
모델 ①: Gaussian VAE. TabSyn / SPT의 model1.py 구조(Transformer VAE)를 따른다.
  Tokenizer (변수 하나 = 토큰 하나) → Transformer 인코더 (μ, log σ²) → z → Transformer 디코더 → Reconstructor
  사전분포 p(z) = N(0, I), 생성: z ~ N(0, I) → 디코더

순수 VAE만 둔다. 모델 간 비교를 위해 다른 장치(확산, β 스케줄링 등)는 넣지 않는다.
원본과 다른 점
  1. 인코더 분산을 학습한다 (encoder_logvar). 표준 VAE의 형태 (SPT는 분산을 0.1로 고정)
  2. Tokenizer.forward 중복 정의를 하나로 정리
  3. sample(): 사전분포에서 z를 뽑아 생성한다. 범주는 디코더가 준 확률분포에서 샘플링한다
  4. Stage 2용 Encoder_model / Decoder_model은 두지 않는다 (모델 ③ TabSyn에서 추가)

Reference:
[1] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/model.py
[2] https://github.com/hoon-ii/SPT/blob/main/SPT/modules/model1.py
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

class VAE(nn.Module):
    def __init__(self, d_numerical, categories, num_layers, hid_dim, n_head = 1, factor = 32, bias = True):
        super(VAE, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head

        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        mu_z = self.encoder_mu(x)
        logvar_z = self.encoder_logvar(x)
        z = self.reparameterize(mu_z, logvar_z)
        h = self.decoder(z[:, 1:])                  # [CLS] 토큰은 복원에 쓰지 않는다

        return h, mu_z, logvar_z

class Reconstructor(nn.Module):
    """디코더 출력 토큰 → 연속 값 / 범주별 점수(logit)"""
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class Model_VAE(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head = 1, factor = 32, bias = True):
        super(Model_VAE, self).__init__()

        self.VAE = VAE(d_numerical, categories, num_layers, d_token, n_head = n_head, factor = factor, bias = bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)
        self.d_token = d_token
        self.n_latent_tokens = d_numerical + len(categories)       # [CLS] 제외

    def forward(self, x_num, x_cat):
        h, mu_z, logvar_z = self.VAE(x_num, x_cat)
        recon_x_num, recon_x_cat = self.Reconstructor(h)

        return recon_x_num, recon_x_cat, mu_z, logvar_z

    """ 사전분포 p(z): 다른 모델(GMVAE 등)은 아래 두 메서드만 바꾼다 """
    def kl(self, mu_z, logvar_z):
        """KL(q(z|x) || N(0, I)). 행별 합의 배치 평균. [CLS] 토큰은 디코더가 쓰지 않으므로 제외"""
        mu, logvar = mu_z[:, 1:], logvar_z[:, 1:]
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim = (1, 2)).mean()

    def sample_z(self, n):
        """z ~ N(0, I)  [n, 토큰 수, d_token]"""
        return torch.randn(n, self.n_latent_tokens, self.d_token, device = self.Reconstructor.weight.device)

    @torch.no_grad()
    def sample(self, n, batch_size = 2048):
        """z ~ p(z) → 디코더 → 연속: 평균, 범주: 확률분포에서 샘플링"""
        nums, cats = [], []
        for i in range(0, n, batch_size):
            z = self.sample_z(min(batch_size, n - i))
            x_num, x_cat = self.Reconstructor(self.VAE.decoder(z))
            nums.append(x_num.cpu())
            # 범주는 디코더가 준 확률분포 p(x|z)에서 뽑는다 (argmax를 쓰면 희귀 범주가 생성되지 않음)
            cats.append(torch.stack([torch.multinomial(F.softmax(l, dim = -1), 1).squeeze(1)
                                     for l in x_cat], dim = 1).cpu())
        return torch.cat(nums), torch.cat(cats)

