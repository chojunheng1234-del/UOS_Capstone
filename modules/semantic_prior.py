"""
의미 기반 사전분포 (모델 ④용). 연구 계획서 5절, 식 (5)~(7).

입력: Datasets 쪽에서 만든 LLM 임베딩 행렬 E [k, p]   (i번째 행 = 범주 번호 i)
출력: 범주별 의미 위치 (W·E)_c [k, d] 와 범주별 사전분포의 표준편차 δ0
      → 모델 ④에서 p(z_sem | c) = N(m_c, δ0² I),  m_c = (W·E)_c + Δ_c

  (5) 중심화한 E의 공분산을 고유분해 → 상위 d개 주성분 W
  (6) 백색화: 주성분 좌표를 √고유값으로 나눔 (한 성분이 거리를 지배하지 않게) → 척도 s를 곱함
  (7) D̄ = 백색화 좌표에서 범주별 최근접 범주까지 거리의 평균
      δ0 = D̄ / √(r² + D̄²),   s = √(1 − δ0²)
      → 의미 위치의 분산(s²)과 범주 내 분산(δ0²)의 합이 1 (표준 정규와 같은 규모)
      → 척도를 맞춘 뒤 최근접 범주 거리 ÷ δ0 = r  (r: 범주 간 분리 비율)

d는 의미 부분 잠재변수의 차원이다. 토큰 구조에서는 범주 변수의 토큰이 z_sem이므로 d = d_token.
"""
#%%
import math

import torch
#%%
def build_semantic_prior(E, d, r = 1.0, batch_size = 4096, device = "cpu"):
    """
    E:  LLM 임베딩 행렬 [k, p] (numpy 또는 tensor)
    d:  사영 차원 (= d_token)
    r:  분리 비율. 클수록 범주끼리 멀리 떨어진 사전분포
    반환: dict(WE [k, d], delta0, s, D_bar, explained)
    """
    E = torch.as_tensor(E, dtype = torch.float64, device = device)
    k, p = E.shape
    assert d <= min(k - 1, p), f"d={d}는 min(k-1, p)={min(k - 1, p)} 이하여야 한다"

    # (5) 주성분분석
    Ec = E - E.mean(dim = 0, keepdim = True)
    cov = Ec.T @ Ec / k                                             # [p, p]
    eigval, eigvec = torch.linalg.eigh(cov)                         # 오름차순
    eigval, eigvec = eigval.flip(0), eigvec.flip(1)
    lam, W = eigval[:d], eigvec[:, :d].T                            # [d], [d, p]

    # (6) 백색화 좌표 (척도 s 적용 전)
    U = (Ec @ W.T) / lam.sqrt()                                     # [k, d]

    # (7) 최근접 범주 거리의 평균 → δ0, s
    nearest = []
    for i in range(0, k, batch_size):
        dist = torch.cdist(U[i:i + batch_size], U)                  # [b, k]
        dist[torch.arange(dist.shape[0]), torch.arange(i, i + dist.shape[0])] = float("inf")   # 자기 자신 제외
        nearest.append(dist.min(dim = 1).values)
    D_bar = torch.cat(nearest).mean().item()
    delta0 = D_bar / math.sqrt(r ** 2 + D_bar ** 2)
    s = math.sqrt(1 - delta0 ** 2)

    return dict(
        WE = (s * U).float().cpu(),                                 # 범주별 의미 위치 (W·E)_c
        delta0 = delta0,                                            # 범주별 사전분포의 표준편차
        s = s,
        D_bar = D_bar,
        explained = (lam.sum() / eigval.clamp(min = 0).sum()).item(),   # 상위 d개 주성분의 설명 분산 비율
    )
