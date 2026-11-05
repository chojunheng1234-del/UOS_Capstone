# UOS Capstone — 모델 모듈

2026-2 서울시립대학교 통계자료분석캡스톤디자인.
이 저장소에는 **모델 모듈만** 둔다. 데이터 읽기·전처리는 별도로 진행하고, 그 결과(텐서)를 이 모듈에 넣는다.
구조는 [SPT](https://github.com/hoon-ii/SPT) / [TabSyn](https://github.com/amazon-science/tabsyn)의 `modules/`를 따른다.

## 구조

```
modules/
├── model1.py    모델 ① Gaussian VAE (Tokenizer → Transformer 인코더 → z → Transformer 디코더 → Reconstructor)
│                  VAE        토큰화 + 인코더(μ, log σ²) + 디코더
│                  Model_VAE  VAE + Reconstructor. sample(n): z ~ N(0, I) → 디코더
├── train1.py    compute_loss (표준 ELBO: 재구성 + KL, β = 1), train_function (검증 기반 조기 종료)
└── utils.py     set_random_seed, get_device
```

## 입력 형식

| 이름 | 형태 | 내용 |
|---|---|---|
| `X_num` | FloatTensor `[N, 연속 변수 수]` | 전처리(표준화 등)가 끝난 연속 변수 |
| `X_cat` | LongTensor `[N, 범주 변수 수]` | 범주 번호 `0 ~ k-1` |
| `categories` | `list[int]` | 범주 변수별 범주 수 `k` (이진 변수는 2) |

## 사용 예

```python
from modules.model1 import Model_VAE
from modules.train1 import train_function
from modules.utils import set_random_seed, get_device

set_random_seed(42); device = get_device()
model = Model_VAE(num_layers=2, d_numerical=X_num.shape[1], categories=categories,
                  d_token=16, n_head=1, factor=32).to(device)

config = dict(seed=42, val_frac=0.05, lr=1e-3, epochs=30, batch_size=512, patience=3)
model, history = train_function(model, X_num, X_cat, config, device)

syn_num, syn_cat = model.sample(n=len(X_num))      # 전처리 척도의 값 / 범주 번호 → 역변환은 전처리 쪽에서
```

## 원칙: 모델 파일에는 그 모델만

모델끼리 비교하기 위해, 각 모델 파일에는 그 모델의 정의만 둔다.
- 모델 ① `model1.py` / `train1.py`: 순수 Gaussian VAE. 사전분포 N(0, I), 손실 = 표준 ELBO (β = 1 고정)
- 네트워크 구조(Tokenizer, Transformer, Reconstructor)와 학습 절차(Adam, 검증 5%, 조기 종료)는 모든 모델이 같게 쓴다
- 확산(Stage 2), β 스케줄링 같은 장치는 해당 모델(③ TabSyn 등)에서만 추가한다

## 하이퍼파라미터 메모

- `d_token`: SPT 기본값 4로는 상품(`parent_asin`, 약 10만 개)을 토큰 하나로 복원하기 어렵다 (Amazon 5% 표본 시험에서 head 상품 복원 정확도 0.30 → `d_token=16`에서 1.00).

## 진행 예정

- [x] 모델 ① Gaussian VAE (`model1.py`, `train1.py`)
- [ ] 모델 ② GMVAE: 사전분포만 혼합정규로 교체
- [ ] 모델 ③ TabSyn: Stage 2 잠재 확산 추가
- [ ] 모델 ④ 제안 모델
