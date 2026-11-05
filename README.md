# UOS Capstone — 모델 모듈

2026-2 서울시립대학교 통계자료분석캡스톤디자인.
이 저장소에는 **모델 모듈만** 둔다. 데이터 읽기·전처리는 별도로 진행하고, 그 결과(텐서)를 이 모듈에 넣는다.
구조는 [SPT](https://github.com/hoon-ii/SPT) / [TabSyn](https://github.com/amazon-science/tabsyn)의 `modules/`를 따른다.

## 구조

```
modules/
├── model1.py    Stage 1 VAE (Tokenizer → Transformer 인코더 → z → Transformer 디코더 → Reconstructor)
│                  Model_VAE      학습·생성용 전체 모델. sample(n): z ~ N(0, I)에서 바로 생성 (모델 ①)
│                  Encoder_model  학습된 인코더만 (Stage 2 확산 모델의 입력 z 추출용)
│                  Decoder_model  학습된 디코더만 (Stage 2에서 생성한 z를 표로 복원)
├── train1.py    compute_loss (MSE + CE + KL), train_function (β 스케줄링, 검증 기반 조기 종료)
└── utils.py     set_random_seed, get_device
```

## 입력 형식

| 이름 | 형태 | 내용 |
|---|---|---|
| `X_num` | FloatTensor `[N, 연속 변수 수]` | 전처리(표준화 등)가 끝난 연속 변수 |
| `X_cat` | LongTensor `[N, 범주 변수 수]` | 범주 번호 `0 ~ k-1` (학습에 없는 범주는 `k` = UNK) |
| `categories` | `list[int]` | 범주 변수별 범주 수 `k` (이진 변수는 2) |

## 사용 예

```python
from modules.model1 import Model_VAE
from modules.train1 import train_function
from modules.utils import set_random_seed, get_device

set_random_seed(42); device = get_device()
model = Model_VAE(num_layers=2, d_numerical=X_num.shape[1], categories=categories,
                  d_token=16, n_head=1, factor=32).to(device)

config = dict(seed=42, val_frac=0.05, lr=1e-3, epochs=30, batch_size=512,
              max_beta=1.0, min_beta=1e-5, beta_patience=3, stop_patience=6, **{"lambda": 0.7})
model, history = train_function(model, X_num, X_cat, config, device)

syn_num, syn_cat = model.sample(n=len(X_num))      # 전처리 척도의 값 / 범주 번호 → 역변환은 전처리 쪽에서
```

## 하이퍼파라미터 메모 (Amazon All_Beauty 5% 표본, 4 epoch 시험)

- `d_token`: SPT 기본값 4로는 상품(`parent_asin`, 약 10만 개)을 토큰 하나로 복원하기 어렵다 (head 상품 복원 정확도 0.30). 16에서 1.00.
- `max_beta`: TabSyn 기본값 `1e-2`는 Stage 2 확산 모델을 전제로 한 값이다. 이 VAE를 **N(0, I)에서 바로 생성**(모델 ①)하려면 `1.0`이 필요하다 (평점 분포 TVD: β=0.01 → 0.295, β=1 → 0.080).

## 진행 예정

- [x] 모델 ① Gaussian VAE (`model1.py`, `train1.py`)
- [ ] 모델 ② GMVAE: 사전분포만 혼합정규로 교체
- [ ] 모델 ③ TabSyn: `model2.py`, `train2.py` (잠재 확산)
- [ ] 모델 ④ 제안 모델
