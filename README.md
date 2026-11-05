# UOS Capstone — 모델 모듈

2026-2 서울시립대학교 통계자료분석캡스톤디자인.
이 저장소에는 **모델 모듈만** 둔다. 데이터 읽기·전처리는 별도로 진행하고, 그 결과(텐서)를 이 모듈에 넣는다.

## 구조

모델마다 **파일 하나에 그 모델 전체**를 둔다. 파일 하나만 읽어도 모델을 이해할 수 있게 하기 위해서다.
그래서 뼈대(Tokenizer, Transformer, VAE, Reconstructor)는 모델 파일마다 같은 코드로 들어 있다.

```
modules/
├── gaussian_vae.py    모델 ① Gaussian VAE   Model_VAE    사전분포 N(0, I)
├── gmvae.py           모델 ② GMVAE          Model_GMVAE  사전분포 혼합정규 (q(y|x)는 [CLS] 토큰에서)
├── train.py           학습 절차 (①·② 공통): 손실 = 재구성(Gaussian NLL + CE) + model.kl(), β = 1, 검증 기반 조기 종료
├── semantic_prior.py  모델 ④용: LLM 임베딩 E → PCA → 백색화 → 범주별 의미 위치 (W·E)_c, δ0
└── utils.py           set_random_seed, get_device
```

모델 구조: Tokenizer (변수 하나 = 토큰 하나) → Transformer 인코더 (μ, log σ²) → z → Transformer 디코더 → Reconstructor
모델끼리 다른 곳은 각 파일 맨 아래 클래스의 사전분포 부분(`kl`, `sample_z`)뿐이다.

디코더 p(x|z) (모든 VAE 모델 공통, Kingma & Welling 2014):
- 연속 변수: N(μ(z), σ²(z)). 디코더가 평균과 log σ²를 함께 출력하고, 생성할 때 이 분포에서 뽑는다
- 범주 변수: softmax 확률분포. 생성할 때 이 분포에서 뽑는다 (argmax를 쓰면 희귀 범주가 생성되지 않음)

**뼈대를 고칠 때는 모든 모델 파일을 똑같이 고친다.** 뼈대가 달라지면 모델 비교가 공정하지 않다.

## 입력 형식

| 이름 | 형태 | 내용 |
|---|---|---|
| `X_num` | FloatTensor `[N, 연속 변수 수]` | 전처리(표준화 등)가 끝난 연속 변수 |
| `X_cat` | LongTensor `[N, 범주 변수 수]` | 범주 번호 `0 ~ k-1` |
| `categories` | `list[int]` | 범주 변수별 범주 수 `k` (이진 변수는 2) |

## 사용 예

```python
from modules.gaussian_vae import Model_VAE    # 모델 ②: from modules.gmvae import Model_GMVAE
from modules.train import train_function
from modules.utils import set_random_seed, get_device

set_random_seed(42); device = get_device()
model = Model_VAE(num_layers=2, d_numerical=X_num.shape[1], categories=categories,
                  d_token=16, n_head=1, factor=32).to(device)

config = dict(seed=42, val_frac=0.05, lr=1e-3, epochs=30, batch_size=512, patience=3)
model, history = train_function(model, X_num, X_cat, config, device)

syn_num, syn_cat = model.sample(n=len(X_num))      # 전처리 척도의 값 / 범주 번호 → 역변환은 전처리 쪽에서
```

모델 ②는 클래스만 바꾸면 된다 (`Model_GMVAE(..., n_components=50)`). 학습 함수와 생성 방법은 같다.

### 의미 기반 사전분포 (모델 ④용)

```python
from modules.semantic_prior import build_semantic_prior

E = np.load("E.npy")                          # Datasets 쪽 결과: [k, p], i번째 행 = 범주 번호 i
prior = build_semantic_prior(E, d=16, r=1.0)  # d = d_token (상품 토큰이 의미 부분 z_sem)
prior["WE"], prior["delta0"]                  # 범주별 의미 위치 [k, 16], 범주별 사전분포의 표준편차
```

| 역할 분담 | 위치 |
|---|---|
| 범주 기술문 → LLM 임베딩 → `E.npy`, 전제 검증 | Datasets (다른 저장소) |
| `E` → PCA → 사전분포 (`semantic_prior.py`), 모델 ④ | 이 저장소 `modules/` |

## 원칙: 모델 파일에는 그 모델만

모델끼리 비교하기 위해, 각 모델 파일에는 그 모델의 정의만 둔다.
- 모델 ① `gaussian_vae.py`: 순수 Gaussian VAE. 사전분포 N(0, I), 손실 = 표준 ELBO (β = 1 고정)
- 모델 ② `gmvae.py`: ①과 같은 뼈대에 사전분포(`kl`, `sample_z`)만 혼합정규. q(y|x)는 인코더 출력의 [CLS] 토큰에서 계산
- 네트워크 구조(Tokenizer, Transformer, Reconstructor)와 학습 절차(Adam, 검증 5%, 조기 종료)는 모든 모델이 같게 쓴다
- 다른 장치는 그 장치를 쓰는 모델의 파일에서만 추가한다

## 하이퍼파라미터 메모

- `d_token`: 4로는 상품(`parent_asin`, 약 10만 개)을 토큰 하나로 복원하기 어렵다 (Amazon 5% 표본 시험에서 head 상품 복원 정확도 0.30 → `d_token=16`에서 1.00).

## 진행 예정

- [x] 모델 ① Gaussian VAE (`gaussian_vae.py`, `train.py`)
- [x] 모델 ② GMVAE (`gmvae.py`)
- [x] 의미 기반 사전분포 생성 (`semantic_prior.py`)
- [ ] 모델 ③
- [ ] 모델 ④ 제안 모델 (`semantic_prior.py`의 결과를 상품 토큰의 사전분포로 사용)

## 참고 문헌

- Kingma & Welling (2014), Auto-Encoding Variational Bayes
- Dilokthanakul et al. (2016), Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders
- Gorishniy et al. (2021), Revisiting Deep Learning Models for Tabular Data (FT-Transformer)

코드 출처: `Tokenizer` ~ `Reconstructor`는 [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) (Apache-2.0)의 코드를 수정해 사용
