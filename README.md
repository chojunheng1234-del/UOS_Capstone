# UOS Capstone — 고카디널리티 정형 데이터 합성

2026-2 서울시립대학교 통계자료분석캡스톤디자인.
코드 구조는 [SPT](https://github.com/hoon-ii/SPT)를 참고했다: **모델 하나 = 폴더 하나**, 각 폴더에서 `main.py`(학습) → `inference.py`(생성·평가) 순으로 실행한다.

## 구조

```
UOS_Capstone/
├── README.md
├── requirements.txt
└── GaussianVAE/                 모델 ① Gaussian VAE  (현재 Amazon 데이터)
    ├── exam.py                  셀(#%%) 단위 실습용: 데이터 → 전처리 → VAE 정의 → 학습 → 생성 → 시각화·평가 → β 스케줄링
    ├── main.py                  학습: 데이터 → 모델 → 학습 → 저장
    ├── inference.py             생성 → 원래 값으로 복원 → 평가
    ├── datasets/
    │   ├── raw_data.py          load_raw_data(config): 원자료 읽기 + 변수 만들기 (데이터셋별 분기)
    │   └── preprocess.py        build_datasets(config): 학습/홀드아웃 분할, 범주 번호, 표준화, 역변환
    ├── modules/
    │   ├── model.py             MixedVAE(공통 인코더·디코더) + GaussianVAE(사전분포 N(0, I))
    │   ├── train.py             KL warm-up, 검증 손실 기준 조기 종료
    │   └── utils.py             인자(get_args), 시드, 장치, 저장 폴더 이름
    └── evaluation/
        └── metrics.py           주변 분포 + 빈도 구간(tail/torso/head)별 조건부 분포 평가
```

실행하면 다음 폴더가 생긴다 (GitHub에는 올리지 않음).
- `data/`: 원자료 (Amazon은 자동으로 내려받음)
- `assets/models/<설정 이름>/`: `model.pth`, `history.csv`, `config.json`, `metrics.csv`
- `syn_data/`: 합성 데이터 CSV

## 실행

```bash
cd GaussianVAE
python main.py                       # 학습 (전체 데이터, 최대 30 epoch)
python inference.py                  # 생성 + 평가
```

빠른 시험 (5% 표본, 3 epoch):
```bash
python main.py --sample_frac 0.05 --epochs 3
python inference.py --sample_frac 0.05 --epochs 3
```

`inference.py`에는 **학습 때와 같은 인자**를 넘겨야 한다. 저장 폴더 이름이 인자로 정해지기 때문이다 (SPT와 같은 방식).

### Colab

```python
!git clone https://github.com/chojunheng1234-del/UOS_Capstone.git
%cd UOS_Capstone/GaussianVAE
!python main.py
!python inference.py
```
런타임 → 런타임 유형 변경 → T4 GPU 권장.

## 데이터: Amazon Reviews 2023 (All_Beauty)

| 변수 | 원천 | 변환 | 형태 |
|---|---|---|---|
| `parent_asin` | 상품 ID | 학습 빈도 내림차순 번호, 학습에 없으면 UNK | 고카디널리티 범주 (102,135개) |
| `rating` | 평점 | 1~5 | 범주 |
| `verified`, `has_image` | 구매 인증, 이미지 첨부 | 0/1 | 범주 |
| `log_votes`, `log_len`, `year` | 유용성 투표, 본문 길이, 작성 시점 | log(1+x) 등 → 표준화 | 연속 |

학습 80% / 홀드아웃 20%로 나눈다. 홀드아웃은 학습에 쓰지 않고, 평가 기준선과 프라이버시 평가에만 쓴다.

## 평가 지표 (`evaluation/metrics.py`)

| 지표 | 의미 |
|---|---|
| `<범주>_TVD`, `<연속>_W1` | 주변 분포 차이 |
| `GR_<구간>` | 해당 빈도 구간 범주의 생성 비율 ÷ 실제 비율 (1에 가까울수록 좋음) |
| `coverage_<구간>` | 한 번 이상 생성된 범주의 비율 |
| `cond_TVD_<변수>_<구간>` | 상품별 조건부 분포 차이 (학습 관측 수로 가중평균) |
| `cond_mean_err_<구간>` | 상품별 연속 변수 평균 차이 |

`real_holdout (기준)` 열은 홀드아웃을 학습 데이터와 비교한 값이다. 실제 데이터끼리도 생기는 자연스러운 차이를 뜻한다.

## 진행 예정

- [x] 모델 ① Gaussian VAE (Amazon)
- [ ] 모델 ② GMVAE: `MixedVAE`를 상속하고 사전분포만 바꾼다
- [ ] 모델 ③ TabSyn, ④ 제안 모델
- [ ] Medicare 데이터: `raw_data.py`에 분기 추가
- [ ] 프라이버시 지표 (DCR)
