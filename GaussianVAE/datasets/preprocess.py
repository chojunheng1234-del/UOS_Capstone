#%%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.raw_data import load_raw_data

TIER_LABELS = ["tail", "torso", "head"]
#%%
class CustomDataset(Dataset):
    """전처리된 분할 하나 (학습 또는 홀드아웃).

    raw_data: 변환 전 값 (평가·확인용)
    X_num:    표준화한 연속 변수              [N, 연속 변수 수]
    X_cat:    범주 번호, 학습에 없는 범주 = k   [N, 범주 변수 수]
    """
    def __init__(self, data, info):
        self.raw_data = data.reset_index(drop=True)
        self.X_num, self.X_cat = encode(self.raw_data, info)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        # idx는 정수 하나 또는 인덱스 리스트 (BatchSampler로 배치 단위 조회 가능)
        return self.X_num[idx], self.X_cat[idx]


def build_datasets(config, raw=None):
    """원자료 → 학습 / 홀드아웃 분할 → 학습 데이터로만 인코딩 정보(info)를 만든다.
    raw: 이미 읽은 load_raw_data(config)의 반환값 (없으면 여기서 읽는다)
    """
    if raw is None:
        raw = load_raw_data(config)
    data, continuous_features, categorical_features, high_card_features, tier_bins, ClfTarget = raw

    # 1. 학습 / 홀드아웃 분할 (홀드아웃은 평가 기준선·프라이버시 평가에만 사용)
    rng = np.random.default_rng(config["seed"])
    is_hold = np.zeros(len(data), dtype=bool)
    is_hold[rng.choice(len(data), size=int(round(len(data) * config["holdout_frac"])), replace=False)] = True
    train, hold = data[~is_hold], data[is_hold]

    # 2. 인코딩 정보: 범주는 학습 빈도 내림차순 번호, 연속은 학습 평균·표준편차
    info = {
        "continuous_features": continuous_features,
        "categorical_features": categorical_features,
        "high_card_features": high_card_features,
        "ClfTarget": ClfTarget,
        "cat_maps": {},        # 범주 번호 → 원래 값
        "categories": [],      # 범주 변수별 범주 수 k
        "train_counts": {},    # 범주 번호별 학습 관측 수
        "tiers": {},           # 고카디널리티 변수: 범주 번호별 빈도 구간
        "stats": {},           # 연속 변수: (평균, 표준편차)
    }
    for col in categorical_features:
        cnt = train[col].value_counts()
        info["cat_maps"][col] = cnt.index.to_numpy()
        info["categories"].append(len(cnt))
        info["train_counts"][col] = cnt.to_numpy()
        if col in high_card_features:
            info["tiers"][col] = np.asarray(
                pd.cut(cnt.to_numpy(), tier_bins, right=False, labels=TIER_LABELS).astype(str))
    for col in continuous_features:
        info["stats"][col] = (float(train[col].mean()), float(train[col].std()))

    return CustomDataset(train, info), CustomDataset(hold, info), info


def encode(data, info):
    """원래 값 → (X_num, X_cat) 텐서"""
    X_num = np.stack([(data[col].to_numpy(dtype=np.float64) - m) / s
                      for col, (m, s) in info["stats"].items()], axis=1)
    X_cat = []
    for col, k in zip(info["categorical_features"], info["categories"]):
        codes = pd.Index(info["cat_maps"][col]).get_indexer(data[col])
        X_cat.append(np.where(codes < 0, k, codes))          # 학습에 없는 범주 → UNK = k
    X_cat = np.stack(X_cat, axis=1)
    return torch.tensor(X_num, dtype=torch.float32), torch.tensor(X_cat, dtype=torch.long)


def decode(X_num, X_cat, info):
    """(X_num, X_cat) → 원래 값의 DataFrame (합성 데이터 복원용)"""
    X_num, X_cat = np.asarray(X_num), np.asarray(X_cat)
    df = pd.DataFrame()
    for j, col in enumerate(info["categorical_features"]):
        df[col] = info["cat_maps"][col][X_cat[:, j]]
    for j, (col, (m, s)) in enumerate(info["stats"].items()):
        df[col] = X_num[:, j] * s + m
    return df
