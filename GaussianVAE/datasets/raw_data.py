#%%
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

AMAZON_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
#%%
def load_raw_data(config):
    """원자료를 읽어 모델에 넣을 변수를 만든다. 데이터셋마다 분기를 추가한다 (SPT 방식).

    반환: data, continuous_features, categorical_features, high_card_features, tier_bins, ClfTarget
      - categorical_features: 모든 범주형 변수 (이진 변수 포함)
      - high_card_features:   그중 고카디널리티 변수. 빈도 구간(tier)별 평가 대상
      - tier_bins:            학습 관측 수 기준 tail / torso / head 경계
    """
    data_dir = Path(config["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    if config["dataset"] == "amazon":
        path = data_dir / "All_Beauty.jsonl.gz"
        if not path.exists():
            print("원자료 내려받는 중 (약 94MB)...")
            urllib.request.urlretrieve(AMAZON_URL, path)

        # user_id는 읽지 않고, 리뷰 본문은 글자 수만, 이미지는 개수만 남긴다
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["parent_asin"], r["rating"], len(r.get("text") or ""), r["helpful_vote"],
                             r["verified_purchase"], r["timestamp"], len(r.get("images") or [])))
        raw = pd.DataFrame(rows, columns=["parent_asin", "rating", "text_len", "helpful_vote",
                                          "verified_purchase", "timestamp", "n_images"])

        ts = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
        data = pd.DataFrame({
            "parent_asin": raw["parent_asin"].astype(str),
            "rating":      raw["rating"].astype(int),
            "verified":    raw["verified_purchase"].astype(bool).astype(int),
            "has_image":   (raw["n_images"] > 0).astype(int),
            "log_votes":   np.log1p(raw["helpful_vote"].clip(lower=0)),
            "log_len":     np.log1p(raw["text_len"]),
            "year":        ts.dt.year + (ts.dt.dayofyear - 1) / 365.25,
        })
        assert data.isna().sum().sum() == 0

        continuous_features = ["log_votes", "log_len", "year"]
        categorical_features = ["parent_asin", "rating", "verified", "has_image"]
        high_card_features = ["parent_asin"]
        tier_bins = [0, 5, 50, np.inf]          # 학습 관측 수: tail <5, torso 5~49, head ≥50
        ClfTarget = "rating"

    else:
        raise ValueError(f"지원하지 않는 데이터셋: {config['dataset']}")

    if config["sample_frac"] < 1.0:             # 빠른 시험용
        data = data.sample(frac=config["sample_frac"], random_state=config["seed"]).reset_index(drop=True)

    return data, continuous_features, categorical_features, high_card_features, tier_bins, ClfTarget
