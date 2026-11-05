#%%
import numpy as np

TIER_LABELS = ["tail", "torso", "head"]
#%%
def tvd(a, b, k):
    """두 범주 표본의 총변동거리"""
    pa = np.bincount(a, minlength=k)[:k] / len(a)
    pb = np.bincount(b, minlength=k)[:k] / len(b)
    return 0.5 * float(np.abs(pa - pb).sum())


def w1(a, b, q=np.linspace(0.0005, 0.9995, 1000)):
    """1차 Wasserstein 거리 (분위수 근사)"""
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def evaluate(real, synthetic, info):
    """실제(학습) 데이터와 비교 대상의 차이. real, synthetic = (X_num, X_cat) numpy 배열.

    - 주변 분포:  범주 변수 TVD, 연속 변수 W1 (표준화 척도)
    - 고카디널리티 변수의 빈도 구간별:
        GR             생성 비율 / 실제 비율 (1에 가까울수록 좋음)
        coverage       한 번 이상 생성된 범주의 비율
        cond_TVD_<변수> 범주별 조건부 분포의 TVD (학습 관측 수로 가중평균)
        cond_mean_err  범주별 연속 변수 평균의 절대 오차 (학습 관측 수로 가중평균)
    """
    (r_num, r_cat), (s_num, s_cat) = real, synthetic
    cont, cats = info["continuous_features"], info["categorical_features"]
    results = {}

    for j, col in enumerate(cats):
        if col not in info["high_card_features"]:
            results[f"{col}_TVD"] = tvd(r_cat[:, j], s_cat[:, j], info["categories"][j])
    for j, col in enumerate(cont):
        results[f"{col}_W1"] = w1(r_num[:, j], s_num[:, j])

    for col in info["high_card_features"]:
        h = cats.index(col); k = info["categories"][h]
        tiers = info["tiers"][col]
        r_cnt = np.bincount(r_cat[:, h], minlength=k + 1)[:k]
        s_cnt = np.bincount(s_cat[:, h], minlength=k + 1)[:k]

        # 범주별 연속 변수 평균
        r_mean = np.stack([np.bincount(r_cat[:, h], weights=r_num[:, j], minlength=k + 1)[:k]
                           for j in range(len(cont))], 1) / np.maximum(r_cnt, 1)[:, None]
        s_mean = np.stack([np.bincount(s_cat[:, h], weights=s_num[:, j], minlength=k + 1)[:k]
                           for j in range(len(cont))], 1) / np.maximum(s_cnt, 1)[:, None]
        mean_err = np.abs(r_mean - s_mean).mean(axis=1)

        # 범주별 다른 범주 변수의 조건부 분포
        cond_tvd = {}
        for j, other in enumerate(cats):
            if other in info["high_card_features"]:
                continue
            kv = info["categories"][j]
            r_tab = np.bincount(r_cat[:, h] * kv + r_cat[:, j], minlength=(k + 1) * kv)[:k * kv].reshape(k, kv)
            s_tab = np.bincount(s_cat[:, h] * kv + s_cat[:, j], minlength=(k + 1) * kv)[:k * kv].reshape(k, kv)
            cond_tvd[other] = 0.5 * np.abs(r_tab / np.maximum(r_cnt, 1)[:, None]
                                           - s_tab / np.maximum(s_cnt, 1)[:, None]).sum(axis=1)

        for g in TIER_LABELS:
            in_g = tiers == g
            results[f"GR_{g}"] = float((s_cnt[in_g].sum() / s_cnt.sum()) / (r_cnt[in_g].sum() / r_cnt.sum()))
            results[f"coverage_{g}"] = float((s_cnt[in_g] > 0).mean())
            both = in_g & (r_cnt > 0) & (s_cnt > 0)          # 양쪽에 모두 있는 범주만 비교
            w = r_cnt[both]
            for other, v in cond_tvd.items():
                results[f"cond_TVD_{other}_{g}"] = float((v[both] * w).sum() / w.sum())
            results[f"cond_mean_err_{g}"] = float((mean_err[both] * w).sum() / w.sum())

    return results
