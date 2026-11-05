#%%
from pathlib import Path

import pandas as pd
import torch

from datasets.preprocess import build_datasets, decode
from evaluation.metrics import evaluate
from modules.model import GaussianVAE
from modules.utils import get_args, set_random_seed, get_device, get_model_dir
#%%
def main():
    """ 0. Reproducibility """
    config = vars(get_args(debug=False))      # 학습 때와 같은 인자를 넘겨야 같은 모델을 찾는다
    set_random_seed(config["seed"])
    device = get_device()
    model_dir = get_model_dir(config)

    """ 1. Load and preprocess the data (학습 때와 같은 시드 → 같은 분할·인코딩) """
    train_dataset, holdout_dataset, info = build_datasets(config)

    """ 2. Load the trained model """
    model = GaussianVAE(
        d_numerical=len(info["continuous_features"]),
        categories=info["categories"],
        d_latent=config["latent_dim"],
        d_hidden=config["hidden_dim"],
        d_emb=config["emb_dim"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.eval()

    """ 3. Synthetic data generation """
    n = config["n_samples"] or len(train_dataset)
    syn_num, syn_cat = model.sample(n)
    syn_df = decode(syn_num, syn_cat, info)
    Path("./syn_data").mkdir(exist_ok=True)
    syn_path = Path("./syn_data") / f"{model_dir.name}.csv"
    syn_df.to_csv(syn_path, index=False)
    print(f"합성 데이터 {len(syn_df):,}행 저장: {syn_path}")
    print(syn_df.head())

    """ 4. Evaluation (실제 = 학습 데이터) """
    real = (train_dataset.X_num.numpy(), train_dataset.X_cat.numpy())
    # 기준선: 홀드아웃. 학습에 없는 범주(UNK)가 있는 행은 비교할 수 없어 제외한다
    hold_num, hold_cat = holdout_dataset.X_num.numpy(), holdout_dataset.X_cat.numpy()
    known = (hold_cat < torch.tensor(info["categories"]).numpy()).all(axis=1)
    results = {
        "real_holdout (기준)": evaluate(real, (hold_num[known], hold_cat[known]), info),
        config["model"]: evaluate(real, (syn_num.numpy(), syn_cat.numpy()), info),
    }
    res = pd.DataFrame(results)
    res.to_csv(model_dir / "metrics.csv")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string())
    print("저장:", model_dir / "metrics.csv")
#%%
if __name__ == "__main__":
    main()
