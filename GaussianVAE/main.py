#%%
import json

import torch

from datasets.preprocess import build_datasets
from modules.model import GaussianVAE
from modules.train import train_function
from modules.utils import get_args, set_random_seed, get_device, get_model_dir
#%%
def main():
    """ 0. Reproducibility """
    config = vars(get_args(debug=False))
    set_random_seed(config["seed"])
    device = get_device()
    print("Current device is", device)

    """ 1. Load and preprocess the data """
    train_dataset, holdout_dataset, info = build_datasets(config)
    print(f"학습 {len(train_dataset):,}행 | 홀드아웃 {len(holdout_dataset):,}행")
    for col, k in zip(info["categorical_features"], info["categories"]):
        print(f"  {col}: 범주 {k:,}개")

    """ 2. Build the model """
    model = GaussianVAE(
        d_numerical=len(info["continuous_features"]),
        categories=info["categories"],
        d_latent=config["latent_dim"],
        d_hidden=config["hidden_dim"],
        d_emb=config["emb_dim"],
    ).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    """ 3. Train """
    model, history = train_function(model, train_dataset, config, device)

    """ 4. Save """
    model_dir = get_model_dir(config)
    torch.save(model.state_dict(), model_dir / "model.pth")
    history.to_csv(model_dir / "history.csv", index=False)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print("저장:", model_dir)
#%%
if __name__ == "__main__":
    main()
