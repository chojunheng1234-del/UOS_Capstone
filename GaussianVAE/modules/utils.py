#%%
import argparse
import random
from pathlib import Path

import numpy as np
import torch
#%%
def get_args(debug=False):
    """학습(main.py)과 생성(inference.py)이 같은 인자를 쓴다. 모델 이름이 인자로 정해지기 때문이다."""
    parser = argparse.ArgumentParser("parameters")

    ### default
    parser.add_argument("--seed", type=int, default=42, help="seed for repeatable results")
    parser.add_argument("--model", type=str, default="GaussianVAE")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset options: amazon")
    parser.add_argument("--data_dir", type=str, default="./data", help="Folder for raw data")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Use a fraction of rows (quick test: 0.05)")
    parser.add_argument("--holdout_frac", type=float, default=0.2, help="Proportion of the holdout split")

    ### model
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension d")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--emb_dim", type=int, default=32, help="Max embedding size of categorical variables")

    ### training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=2, help="Epochs for KL warm-up (beta 0 -> 1)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Validation split within the training data")

    ### inference
    parser.add_argument("--n_samples", type=int, default=0, help="Number of synthetic rows (0 = training size)")

    return parser.parse_args(args=[]) if debug else parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_dir(config):
    """하이퍼파라미터 조합으로 저장 폴더 이름을 만든다 (SPT 방식)."""
    name = (f"{config['dataset']}_{config['latent_dim']}_{config['hidden_dim']}_{config['emb_dim']}"
            f"_{config['lr']}_{config['batch_size']}_{config['sample_frac']}_{config['seed']}")
    model_dir = Path("./assets/models") / name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
