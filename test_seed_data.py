from ntd.datasets import SEED_IV_DIF

import types
import yaml
import numpy as np


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
config = load_config("/home/ravindu/fyp/timevqvae-impl/configs/config_eeg.yaml")
args = {
    "data_dir": "/home/ravindu/fyp/seed_iv_h5",
    "dataset_name": "seed_iv",
    "in_channels": 12,
    "sequence_length": 800,
    "n_classes": 4,
    "mask_ratio": 0.5,
    "batch_size": 8,
    "num_workers": 2,
    "devices": 1,
    "save_dir": "saved_models/emotion_transfer",
    "wandb_project": "emotion-transfer",
    "run_name": "emotion-transfer-mask0.5",
    "val_check_interval": 0.5,
    "log_every_n_steps": 50,
    "seed": 42,
    "config":config,
    }   
args = types.SimpleNamespace(**args)


data = SEED_IV_DIF(root_dir="/home/ravindu/fyp/seed_iv_h5", sequence_length=800)

item = data[0]
signal = item["signal"]
try:
    print(signal.shape)
except AttributeError:
    print(np.array(signal).shape)