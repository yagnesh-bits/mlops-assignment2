"""
scripts/run_preprocess.py
DVC stage: Split raw data into train/val/test.
"""
import yaml
from pathlib import Path
from src.data.preprocess import split_dataset

params = yaml.safe_load(open("params.yaml"))["data"]

n_train, n_val, n_test = split_dataset(
    raw_dir="data/raw",
    processed_dir="data/processed",
    train_ratio=params["train_ratio"],
    val_ratio=params["val_ratio"],
    seed=params["seed"],
)
print(f"Split complete → train:{n_train}  val:{n_val}  test:{n_test}")
