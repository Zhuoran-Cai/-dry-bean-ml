"""数据加载模块：读取配置文件与原始/预处理后数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

CLASS_COL = "Class"


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_preprocess_config(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or project_root / "config" / "preprocess.yaml"
    return load_config(path)


def load_train_config(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or project_root / "config" / "train.yaml"
    return load_config(path)


def load_raw_datasets(project_root: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """从 data/ 加载原始 train/val/test CSV。"""
    data_dir = project_root / config["paths"]["data_dir"]
    return {
        "train": pd.read_csv(data_dir / config["paths"]["train_file"]),
        "val": pd.read_csv(data_dir / config["paths"]["val_file"]),
        "test": pd.read_csv(data_dir / config["paths"]["test_file"]),
    }


def load_processed_data(project_root: Path, train_config: dict[str, Any]) -> dict[str, Any]:
    """从 results/processed/ 加载预处理后的 train/val/test 数据。"""
    proc_dir = project_root / train_config["paths"]["processed_dir"]
    splits: dict[str, tuple] = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(proc_dir / f"{name}_processed.csv")
        feature_cols = [c for c in df.columns if c != CLASS_COL]
        splits[name] = (df[feature_cols].values, df[CLASS_COL].values, feature_cols)

    label_encoder = LabelEncoder()
    label_encoder.fit(splits["train"][1])

    return {
        "X_train": splits["train"][0],
        "y_train": label_encoder.transform(splits["train"][1]),
        "X_val": splits["val"][0],
        "y_val": label_encoder.transform(splits["val"][1]),
        "X_test": splits["test"][0],
        "y_test": label_encoder.transform(splits["test"][1]),
        "feature_names": splits["train"][2],
        "class_names": list(label_encoder.classes_),
        "label_encoder": label_encoder,
    }
