"""静态结果数据加载。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "figures"
EXPERIMENTS = RESULTS / "experiments"
PROCESSED = RESULTS / "processed"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_preprocess_summary() -> dict:
    return load_json(REPORTS / "preprocess_summary.json")


def load_experiment_summary() -> dict:
    return load_json(EXPERIMENTS / "experiment_summary.json")


def load_accuracy_comparison() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS / "accuracy_comparison.csv")


def load_inference_speed() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS / "inference_speed.csv")


def load_overfitting() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS / "overfitting.csv")


def load_robustness() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS / "robustness.csv")


def load_iqr_report() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "iqr_outlier_report.csv")


def load_mi_report() -> pd.DataFrame:
    """Mutual Information 特征排名报告。"""
    return pd.read_csv(REPORTS / "mi_feature_report.csv")


def load_pca_report() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "pca_variance_report.csv")


def load_class_distribution() -> pd.Series:
    train = pd.read_csv(PROCESSED / "train_processed.csv")
    return train["Class"].value_counts().sort_index()


def figure_path(name: str) -> Path | None:
    path = FIGURES / name
    return path if path.exists() else None
