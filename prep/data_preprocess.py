"""干豆数据集：数据清理与特征工程。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from prep.data_loader import load_preprocess_config, load_raw_datasets

CLASS_COL = "Class"
BASE_FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
ENGINEERED_FEATURES = [
    "AxisRatio", "AreaPerimeterRatio", "ConvexAreaRatio",
    "EquivDiameterRatio", "RoundnessCompactness",
]
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def clean_labels(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """标签标准化：去空格、转大写、OCR 修正（0→O, 3→E）。"""
    out = df.copy()
    labels = out[CLASS_COL].astype(str).str.strip().str.upper()
    for old, new in config["label"]["ocr_replace"].items():
        labels = labels.str.replace(old, new, regex=False)
    valid = set(config["label"]["valid_classes"])
    if unknown := set(labels.unique()) - valid:
        raise ValueError(f"存在无法映射的标签: {sorted(unknown)}")
    out[CLASS_COL] = labels
    return out


def clean_numeric(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """数值字符串清洗 + 非法值转缺失值。"""
    out = df.copy()
    suffix = config["cleaning"]["numeric_suffix_strip"]
    for col in [c for c in out.columns if c != CLASS_COL]:
        s = out[col].astype(str).str.strip()
        if suffix:
            s = s.str.replace(suffix, "", regex=False)
        out[col] = pd.to_numeric(s, errors="coerce")
    return out


def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 5 个形态比例特征。"""
    out = df.copy()

    def ratio(a: str, b: str) -> pd.Series:
        return (out[a] / out[b].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    out["AxisRatio"] = ratio("MajorAxisLength", "MinorAxisLength")
    out["AreaPerimeterRatio"] = ratio("Area", "Perimeter")
    out["ConvexAreaRatio"] = ratio("ConvexArea", "Area")
    out["EquivDiameterRatio"] = ratio("EquivDiameter", "MajorAxisLength")
    out["RoundnessCompactness"] = out["roundness"] * out["Compactness"]
    return out


def _fit_transform_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    transformer,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train[cols] = transformer.fit_transform(train[cols])
    val[cols] = transformer.transform(val[cols])
    test[cols] = transformer.transform(test[cols])
    return train, val, test


def compute_mutual_information(
    train: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> pd.Series:
    """计算各特征与类别的互信息得分。"""
    y = LabelEncoder().fit_transform(train[CLASS_COL])
    scores = mutual_info_classif(
        train[feature_cols], y, random_state=random_state, discrete_features=False
    )
    return pd.Series(scores, index=feature_cols)


def select_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    mi_threshold: float,
    multi_corr: float,
    random_state: int = 42,
) -> list[str]:
    """互信息初筛 + Pearson 多重共线性去冗余。"""
    mi_scores = compute_mutual_information(train, feature_cols, random_state)
    candidates = mi_scores[mi_scores >= mi_threshold].index.tolist()
    if not candidates:
        candidates = mi_scores.nlargest(5).index.tolist()

    selected = candidates.copy()
    corr_mat = train[candidates].corr().abs()
    for i, col_i in enumerate(candidates):
        if col_i not in selected:
            continue
        for col_j in candidates[i + 1:]:
            if col_j not in selected:
                continue
            if corr_mat.loc[col_i, col_j] >= multi_corr:
                drop = col_j if mi_scores[col_i] >= mi_scores[col_j] else col_i
                if drop in selected:
                    selected.remove(drop)
    return selected


def clean_data(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, pd.DataFrame], int]:
    """执行清洗至中位数填补（步骤 1~6），供预处理与分析复用。"""
    data = load_raw_datasets(project_root, config)
    for key in data:
        data[key] = clean_labels(data[key], config)
        data[key] = clean_numeric(data[key], config)
        data[key] = build_ratio_features(data[key])

    before = len(data["train"])
    data["train"] = data["train"].drop_duplicates().reset_index(drop=True)
    dup_removed = before - len(data["train"])

    imputer = SimpleImputer(strategy="median")
    data["train"], data["val"], data["test"] = _fit_transform_splits(
        data["train"], data["val"], data["test"], ALL_FEATURES, imputer
    )
    return data, dup_removed


def run_preprocess(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """执行完整预处理：清洗 → 特征选择 → 标准化 → 保存。"""
    root = Path(project_root or Path(__file__).resolve().parent.parent)
    config = load_preprocess_config(root, Path(config_path) if config_path else None)

    data, dup_removed = clean_data(config, root)
    fs = config["feature_selection"]
    rs = config.get("pca", {}).get("random_state", 42)
    selected = select_features(
        data["train"],
        ALL_FEATURES,
        fs["mi_threshold"],
        fs["multicollinearity_threshold"],
        rs,
    )

    scaler = StandardScaler()
    data["train"], data["val"], data["test"] = _fit_transform_splits(
        data["train"], data["val"], data["test"], selected, scaler
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(config["label"]["valid_classes"])

    out_dir = root / config["paths"]["results_dir"] / "processed"
    reports_dir = root / config["paths"]["results_dir"] / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df[selected + [CLASS_COL]].to_csv(out_dir / f"{name}_processed.csv", index=False)

    summary = {
        "train_samples": len(data["train"]),
        "val_samples": len(data["val"]),
        "test_samples": len(data["test"]),
        "duplicates_removed": dup_removed,
        "total_features_before_selection": len(ALL_FEATURES),
        "selected_features": selected,
        "class_names": list(label_encoder.classes_),
    }
    with open(reports_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"预处理完成 | train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")
    print(f"删除重复行: {dup_removed} | 选中特征({len(selected)}): {selected}")

    return {
        "X_train": data["train"][selected].values,
        "y_train": label_encoder.transform(data["train"][CLASS_COL]),
        "X_val": data["val"][selected].values,
        "y_val": label_encoder.transform(data["val"][CLASS_COL]),
        "X_test": data["test"][selected].values,
        "y_test": label_encoder.transform(data["test"][CLASS_COL]),
        "feature_names": selected,
        "class_names": list(label_encoder.classes_),
        "label_encoder": label_encoder,
    }


if __name__ == "__main__":
    run_preprocess()
