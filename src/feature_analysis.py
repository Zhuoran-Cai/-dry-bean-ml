"""特征分析：IQR 异常值检测、MI 排名、Pearson 热力图、PCA 可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from prep.data_loader import load_preprocess_config
from prep.data_preprocess import (
    ALL_FEATURES,
    CLASS_COL,
    clean_data,
    compute_mutual_information,
    select_features,
)
from src.plot_style import (
    BOX_COLOR,
    CLASS_COLORS,
    _save,
    _style_ax,
    plot_mi_ranking,
    plot_pearson_heatmap,
    setup_style,
)


def analyze_iqr(train: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """IQR 异常值分析（仅检测，不删除）。"""
    rows = []
    for col in ALL_FEATURES:
        s = train[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        cnt = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "feature": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_bound": lo, "upper_bound": hi,
            "outlier_count": cnt, "outlier_ratio": cnt / len(s),
        })
    return pd.DataFrame(rows)


def mi_report(train: pd.DataFrame, selected: list[str], random_state: int = 42) -> pd.DataFrame:
    """Mutual Information 特征排名报告。"""
    scores = compute_mutual_information(train, ALL_FEATURES, random_state)
    return pd.DataFrame({
        "feature": ALL_FEATURES,
        "mutual_information": scores.reindex(ALL_FEATURES).values,
        "selected": [c in selected for c in ALL_FEATURES],
    }).sort_values("mutual_information", ascending=False)


def plot_iqr_boxplots(train: pd.DataFrame, save_path: Path) -> None:
    setup_style()
    n = len(ALL_FEATURES)
    n_cols, n_rows = 7, int(np.ceil(n / 7))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    for i, col in enumerate(ALL_FEATURES):
        bp = axes.flat[i].boxplot(train[col].dropna(), patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_COLOR)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("#333333")
        axes.flat[i].set_title(col, fontsize=8)
        _style_ax(axes.flat[i])
    for j in range(n, len(axes.flat)):
        axes.flat[j].axis("off")
    fig.suptitle("IQR 箱线图（训练集）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_pca_2d(train: pd.DataFrame, features: list[str], save_path: Path, random_state: int = 42) -> pd.DataFrame:
    """PCA 二维可视化：特征筛选 → StandardScaler → PCA(n_components=2)。"""
    setup_style()
    X_scaled = StandardScaler().fit_transform(train[features].values)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    y = train[CLASS_COL].values

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(sorted(train[CLASS_COL].unique())):
        m = y == cls
        ax.scatter(coords[m, 0], coords[m, 1], label=cls, alpha=0.55, s=16,
                   color=CLASS_COLORS[i % len(CLASS_COLORS)], edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title("PCA 二维可视化（训练集）")
    ax.legend(fontsize=8, loc="best", markerscale=1.5)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, save_path)

    return pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })


def run_feature_analysis(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """运行 IQR / MI / Pearson 热力图 / PCA 分析并保存报告与图表。"""
    root = Path(project_root or Path(__file__).resolve().parent.parent)
    config = load_preprocess_config(root, Path(config_path) if config_path else None)
    data, _ = clean_data(config, root)
    train = data["train"]

    fs = config["feature_selection"]
    rs = config["pca"]["random_state"]
    selected = select_features(
        train, ALL_FEATURES, fs["mi_threshold"], fs["multicollinearity_threshold"], rs
    )

    reports = root / config["paths"]["results_dir"] / "reports"
    figures = root / config["paths"]["results_dir"] / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    iqr = analyze_iqr(train, config["iqr"]["multiplier"])
    iqr.to_csv(reports / "iqr_outlier_report.csv", index=False)

    mi_df = mi_report(train, selected, rs)
    mi_df.to_csv(reports / "mi_feature_report.csv", index=False)
    plot_mi_ranking(mi_df, figures / "mi_feature_ranking.png")

    plot_pearson_heatmap(train, ALL_FEATURES, figures / "pearson_heatmap.png")

    pca = plot_pca_2d(train, selected, figures / "pca_2d_visualization.png", rs)
    pca.to_csv(reports / "pca_variance_report.csv", index=False)
    plot_iqr_boxplots(train, figures / "iqr_boxplots.png")

    print(f"分析完成 | 报告: {reports} | 图表: {figures}")
    print(f"筛选特征({len(selected)}): {selected}")
    return {"selected_features": selected, "iqr_report": iqr, "mi_report": mi_df, "pca_report": pca}


if __name__ == "__main__":
    run_feature_analysis()
