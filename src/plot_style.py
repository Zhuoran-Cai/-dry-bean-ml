"""实验图表统一样式与绑图函数。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 四算法统一色系
ALGO_COLORS = {
    "logistic_regression": "#4A7BB7",
    "knn": "#5CB8A8",
    "svm": "#E07A5F",
    "random_forest": "#8B6BB8",
}
ALGO_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
}
TRAIN_COLOR = "#4A7BB7"
TEST_COLOR = "#E07A5F"
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E0E0E0"
# 7 个豆类别在 PCA 等图中的配色
CLASS_COLORS = ["#4A7BB7", "#5CB8A8", "#E07A5F", "#8B6BB8", "#F2B950", "#6B9E3E", "#C44E52"]
BOX_COLOR = "#4A7BB7"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#333333",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#DDDDDD",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both")


def plot_pearson_heatmap(train: pd.DataFrame, feature_cols: list[str], save_path: Path) -> None:
    """特征间 Pearson 相关性热力图。"""
    setup_style()
    corr = train[feature_cols].corr()
    n = len(feature_cols)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(feature_cols, fontsize=7)
    ax.set_title("特征间 Pearson 相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, save_path)


def plot_mi_ranking(mi_df: pd.DataFrame, save_path: Path) -> None:
    """Mutual Information 特征排名条形图。"""
    setup_style()
    ranked = mi_df.sort_values("mutual_information", ascending=True)
    colors = [BOX_COLOR if sel else "#CCCCCC" for sel in ranked["selected"]]
    fig, ax = plt.subplots(figsize=(9, max(5, len(ranked) * 0.32)))
    ax.barh(ranked["feature"], ranked["mutual_information"], color=colors, edgecolor="white")
    ax.set_xlabel("Mutual Information")
    ax.set_title("Mutual Information 特征排名（与类别）")
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, save_path)


def plot_accuracy_comparison(results: list[dict], save_path: Path) -> None:
    """测试集精度对比 — 横向条形图。"""
    setup_style()
    sorted_res = sorted(results, key=lambda r: r["test_acc"])
    names = [r["name"] for r in sorted_res]
    accs = [r["test_acc"] for r in sorted_res]
    colors = [ALGO_COLORS[r["key"]] for r in sorted_res]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, accs, color=colors, height=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("测试集准确率")
    ax.set_title("测试集精度对比")
    ax.set_xlim(0, 1.02)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va="center", fontsize=10, color="#333333")
    _style_ax(ax)
    _save(fig, save_path)


def plot_loss_curves(results: list[dict], save_path: Path) -> None:
    """Loss 曲线对比 — 折线图。"""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, res in zip(axes.flatten(), results):
        c = res["curve"]
        color = ALGO_COLORS[res["key"]]
        ax.plot(c["steps"], c["train"], color=color, linewidth=2,
                marker="o", markersize=4, label="Train")
        ax.plot(c["steps"], c["val"], color=color, linewidth=2,
                marker="s", markersize=4, linestyle="--", alpha=0.85, label="Val")
        subtitle = "" if res["has_loss_curve"] else "（参数扫描）"
        ax.set_title(f"{res['name']}{subtitle}", fontsize=11)
        ax.set_xlabel(c["xlabel"])
        ax.set_ylabel(c["ylabel"])
        ax.legend(fontsize=9)
        _style_ax(ax)
    fig.suptitle("Loss 曲线对比", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_inference_speed(results: list[dict], save_path: Path) -> None:
    """推理速度对比 — 横向条形图。"""
    setup_style()
    sorted_res = sorted(results, key=lambda r: r["inference_ms"])
    names = [r["name"] for r in sorted_res]
    times = [r["inference_ms"] for r in sorted_res]
    colors = [ALGO_COLORS[r["key"]] for r in sorted_res]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, times, color=colors, height=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("单样本推理时间 (ms/样本)")
    ax.set_title("算法推理速度对比（逐样本 predict）")
    for bar, t in zip(bars, times):
        ax.text(t + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.4f}", va="center", fontsize=10, color="#333333")
    _style_ax(ax)
    _save(fig, save_path)


def plot_robustness(robust_df: pd.DataFrame, save_path: Path) -> None:
    """鲁棒性对比 — 多折线图。"""
    setup_style()
    noise_labels = {
        "gaussian": "高斯噪声",
        "uniform": "均匀噪声",
        "salt_pepper": "椒盐噪声",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, noise_type in zip(axes, robust_df["noise_type"].unique()):
        sub = robust_df[robust_df["noise_type"] == noise_type]
        for algo in sub["algorithm"].unique():
            s = sub[sub["algorithm"] == algo].sort_values("noise_level")
            ax.plot(s["noise_level"], s["test_acc"],
                    color=ALGO_COLORS[algo], linewidth=2, marker="o", markersize=5,
                    label=ALGO_LABELS[algo])
        ax.set_title(noise_labels.get(noise_type, noise_type))
        ax.set_xlabel("噪声强度")
        ax.set_ylabel("测试集准确率")
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=8, loc="lower left")
        _style_ax(ax)
    fig.suptitle("鲁棒性对比", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_overfitting(results: list[dict], save_path: Path) -> None:
    """过拟合分析 — Train/Test 分组柱状图。"""
    setup_style()
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [r["train_acc"] for r in results], width,
           label="Train", color=TRAIN_COLOR, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, [r["test_acc"] for r in results], width,
           label="Test", color=TEST_COLOR, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("准确率")
    ax.set_title("过拟合分析（Train / Test）")
    ax.set_ylim(0, 1.08)
    ax.legend()
    for i, r in enumerate(results):
        ax.text(i - width / 2, r["train_acc"] + 0.01, f"{r['train_acc']:.3f}",
                ha="center", fontsize=8, color=TRAIN_COLOR)
        ax.text(i + width / 2, r["test_acc"] + 0.01, f"{r['test_acc']:.3f}",
                ha="center", fontsize=8, color=TEST_COLOR)
    _style_ax(ax)
    _save(fig, save_path)


def plot_rf_feature_importance(model, feature_names: list[str], save_path: Path) -> None:
    """课堂外算法分析 — Random Forest 特征重要性图。"""
    setup_style()
    importances = model.feature_importances_
    idx = np.argsort(importances)
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    n = len(values)
    # 重要性由低到高排列；颜色由浅紫到深紫，顶部（最重要）最深
    purple_gradient = plt.cm.Purples(np.linspace(0.35, 0.92, n))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values, color=purple_gradient, height=0.6, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("特征重要性")
    ax.set_title("Random Forest 特征重要性")
    for bar, v in zip(ax.patches, values):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9, color="#333333")
    _style_ax(ax)
    _save(fig, save_path)


def plot_confusion_matrices(
    results: list[dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """分类错误分析 — 混淆矩阵热力图。"""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for ax, res in zip(axes.flatten(), results):
        y_pred = res["model"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{res['name']}", fontsize=12, fontweight="bold")
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

    fig.suptitle("分类错误分析 — 混淆矩阵", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
