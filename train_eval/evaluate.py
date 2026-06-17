"""测试评估模块：测试集评估、鲁棒性实验、对比分析与图表输出。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import knn, logistic_regression, random_forest, svm
from src.inference import inference_time_ms
from src.plot_style import (
    plot_accuracy_comparison,
    plot_confusion_matrices,
    plot_inference_speed,
    plot_loss_curves,
    plot_overfitting,
    plot_rf_feature_importance,
    plot_robustness,
)
from train_eval.train import ALGORITHMS

MODULES = {
    "logistic_regression": logistic_regression,
    "knn": knn,
    "svm": svm,
    "random_forest": random_forest,
}


def add_noise(X: np.ndarray, noise_type: str, level: float, seed: int) -> np.ndarray:
    """对 StandardScaler 标准化后的特征施加噪声；level 为单位标准差倍数。"""
    rng = np.random.default_rng(seed)
    Xn = X.copy()

    if noise_type == "gaussian":
        Xn += rng.normal(0, level, X.shape)
    elif noise_type == "uniform":
        Xn += rng.uniform(-level, level, X.shape)
    elif noise_type == "salt_pepper":
        mask = rng.random(X.shape) < level
        col_min, col_max = X.min(axis=0), X.max(axis=0)
        salt = rng.random(X.shape) < 0.5
        Xn[mask & salt] = col_max[np.where(mask & salt)[1]]
        Xn[mask & ~salt] = col_min[np.where(mask & ~salt)[1]]
    else:
        raise ValueError(f"未知噪声类型: {noise_type}")
    return Xn


def load_trained_results(data: dict, config: dict, project_root: Path) -> list[dict]:
    """从已保存模型加载，重建测试结果。"""
    model_dir = project_root / config["paths"]["models_dir"]
    experiment = config["experiment"]
    rs = config["experiment"]["random_state"]
    results = []

    for key, mod in ALGORITHMS:
        model = joblib.load(model_dir / f"{key}.pkl")
        params = config[key]
        curve_fn = {
            "logistic_regression": lambda: mod.loss_curve(
                data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, rs
            ),
            "knn": lambda: mod.k_selection_curve(
                data["X_train"], data["y_train"], data["X_val"], data["y_val"], params
            ),
            "svm": lambda: mod.c_parameter_curve(
                data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, rs
            ),
            "random_forest": lambda: mod.staged_loss_curve(
                data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, rs
            ),
        }[key]

        train_acc = (
            mod.train_accuracy_exclude_self(
                model, data["X_train"], data["y_train"], params["n_neighbors"]
            )
            if key == "knn"
            else mod.accuracy(model, data["X_train"], data["y_train"])
        )

        results.append({
            "key": mod.KEY,
            "name": mod.NAME,
            "model": model,
            "train_acc": train_acc,
            "val_acc": mod.accuracy(model, data["X_val"], data["y_val"]),
            "test_acc": mod.accuracy(model, data["X_test"], data["y_test"]),
            "inference_ms": inference_time_ms(model, data["X_test"], experiment),
            "curve": curve_fn(),
            "has_loss_curve": key != "knn",
        })
    return results


def run_evaluate(
    data: dict,
    config: dict,
    project_root: Path,
    results: list[dict] | None = None,
) -> dict:
    """在测试集上评估模型，输出对比报告与图表。"""
    root = Path(project_root)
    exp_dir = root / config["paths"]["experiment_dir"]
    fig_dir = root / config["paths"]["figures_dir"]
    for d in (exp_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    if results is None:
        print("加载已训练模型...")
        results = load_trained_results(data, config, root)

    print("=" * 50)
    print("测试与实验分析")
    print("=" * 50)

    baseline = {r["key"]: r["test_acc"] for r in results}

    pd.DataFrame([{
        "algorithm": r["key"], "name": r["name"],
        "train_acc": r["train_acc"], "val_acc": r["val_acc"], "test_acc": r["test_acc"],
    } for r in results]).to_csv(exp_dir / "accuracy_comparison.csv", index=False)
    plot_accuracy_comparison(results, fig_dir / "accuracy_comparison.png")

    plot_loss_curves(results, fig_dir / "loss_curves.png")

    pd.DataFrame([{
        "algorithm": r["key"], "name": r["name"],
        "inference_ms_per_sample": r["inference_ms"],
    } for r in results]).to_csv(exp_dir / "inference_speed.csv", index=False)
    plot_inference_speed(results, fig_dir / "inference_speed.png")

    print("\n鲁棒性实验（测试集注入噪声，评估已训练模型）...")
    rb = config["robustness"]
    rs = config["experiment"]["random_state"]
    robust_rows = []
    for ai, (key, mod) in enumerate(ALGORITHMS):
        model = next(r["model"] for r in results if r["key"] == key)
        for ni, noise_type in enumerate(rb["noise_types"]):
            levels = rb.get("salt_pepper_levels", rb["noise_levels"]) if noise_type == "salt_pepper" else rb["noise_levels"]
            for li, level in enumerate(levels):
                seed = rs + ai * 100 + ni * 10 + li
                X_test_noisy = add_noise(data["X_test"], noise_type, level, seed)
                test_acc = mod.accuracy(model, X_test_noisy, data["y_test"])
                robust_rows.append({
                    "algorithm": key,
                    "noise_type": noise_type,
                    "noise_level": level,
                    "test_acc": test_acc,
                    "acc_drop": baseline[key] - test_acc,
                })
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(exp_dir / "robustness.csv", index=False)
    plot_robustness(robust_df, fig_dir / "robustness.png")

    overfit_df = pd.DataFrame([{
        "algorithm": r["key"], "name": r["name"],
        "train_acc": r["train_acc"], "test_acc": r["test_acc"],
        "overfit_gap": r["train_acc"] - r["test_acc"],
    } for r in results])
    overfit_df.to_csv(exp_dir / "overfitting.csv", index=False)
    plot_overfitting(results, fig_dir / "overfitting.png")

    rf_res = next(r for r in results if r["key"] == "random_forest")
    plot_rf_feature_importance(
        rf_res["model"], data["feature_names"], fig_dir / "rf_feature_importance.png"
    )
    plot_confusion_matrices(
        results, data["X_test"], data["y_test"], data["class_names"],
        fig_dir / "confusion_matrices.png",
    )

    summary = {
        "best_test_acc": max(results, key=lambda x: x["test_acc"])["name"],
        "fastest": min(results, key=lambda x: x["inference_ms"])["name"],
        "most_robust": robust_df.groupby("algorithm")["acc_drop"].mean().idxmin(),
        "least_overfit": overfit_df.loc[overfit_df["overfit_gap"].idxmin(), "name"],
    }
    with open(exp_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n测试评估完成")
    print(f"  报告: {exp_dir}")
    print(f"  图表: {fig_dir}")
    print(f"  最高测试精度: {summary['best_test_acc']}")
    print("=" * 50)

    return {"results": results, "robustness": robust_df, "summary": summary}
