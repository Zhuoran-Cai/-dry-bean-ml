"""算法训练模块：调用各分类器完成训练并保存模型。"""

from __future__ import annotations

from pathlib import Path

import joblib

from src import knn, logistic_regression, random_forest, svm

ALGORITHMS = [
    ("logistic_regression", logistic_regression),
    ("knn", knn),
    ("svm", svm),
    ("random_forest", random_forest),
]


def run_train(data: dict, config: dict, project_root: Path) -> list[dict]:
    """训练四种算法，保存模型文件，返回训练结果（含 model、curve 等）。"""
    model_dir = project_root / config["paths"]["models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("算法训练")
    print("=" * 50)

    results = []
    for key, mod in ALGORITHMS:
        print(f"\n训练 {key} ...")
        res = mod.run(data, config)
        joblib.dump(res["model"], model_dir / f"{key}.pkl")
        results.append(res)
        print(f"  train={res['train_acc']:.4f}  val={res['val_acc']:.4f}  test={res['test_acc']:.4f}")

    print("\n训练完成")
    print(f"  模型保存: {model_dir}")
    return results
