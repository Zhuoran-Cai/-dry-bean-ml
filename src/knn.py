"""KNN（基于距离的模型）。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.inference import inference_time_ms

NAME = "KNN"
KEY = "knn"


def build_model(params: dict) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric=params["metric"],
    )


def train(model: KNeighborsClassifier, X_train, y_train) -> KNeighborsClassifier:
    model.fit(X_train, y_train)
    return model


def accuracy(model, X, y) -> float:
    return float(accuracy_score(y, model.predict(X)))


def train_accuracy_exclude_self(clf: KNeighborsClassifier, X_train, y_train, k: int) -> float:
    """训练集精度：预测时排除自身近邻，避免 KNN 在训练集上恒为 100%。"""
    n = len(y_train)
    n_neighbors = min(k + 1, n)
    distances, indices = clf.kneighbors(X_train, n_neighbors=n_neighbors)
    if n_neighbors > k:
        indices = indices[:, 1 : k + 1]
        distances = distances[:, 1 : k + 1]

    preds = []
    for i in range(n):
        labels = y_train[indices[i]]
        if clf.weights == "uniform":
            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        else:
            weights = 1.0 / (distances[i] + 1e-8)
            classes = np.unique(labels)
            preds.append(max(classes, key=lambda c: weights[labels == c].sum()))
    return float(accuracy_score(y_train, preds))


def k_selection_curve(X_train, y_train, X_val, y_val, params: dict) -> dict:
    """非训练型算法：用不同 k 值的验证集精度代替 loss 曲线。"""
    k_values = params["k_curve_range"]
    train_acc, val_acc = [], []
    for k in k_values:
        clf = KNeighborsClassifier(
            n_neighbors=k,
            weights=params["weights"],
            metric=params["metric"],
        )
        clf.fit(X_train, y_train)
        train_acc.append(train_accuracy_exclude_self(clf, X_train, y_train, k))
        val_acc.append(accuracy(clf, X_val, y_val))
    return {
        "steps": k_values,
        "train": train_acc,
        "val": val_acc,
        "ylabel": "Accuracy",
        "xlabel": "k (n_neighbors)",
    }


def run(data: dict, config: dict) -> dict:
    params = config["knn"]
    model = build_model(params)
    model = train(model, data["X_train"], data["y_train"])
    return {
        "key": KEY,
        "name": NAME,
        "model": model,
        "train_acc": train_accuracy_exclude_self(
            model, data["X_train"], data["y_train"], params["n_neighbors"]
        ),
        "val_acc": accuracy(model, data["X_val"], data["y_val"]),
        "test_acc": accuracy(model, data["X_test"], data["y_test"]),
        "inference_ms": inference_time_ms(model, data["X_test"], config["experiment"]),
        "curve": k_selection_curve(data["X_train"], data["y_train"], data["X_val"], data["y_val"], params),
        "has_loss_curve": False,
    }
