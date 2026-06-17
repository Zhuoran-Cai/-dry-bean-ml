"""逻辑回归（线性模型）。"""

from __future__ import annotations

from src.inference import inference_time_ms
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, log_loss

NAME = "Logistic Regression"
KEY = "logistic_regression"


def build_model(params: dict, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        solver="lbfgs",
        random_state=random_state,
    )


def train(model: LogisticRegression, X_train, y_train) -> LogisticRegression:
    model.fit(X_train, y_train)
    return model


def accuracy(model, X, y) -> float:
    return float(accuracy_score(y, model.predict(X)))


def loss_curve(X_train, y_train, X_val, y_val, params: dict, random_state: int) -> dict:
    """SGD 逐步训练，记录 log loss 曲线。"""
    classes = np.unique(y_train)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1.0 / (params["C"] * len(X_train)),
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )
    epochs = params["loss_curve_epochs"]
    train_loss, val_loss, steps = [], [], []
    for epoch in range(1, epochs + 1):
        clf.partial_fit(X_train, y_train, classes=classes)
        train_loss.append(log_loss(y_train, clf.predict_proba(X_train)))
        val_loss.append(log_loss(y_val, clf.predict_proba(X_val)))
        steps.append(epoch)
    return {
        "steps": steps,
        "train": train_loss,
        "val": val_loss,
        "ylabel": "Log Loss",
        "xlabel": "Epoch",
    }


def run(data: dict, config: dict) -> dict:
    params = config["logistic_regression"]
    rs = config["experiment"]["random_state"]
    model = build_model(params, rs)
    model = train(model, data["X_train"], data["y_train"])
    return {
        "key": KEY,
        "name": NAME,
        "model": model,
        "train_acc": accuracy(model, data["X_train"], data["y_train"]),
        "val_acc": accuracy(model, data["X_val"], data["y_val"]),
        "test_acc": accuracy(model, data["X_test"], data["y_test"]),
        "inference_ms": inference_time_ms(model, data["X_test"], config["experiment"]),
        "curve": loss_curve(data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, rs),
        "has_loss_curve": True,
    }
