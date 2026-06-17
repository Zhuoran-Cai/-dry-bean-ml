"""随机森林（集成学习与树模型）。"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.inference import inference_time_ms

NAME = "Random Forest"
KEY = "random_forest"


def build_model(params: dict, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )


def train(model: RandomForestClassifier, X_train, y_train) -> RandomForestClassifier:
    model.fit(X_train, y_train)
    return model


def accuracy(model, X, y) -> float:
    return float(accuracy_score(y, model.predict(X)))


def staged_loss_curve(X_train, y_train, X_val, y_val, params: dict, random_state: int) -> dict:
    """逐步增加树数量，记录 log loss 变化。"""
    step = params["loss_curve_step"]
    max_trees = params["n_estimators"]
    steps = list(range(step, max_trees + 1, step))
    if steps[-1] != max_trees:
        steps.append(max_trees)

    rf = RandomForestClassifier(
        n_estimators=step,
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )
    train_loss, val_loss = [], []
    for n in steps:
        rf.n_estimators = n
        rf.fit(X_train, y_train)
        train_loss.append(log_loss(y_train, rf.predict_proba(X_train)))
        val_loss.append(log_loss(y_val, rf.predict_proba(X_val)))

    return {
        "steps": steps,
        "train": train_loss,
        "val": val_loss,
        "ylabel": "Log Loss",
        "xlabel": "n_estimators",
    }


def run(data: dict, config: dict) -> dict:
    params = config["random_forest"]
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
        "curve": staged_loss_curve(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, rs
        ),
        "has_loss_curve": True,
    }
