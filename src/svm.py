"""SVM（最大间隔与核方法，课堂外自主实现）。"""

from __future__ import annotations

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from src.inference import inference_time_ms

NAME = "SVM"
KEY = "svm"


def build_model(params: dict, random_state: int) -> SVC:
    return SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        random_state=random_state,
    )


def train(model: SVC, X_train, y_train) -> SVC:
    model.fit(X_train, y_train)
    return model


def accuracy(model, X, y) -> float:
    return float(accuracy_score(y, model.predict(X)))


def c_parameter_curve(X_train, y_train, X_val, y_val, params: dict, random_state: int) -> dict:
    """记录不同 C 下训练/验证精度，反映模型拟合过程。"""
    c_values = params["c_curve_values"]
    train_acc, val_acc = [], []
    for c in c_values:
        clf = SVC(kernel=params["kernel"], C=c, gamma=params["gamma"], random_state=random_state)
        clf.fit(X_train, y_train)
        train_acc.append(accuracy(clf, X_train, y_train))
        val_acc.append(accuracy(clf, X_val, y_val))
    return {
        "steps": c_values,
        "train": train_acc,
        "val": val_acc,
        "ylabel": "Accuracy",
        "xlabel": "C",
    }


def run(data: dict, config: dict) -> dict:
    params = config["svm"]
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
        "curve": c_parameter_curve(data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, rs),
        "has_loss_curve": True,
    }
