"""模型推理耗时度量。"""

from __future__ import annotations

import time

import numpy as np


def _probe_single_ms(model, Xs: np.ndarray, n_probe: int = 20) -> float:
    n = min(n_probe, len(Xs))
    model.predict(Xs[:1])
    start = time.perf_counter()
    for i in range(n):
        model.predict(Xs[i : i + 1])
    return (time.perf_counter() - start) / n * 1000


def _single_sample_ms(model, Xs: np.ndarray, repeats: int) -> float:
    n = len(Xs)
    model.predict(Xs[:1])
    start = time.perf_counter()
    for _ in range(repeats):
        for i in range(n):
            model.predict(Xs[i : i + 1])
    return (time.perf_counter() - start) / repeats / n * 1000


def _batch_ms(model, Xs: np.ndarray, repeats: int) -> float:
    n = len(Xs)
    model.predict(Xs[:1])
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(Xs)
    return (time.perf_counter() - start) / repeats / n * 1000


def inference_time_ms(model, X: np.ndarray, experiment: dict) -> float:
    """单样本在线推理平均耗时 (ms)。

    默认对测试集逐样本调用 predict，整集重复多次取平均，避免线性模型批量
    predict 的向量化摊薄导致耗时被严重低估。若单样本探测耗时过高（如随机森林），
    则回退为对采样子集的批量 predict。
    """
    repeats = experiment["inference_repeats"]
    sample_size = experiment.get("inference_sample_size", len(X))
    fallback_ms = experiment.get("inference_batch_fallback_ms", 5.0)

    n = min(sample_size, len(X))
    Xs = X[:n]

    if _probe_single_ms(model, Xs) > fallback_ms:
        return _batch_ms(model, Xs, repeats)
    return _single_sample_ms(model, Xs, repeats)
