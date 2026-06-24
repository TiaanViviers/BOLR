from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - logsumexp(values)


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(values))


def softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values > 0.0
    result = np.empty_like(values, dtype=float)
    result[positive] = values[positive] + np.log1p(np.exp(-values[positive]))
    result[~positive] = np.log1p(np.exp(values[~positive]))
    return result


def cross_entropy(target: np.ndarray, predicted_log_probs: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    predicted_log_probs = np.asarray(predicted_log_probs, dtype=float)
    return float(-(target * predicted_log_probs).sum())
