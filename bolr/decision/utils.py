from __future__ import annotations

from math import erf, sqrt

import numpy as np


def stable_rank_order(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.argsort(-scores, kind="mergesort")


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = stable_rank_order(scores)
    ranks = np.empty(scores.size, dtype=int)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def binary_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probs, 1e-300, 1.0)
    return float(-(probs * np.log(clipped)).sum())


def effective_count(probabilities: np.ndarray) -> float:
    return float(np.exp(binary_entropy(probabilities)))


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(x, dtype=float)
    result = 0.5 * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))
    if np.isscalar(x):
        return float(result)
    return result
