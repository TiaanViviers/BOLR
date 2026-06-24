from __future__ import annotations

import numpy as np

from bolr.decision.utils import binary_entropy


def realized_best_distribution(utilities: np.ndarray, *, tolerance: float = 0.0) -> np.ndarray:
    utilities = np.asarray(utilities, dtype=float)
    best = np.max(utilities)
    mask = utilities >= best - tolerance
    target = np.zeros(utilities.size, dtype=float)
    target[mask] = 1.0 / float(mask.sum())
    return target


def realized_top_k_indicator(utilities: np.ndarray, k: int) -> np.ndarray:
    utilities = np.asarray(utilities, dtype=float)
    order = np.argsort(-utilities, kind="mergesort")
    indicator = np.zeros(utilities.size, dtype=float)
    indicator[order[:k]] = 1.0
    return indicator


def probability_best_brier(probability_best: np.ndarray, utilities: np.ndarray, *, tolerance: float = 0.0) -> float:
    target = realized_best_distribution(utilities, tolerance=tolerance)
    diff = np.asarray(probability_best, dtype=float) - target
    return float(np.sum(diff * diff))


def top_k_brier(probability_top_k: np.ndarray, utilities: np.ndarray, *, k: int) -> float:
    target = realized_top_k_indicator(utilities, k)
    diff = np.asarray(probability_top_k, dtype=float) - target
    return float(np.sum(diff * diff))


def region_coverage(region_indices: np.ndarray, utilities: np.ndarray, *, tolerance: float = 0.0) -> bool:
    region_indices = np.asarray(region_indices, dtype=int)
    target = realized_best_distribution(utilities, tolerance=tolerance)
    return bool(np.any(target[region_indices] > 0.0))


def probability_best_entropy(probability_best: np.ndarray) -> float:
    return binary_entropy(np.asarray(probability_best, dtype=float))
