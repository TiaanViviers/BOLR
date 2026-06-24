from __future__ import annotations

import numpy as np


def entropy_from_scores(scores: np.ndarray) -> float:
    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs /= probs.sum()
    return float(-(probs * np.log(np.clip(probs, 1e-300, 1.0))).sum())


def rank_of_selected(utilities: np.ndarray, selected_index: int) -> int:
    order = np.argsort(-utilities, kind="mergesort")
    rank_lookup = np.empty_like(order)
    rank_lookup[order] = np.arange(1, utilities.size + 1)
    return int(rank_lookup[selected_index])


def top_fraction_membership(rank: int, n_candidates: int, fraction: float) -> bool:
    cutoff = max(1, int(np.ceil(fraction * n_candidates)))
    return rank <= cutoff


def maximum_drawdown(cumulative_values: np.ndarray) -> float:
    running_max = np.maximum.accumulate(cumulative_values)
    drawdowns = running_max - cumulative_values
    return float(drawdowns.max())

