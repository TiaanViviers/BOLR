from __future__ import annotations

import numpy as np


def stationary_interaction_matrix(
    candidate_dim: int,
    context_dim: int,
    seed: int | None = None,
    scale: float = 0.35,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    row_grid = np.linspace(-1.0, 1.0, candidate_dim, dtype=float)
    col_grid = np.linspace(-1.0, 1.0, context_dim, dtype=float)
    base = np.outer(np.cos(np.pi * row_grid / 2.0), 1.0 + 0.25 * col_grid)
    perturbation = rng.normal(scale=scale, size=(candidate_dim, context_dim))
    return base + perturbation


def drifting_interaction_sequence(
    candidate_dim: int,
    context_dim: int,
    n_steps: int,
    seed: int | None = None,
    drift_scale: float = 0.03,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sequence = np.empty((n_steps, candidate_dim, context_dim), dtype=float)
    sequence[0] = stationary_interaction_matrix(candidate_dim, context_dim, seed=seed)
    for t in range(1, n_steps):
        drift = rng.normal(scale=drift_scale, size=(candidate_dim, context_dim))
        sequence[t] = sequence[t - 1] + drift
    return sequence

