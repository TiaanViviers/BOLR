from __future__ import annotations

import numpy as np


def build_explicit_design(candidate_basis: np.ndarray, context_vector: np.ndarray) -> np.ndarray:
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    context_vector = np.asarray(context_vector, dtype=float)
    if context_vector.ndim != 1:
        raise ValueError("context_vector must be one-dimensional.")
    return np.kron(context_vector.reshape(1, -1), candidate_basis)


def structured_scores(
    candidate_basis: np.ndarray,
    interaction_matrix: np.ndarray,
    context_vector: np.ndarray,
) -> np.ndarray:
    return np.asarray(candidate_basis, dtype=float) @ np.asarray(interaction_matrix, dtype=float) @ np.asarray(
        context_vector, dtype=float
    )


def theta_from_matrix(interaction_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(interaction_matrix, dtype=float).reshape(-1, order="F")


def matrix_from_theta(theta: np.ndarray, candidate_dim: int, context_dim: int) -> np.ndarray:
    return np.asarray(theta, dtype=float).reshape((candidate_dim, context_dim), order="F")
