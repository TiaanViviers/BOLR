from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DailyDesign:
    candidate_basis: np.ndarray
    context_vector: np.ndarray
    static_scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        candidate_basis = np.asarray(self.candidate_basis, dtype=float)
        context_vector = np.asarray(self.context_vector, dtype=float)
        if candidate_basis.ndim != 2:
            raise ValueError("candidate_basis must be two-dimensional.")
        if context_vector.ndim != 1:
            raise ValueError("context_vector must be one-dimensional.")
        if self.static_scores is not None and np.asarray(self.static_scores, dtype=float).shape != (candidate_basis.shape[0],):
            raise ValueError("static_scores must match the candidate dimension.")
        object.__setattr__(self, "candidate_basis", candidate_basis)
        object.__setattr__(self, "context_vector", context_vector)
        if self.static_scores is not None:
            object.__setattr__(self, "static_scores", np.asarray(self.static_scores, dtype=float))

    @property
    def candidate_dim(self) -> int:
        return int(self.candidate_basis.shape[1])

    @property
    def context_dim(self) -> int:
        return int(self.context_vector.size)

    @property
    def state_dim(self) -> int:
        return self.candidate_dim * self.context_dim

    def explicit_matrix(self) -> np.ndarray:
        return build_explicit_design(self.candidate_basis, self.context_vector)

    def scores(self, theta: np.ndarray) -> np.ndarray:
        interaction_matrix = matrix_from_theta(theta, self.candidate_dim, self.context_dim)
        dynamic_scores = structured_scores(self.candidate_basis, interaction_matrix, self.context_vector)
        if self.static_scores is None:
            return dynamic_scores
        return self.static_scores + dynamic_scores


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
