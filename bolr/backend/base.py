from __future__ import annotations

from typing import Protocol

import numpy as np


class NumericalBackend(Protocol):
    def block_forward(self, design: np.ndarray, state: np.ndarray) -> np.ndarray: ...
    def block_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray: ...
    def composite_forward(self, design: np.ndarray, state: np.ndarray, static_scores: np.ndarray | None = None) -> np.ndarray: ...
    def composite_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray: ...
    def observation_value_gradient_hvp(self, scores: np.ndarray, observation: object, vector: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]: ...
    def cholesky_factor(self, matrix: np.ndarray) -> np.ndarray: ...
    def cholesky_solve(self, factor: np.ndarray, rhs: np.ndarray) -> np.ndarray: ...


class NumpyBackend:
    def block_forward(self, design: np.ndarray, state: np.ndarray) -> np.ndarray:
        return np.asarray(design, dtype=float) @ np.asarray(state, dtype=float)

    def block_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        return np.asarray(design, dtype=float).T @ np.asarray(score_vector, dtype=float)

    def composite_forward(self, design: np.ndarray, state: np.ndarray, static_scores: np.ndarray | None = None) -> np.ndarray:
        result = np.asarray(design, dtype=float) @ np.asarray(state, dtype=float)
        if static_scores is not None:
            result = result + np.asarray(static_scores, dtype=float)
        return result

    def composite_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        return np.asarray(design, dtype=float).T @ np.asarray(score_vector, dtype=float)

    def observation_value_gradient_hvp(self, scores: np.ndarray, observation: object, vector: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        raise NotImplementedError("Observation kernels are supplied by observation models, not the generic backend.")

    def cholesky_factor(self, matrix: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(np.asarray(matrix, dtype=float))

    def cholesky_solve(self, factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(factor, rhs)
        return np.linalg.solve(factor.T, y)
