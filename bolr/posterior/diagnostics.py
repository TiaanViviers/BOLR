from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CholeskyResult:
    factor: np.ndarray
    jitter: float
    attempts: int


def jittered_cholesky(
    matrix: np.ndarray,
    initial_jitter: float = 1e-10,
    max_attempts: int = 8,
    growth_factor: float = 10.0,
) -> CholeskyResult:
    matrix = np.asarray(matrix, dtype=float)
    eye = np.eye(matrix.shape[0], dtype=float)
    jitter = 0.0
    for attempt in range(max_attempts):
        try:
            factor = np.linalg.cholesky(matrix + jitter * eye)
            return CholeskyResult(factor=factor, jitter=jitter, attempts=attempt + 1)
        except np.linalg.LinAlgError:
            jitter = initial_jitter if jitter == 0.0 else jitter * growth_factor
    raise np.linalg.LinAlgError("Cholesky factorization failed after jitter escalation.")


def covariance_condition_number(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=float))
    positive = eigenvalues[eigenvalues > 0.0]
    if positive.size == 0:
        return float("inf")
    return float(positive[-1] / positive[0])

