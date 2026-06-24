from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from bolr.model.penalties import QuadraticPenalty
from bolr.posterior.diagnostics import covariance_condition_number, jittered_cholesky


@dataclass(frozen=True)
class StructuredGaussianPrior:
    mean: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    penalty: QuadraticPenalty
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ProcessNoiseModel:
    covariance: np.ndarray
    family: str
    metadata: Mapping[str, object]


def _spd_inverse(matrix: np.ndarray) -> np.ndarray:
    chol = jittered_cholesky(matrix)
    eye = np.eye(matrix.shape[0], dtype=float)
    solved = np.linalg.solve(chol.factor, eye)
    return np.linalg.solve(chol.factor.T, solved)


def prior_from_penalty(
    penalty: QuadraticPenalty,
    mean: np.ndarray | None = None,
    smooth_weight: float = 1.0,
    ridge: float = 1e-6,
) -> StructuredGaussianPrior:
    if ridge <= 0.0:
        raise ValueError("ridge must be positive for a proper Gaussian prior.")
    mean = np.zeros(penalty.dimension, dtype=float) if mean is None else np.asarray(mean, dtype=float).reshape(-1)
    if mean.size != penalty.dimension:
        raise ValueError("Prior mean dimension mismatch.")
    precision = smooth_weight * penalty.matrix + ridge * np.eye(penalty.dimension, dtype=float)
    covariance = _spd_inverse(precision)
    precision_eig = np.linalg.eigvalsh(precision)
    covariance_eig = np.linalg.eigvalsh(covariance)
    return StructuredGaussianPrior(
        mean=mean,
        precision=precision,
        covariance=covariance,
        penalty=penalty,
        metadata={
            "smooth_weight": smooth_weight,
            "ridge": ridge,
            "penalty_rank": penalty.rank(),
            "precision_eigenvalues": precision_eig,
            "covariance_eigenvalues": covariance_eig,
            "precision_min_eigenvalue": float(np.min(precision_eig)),
            "precision_max_eigenvalue": float(np.max(precision_eig)),
            "covariance_min_eigenvalue": float(np.min(covariance_eig)),
            "covariance_max_eigenvalue": float(np.max(covariance_eig)),
            "condition_number": covariance_condition_number(covariance),
            "effective_prior_variance": float(np.mean(np.diag(covariance))),
        },
    )


def isotropic_process_noise(dimension: int, scale: float) -> ProcessNoiseModel:
    if scale < 0.0:
        raise ValueError("scale must be non-negative.")
    cov = scale * np.eye(dimension, dtype=float)
    return ProcessNoiseModel(covariance=cov, family="isotropic_random_walk", metadata={"scale": scale})


def diagonal_process_noise(diagonal: np.ndarray) -> ProcessNoiseModel:
    diagonal = np.asarray(diagonal, dtype=float).reshape(-1)
    if np.any(diagonal < 0.0):
        raise ValueError("Diagonal process variances must be non-negative.")
    cov = np.diag(diagonal)
    return ProcessNoiseModel(covariance=cov, family="diagonal_random_walk", metadata={"diagonal": diagonal})


def penalty_shaped_process_noise(
    penalty: QuadraticPenalty,
    scale: float,
    properization: float,
) -> ProcessNoiseModel:
    if scale < 0.0 or properization <= 0.0:
        raise ValueError("scale must be non-negative and properization positive.")
    precision = penalty.matrix + properization * np.eye(penalty.dimension, dtype=float)
    covariance = scale * _spd_inverse(precision)
    eig_penalty = np.linalg.eigvalsh(penalty.matrix)
    eig_cov = np.linalg.eigvalsh(covariance)
    return ProcessNoiseModel(
        covariance=covariance,
        family="penalty_shaped_random_walk",
        metadata={
            "scale": scale,
            "properization": properization,
            "penalty_eigenvalues": eig_penalty,
            "process_eigenvalues": eig_cov,
            "min_drift_variance": float(np.min(eig_cov)),
            "max_drift_variance": float(np.max(eig_cov)),
            "smoothest_mode_drift_variance": float(np.max(eig_cov)),
            "roughest_mode_drift_variance": float(np.min(eig_cov)),
            "condition_number": float(np.max(eig_cov) / np.min(eig_cov)),
        },
    )
