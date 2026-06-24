from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class QuadraticPenalty:
    matrix: np.ndarray
    dimension: int
    name: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Penalty matrix must be square.")
        if matrix.shape[0] != self.dimension:
            raise ValueError("Penalty dimension mismatch.")
        if not np.isfinite(matrix).all():
            raise ValueError("Penalty matrix contains non-finite values.")
        sym = 0.5 * (matrix + matrix.T)
        if np.max(np.abs(sym - matrix)) > 1e-10:
            raise ValueError("Penalty matrix is materially asymmetric.")
        eigenvalues = np.linalg.eigvalsh(sym)
        if np.min(eigenvalues) < -1e-10:
            raise ValueError("Penalty matrix must be PSD within tolerance.")
        object.__setattr__(self, "matrix", sym)

    def value(self, beta: np.ndarray) -> float:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        return 0.5 * float(beta @ self.matrix @ beta)

    def gradient(self, beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        return self.matrix @ beta

    def hessian(self) -> np.ndarray:
        return self.matrix

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    def nullity(self) -> int:
        return self.dimension - self.rank()

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.matrix)

    def is_psd(self) -> bool:
        return bool(np.min(self.eigenvalues()) >= -1e-10)

    def properize(self, ridge: float) -> "QuadraticPenalty":
        if ridge <= 0.0:
            raise ValueError("ridge must be positive for properization.")
        return QuadraticPenalty(
            matrix=self.matrix + ridge * np.eye(self.dimension, dtype=float),
            dimension=self.dimension,
            name=f"{self.name}_proper",
            metadata={**dict(self.metadata), "ridge": ridge},
        )

    def scaled(self, weight: float) -> "QuadraticPenalty":
        if weight < 0.0:
            raise ValueError("weight must be non-negative.")
        return QuadraticPenalty(
            matrix=weight * self.matrix,
            dimension=self.dimension,
            name=f"{self.name}_scaled",
            metadata={**dict(self.metadata), "weight": weight},
        )

    def sum(self, other: "QuadraticPenalty") -> "QuadraticPenalty":
        if self.dimension != other.dimension:
            raise ValueError("Penalty dimensions must match.")
        return QuadraticPenalty(
            matrix=self.matrix + other.matrix,
            dimension=self.dimension,
            name=f"{self.name}_plus_{other.name}",
            metadata={**dict(self.metadata), "other": other.name},
        )


def difference_matrix(size: int, order: int) -> np.ndarray:
    if order not in {1, 2}:
        raise ValueError("Only first- and second-order differences are supported.")
    if size <= order:
        raise ValueError("size must exceed difference order.")
    if order == 1:
        mat = np.zeros((size - 1, size), dtype=float)
        for i in range(size - 1):
            mat[i, i] = -1.0
            mat[i, i + 1] = 1.0
        return mat
    mat = np.zeros((size - 2, size), dtype=float)
    for i in range(size - 2):
        mat[i, i] = 1.0
        mat[i, i + 1] = -2.0
        mat[i, i + 2] = 1.0
    return mat


def difference_penalty(size: int, order: int, name: str | None = None) -> QuadraticPenalty:
    d = difference_matrix(size, order)
    return QuadraticPenalty(
        matrix=d.T @ d,
        dimension=size,
        name=name or f"difference_order_{order}",
        metadata={"order": order, "size": size},
    )


def tensor_product_penalty(
    entry_dim: int,
    stop_dim: int,
    entry_order: int = 2,
    stop_order: int = 2,
    entry_weight: float = 1.0,
    stop_weight: float = 1.0,
    ridge: float = 0.0,
) -> QuadraticPenalty:
    re = difference_penalty(entry_dim, entry_order, "entry").matrix
    rr = difference_penalty(stop_dim, stop_order, "stop").matrix
    penalty = entry_weight * np.kron(re, np.eye(stop_dim)) + stop_weight * np.kron(np.eye(entry_dim), rr)
    if ridge > 0.0:
        penalty = penalty + ridge * np.eye(entry_dim * stop_dim)
    return QuadraticPenalty(
        matrix=penalty,
        dimension=entry_dim * stop_dim,
        name="tensor_surface",
        metadata={
            "entry_dim": entry_dim,
            "stop_dim": stop_dim,
            "entry_order": entry_order,
            "stop_order": stop_order,
            "entry_weight": entry_weight,
            "stop_weight": stop_weight,
            "ridge": ridge,
        },
    )


def project_penalty(raw_penalty: QuadraticPenalty, lift_matrix: np.ndarray, name: str | None = None) -> QuadraticPenalty:
    lift_matrix = np.asarray(lift_matrix, dtype=float)
    if lift_matrix.shape[0] != raw_penalty.dimension:
        raise ValueError("Lift matrix row dimension must match raw penalty dimension.")
    projected = lift_matrix.T @ raw_penalty.matrix @ lift_matrix
    return QuadraticPenalty(
        matrix=projected,
        dimension=lift_matrix.shape[1],
        name=name or f"{raw_penalty.name}_projected",
        metadata={**dict(raw_penalty.metadata), "lift_shape": lift_matrix.shape},
    )


def context_matrix_penalty(
    candidate_penalty: QuadraticPenalty,
    context_penalty_matrix: np.ndarray,
    context_weight: float = 1.0,
    candidate_weight: float = 1.0,
    ridge: float = 0.0,
) -> QuadraticPenalty:
    rc = candidate_penalty.matrix
    rm = np.asarray(context_penalty_matrix, dtype=float)
    if rm.ndim != 2 or rm.shape[0] != rm.shape[1]:
        raise ValueError("Context penalty matrix must be square.")
    pc = rc.shape[0]
    pm = rm.shape[0]
    precision = candidate_weight * np.kron(np.eye(pm), rc) + context_weight * np.kron(rm, np.eye(pc))
    if ridge > 0.0:
        precision = precision + ridge * np.eye(pc * pm)
    return QuadraticPenalty(
        matrix=precision,
        dimension=pc * pm,
        name="context_matrix_penalty",
        metadata={
            "candidate_dimension": pc,
            "context_dimension": pm,
            "candidate_weight": candidate_weight,
            "context_weight": context_weight,
            "ridge": ridge,
        },
    )
