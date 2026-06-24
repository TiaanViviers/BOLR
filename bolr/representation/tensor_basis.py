from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.config.foundation import TensorBasisConfig
from bolr.representation.spline_basis import BSplineBasis1D


@dataclass(frozen=True)
class CandidateBasis:
    original_basis: np.ndarray
    centered_basis: np.ndarray
    reduced_basis: np.ndarray
    column_mean: np.ndarray
    projection: np.ndarray
    lift_matrix: np.ndarray


class TensorProductBasis:
    def __init__(self, config: TensorBasisConfig) -> None:
        self.config = config
        self.entry_basis = BSplineBasis1D(config.entry_basis)
        self.stop_basis = BSplineBasis1D(config.stop_basis)
        self.state: CandidateBasis | None = None

    def fit(self, coordinates: np.ndarray) -> "TensorProductBasis":
        coordinates = np.asarray(coordinates, dtype=float)
        entry_part = self.entry_basis.fit_transform(coordinates[:, 0])
        stop_part = self.stop_basis.fit_transform(coordinates[:, 1])
        original = _rowwise_kronecker(entry_part, stop_part)
        centered = original - original.mean(axis=0, keepdims=True) if self.config.center else original.copy()
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        del u
        mask = s > self.config.rank_tol
        projection = vt[mask].T
        reduced = centered @ projection
        self.state = CandidateBasis(
            original_basis=original,
            centered_basis=centered,
            reduced_basis=reduced,
            column_mean=original.mean(axis=0),
            projection=projection,
            lift_matrix=projection,
        )
        return self

    def transform(self, coordinates: np.ndarray) -> CandidateBasis:
        if self.state is None:
            raise RuntimeError("TensorProductBasis must be fitted before transform().")
        coordinates = np.asarray(coordinates, dtype=float)
        entry_part = self.entry_basis.transform(coordinates[:, 0])
        stop_part = self.stop_basis.transform(coordinates[:, 1])
        original = _rowwise_kronecker(entry_part, stop_part)
        centered = original - self.state.column_mean if self.config.center else original.copy()
        reduced = centered @ self.state.projection
        return CandidateBasis(
            original_basis=original,
            centered_basis=centered,
            reduced_basis=reduced,
            column_mean=self.state.column_mean,
            projection=self.state.projection,
            lift_matrix=self.state.lift_matrix,
        )

    def fit_transform(self, coordinates: np.ndarray) -> CandidateBasis:
        self.fit(coordinates)
        return self.transform(coordinates)


def _rowwise_kronecker(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)
