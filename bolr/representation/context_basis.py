from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from bolr.config.foundation import SelectedColumnsContextConfig


class ContextBasis(Protocol):
    def fit(self, rows: Sequence[Mapping[str, float]]) -> "ContextBasis":
        ...

    def transform(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        ...

    @property
    def output_dim(self) -> int:
        ...


@dataclass(frozen=True)
class SelectedColumnsContextState:
    means: np.ndarray
    scales: np.ndarray
    feature_names: tuple[str, ...]


class SelectedColumnsContextBasis:
    def __init__(self, config: SelectedColumnsContextConfig) -> None:
        self.config = config
        self.state: SelectedColumnsContextState | None = None

    @property
    def output_dim(self) -> int:
        base = len(self.config.columns)
        return base + int(self.config.add_intercept)

    def fit(self, rows: Sequence[Mapping[str, float]]) -> "SelectedColumnsContextBasis":
        matrix = _rows_to_matrix(rows, self.config.columns)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales = np.where(scales == 0.0, 1.0, scales)
        self.state = SelectedColumnsContextState(
            means=means,
            scales=scales,
            feature_names=self.feature_names,
        )
        return self

    def transform(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("SelectedColumnsContextBasis must be fitted before transform().")
        matrix = _rows_to_matrix(rows, self.config.columns)
        if self.config.scale:
            matrix = (matrix - self.state.means) / self.state.scales
        if self.config.add_intercept:
            intercept = np.ones((matrix.shape[0], 1), dtype=float)
            matrix = np.hstack([intercept, matrix])
        return matrix

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.config.add_intercept:
            return ("intercept", *self.config.columns)
        return self.config.columns


def _rows_to_matrix(rows: Sequence[Mapping[str, float]], columns: Sequence[str]) -> np.ndarray:
    return np.array([[float(row[column]) for column in columns] for row in rows], dtype=float)
