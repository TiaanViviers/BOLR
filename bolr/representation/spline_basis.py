from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.config.foundation import SplineAxisConfig


@dataclass(frozen=True)
class BSplineState:
    knots: np.ndarray
    x_min: float
    x_max: float
    n_basis: int
    degree: int


class BSplineBasis1D:
    def __init__(self, config: SplineAxisConfig) -> None:
        self.config = config
        self.state: BSplineState | None = None

    def fit(self, values: np.ndarray) -> "BSplineBasis1D":
        x = np.asarray(values, dtype=float)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max <= x_min:
            raise ValueError("Spline basis requires non-degenerate coordinates.")

        n_internal = self.config.n_basis - self.config.degree - 1
        if n_internal > 0:
            internal = np.linspace(0.0, 1.0, n_internal + 2, dtype=float)[1:-1]
        else:
            internal = np.array([], dtype=float)
        knots = np.concatenate(
            [
                np.zeros(self.config.degree + 1, dtype=float),
                internal,
                np.ones(self.config.degree + 1, dtype=float),
            ]
        )
        self.state = BSplineState(
            knots=knots,
            x_min=x_min,
            x_max=x_max,
            n_basis=self.config.n_basis,
            degree=self.config.degree,
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("BSplineBasis1D must be fitted before transform().")
        x = np.asarray(values, dtype=float)
        x_scaled = (x - self.state.x_min) / (self.state.x_max - self.state.x_min)
        x_scaled = np.clip(x_scaled, 0.0, 1.0)
        basis = np.zeros((x_scaled.size, self.state.n_basis), dtype=float)
        for row_idx, point in enumerate(x_scaled):
            basis[row_idx] = self._evaluate_point(point)
        basis /= basis.sum(axis=1, keepdims=True)
        return basis

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def _evaluate_point(self, x: float) -> np.ndarray:
        assert self.state is not None
        knots = self.state.knots
        degree = self.state.degree
        n_basis = self.state.n_basis
        basis = np.zeros(n_basis, dtype=float)
        for i in range(n_basis):
            left = knots[i]
            right = knots[i + 1]
            in_interval = left <= x < right
            if x == 1.0 and right == 1.0 and i == n_basis - 1:
                in_interval = True
            basis[i] = 1.0 if in_interval else 0.0

        for k in range(1, degree + 1):
            next_basis = np.zeros(n_basis, dtype=float)
            for i in range(n_basis):
                left_num = x - knots[i]
                left_den = knots[i + k] - knots[i]
                right_num = knots[i + k + 1] - x
                right_den = knots[i + k + 1] - knots[i + 1]

                left_term = 0.0 if left_den == 0.0 else (left_num / left_den) * basis[i]
                right_basis = basis[i + 1] if i + 1 < n_basis else 0.0
                right_term = 0.0 if right_den == 0.0 else (right_num / right_den) * right_basis
                next_basis[i] = left_term + right_term
            basis = next_basis
        return basis
