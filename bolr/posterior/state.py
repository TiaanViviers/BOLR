from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from bolr.posterior.diagnostics import covariance_condition_number, jittered_cholesky
from bolr.representation.score_design import DailyDesign


@dataclass(frozen=True)
class GaussianPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    state_layout: dict[str, Any] | None = None
    timestamp: str | None = None
    version: str = "phase_c_reference"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.ndim != 1:
            raise ValueError("mean must be one-dimensional.")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance shape must match mean dimension.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def precision(self) -> np.ndarray:
        return np.linalg.inv(self.covariance)

    def cholesky(self):
        return jittered_cholesky(self.covariance)

    def with_diagnostics(self, **updates: Any) -> "GaussianPosterior":
        merged = dict(self.diagnostics)
        merged.update(updates)
        return replace(self, diagnostics=merged)

    def score_mean(self, design: DailyDesign) -> np.ndarray:
        return design.scores(self.mean)

    def score_covariance(self, design: DailyDesign) -> np.ndarray:
        explicit = design.explicit_matrix()
        return explicit @ self.covariance @ explicit.T

    def selected_score_covariance(self, design: DailyDesign, indices: np.ndarray) -> np.ndarray:
        explicit = design.explicit_matrix()[np.asarray(indices, dtype=int)]
        return explicit @ self.covariance @ explicit.T

    def score_variance(self, design: DailyDesign) -> np.ndarray:
        explicit = design.explicit_matrix()
        return np.einsum("ij,jk,ik->i", explicit, self.covariance, explicit)

    def sample_parameters(
        self,
        n_samples: int,
        seed: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed) if rng is None else rng
        chol = self.cholesky().factor
        if antithetic:
            half = (n_samples + 1) // 2
            z = rng.standard_normal((half, self.dimension))
            z = np.vstack([z, -z])[:n_samples]
        else:
            z = rng.standard_normal((n_samples, self.dimension))
        return self.mean + z @ chol.T

    def sample_scores(
        self,
        design: DailyDesign,
        n_samples: int,
        seed: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        samples = self.sample_parameters(n_samples=n_samples, seed=seed, rng=rng, antithetic=antithetic)
        return np.stack([design.scores(sample) for sample in samples], axis=0)

    def summary(self) -> dict[str, float]:
        return {
            "dimension": float(self.dimension),
            "trace": float(np.trace(self.covariance)),
            "condition_number": covariance_condition_number(self.covariance),
        }
