from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.posterior.state import GaussianPosterior


@dataclass(frozen=True)
class IsotropicRandomWalkDynamics:
    process_variance: float

    def __post_init__(self) -> None:
        if self.process_variance < 0.0:
            raise ValueError("process_variance must be non-negative.")

    def predict(self, posterior: GaussianPosterior, dt: float = 1.0) -> GaussianPosterior:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        process_covariance = self.process_variance * dt * np.eye(posterior.dimension, dtype=float)
        predicted = GaussianPosterior(
            mean=posterior.mean.copy(),
            covariance=posterior.covariance + process_covariance,
            state_layout=posterior.state_layout,
            timestamp=posterior.timestamp,
            version=posterior.version,
            diagnostics=dict(posterior.diagnostics),
        )
        return predicted.with_diagnostics(
            predicted_by="isotropic_random_walk",
            process_variance=self.process_variance,
            dt=dt,
        )

