from __future__ import annotations

from dataclasses import dataclass

from bolr.posterior.state import GaussianPosterior


@dataclass(frozen=True)
class CovarianceDiscountDynamics:
    discount_factor: float

    def __post_init__(self) -> None:
        if not (0.0 < self.discount_factor <= 1.0):
            raise ValueError("discount_factor must lie in (0, 1].")

    def predict(self, posterior: GaussianPosterior, dt: float = 1.0) -> GaussianPosterior:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        scale = self.discount_factor ** (-dt)
        predicted = GaussianPosterior(
            mean=posterior.mean.copy(),
            covariance=posterior.covariance * scale,
            state_layout=posterior.state_layout,
            timestamp=posterior.timestamp,
            version=posterior.version,
            diagnostics=dict(posterior.diagnostics),
        )
        return predicted.with_diagnostics(
            predicted_by="covariance_discount",
            discount_factor=self.discount_factor,
            dt=dt,
        )

