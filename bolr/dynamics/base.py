from __future__ import annotations

from typing import Protocol

from bolr.posterior.state import GaussianPosterior


class DynamicsModel(Protocol):
    def predict(self, posterior: GaussianPosterior, dt: float = 1.0) -> GaussianPosterior:
        ...

