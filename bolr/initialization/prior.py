from __future__ import annotations

import numpy as np

from bolr.posterior.state import GaussianPosterior


def make_initial_dynamic_prior(dimension: int, sigma0: float) -> GaussianPosterior:
    covariance = (sigma0**2) * np.eye(dimension, dtype=float)
    return GaussianPosterior(mean=np.zeros(dimension, dtype=float), covariance=covariance)

