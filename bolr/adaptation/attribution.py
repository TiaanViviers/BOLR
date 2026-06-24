from __future__ import annotations

import numpy as np

from bolr.model.state_layout import StateLayout
from bolr.posterior.diagnostics import jittered_cholesky


def block_innovation_attribution(
    layout: StateLayout,
    predictive_mean: np.ndarray,
    predictive_covariance: np.ndarray,
    posterior_mean: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> dict[str, dict[str, float]]:
    predictive_mean = np.asarray(predictive_mean, dtype=float)
    predictive_covariance = np.asarray(predictive_covariance, dtype=float)
    posterior_mean = np.asarray(posterior_mean, dtype=float)
    delta = posterior_mean - predictive_mean
    mahal = {}
    euclid = {}
    for block in layout.blocks:
        sl = slice(block.start, block.stop)
        d = delta[sl]
        cov = predictive_covariance[sl, sl]
        chol = jittered_cholesky(cov).factor
        whitened = np.linalg.solve(chol, d)
        mahal[block.name] = float(whitened @ whitened)
        euclid[block.name] = float(d @ d)
    denom = sum(value + epsilon for value in mahal.values())
    return {
        block.name: {
            "euclidean_update_energy": euclid[block.name],
            "mahalanobis_update_energy": mahal[block.name],
            "attribution_weight": float((mahal[block.name] + epsilon) / denom),
        }
        for block in layout.blocks
    }
