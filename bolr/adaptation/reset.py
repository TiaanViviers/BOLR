from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.model.state_layout import StateLayout
from bolr.posterior.state import GaussianPosterior


@dataclass(frozen=True)
class PendingReset:
    block_name: str
    strength: float
    anchor_mean: np.ndarray
    anchor_covariance: np.ndarray


def apply_partial_reset(posterior: GaussianPosterior, layout: StateLayout, reset: PendingReset) -> GaussianPosterior:
    if not (0.0 <= reset.strength <= 1.0):
        raise ValueError("reset strength must lie in [0, 1].")
    block = layout._block(reset.block_name)
    sl = slice(block.start, block.stop)
    mean = posterior.mean.copy()
    covariance = posterior.covariance.copy()
    a = 1.0 - reset.strength
    mean[sl] = reset.anchor_mean + a * (mean[sl] - reset.anchor_mean)
    covariance[sl, :] *= a
    covariance[:, sl] *= a
    covariance[sl, sl] += (1.0 - a * a) * reset.anchor_covariance
    return GaussianPosterior(
        mean=mean,
        covariance=covariance,
        state_layout=posterior.state_layout,
        timestamp=posterior.timestamp,
        version=posterior.version,
        diagnostics={**posterior.diagnostics, "partial_reset_block": reset.block_name, "partial_reset_strength": reset.strength},
    )
