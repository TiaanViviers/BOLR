import numpy as np

from bolr.adaptation.reset import PendingReset, apply_partial_reset
from bolr.model.state_layout import make_state_layout
from bolr.posterior.state import GaussianPosterior


def test_partial_reset_scales_mean_and_cross_covariance() -> None:
    layout = make_state_layout([{"name": "surface", "shape": (2,)}, {"name": "residual", "shape": (1,)}])
    posterior = GaussianPosterior(mean=np.array([2.0, -1.0, 0.5]), covariance=np.array([[2.0, 0.3, 0.4], [0.3, 1.5, -0.2], [0.4, -0.2, 0.8]]))
    reset = PendingReset("surface", 0.5, np.zeros(2), np.eye(2))
    updated = apply_partial_reset(posterior, layout, reset)
    assert np.allclose(updated.mean[:2], np.array([1.0, -0.5]))
    assert np.allclose(updated.covariance[:2, 2], 0.5 * posterior.covariance[:2, 2])
    assert np.min(np.linalg.eigvalsh(updated.covariance)) > 0.0
