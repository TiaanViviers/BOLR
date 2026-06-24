import numpy as np

from bolr.model.diagnostics import block_innovation_diagnostics, roughness_diagnostics
from bolr.model.penalties import difference_penalty
from bolr.model.structured import prior_from_penalty


def test_roughness_diagnostics_use_prior_precision_when_supplied() -> None:
    penalty = difference_penalty(4, 2)
    prior = prior_from_penalty(penalty, smooth_weight=2.5, ridge=0.4)
    state = np.array([0.1, 0.2, -0.1, 0.05])
    diagnostics = roughness_diagnostics(
        state,
        penalty,
        prior_mean=prior.mean,
        prior_precision=prior.precision,
    )
    expected = float(state @ prior.precision @ state)
    assert np.isclose(diagnostics["prior_standardized_norm"], expected)


def test_block_innovation_diagnostics_are_zero_for_zero_update() -> None:
    mean = np.array([0.2, -0.1])
    covariance = np.array([[1.5, 0.1], [0.1, 0.8]])
    diagnostics = block_innovation_diagnostics(mean, covariance, mean)
    assert diagnostics["update_l2"] == 0.0
    assert diagnostics["update_mahalanobis"] == 0.0
