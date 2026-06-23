import numpy as np

from bolr.config.foundation import SoftTargetConfig
from bolr.numerics.stable_math import log_softmax, softmax
from bolr.targets.soft_target import build_soft_target_observation


def test_softmax_is_stable_on_large_values() -> None:
    values = np.array([1000.0, 1001.0, 1002.0])
    probs = softmax(values)
    assert np.isfinite(log_softmax(values)).all()
    assert np.allclose(probs.sum(), 1.0)


def test_tolerance_aware_soft_target_can_no_update_degenerate_day() -> None:
    utilities = np.array([1.0, 1.0 + 1e-5, 1.0 - 1e-5])
    observation = build_soft_target_observation(
        utilities,
        SoftTargetConfig(absolute_tolerance=1e-4, relative_tolerance=0.0, no_update_if_degenerate=True),
    )
    assert observation.type == "NO_UPDATE"
    assert observation.update_weight == 0.0
    assert observation.metadata["tolerance_group_count"] == 1


def test_soft_target_returns_probability_vector() -> None:
    utilities = np.array([-2.0, 0.0, 3.0, 4.0])
    observation = build_soft_target_observation(utilities, SoftTargetConfig(kappa=1.5, eta=0.75))
    assert observation.type == "SOFT_TARGET"
    assert np.allclose(observation.target_probabilities.sum(), 1.0)
    assert observation.update_weight == 0.75
