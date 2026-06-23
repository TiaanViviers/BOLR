import numpy as np

from bolr.config.foundation import SoftTargetConfig
from bolr.numerics.derivatives import loss_score_observed_information
from bolr.targets.soft_target import build_soft_target_observation


def test_score_observed_information_is_psd() -> None:
    scores = np.array([0.2, -0.7, 1.3, 0.5])
    curvature = loss_score_observed_information(scores)
    eigenvalues = np.linalg.eigvalsh(curvature)
    assert np.min(eigenvalues) >= -1e-12


def test_degenerate_day_has_zero_update_weight() -> None:
    observation = build_soft_target_observation(
        np.zeros(5),
        SoftTargetConfig(absolute_tolerance=0.0, relative_tolerance=0.0),
    )
    assert observation.update_weight == 0.0
