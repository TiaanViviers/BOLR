from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.config.foundation import CrossGroupLogisticConfig, OrderedPartitionConfig, OrderedPartitionToleranceConfig
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_c_candidate_b_exact_matches_python_derivatives() -> None:
    backend = CBackend()
    observation = OrderedPartitionBuilder(
        OrderedPartitionConfig(tolerance=OrderedPartitionToleranceConfig(absolute_tolerance=0.15), positive_threshold=0.0)
    ).build(np.array([1.4, 1.3, 0.2, -0.8]), date="2024-01-01")
    scores = np.array([0.5, 0.2, -0.1, -0.4])
    vector = np.array([0.1, -0.3, 0.2, 0.5])
    config = CrossGroupLogisticConfig(normalize_pair_losses=True, sampled_pair_budget=None)
    python_model = CrossGroupLogisticObservationModel(config)

    value, gradient, hvp = backend.candidate_b_value_gradient_hvp(scores, observation, config, vector)

    assert np.isclose(value, python_model.log_factor(scores, observation))
    assert np.allclose(gradient, python_model.score_gradient(scores, observation), atol=1e-12, rtol=1e-12)
    assert np.allclose(hvp, python_model.score_curvature_hvp(scores, vector, observation), atol=1e-12, rtol=1e-12)

