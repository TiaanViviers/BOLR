from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.config.foundation import CrossGroupLogisticConfig, OrderedPartitionConfig
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_c_candidate_b_sampled_matches_python_materialized_pairs() -> None:
    backend = CBackend()
    observation = OrderedPartitionBuilder(OrderedPartitionConfig()).build(np.array([2.0, 1.6, 0.2, -0.5, -1.2]), date="2024-01-03")
    scores = np.array([0.6, 0.3, 0.1, -0.4, -0.8])
    vector = np.array([0.2, -0.1, 0.4, -0.3, 0.6])
    config = CrossGroupLogisticConfig(normalize_pair_losses=True, sampled_pair_budget=2, sampled_with_replacement=False, sampling_seed=7)
    python_model = CrossGroupLogisticObservationModel(config)

    materialized = backend.materialize_candidate_b_pairs(observation, config)
    with backend.candidate_b_sampled_observation(observation, config) as handle:
        diagnostics = handle.diagnostics()
        value, gradient, hvp = backend._evaluate_operator(handle.operator(), scores, vector)

    assert diagnostics.used_pair_count == materialized["used_pair_count"]
    assert diagnostics.duplicate_sample_count == materialized["duplicate_sample_count"]
    assert np.isclose(value, python_model.log_factor(scores, observation))
    assert np.allclose(gradient, python_model.score_gradient(scores, observation), atol=1e-12, rtol=1e-12)
    assert np.allclose(hvp, python_model.score_curvature_hvp(scores, vector, observation), atol=1e-12, rtol=1e-12)

