from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.config.foundation import SoftTargetConfig
from bolr.targets.soft_target import SoftTargetBuilder


def test_c_candidate_a_target_matches_python_builder() -> None:
    backend = CBackend()
    config = SoftTargetConfig(kappa=1.25, eta=0.8, clip=3.5, absolute_tolerance=0.1, relative_tolerance=0.2)
    utilities = np.array([3.0, 2.7, -0.5, -1.2])
    python_observation = SoftTargetBuilder(config).build(utilities)

    target, update_weight, diagnostics = backend.build_candidate_a_target(utilities, config)

    assert np.allclose(target, python_observation.target_probabilities, atol=1e-12, rtol=1e-12)
    assert np.isclose(update_weight, python_observation.update_weight)
    assert diagnostics.candidate_count == utilities.size
    assert diagnostics.positive_candidate_count == int((utilities > 0.0).sum())
    assert np.isclose(diagnostics.target_entropy, python_observation.metadata["target_entropy"])
    assert np.isclose(diagnostics.utility_scale, python_observation.metadata["utility_scale"])
    assert np.isclose(diagnostics.clipping_fraction, python_observation.metadata["clipping_fraction"])

