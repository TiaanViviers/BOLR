from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend, CGaussianState


def test_c_sampling_statistics_are_reasonable() -> None:
    backend = CBackend()
    mean = np.array([0.25, -0.5], dtype=np.float64)
    covariance = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64)
    with CGaussianState(mean, covariance, state_layout_hash=7, model_schema_hash=11, library=backend.library) as state:
        with backend.rng(seed=321, stream=9) as rng:
            samples, _ = backend.sample_gaussian_state(state, rng, 4000)
    sample_mean = samples.mean(axis=0)
    sample_cov = np.cov(samples, rowvar=False)
    assert np.allclose(sample_mean, mean, atol=0.06)
    assert np.allclose(sample_cov, covariance, atol=0.08)
