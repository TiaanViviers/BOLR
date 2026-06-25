from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend, CGaussianState


def test_c_gaussian_sampling_antithetic_pairs() -> None:
    backend = CBackend()
    mean = np.array([0.5, -0.25], dtype=np.float64)
    covariance = np.array([[1.0, 0.3], [0.3, 2.0]], dtype=np.float64)
    with CGaussianState(mean, covariance, state_layout_hash=7, model_schema_hash=11, library=backend.library) as state:
        with backend.rng(seed=123, stream=7) as rng:
            samples, diagnostics = backend.sample_gaussian_state(state, rng, 6, antithetic=True)
    assert samples.shape == (6, 2)
    assert diagnostics.antithetic is True
    assert diagnostics.sample_count == 6
    assert np.allclose(samples[0] + samples[3], 2.0 * mean)
    assert np.allclose(samples[1] + samples[4], 2.0 * mean)
    assert np.allclose(samples[2] + samples[5], 2.0 * mean)
