from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CCheckpointState, CGaussianState, CLibrary


def test_c_gaussian_state_round_trip_and_prediction() -> None:
    library = CLibrary()
    mean = np.array([0.25, -0.5], dtype=np.float64)
    covariance = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    process_noise = np.array([[0.2, 0.0], [0.0, 0.3]], dtype=np.float64)

    with CGaussianState(mean, covariance, state_layout_hash=7, model_schema_hash=11, library=library) as state:
        assert state.dimension == 2
        assert state.step_index == 0
        assert np.allclose(state.mean(), mean)
        assert np.allclose(state.covariance(), covariance)

        with state.export_checkpoint() as checkpoint:
            payload = checkpoint.to_bytes()
            with CCheckpointState.from_bytes(payload, library=library) as decoded:
                with CGaussianState.import_checkpoint(decoded, library=library) as restored:
                    assert restored.step_index == 0
                    assert np.allclose(restored.mean(), mean)
                    assert np.allclose(restored.covariance(), covariance)

        with state.predict_additive(process_noise)[0] as predictive:
            assert predictive.step_index == 1
            assert np.allclose(predictive.mean(), mean)
            assert np.allclose(predictive.covariance(), covariance + process_noise)
