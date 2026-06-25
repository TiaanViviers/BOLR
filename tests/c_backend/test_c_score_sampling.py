from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock
from bolr.posterior.state import GaussianPosterior


def _artifacts(backend: CBackend, design: np.ndarray) -> tuple[object, object]:
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", np.asarray(design, dtype=float))], {})
    artifacts = backend.model_artifacts(model, {})
    return model, artifacts


def test_c_score_sampling_matches_linear_transform() -> None:
    backend = CBackend()
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    _, artifacts = _artifacts(backend, design)
    try:
        state_samples = np.array([[1.0, 2.0], [0.5, -1.5], [-0.2, 0.3]], dtype=np.float64)
        scores, diagnostics = backend.score_samples_from_state_samples(artifacts, state_samples, None)
        assert diagnostics.sample_count == 3
        assert diagnostics.candidate_count == 3
        assert np.allclose(scores, state_samples @ design.T)

        posterior = GaussianPosterior(mean=np.array([0.0, 0.0]), covariance=np.eye(2))
        with artifacts.state_from_posterior(posterior) as state:
            with backend.rng(seed=123, stream=7) as rng:
                direct_scores, _ = backend.sample_posterior_scores(state, artifacts, None, rng, 5, antithetic=True)
            with backend.rng(seed=123, stream=7) as rng:
                state_draws, _ = backend.sample_gaussian_state(state, rng, 5, antithetic=True)
        assert np.allclose(direct_scores, state_draws @ design.T)
    finally:
        artifacts.close()
