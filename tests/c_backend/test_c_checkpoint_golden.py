from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bolr.backend.c_backend import CBackend, CReplayEngine, make_restore_context
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.posterior.state import GaussianPosterior


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"


def _fixture_model() -> CompositeScoreModel:
    phi = np.array([[1.0, 0.0], [0.2, 0.8], [-0.3, 0.4]], dtype=float)
    return CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, np.array([0.1, -0.2], dtype=float), {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )


def test_golden_checkpoint_fixtures_decode() -> None:
    meta = json.loads((FIXTURE_DIR / "c_checkpoint_v1.json").read_text())
    ready = (FIXTURE_DIR / "c_checkpoint_ready_v1.bin").read_bytes()
    pending = (FIXTURE_DIR / "c_checkpoint_pending_v1.bin").read_bytes()
    assert ready[:8] == b"BOLRCP01"
    assert pending[:8] == b"BOLRCP01"
    assert len(ready) == meta["ready"]["file_size"]
    assert len(pending) == meta["pending"]["file_size"]

    backend = CBackend()
    model = _fixture_model()
    prior = GaussianPosterior(mean=np.zeros(2), covariance=np.eye(2))
    with backend.model_artifacts(model, {}) as artifacts:
        _ = artifacts.state_from_posterior(prior)
        ctx = make_restore_context(artifacts)
        ready_engine = CReplayEngine.decode_checkpoint(ready, ctx, library=backend.library)
        pending_engine = CReplayEngine.decode_checkpoint(pending, ctx, library=backend.library)
        try:
            assert ready_engine.phase == meta["ready"]["expected_restored_phase"]
            assert pending_engine.phase == meta["pending"]["expected_restored_phase"]
            assert pending_engine.pending_selected_index == meta["pending"]["selected_index"]
        finally:
            ready_engine.close()
            pending_engine.close()
