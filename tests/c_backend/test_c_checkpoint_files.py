from __future__ import annotations

import numpy as np
import pytest

from bolr.backend.c_backend import CReplayEngine, make_restore_context
from bolr.config.foundation import DecisionPolicyConfig
from bolr.posterior.state import GaussianPosterior
from tests.c_backend.test_c_replay import _candidate_a_model, _fixed_transition


def _begin_day_kwargs() -> dict:
    return {
        "ranking_sample_count": 20,
        "chunk_size": 5,
        "top_k_values": (1, 2),
        "antithetic": True,
        "retention": "sample_zero",
    }


def test_c_replay_checkpoint_write_and_read_file(tmp_path) -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)
    checkpoint_path = tmp_path / "replay.ckpt"

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_posterior:
                with backend.rng(seed=37, stream=4) as rng:
                    with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                        with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                            decision, _ = engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            engine.write_checkpoint(checkpoint_path)
                            context = make_restore_context(artifacts)
                            with CReplayEngine.read_checkpoint(checkpoint_path, context) as restored:
                                assert restored.phase == 2
                                assert restored.pending_selected_index == decision.selected_index
                                assert np.allclose(
                                    restored.posterior_mean(artifacts.state_dimension),
                                    engine.posterior_mean(artifacts.state_dimension),
                                )
        finally:
            c_posterior.close()


def test_c_replay_checkpoint_write_replace_existing(tmp_path) -> None:
    from bolr.backend.c_api import CCheckpointIOError
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)
    checkpoint_path = tmp_path / "replay.ckpt"

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_posterior:
                with backend.rng(seed=43, stream=6) as rng:
                    with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                        engine.write_checkpoint(checkpoint_path, replace_existing=True)
                        with pytest.raises(CCheckpointIOError):
                            engine.write_checkpoint(checkpoint_path, replace_existing=False)
                        engine.write_checkpoint(checkpoint_path, replace_existing=True)
                        with CReplayEngine.read_checkpoint(checkpoint_path, artifacts) as restored:
                            assert restored.phase == 1
        finally:
            c_posterior.close()
