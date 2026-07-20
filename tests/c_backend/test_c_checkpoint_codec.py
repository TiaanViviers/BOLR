from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bolr.backend.c_backend import CReplayEngine, make_restore_context
from bolr.config.foundation import DecisionPolicyConfig
from bolr.posterior.state import GaussianPosterior
from tests.c_backend.test_c_replay import _candidate_a_model, _fixed_transition


def _begin_day_kwargs() -> dict:
    return {
        "ranking_sample_count": 24,
        "chunk_size": 6,
        "top_k_values": (1, 2),
        "antithetic": True,
        "retention": "sample_zero",
    }


def test_c_replay_checkpoint_ready_round_trip_bytes() -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_posterior:
                with backend.rng(seed=11, stream=3) as rng:
                    with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                        assert engine.phase == 1
                        payload = engine.encode_checkpoint()
                        with CReplayEngine.decode_checkpoint(payload, artifacts) as restored:
                            assert restored.phase == 1
                            assert np.allclose(
                                restored.posterior_mean(artifacts.state_dimension),
                                engine.posterior_mean(artifacts.state_dimension),
                            )
                            assert np.allclose(
                                restored.posterior_covariance(artifacts.state_dimension),
                                engine.posterior_covariance(artifacts.state_dimension),
                            )
        finally:
            c_posterior.close()


def test_c_replay_checkpoint_pending_round_trip_bytes() -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_posterior:
                with backend.rng(seed=19, stream=1) as rng:
                    with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                        with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                            decision, _ = engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            assert engine.phase == 2
                            payload = engine.encode_checkpoint()
                            context = make_restore_context(artifacts)
                            with CReplayEngine.decode_checkpoint(payload, context) as restored:
                                assert restored.phase == 2
                                assert restored.pending_selected_index == decision.selected_index
        finally:
            c_posterior.close()


def test_c_replay_checkpoint_encode_is_deterministic() -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_posterior:
                with backend.rng(seed=23, stream=5) as rng:
                    with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                        first = engine.encode_checkpoint()
                        second = engine.encode_checkpoint()
                        assert first == second
        finally:
            c_posterior.close()


def test_c_replay_checkpoint_size_report() -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_posterior:
                with backend.rng(seed=31, stream=2) as rng:
                    with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                        with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                            engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            report = engine.checkpoint_size_report()
                            payload = engine.encode_checkpoint()
                            assert report.total_bytes == len(payload)
                            assert report.header_bytes == 180
                            assert report.state_dimension == artifacts.state_dimension
                            assert report.candidate_count == artifacts.candidate_count
                            assert report.directory_bytes + report.header_bytes + report.payload_bytes == report.total_bytes
        finally:
            c_posterior.close()
