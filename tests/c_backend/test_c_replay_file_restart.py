from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CReplayEngine, make_restore_context
from bolr.config.foundation import DecisionPolicyConfig
from bolr.posterior.state import GaussianPosterior
from bolr.targets.soft_target import SoftTargetBuilder
from tests.c_backend.test_c_replay import _adaptive_policy, _candidate_a_model, _fixed_transition


def _begin_day_kwargs() -> dict:
    return {
        "ranking_sample_count": 24,
        "chunk_size": 6,
        "top_k_values": (1, 2),
        "antithetic": True,
        "retention": "sample_zero",
    }


def _run_candidate_a_days(backend, engine, artifacts, decision_policy, days) -> list[int]:
    selected: list[int] = []
    builder = SoftTargetBuilder()
    for values, date in days:
        observation = builder.build(np.asarray(values, dtype=float), date=date)
        decision, _ = engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
        selected.append(int(decision.selected_index))
        with backend.candidate_a_observation(observation) as obs_handle:
            engine.finish_day(artifacts, obs_handle)
    return selected


def test_c_candidate_a_file_restart_matches_uninterrupted_sequence(tmp_path) -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)
    days = [
        ([1.0, 0.3, -0.2], "2026-01-01"),
        ([0.2, 0.9, -0.1], "2026-01-02"),
        ([-0.4, 0.1, 0.7], "2026-01-03"),
    ]
    checkpoint_path = tmp_path / "mid_ready.ckpt"

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior_direct = artifacts.state_from_posterior(prior)
        c_posterior_restart = artifacts.state_from_posterior(prior)
        try:
            with c_posterior_direct, c_posterior_restart:
                with backend.rng(seed=51, stream=2) as rng_direct, backend.rng(seed=51, stream=2) as rng_restart:
                    with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                        with backend.replay_engine_fixed(c_posterior_direct, transition, rng_direct) as direct_engine:
                            direct_selected = _run_candidate_a_days(backend, direct_engine, artifacts, decision_policy, days)
                            direct_mean = direct_engine.posterior_mean(artifacts.state_dimension)

                        with backend.replay_engine_fixed(c_posterior_restart, transition, rng_restart) as restart_engine:
                            restart_selected = _run_candidate_a_days(backend, restart_engine, artifacts, decision_policy, days[:2])
                            restart_engine.write_checkpoint(checkpoint_path)
                            with CReplayEngine.read_checkpoint(checkpoint_path, artifacts) as resumed_engine:
                                restart_selected.extend(_run_candidate_a_days(backend, resumed_engine, artifacts, decision_policy, days[2:]))
                                resumed_mean = resumed_engine.posterior_mean(artifacts.state_dimension)

                            assert restart_selected == direct_selected
                            assert np.allclose(resumed_mean, direct_mean)
        finally:
            c_posterior_direct.close()
            c_posterior_restart.close()


def test_c_candidate_a_pending_file_restart_matches_twin(tmp_path) -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    builder = SoftTargetBuilder()
    day1 = builder.build(np.array([1.0, 0.3, -0.2]), date="2026-01-01")
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)
    checkpoint_path = tmp_path / "pending.ckpt"

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior_a = artifacts.state_from_posterior(prior)
        c_posterior_b = artifacts.state_from_posterior(prior)
        try:
            with c_posterior_a, c_posterior_b:
                with backend.rng(seed=59, stream=3) as rng_a, backend.rng(seed=59, stream=3) as rng_b:
                    with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                        with backend.replay_engine_fixed(c_posterior_a, transition, rng_a) as interrupted_engine:
                            interrupted_decision, _ = interrupted_engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            interrupted_engine.write_checkpoint(checkpoint_path)

                        with backend.replay_engine_fixed(c_posterior_b, transition, rng_b) as direct_engine:
                            direct_decision, _ = direct_engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            with backend.candidate_a_observation(day1) as observation:
                                direct_engine.finish_day(artifacts, observation)

                            with CReplayEngine.read_checkpoint(checkpoint_path, artifacts) as restored_engine:
                                assert restored_engine.pending_selected_index == interrupted_decision.selected_index
                                with backend.candidate_a_observation(day1) as observation:
                                    restored_engine.finish_day(artifacts, observation)
                                assert interrupted_decision.selected_index == direct_decision.selected_index
                                assert np.allclose(
                                    restored_engine.posterior_mean(artifacts.state_dimension),
                                    direct_engine.posterior_mean(artifacts.state_dimension),
                                )
        finally:
            c_posterior_a.close()
            c_posterior_b.close()


def test_c_adaptive_file_restart_matches_uninterrupted_sequence(tmp_path) -> None:
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    policy = _adaptive_policy(np.eye(model.layout.total_dimension) * 0.05)
    builder = SoftTargetBuilder()
    day1 = builder.build(np.array([1.0, 0.3, -0.2]), date="2026-01-01")
    day2 = builder.build(np.array([0.2, 0.9, -0.1]), date="2026-01-02")
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    checkpoint_path = tmp_path / "adaptive.ckpt"

    with backend.model_artifacts(model, {}) as artifacts:
        c_policy = backend.adaptive_policy(policy, model.layout)
        c_state_direct = backend.adaptive_state(c_policy)
        c_state_restart = backend.adaptive_state(c_policy)
        c_posterior_direct = artifacts.state_from_posterior(prior)
        c_posterior_restart = artifacts.state_from_posterior(prior)
        try:
            with c_policy, c_state_direct, c_state_restart, c_posterior_direct, c_posterior_restart:
                with backend.rng(seed=67, stream=8) as rng_direct, backend.rng(seed=67, stream=8) as rng_restart:
                    with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                        with backend.replay_engine_adaptive(c_posterior_direct, c_policy, c_state_direct, rng_direct) as direct_engine:
                            direct_day1, _ = direct_engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            with backend.candidate_a_observation(day1) as observation:
                                direct_engine.finish_day(artifacts, observation)
                            direct_day2, _ = direct_engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            with backend.candidate_a_observation(day2) as observation:
                                direct_engine.finish_day(artifacts, observation)
                            direct_mean = direct_engine.posterior_mean(artifacts.state_dimension)

                        with backend.replay_engine_adaptive(c_posterior_restart, c_policy, c_state_restart, rng_restart) as restart_engine:
                            restart_day1, _ = restart_engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                            with backend.candidate_a_observation(day1) as observation:
                                restart_engine.finish_day(artifacts, observation)
                            restart_engine.write_checkpoint(checkpoint_path)
                            context = make_restore_context(artifacts, adaptive_policy=c_policy)
                            with CReplayEngine.read_checkpoint(checkpoint_path, context) as resumed_engine:
                                restart_day2, _ = resumed_engine.begin_day(artifacts, decision_policy, **_begin_day_kwargs())
                                with backend.candidate_a_observation(day2) as observation:
                                    resumed_engine.finish_day(artifacts, observation)
                                assert restart_day1.selected_index == direct_day1.selected_index
                                assert restart_day2.selected_index == direct_day2.selected_index
                                assert np.allclose(
                                    resumed_engine.posterior_mean(artifacts.state_dimension),
                                    direct_mean,
                                )
        finally:
            c_posterior_direct.close()
            c_posterior_restart.close()
            c_state_direct.close()
            c_state_restart.close()
            c_policy.close()
