from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy
from bolr.backend.c_backend import CBackend
from bolr.config.foundation import (
    AdaptiveTransitionConfig,
    BlockAdaptationConfig,
    CrossGroupLogisticConfig,
    DecisionPolicyConfig,
    OrderedPartitionConfig,
    RegionDefinitionConfig,
    SurpriseStandardizerConfig,
    BOCPDConfig,
)
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.posterior.state import GaussianPosterior
from bolr.targets.ordered_partition import OrderedPartitionBuilder
from bolr.targets.soft_target import SoftTargetBuilder


def _fixed_transition(dimension: int) -> SimpleNamespace:
    return SimpleNamespace(
        family="additive",
        process_noise=np.eye(dimension, dtype=float) * 0.05,
        global_discount=0.0,
        block_discount_scales=None,
    )


def _adaptive_policy(q: np.ndarray) -> AdaptiveAdditiveTransitionPolicy:
    return AdaptiveAdditiveTransitionPolicy(
        q,
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=2.0, decay=0.0),),
        ),
    )


def _candidate_a_model() -> CompositeScoreModel:
    phi = np.array([[1.0, 0.0], [0.2, 0.8], [-0.3, 0.4]])
    static_coeffs = np.array([0.1, -0.2])
    return CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )


def _candidate_b_model() -> CompositeScoreModel:
    phi = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5]])
    static_coeffs = np.array([0.2, -0.1])
    return CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )


def test_c_adaptive_replay_checkpoint_resume_matches_uninterrupted_sequence() -> None:
    backend = CBackend()
    model = _candidate_a_model()
    policy = _adaptive_policy(np.eye(model.layout.total_dimension) * 0.05)
    builder = SoftTargetBuilder()
    day1 = builder.build(np.array([1.0, 0.3, -0.2]), date="2026-01-01")
    day2 = builder.build(np.array([0.2, 0.9, -0.1]), date="2026-01-02")
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))

    with backend.model_artifacts(model, {}) as artifacts:
        c_policy = backend.adaptive_policy(policy, model.layout)
        c_state = backend.adaptive_state(c_policy)
        c_posterior = artifacts.state_from_posterior(prior)
        try:
            with c_policy, c_state, c_posterior:
                with backend.rng(seed=41, stream=9) as rng_a, backend.rng(seed=41, stream=9) as rng_b:
                    with backend.replay_engine_adaptive(c_posterior, c_policy, c_state, rng_a) as resumed_engine:
                        with backend.replay_engine_adaptive(c_posterior, c_policy, c_state, rng_b) as direct_engine:
                            with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                                resumed_decision, _ = resumed_engine.begin_day(
                                    artifacts,
                                    decision_policy,
                                    ranking_sample_count=32,
                                    chunk_size=7,
                                    top_k_values=(1, 2),
                                    antithetic=True,
                                    retention="sample_zero",
                                )
                                with resumed_engine.export_checkpoint() as checkpoint:
                                    with backend.replay_engine_import_adaptive(checkpoint, c_policy) as restored_engine:
                                        with backend.candidate_a_observation(day1) as day1_observation:
                                            restored_engine.finish_day(artifacts, day1_observation)
                                        direct_decision, _ = direct_engine.begin_day(
                                            artifacts,
                                            decision_policy,
                                            ranking_sample_count=32,
                                            chunk_size=7,
                                            top_k_values=(1, 2),
                                            antithetic=True,
                                            retention="sample_zero",
                                        )
                                        with backend.candidate_a_observation(day1) as day1_observation:
                                            direct_engine.finish_day(artifacts, day1_observation)

                                        restored_day2, _ = restored_engine.begin_day(
                                            artifacts,
                                            decision_policy,
                                            ranking_sample_count=32,
                                            chunk_size=7,
                                            top_k_values=(1, 2),
                                            antithetic=True,
                                            retention="sample_zero",
                                        )
                                        direct_day2, _ = direct_engine.begin_day(
                                            artifacts,
                                            decision_policy,
                                            ranking_sample_count=32,
                                            chunk_size=7,
                                            top_k_values=(1, 2),
                                            antithetic=True,
                                            retention="sample_zero",
                                        )
                                        with backend.candidate_a_observation(day2) as day2_observation:
                                            restored_engine.finish_day(artifacts, day2_observation)
                                        with backend.candidate_a_observation(day2) as day2_observation:
                                            direct_engine.finish_day(artifacts, day2_observation)

                                        assert resumed_decision.selected_index == direct_decision.selected_index
                                        assert restored_day2.selected_index == direct_day2.selected_index
                                        assert np.allclose(
                                            restored_engine.posterior_mean(artifacts.state_dimension),
                                            direct_engine.posterior_mean(artifacts.state_dimension),
                                        )
                                        assert np.allclose(
                                            restored_engine.posterior_covariance(artifacts.state_dimension),
                                            direct_engine.posterior_covariance(artifacts.state_dimension),
                                        )
        finally:
            c_posterior.close()
            c_state.close()
            c_policy.close()


@pytest.mark.parametrize("sampled", [False, True])
def test_c_candidate_b_replay_checkpoint_resume_matches_uninterrupted_sequence(sampled: bool) -> None:
    backend = CBackend()
    model = _candidate_b_model()
    builder = OrderedPartitionBuilder(OrderedPartitionConfig())
    observation = builder.build(np.array([1.0, 0.3, -0.4, 0.2]), date="2026-02-01")
    config = CrossGroupLogisticConfig(sampled_pair_budget=4 if sampled else None, sampling_seed=17)
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)

    with backend.model_artifacts(model, {}) as artifacts:
        c_posterior_a = artifacts.state_from_posterior(prior)
        c_posterior_b = artifacts.state_from_posterior(prior)
        try:
            with c_posterior_a, c_posterior_b:
                with backend.rng(seed=29, stream=4) as rng_a, backend.rng(seed=29, stream=4) as rng_b:
                    with backend.replay_engine_fixed(c_posterior_a, transition, rng_a) as resumed_engine:
                        with backend.replay_engine_fixed(c_posterior_b, transition, rng_b) as direct_engine:
                            with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                                resumed_decision, _ = resumed_engine.begin_day(
                                    artifacts,
                                    decision_policy,
                                    ranking_sample_count=24,
                                    chunk_size=5,
                                    top_k_values=(1, 2),
                                    antithetic=True,
                                    retention="sample_zero",
                                )
                                with resumed_engine.export_checkpoint() as checkpoint:
                                    with backend.replay_engine_import_fixed(checkpoint) as restored_engine:
                                        if sampled:
                                            obs_handle = backend.candidate_b_sampled_observation(observation, config)
                                        else:
                                            obs_handle = backend.candidate_b_exact_observation(observation)
                                        with obs_handle:
                                            restored_engine.finish_day(artifacts, obs_handle)
                                            direct_decision, _ = direct_engine.begin_day(
                                                artifacts,
                                                decision_policy,
                                                ranking_sample_count=24,
                                                chunk_size=5,
                                                top_k_values=(1, 2),
                                                antithetic=True,
                                                retention="sample_zero",
                                            )
                                            direct_engine.finish_day(artifacts, obs_handle)

                                        assert resumed_decision.selected_index == direct_decision.selected_index
                                        assert np.allclose(
                                            restored_engine.posterior_mean(artifacts.state_dimension),
                                            direct_engine.posterior_mean(artifacts.state_dimension),
                                        )
                                        assert np.allclose(
                                            restored_engine.posterior_covariance(artifacts.state_dimension),
                                            direct_engine.posterior_covariance(artifacts.state_dimension),
                                        )
        finally:
            c_posterior_a.close()
            c_posterior_b.close()


def test_c_replay_transactional_failures_preserve_phase_and_allow_recovery() -> None:
    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)
    observation = SoftTargetBuilder().build(np.array([0.8, 0.1, -0.4]), date="2026-03-01")
    mismatch_model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("other", np.eye(2))], {})

    with backend.model_artifacts(model, {}) as artifacts:
        with backend.model_artifacts(mismatch_model, {}) as mismatch_artifacts:
            c_posterior = artifacts.state_from_posterior(prior)
            try:
                with c_posterior:
                    with backend.rng(seed=13, stream=6) as rng:
                        with backend.replay_engine_fixed(c_posterior, transition, rng) as engine:
                            with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as decision_policy:
                                with pytest.raises(RuntimeError):
                                    engine.begin_day(
                                        artifacts,
                                        decision_policy,
                                        ranking_sample_count=16,
                                        chunk_size=4,
                                        top_k_values=(1,),
                                        antithetic=True,
                                        retention="sample_zero",
                                        region_config=RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold"),
                                    )
                                assert engine.phase == 1
                                decision, _ = engine.begin_day(
                                    artifacts,
                                    decision_policy,
                                    ranking_sample_count=16,
                                    chunk_size=4,
                                    top_k_values=(1,),
                                    antithetic=True,
                                    retention="sample_zero",
                                )
                                assert engine.phase == 2
                                assert engine.pending_selected_index == decision.selected_index
                                with backend.candidate_a_observation(observation) as obs_handle:
                                    with pytest.raises(RuntimeError):
                                        engine.finish_day(mismatch_artifacts, obs_handle)
                                    assert engine.phase == 2
                                    engine.finish_day(artifacts, obs_handle)
                                assert engine.phase == 1
            finally:
                c_posterior.close()
