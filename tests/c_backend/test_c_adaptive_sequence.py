from __future__ import annotations

import numpy as np

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy
from bolr.backend.c_backend import CBackend, CInferenceWorkspace
from bolr.config.foundation import AdaptiveTransitionConfig, BlockAdaptationConfig, BOCPDConfig, OrderedPartitionConfig, SurpriseStandardizerConfig
from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.diagnostics import innovation_diagnostics
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.ordered_partition import OrderedPartitionBuilder
from bolr.targets.soft_target import SoftTargetBuilder


def _adaptive_policy(q: np.ndarray) -> AdaptiveAdditiveTransitionPolicy:
    return AdaptiveAdditiveTransitionPolicy(
        q,
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=2.0, decay=0.0),),
        ),
    )


def _candidate_a_sequence() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.2, 0.8], [-0.3, 0.4]])
    static_coeffs = np.array([0.1, -0.2])
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    policy = _adaptive_policy(np.eye(2) * 0.05)
    observation_model = SoftTargetObservationModel()
    builder = SoftTargetBuilder()
    python_state = policy.initial_state(layout=model.layout)
    python_posterior = GaussianPosterior(mean=np.zeros(2), covariance=np.eye(2))
    with backend.model_artifacts(model, {}) as artifacts:
        c_policy = backend.adaptive_policy(policy, model.layout)
        c_state = backend.adaptive_state(c_policy)
        c_posterior = artifacts.state_from_posterior(python_posterior)
        try:
            with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                for day, utilities in enumerate((np.array([1.0, 0.3, -0.2]), np.array([0.2, 0.9, -0.1]))):
                    python_predictive, python_state, _ = policy.predict(python_posterior, python_state, layout=model.layout)
                    c_predictive, _ = backend.adaptive_predict(c_policy, c_state, c_posterior)
                    observation = builder.build(utilities, date=f"2026-01-0{day + 1}")
                    python_result = laplace_update_composite(python_predictive, model, {}, observation, observation_model=observation_model)
                    with c_predictive:
                        with backend.candidate_a_observation(observation) as c_obs:
                            c_posterior_next, c_laplace = backend.laplace_update(c_predictive, artifacts, backend.score_context(model, {}), c_obs, workspace)
                            c_post = c_posterior_next.to_posterior(state_layout=model.layout.metadata())
                            log_prior = observation_model.log_factor(model.scores({}, python_predictive.mean), observation)
                            log_post = observation_model.log_factor(model.scores({}, python_result.posterior.mean), observation)
                            innovation = innovation_diagnostics(
                                python_predictive.mean,
                                python_predictive.covariance,
                                python_result.posterior.mean,
                                python_result.posterior.covariance,
                                log_prior,
                                log_post,
                            )
                            python_state, _ = policy.observe_update(
                                predictive_posterior=python_predictive,
                                posterior=python_result.posterior,
                                observation_diagnostics={},
                                block_diagnostics={},
                                policy_state=python_state,
                                layout=model.layout,
                                predictive_scores=model.scores({}, python_predictive.mean),
                                posterior_scores=model.scores({}, python_result.posterior.mean),
                                observation=observation,
                                observation_model=observation_model,
                                date=f"2026-01-0{day + 1}",
                            )
                            c_diag = backend.adaptive_observe(
                                c_policy,
                                c_state,
                                c_predictive,
                                c_posterior_next,
                                log_factor_at_predictive_mean=log_prior,
                                log_factor_at_posterior_mode=log_post,
                                effective_strength=float(observation.update_weight),
                                information_size=float(max(observation.metadata.get("tolerance_group_count", 1), 1)),
                                mahalanobis_update=float(innovation["state_update_mahalanobis"]),
                                gaussian_kl=float(innovation["gaussian_kl"]),
                                objective_improvement=float(innovation["objective_improvement"]),
                            )
                            assert np.allclose(c_post.mean, python_result.posterior.mean, atol=1e-8, rtol=1e-8)
                            assert np.allclose(c_state.block_multipliers(), np.array([python_state.block_multipliers["surface"]]), atol=1e-8, rtol=1e-8)
                            assert np.allclose(c_state.run_length_posterior(), np.exp(python_state.change_detector_state.log_run_length_posterior), atol=1e-8, rtol=1e-8)
                            assert np.isclose(c_diag.change_probability, python_state.change_detector_state and np.exp(python_state.change_detector_state.log_run_length_posterior[0]))
                            c_posterior.close()
                            c_posterior = c_posterior_next
                    python_posterior = python_result.posterior
        finally:
            c_posterior.close()
            c_state.close()
            c_policy.close()


def _candidate_b_sequence() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5]])
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi)], {})
    policy = _adaptive_policy(np.eye(2) * 0.05)
    observation_model = CrossGroupLogisticObservationModel()
    builder = OrderedPartitionBuilder(OrderedPartitionConfig())
    python_state = policy.initial_state(layout=model.layout)
    python_posterior = GaussianPosterior(mean=np.zeros(2), covariance=np.eye(2))
    with backend.model_artifacts(model, {}) as artifacts:
        c_policy = backend.adaptive_policy(policy, model.layout)
        c_state = backend.adaptive_state(c_policy)
        c_posterior = artifacts.state_from_posterior(python_posterior)
        try:
            with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                for day, utilities in enumerate((np.array([1.0, 0.2, -0.4, 0.3]), np.array([0.0, 0.5, 0.6, -0.1]))):
                    python_predictive, python_state, _ = policy.predict(python_posterior, python_state, layout=model.layout)
                    c_predictive, _ = backend.adaptive_predict(c_policy, c_state, c_posterior)
                    observation = builder.build(utilities, date=f"2026-02-0{day + 1}")
                    python_result = laplace_update_composite(python_predictive, model, {}, observation, observation_model=observation_model)
                    with c_predictive:
                        with backend.candidate_b_exact_observation(observation) as c_obs:
                            c_posterior_next, _ = backend.laplace_update(c_predictive, artifacts, backend.score_context(model, {}), c_obs, workspace)
                            log_prior = observation_model.log_factor(model.scores({}, python_predictive.mean), observation)
                            log_post = observation_model.log_factor(model.scores({}, python_result.posterior.mean), observation)
                            innovation = innovation_diagnostics(
                                python_predictive.mean,
                                python_predictive.covariance,
                                python_result.posterior.mean,
                                python_result.posterior.covariance,
                                log_prior,
                                log_post,
                            )
                            python_state, _ = policy.observe_update(
                                predictive_posterior=python_predictive,
                                posterior=python_result.posterior,
                                observation_diagnostics={},
                                block_diagnostics={},
                                policy_state=python_state,
                                layout=model.layout,
                                predictive_scores=model.scores({}, python_predictive.mean),
                                posterior_scores=model.scores({}, python_result.posterior.mean),
                                observation=observation,
                                observation_model=observation_model,
                                date=f"2026-02-0{day + 1}",
                            )
                            backend.adaptive_observe(
                                c_policy,
                                c_state,
                                c_predictive,
                                c_posterior_next,
                                log_factor_at_predictive_mean=log_prior,
                                log_factor_at_posterior_mode=log_post,
                                effective_strength=float(observation.update_weight),
                                information_size=float(max(observation.metadata.get("possible_pair_count", 1), 1)),
                                mahalanobis_update=float(innovation["state_update_mahalanobis"]),
                                gaussian_kl=float(innovation["gaussian_kl"]),
                                objective_improvement=float(innovation["objective_improvement"]),
                            )
                            assert np.allclose(c_state.block_multipliers(), np.array([python_state.block_multipliers["surface"]]), atol=1e-8, rtol=1e-8)
                            c_posterior.close()
                            c_posterior = c_posterior_next
                    python_posterior = python_result.posterior
        finally:
            c_posterior.close()
            c_state.close()
            c_policy.close()


def test_c_adaptive_candidate_a_sequence_matches_python() -> None:
    _candidate_a_sequence()


def test_c_adaptive_candidate_b_sequence_matches_python() -> None:
    _candidate_b_sequence()
