from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend, CInferenceWorkspace, CNewtonConfig
from bolr.config.foundation import CrossGroupLogisticConfig
from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_c_candidate_b_laplace_matches_python() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5]])
    static_coeffs = np.array([0.2, -0.1])
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    observation = OrderedPartitionBuilder().build(np.array([1.0, 0.3, -0.4, 0.2]), date="2024-02-01")
    observation_model = CrossGroupLogisticObservationModel()
    python_result = laplace_update_composite(prior, model, {}, observation, observation_model=observation_model)

    with backend.model_artifacts(model, {}) as artifacts:
        context_handle = backend.score_context(model, {})
        with artifacts.state_from_posterior(prior) as predictive:
            with backend.candidate_b_exact_observation(observation) as obs_handle:
                with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                    posterior_handle, diagnostics = backend.laplace_update(
                        predictive,
                        artifacts,
                        context_handle,
                        obs_handle,
                        workspace,
                        CNewtonConfig(),
                    )
                    with posterior_handle:
                        c_posterior = posterior_handle.to_posterior(state_layout=model.layout.metadata())

    assert np.allclose(c_posterior.mean, python_result.posterior.mean, atol=1e-8, rtol=1e-8)
    assert np.allclose(c_posterior.covariance, python_result.posterior.covariance, atol=1e-8, rtol=1e-8)
    assert diagnostics.newton.converged


def test_c_candidate_b_sampled_laplace_matches_python() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5], [0.2, 0.3]])
    static_coeffs = np.array([0.2, -0.1])
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    observation = OrderedPartitionBuilder().build(np.array([1.1, 0.9, 0.4, -0.2, -0.6]), date="2024-02-02")
    config = CrossGroupLogisticConfig(normalize_pair_losses=True, sampled_pair_budget=2, sampled_with_replacement=False, sampling_seed=11)
    observation_model = CrossGroupLogisticObservationModel(config)
    python_result = laplace_update_composite(prior, model, {}, observation, observation_model=observation_model)

    with backend.model_artifacts(model, {}) as artifacts:
        context_handle = backend.score_context(model, {})
        with artifacts.state_from_posterior(prior) as predictive:
            with backend.candidate_b_sampled_observation(observation, config) as obs_handle:
                with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                    posterior_handle, diagnostics = backend.laplace_update(
                        predictive,
                        artifacts,
                        context_handle,
                        obs_handle,
                        workspace,
                        CNewtonConfig(),
                    )
                    with posterior_handle:
                        c_posterior = posterior_handle.to_posterior(state_layout=model.layout.metadata())

    assert np.allclose(c_posterior.mean, python_result.posterior.mean, atol=1e-8, rtol=1e-8)
    assert np.allclose(c_posterior.covariance, python_result.posterior.covariance, atol=1e-8, rtol=1e-8)
    assert diagnostics.newton.converged
