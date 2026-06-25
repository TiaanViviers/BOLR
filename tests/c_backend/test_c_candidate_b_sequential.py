from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend, CInferenceWorkspace, CNewtonConfig
from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_c_candidate_b_sequential_updates_match_python() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5]])
    static_coeffs = np.array([0.2, -0.1])
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    observation_model = CrossGroupLogisticObservationModel()
    builder = OrderedPartitionBuilder()
    process_noise = np.eye(model.layout.total_dimension) * 0.05
    utilities_by_day = [
        np.array([1.0, 0.3, -0.4, 0.2]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.1, 0.8, -0.2]),
    ]
    python_posterior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))

    with backend.model_artifacts(model, {}) as artifacts:
        context_handle = backend.score_context(model, {})
        c_state = artifacts.state_from_posterior(python_posterior)
        try:
            with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                for day, utilities in enumerate(utilities_by_day):
                    python_predictive = GaussianPosterior(
                        mean=python_posterior.mean.copy(),
                        covariance=python_posterior.covariance + process_noise,
                        state_layout=python_posterior.state_layout,
                    )
                    c_predictive, _ = c_state.predict_additive(process_noise)
                    observation = builder.build(utilities, date=f"2024-02-0{day + 1}")
                    python_result = laplace_update_composite(
                        python_predictive,
                        model,
                        {},
                        observation,
                        observation_model=observation_model,
                    )
                    with c_predictive:
                        with backend.candidate_b_exact_observation(observation) as obs_handle:
                            c_posterior_handle, _ = backend.laplace_update(
                                c_predictive,
                                artifacts,
                                context_handle,
                                obs_handle,
                                workspace,
                                CNewtonConfig(),
                            )
                            with c_posterior_handle:
                                c_posterior = c_posterior_handle.to_posterior(state_layout=model.layout.metadata())
                                assert np.allclose(c_posterior.mean, python_result.posterior.mean, atol=1e-8, rtol=1e-8), day
                                assert np.allclose(c_posterior.covariance, python_result.posterior.covariance, atol=1e-8, rtol=1e-8), day
                        c_state.close()
                        c_state = artifacts.state_from_posterior(c_posterior)
                    python_posterior = python_result.posterior
        finally:
            c_state.close()
