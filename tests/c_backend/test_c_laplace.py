from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend, CInferenceWorkspace, CNewtonConfig
from bolr.inference.laplace import laplace_update_composite
from bolr.inference.newton import NewtonOptions
from bolr.model.composite import CompositeScoreModel
from bolr.model.priors import BlockPriorSpec, assemble_block_prior
from bolr.model.score_blocks import ContextInteractionBlock, DynamicSurfaceBlock, GraphResidualBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder


def _assert_close(name: str, c_value: np.ndarray, py_value: np.ndarray, *, atol: float = 1e-9, rtol: float = 1e-8) -> None:
    if not np.allclose(c_value, py_value, atol=atol, rtol=rtol):
        diff = np.abs(c_value - py_value)
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(f"{name} mismatch at {idx}: c={c_value[idx]!r} python={py_value[idx]!r} max_abs={diff[idx]!r}")


def test_c_laplace_matches_python_for_multiblock_model() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.4, 0.6], [-0.2, 0.7]])
    graph_basis = np.array([[1.0], [0.0], [-1.0]])
    static = StaticBaselineBlock("baseline", phi, np.array([0.1, -0.05]), {})
    surface = DynamicSurfaceBlock("surface", phi)
    context = ContextInteractionBlock("context", phi)
    graph = GraphResidualBlock("graph", graph_basis)
    batch = {"context_vector": np.array([1.0, 0.3])}
    model = CompositeScoreModel.from_blocks([static], [surface, context, graph], batch)
    prior = assemble_block_prior(
        model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("context", isotropic_scale=1.0),
            BlockPriorSpec("graph", isotropic_scale=1.0),
        ],
    )
    observation = SoftTargetBuilder().build(np.array([1.0, -0.2, 0.4]))
    python_result = laplace_update_composite(
        prior,
        model,
        batch,
        observation,
        observation_model=SoftTargetObservationModel(),
        options=NewtonOptions(),
    )

    with backend.model_artifacts(model, batch) as artifacts:
        context_handle = backend.score_context(model, batch)
        with artifacts.state_from_posterior(prior) as predictive:
            with backend.candidate_a_observation(observation) as obs_handle:
                with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                    posterior_handle, diagnostics = backend.laplace_update(
                        predictive,
                        artifacts,
                        context_handle,
                        obs_handle,
                        workspace,
                        CNewtonConfig.from_python_options(NewtonOptions()),
                    )
                    with posterior_handle:
                        c_posterior = posterior_handle.to_posterior(state_layout=model.layout.metadata())

    _assert_close("posterior mean", c_posterior.mean, python_result.posterior.mean)
    _assert_close("posterior covariance", c_posterior.covariance, python_result.posterior.covariance, atol=1e-8, rtol=1e-8)
    assert diagnostics.newton.converged
