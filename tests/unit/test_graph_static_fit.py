import numpy as np

from bolr.initialization.static_surface import fit_static_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph_residual import graph_residual_prior
from bolr.model.score_blocks import DynamicSurfaceBlock, GraphResidualBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder
from bolr.data.candidate_grid import CandidateGrid
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.graph_residual import build_graph_residual_basis


def test_graph_static_composite_fit_converges() -> None:
    grid = CandidateGrid(
        config_ids=np.arange(9),
        entry_values=np.repeat(np.array([0.1, 0.2, 0.3]), 3),
        stop_values=np.tile(np.array([0.1, 0.2, 0.3]), 3),
        pair_to_id={(float(e), float(s)): idx for idx, (e, s) in enumerate(zip(np.repeat([0.1, 0.2, 0.3], 3), np.tile([0.1, 0.2, 0.3], 3), strict=True))},
        grid_shape=(3, 3),
    )
    graph = build_canonical_grid_graph(grid)
    phi = np.column_stack([np.linspace(-1.0, 1.0, 9), np.tile(np.array([-1.0, 0.0, 1.0]), 3)])
    residual = build_graph_residual_basis(graph, phi, residual_dimension=2)
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi), GraphResidualBlock("local", residual.basis)], {})
    residual_prior = graph_residual_prior(residual, graph_energy_weight=3.0, ridge_weight=0.2)
    prior_precision = np.zeros((model.layout.total_dimension, model.layout.total_dimension), dtype=float)
    prior_precision[:2, :2] = 0.5 * np.eye(2)
    prior_precision[2:, 2:] = residual_prior.precision
    truth = phi @ np.array([0.4, -0.3]) + residual.basis @ np.array([0.1, -0.05])
    observations = [SoftTargetBuilder().build(truth), SoftTargetBuilder().build(truth + 0.02)]
    fit = fit_static_composite(
        model,
        [{}, {}],
        observations,
        prior_precision=prior_precision,
        observation_model=SoftTargetObservationModel(),
    )
    assert fit.converged
    assert fit.gradient_norm < 1e-5
