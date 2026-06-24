import numpy as np

from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.graph_residual import build_graph_residual_basis, graph_residual_prior, smooth_plus_local_diagnostics
from bolr.model.priors import BlockPriorSpec, assemble_block_prior
from bolr.model.score_blocks import DynamicSurfaceBlock, GraphResidualBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder
from bolr.data.candidate_grid import CandidateGrid


def _small_grid() -> CandidateGrid:
    return CandidateGrid(
        config_ids=np.arange(9),
        entry_values=np.repeat(np.array([0.1, 0.2, 0.3]), 3),
        stop_values=np.tile(np.array([0.1, 0.2, 0.3]), 3),
        pair_to_id={(float(e), float(s)): idx for idx, (e, s) in enumerate(zip(np.repeat([0.1, 0.2, 0.3], 3), np.tile([0.1, 0.2, 0.3], 3), strict=True))},
        grid_shape=(3, 3),
    )


def test_smooth_truth_keeps_local_residual_small() -> None:
    grid = _small_grid()
    graph = build_canonical_grid_graph(grid)
    phi = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 9),
            np.tile(np.array([-1.0, 0.0, 1.0]), 3),
        ]
    )
    residual = build_graph_residual_basis(graph, phi, residual_dimension=2)
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi), GraphResidualBlock("local", residual.basis)], {})
    structured = graph_residual_prior(residual, graph_energy_weight=4.0, ridge_weight=0.3)
    prior = assemble_block_prior(
        model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("local", covariance=structured.covariance, mean=structured.mean),
        ],
    )
    truth_surface = np.array([0.7, -0.4])
    observation = SoftTargetBuilder().build(phi @ truth_surface)
    result = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    local_slice = model.layout.slice_for("local")
    surface_slice = model.layout.slice_for("surface")
    diagnostics = smooth_plus_local_diagnostics(
        phi @ result.posterior.mean[surface_slice],
        residual.basis @ result.posterior.mean[local_slice],
        graph,
    )
    assert result.newton_result.converged
    assert diagnostics["local_to_total_norm_ratio"] < 0.35


def test_sharp_boundary_truth_benefits_from_local_residual() -> None:
    grid = _small_grid()
    graph = build_canonical_grid_graph(grid)
    phi = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 9),
            np.tile(np.array([-1.0, 0.0, 1.0]), 3),
        ]
    )
    residual = build_graph_residual_basis(graph, phi, residual_dimension=4)
    smooth_model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi)], {})
    hybrid_model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi), GraphResidualBlock("local", residual.basis)], {})
    truth = np.where((grid.entry_values >= 0.2) & (grid.stop_values <= 0.2), 1.0, -0.2)
    truth = truth - np.mean(truth)
    observation = SoftTargetBuilder().build(truth)
    smooth_prior = assemble_block_prior(smooth_model.layout, [BlockPriorSpec("surface", isotropic_scale=1.0)])
    residual_prior = graph_residual_prior(residual, graph_energy_weight=4.0, ridge_weight=0.3)
    hybrid_prior = assemble_block_prior(
        hybrid_model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("local", covariance=residual_prior.covariance, mean=residual_prior.mean),
        ],
    )
    smooth_result = laplace_update_composite(smooth_prior, smooth_model, {}, observation, observation_model=SoftTargetObservationModel())
    hybrid_result = laplace_update_composite(hybrid_prior, hybrid_model, {}, observation, observation_model=SoftTargetObservationModel())
    smooth_scores = smooth_model.scores({}, smooth_result.posterior.mean)
    hybrid_scores = hybrid_model.scores({}, hybrid_result.posterior.mean)
    smooth_rmse = float(np.sqrt(np.mean((smooth_scores - truth) ** 2)))
    hybrid_rmse = float(np.sqrt(np.mean((hybrid_scores - truth) ** 2)))
    assert hybrid_rmse < smooth_rmse
