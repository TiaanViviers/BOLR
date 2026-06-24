from pathlib import Path

import numpy as np

from bolr.checkpoint.reader import read_checkpoint
from bolr.checkpoint.state import HistoricalReplayCheckpoint
from bolr.checkpoint.writer import write_checkpoint_atomic
from bolr.config.foundation import CandidateGridConfig, HistoricalRunConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.inference.laplace import laplace_update_composite
from bolr.initialization.static_surface import fit_static_surface
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.graph_residual import (
    build_graph_residual_basis,
    graph_penalty_shaped_process_noise,
    graph_residual_prior,
    validate_checkpoint_graph_metadata,
    validate_residual_basis_compatibility,
)
from bolr.model.priors import BlockDynamicsSpec, BlockPriorSpec, assemble_block_prior, assemble_block_process_noise
from bolr.model.score_blocks import DynamicSurfaceBlock, GraphResidualBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.targets.soft_target import SoftTargetBuilder


def test_historical_graph_residual_smoke_and_checkpoint_metadata(tmp_path) -> None:
    grid = load_candidate_grid("data/YM_grid.csv", CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet("data/YM_full.parquet", candidate_grid=grid)
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(grid.entry_values, grid.stop_values)
    basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=6, degree=3),
            stop_basis=SplineAxisConfig(n_basis=8, degree=3),
        )
    ).fit_transform(coordinates)
    candidate_basis = basis.reduced_basis
    graph = build_canonical_grid_graph(grid)
    residual = build_graph_residual_basis(graph, candidate_basis, residual_dimension=8)
    validate_residual_basis_compatibility(residual, graph, candidate_basis)

    target_builder = SoftTargetBuilder()
    observation_model = SoftTargetObservationModel()
    warm_up_observations = [
        target_builder.build(dataset.day_frame(date)["pnl"].to_numpy(dtype=float), date=date)
        for date in dataset.dates[:504]
    ]
    static_surface = fit_static_surface(candidate_basis, warm_up_observations, observation_model=observation_model)
    static = StaticBaselineBlock("baseline", candidate_basis, static_surface.coefficients, {})
    model = CompositeScoreModel.from_blocks([static], [DynamicSurfaceBlock("surface", candidate_basis), GraphResidualBlock("local", residual.basis)], {})
    residual_prior = graph_residual_prior(residual, graph_energy_weight=3.0, ridge_weight=0.2)
    prior = assemble_block_prior(
        model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("local", covariance=residual_prior.covariance, mean=residual_prior.mean),
        ],
    )
    process_noise = assemble_block_process_noise(
        model.layout,
        [
            BlockDynamicsSpec("surface", isotropic_process_variance=0.05),
            BlockDynamicsSpec("local", process_noise=graph_penalty_shaped_process_noise(residual, scale=0.01, properization=0.2)),
        ],
    )

    posterior = prior
    replay_dates = dataset.dates[504:507]
    for date in replay_dates:
        predicted = GaussianPosterior(mean=posterior.mean.copy(), covariance=posterior.covariance + process_noise)
        dataset.get_predictors(date)
        observation = target_builder.build(dataset.reveal_outcomes(date).pnl, date=date)
        result = laplace_update_composite(predicted, model, {}, observation, observation_model=observation_model)
        assert result.newton_result.converged
        posterior = result.posterior

    checkpoint_path = Path(tmp_path) / "graph_checkpoint.json"
    write_checkpoint_atomic(
        HistoricalReplayCheckpoint(
            schema_version="phase_i_v1",
            run_id="graph_smoke",
            configuration={"warm_up_days": HistoricalRunConfig().warm_up_days},
            last_completed_date=replay_dates[-1],
            static_alpha=static_surface.coefficients,
            posterior_mean=posterior.mean,
            posterior_covariance=posterior.covariance,
            output_row_counts={"predictions": len(replay_dates)},
            graph_metadata={
                "graph_schema_version": "phase_i_v1",
                "candidate_grid_fingerprint": graph.metadata["graph_definition_hash"],
                "smooth_basis_fingerprint": residual.metadata["smooth_basis_fingerprint"],
                "graph_definition_hash": graph.metadata["graph_definition_hash"],
                "residual_basis_hash": residual.metadata["residual_basis_hash"],
                "residual_dimension": residual.basis.shape[1],
                "laplacian_type": graph.metadata["laplacian_type"],
                "edge_weights": residual.metadata["edge_weights"],
                "residual_eigenvalues": residual.eigenvalues.tolist(),
            },
        ),
        checkpoint_path,
    )
    loaded = read_checkpoint(checkpoint_path)
    assert loaded.graph_metadata is not None
    validate_checkpoint_graph_metadata(loaded.graph_metadata, residual, graph)
    assert loaded.graph_metadata["residual_dimension"] == 8
    assert np.min(np.linalg.eigvalsh(posterior.covariance)) > 0.0


def test_graph_residual_block_accepts_candidate_b_reduced_case() -> None:
    from bolr.config.foundation import CrossGroupLogisticConfig, OrderedPartitionConfig
    from bolr.model.composite import CompositeScoreModel
    from bolr.model.score_blocks import DynamicSurfaceBlock, GraphResidualBlock
    from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
    from bolr.targets.ordered_partition import OrderedPartitionBuilder

    phi = np.array(
        [
            [1.0, 0.0],
            [0.2, 0.6],
            [-0.3, 0.4],
            [-0.5, -0.1],
        ]
    )
    psi = np.array(
        [
            [0.7, 0.1],
            [0.1, 0.4],
            [-0.4, 0.2],
            [-0.4, -0.7],
        ]
    )
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi), GraphResidualBlock("local", psi)], {})
    prior = assemble_block_prior(
        model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("local", isotropic_scale=1.0),
        ],
    )
    observation = OrderedPartitionBuilder(OrderedPartitionConfig()).build(np.array([2.0, 1.0, -0.1, -0.4]), date="2026-01-01")
    result = laplace_update_composite(
        prior,
        model,
        {},
        observation,
        observation_model=CrossGroupLogisticObservationModel(CrossGroupLogisticConfig()),
    )
    assert result.newton_result.converged
