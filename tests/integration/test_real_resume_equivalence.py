from pathlib import Path

import numpy as np

from bolr.config.foundation import CandidateGridConfig, HistoricalRunConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.prequential_runner import run_historical_replay
from bolr.initialization.static_surface import fit_static_surface
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.targets.soft_target import SoftTargetBuilder


def test_real_resume_equivalence(tmp_path) -> None:
    grid = load_candidate_grid("data/YM_grid.csv", CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet("data/YM_full.parquet", candidate_grid=grid)
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(grid.entry_values, grid.stop_values)
    candidate_basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=6, degree=3),
            stop_basis=SplineAxisConfig(n_basis=8, degree=3),
        )
    ).fit_transform(coordinates).reduced_basis
    config = HistoricalRunConfig(warm_up_days=504, outputs_dir=str(tmp_path / "outputs"))
    replay_dates = dataset.dates[504:514]
    target_builder = SoftTargetBuilder(config.target)
    observation_model = SoftTargetObservationModel()
    warm_up_observations = [
        target_builder.build(dataset.day_frame(date)["pnl"].to_numpy(dtype=float), date=date)
        for date in dataset.dates[:504]
    ]
    static_surface = fit_static_surface(candidate_basis, warm_up_observations, config.static_surface, observation_model=observation_model)
    static_scores = candidate_basis @ static_surface.coefficients

    continuous_dataset = HistoricalDataset.from_parquet("data/YM_full.parquet", candidate_grid=grid)
    continuous = run_historical_replay(
        continuous_dataset,
        candidate_basis,
        target_builder=target_builder,
        observation_model=observation_model,
        config=config,
        run_label="resume_continuous",
        replay_dates=replay_dates,
        static_surface=static_surface,
        static_scores=static_scores,
        write_outputs_flag=False,
    )

    interrupted_dataset = HistoricalDataset.from_parquet("data/YM_full.parquet", candidate_grid=grid)
    first_half = run_historical_replay(
        interrupted_dataset,
        candidate_basis,
        target_builder=target_builder,
        observation_model=observation_model,
        config=config,
        run_label="resume_split",
        replay_dates=replay_dates[:5],
        static_surface=static_surface,
        static_scores=static_scores,
        write_outputs_flag=False,
    )
    resumed_dataset = HistoricalDataset.from_parquet("data/YM_full.parquet", candidate_grid=grid)
    second_half = run_historical_replay(
        resumed_dataset,
        candidate_basis,
        target_builder=target_builder,
        observation_model=observation_model,
        config=config,
        run_label="resume_split",
        replay_dates=replay_dates[5:],
        initial_posterior=first_half.final_posterior,
        static_surface=static_surface,
        static_scores=static_scores,
        write_outputs_flag=False,
    )
    combined_predictions = np.concatenate(
        [first_half.predictions["selected_predicted_score"].to_numpy(), second_half.predictions["selected_predicted_score"].to_numpy()]
    )
    assert np.max(np.abs(continuous.final_posterior.mean - second_half.final_posterior.mean)) < 1e-12
    assert np.max(np.abs(continuous.final_posterior.covariance - second_half.final_posterior.covariance)) < 1e-12
    assert np.allclose(continuous.predictions["selected_predicted_score"].to_numpy(), combined_predictions)
