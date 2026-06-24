import numpy as np
import pandas as pd

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy
from bolr.config.foundation import AdaptiveTransitionConfig, BlockAdaptationConfig, BOCPDConfig, HistoricalDatasetConfig, HistoricalRunConfig, SurpriseStandardizerConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.prequential_runner import CompositeReplayExperiment, run_composite_historical_replay
from bolr.initialization.static_surface import StaticSurfaceFit
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.soft_target import SoftTargetBuilder


def _dataset(first_day_pnl):
    dates = ["2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01"]
    frame = pd.DataFrame(
        {
            "date": sum(([d, d] for d in dates), []),
            "config_id": [0, 1] * len(dates),
            "entry_percentage": [0.1, 0.2] * len(dates),
            "sl_trail_percentage": [0.3, 0.4] * len(dates),
            "pnl": list(first_day_pnl) + [1.2, 1.9, 2.5, 0.5, 0.3, 2.7],
        }
    )
    grid = CandidateGrid(
        config_ids=np.array([0, 1]),
        entry_values=np.array([0.1, 0.2]),
        stop_values=np.array([0.3, 0.4]),
        pair_to_id={(0.1, 0.3): 0, (0.2, 0.4): 1},
        grid_shape=(1, 2),
    )
    dataset = HistoricalDataset(
        frame,
        candidate_grid=grid,
        config=HistoricalDatasetConfig(
            expected_rows=8,
            expected_columns=5,
            expected_dates=4,
            expected_rows_per_date=2,
            expected_start_date="2021-01-29",
            expected_end_date="2021-02-01",
        ),
    )
    return dataset


def _experiment():
    phi = np.array([[1.0], [0.5]])
    static_surface = StaticSurfaceFit(coefficients=np.array([0.1]), objective=0.0, gradient_norm=0.0, iterations=1, converged=True, regularization=0.1, diagnostics={})
    model = CompositeScoreModel.from_blocks([StaticBaselineBlock("baseline", phi, static_surface.coefficients, {})], [DynamicSurfaceBlock("surface", phi)], {})
    policy = AdaptiveAdditiveTransitionPolicy(
        np.array([[0.05]]),
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=2.0, decay=0.0),),
        ),
    )
    return CompositeReplayExperiment(
        score_model=model,
        initial_posterior=GaussianPosterior(mean=np.zeros(1), covariance=np.eye(1)),
        transition_policy=policy,
        target_builder=SoftTargetBuilder(),
        observation_model=SoftTargetObservationModel(),
        static_surface=static_surface,
        batch_builder=lambda date, predictor_batch: {},
    )


def test_day_t_outcome_does_not_change_day_t_prediction(tmp_path) -> None:
    config = HistoricalRunConfig(warm_up_days=2, outputs_dir=str(tmp_path / "outputs"))
    result_a = run_composite_historical_replay(_dataset([1.0, 2.0]), _experiment(), config=config, replay_dates=("2021-01-31", "2021-02-01"), write_outputs_flag=False)
    result_b = run_composite_historical_replay(_dataset([10.0, -10.0]), _experiment(), config=config, replay_dates=("2021-01-31", "2021-02-01"), write_outputs_flag=False)
    assert np.isclose(result_a.predictions.iloc[0]["selected_predicted_score"], result_b.predictions.iloc[0]["selected_predicted_score"])


def test_adaptive_runner_resume_matches_uninterrupted(tmp_path) -> None:
    config = HistoricalRunConfig(warm_up_days=2, outputs_dir=str(tmp_path / "outputs"))
    dataset = _dataset([1.0, 2.0])
    experiment = _experiment()
    full = run_composite_historical_replay(dataset, experiment, config=config, replay_dates=("2021-01-31", "2021-02-01"), write_outputs_flag=False)
    split_first = run_composite_historical_replay(_dataset([1.0, 2.0]), experiment, config=config, replay_dates=("2021-01-31",), write_outputs_flag=False)
    split_second = run_composite_historical_replay(
        _dataset([1.0, 2.0]),
        CompositeReplayExperiment(
            score_model=experiment.score_model,
            initial_posterior=split_first.final_posterior,
            transition_policy=experiment.transition_policy,
            target_builder=experiment.target_builder,
            observation_model=experiment.observation_model,
            static_surface=experiment.static_surface,
            batch_builder=experiment.batch_builder,
        ),
        config=config,
        replay_dates=("2021-02-01",),
        initial_policy_state=split_first.final_policy_state,
        write_outputs_flag=False,
    )
    assert np.allclose(full.final_posterior.mean, split_second.final_posterior.mean)
    assert np.allclose(full.final_posterior.covariance, split_second.final_posterior.covariance)
