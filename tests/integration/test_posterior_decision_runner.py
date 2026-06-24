import numpy as np
import pandas as pd

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy
from bolr.config.foundation import (
    AdaptiveTransitionConfig,
    BlockAdaptationConfig,
    BOCPDConfig,
    DecisionPolicyConfig,
    HistoricalDatasetConfig,
    HistoricalRunConfig,
    PosteriorSamplingConfig,
    RegionDefinitionConfig,
    SurpriseStandardizerConfig,
)
from bolr.data.candidate_grid import CandidateGrid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.decision.policies import HighestMassRegionDecisionPolicy, MaximumProbabilityBestDecisionPolicy
from bolr.evaluation.prequential_runner import CompositeReplayExperiment, run_composite_historical_replay
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder


def _dataset() -> tuple[HistoricalDataset, CandidateGrid]:
    dates = ["2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01"]
    frame = pd.DataFrame(
        {
            "date": sum(([d, d, d, d] for d in dates), []),
            "config_id": [0, 1, 2, 3] * len(dates),
            "entry_percentage": [0.1, 0.1, 0.2, 0.2] * len(dates),
            "sl_trail_percentage": [0.1, 0.2, 0.1, 0.2] * len(dates),
            "pnl": [
                1.0, 0.9, 0.1, 0.0,
                1.1, 1.0, 0.0, -0.1,
                0.2, 0.1, 1.2, 1.1,
                0.3, 0.4, 1.0, 0.9,
            ],
        }
    )
    grid = CandidateGrid(
        config_ids=np.arange(4),
        entry_values=np.array([0.1, 0.1, 0.2, 0.2]),
        stop_values=np.array([0.1, 0.2, 0.1, 0.2]),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.2, 0.1): 2, (0.2, 0.2): 3},
        grid_shape=(2, 2),
    )
    dataset = HistoricalDataset(
        frame,
        candidate_grid=grid,
        config=HistoricalDatasetConfig(
            expected_rows=16,
            expected_columns=5,
            expected_dates=4,
            expected_rows_per_date=4,
            expected_start_date="2021-01-29",
            expected_end_date="2021-02-01",
        ),
    )
    return dataset, grid


def _experiment(grid: CandidateGrid, decision_policy):
    phi = np.eye(grid.n_candidates)
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, np.zeros(grid.n_candidates), {"fit": "test"})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    transition = AdaptiveAdditiveTransitionPolicy(
        np.eye(grid.n_candidates) * 0.05,
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=1.5, decay=0.2),),
        ),
    )
    return CompositeReplayExperiment(
        score_model=model,
        initial_posterior=make_initial_dynamic_prior(grid.n_candidates, sigma0=1.0),
        transition_policy=transition,
        target_builder=SoftTargetBuilder(),
        observation_model=SoftTargetObservationModel(),
        static_surface=type("StaticSurface", (), {"coefficients": np.zeros(grid.n_candidates), "objective": 0.0, "gradient_norm": 0.0, "iterations": 0, "converged": True, "regularization": 0.0})(),
        batch_builder=lambda date, predictor_batch: {},
        decision_policy=decision_policy,
        sampling_config=PosteriorSamplingConfig(sample_count=64, seed=5, retain_score_samples=True),
        region_definition=RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold"),
        graph=build_canonical_grid_graph(grid),
    )


def test_runner_emits_decision_outputs_and_checkpoint_metadata(tmp_path) -> None:
    dataset, grid = _dataset()
    region_policy = HighestMassRegionDecisionPolicy(
        DecisionPolicyConfig(
            family="highest_mass_region",
            region_selection_statistic="probability_best",
            representative_policy="weighted_medoid",
        )
    )
    result = run_composite_historical_replay(
        dataset,
        _experiment(grid, region_policy),
        config=HistoricalRunConfig(warm_up_days=2, outputs_dir=str(tmp_path / "outputs")),
        write_outputs_flag=True,
    )
    assert len(result.predictions) == 2
    assert "decision_policy" in result.predictions.columns
    assert "selected_probability_best" in result.predictions.columns
    assert "probability_best_brier" in result.daily_metrics.columns
    checkpoint_text = (result.run_dir / "checkpoints" / "latest.json").read_text()
    assert "decision_policy_family" in checkpoint_text
    assert "posterior_sample_count" in checkpoint_text


def test_runner_is_resume_stable_for_probability_best_policy(tmp_path) -> None:
    dataset, grid = _dataset()
    policy = MaximumProbabilityBestDecisionPolicy()
    experiment = _experiment(grid, policy)
    config = HistoricalRunConfig(warm_up_days=2, outputs_dir=str(tmp_path / "outputs"))

    full = run_composite_historical_replay(dataset, experiment, config=config, write_outputs_flag=False)
    first = run_composite_historical_replay(dataset, experiment, config=config, replay_dates=(dataset.dates[2],), write_outputs_flag=False)
    second = run_composite_historical_replay(
        dataset,
        CompositeReplayExperiment(
            score_model=experiment.score_model,
            initial_posterior=first.final_posterior,
            transition_policy=experiment.transition_policy,
            target_builder=experiment.target_builder,
            observation_model=experiment.observation_model,
            static_surface=experiment.static_surface,
            batch_builder=experiment.batch_builder,
            decision_policy=experiment.decision_policy,
            sampling_config=experiment.sampling_config,
            region_definition=experiment.region_definition,
            graph=experiment.graph,
        ),
        config=config,
        replay_dates=(dataset.dates[3],),
        initial_policy_state=first.final_policy_state,
        write_outputs_flag=False,
    )
    assert np.allclose(full.final_posterior.mean, second.final_posterior.mean)
    assert np.allclose(full.final_posterior.covariance, second.final_posterior.covariance)
