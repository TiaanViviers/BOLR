from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping
import uuid

import numpy as np
import pandas as pd

from bolr.adaptation.policy import FixedAdditiveTransitionPolicy, TransitionPolicyState
from bolr.checkpoint.state import HistoricalReplayCheckpoint
from bolr.checkpoint.writer import write_checkpoint_atomic
from bolr.config.foundation import HistoricalRunConfig, PosteriorSamplingConfig, RegionDefinitionConfig, SoftTargetConfig
from bolr.data.historical_dataset import HistoricalDataset
from bolr.decision.metrics import probability_best_brier, probability_best_entropy, region_coverage, top_k_brier
from bolr.decision.policies import DecisionPolicy, PosteriorMeanDecisionPolicy
from bolr.decision.prediction import build_posterior_prediction
from bolr.evaluation.baselines import compute_reference_baselines
from bolr.evaluation.metrics import entropy_from_scores, maximum_drawdown, rank_of_selected, top_fraction_membership
from bolr.evaluation.outputs import ensure_run_directory, write_json, write_parquet
from bolr.inference.laplace import laplace_update_composite
from bolr.inference.newton import NewtonOptions
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.initialization.static_surface import StaticSurfaceFit, fit_static_surface
from bolr.model.composite import CompositeScoreModel
from bolr.model.diagnostics import innovation_diagnostics
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.base import ObservationModel
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.base import TargetBuilder
from bolr.targets.soft_target import SoftTargetBuilder


@dataclass(frozen=True)
class HistoricalReplayResult:
    run_id: str
    run_dir: Path
    predictions: pd.DataFrame
    daily_metrics: pd.DataFrame
    posterior_diagnostics: pd.DataFrame
    target_diagnostics: pd.DataFrame
    update_diagnostics: pd.DataFrame
    baselines: pd.DataFrame
    static_surface: StaticSurfaceFit
    final_posterior: GaussianPosterior
    final_policy_state: TransitionPolicyState | None = None


@dataclass(frozen=True)
class CompositeReplayExperiment:
    score_model: CompositeScoreModel
    initial_posterior: GaussianPosterior
    transition_policy: object
    target_builder: TargetBuilder
    observation_model: ObservationModel
    static_surface: StaticSurfaceFit
    batch_builder: Callable[[str, object], object]
    decision_policy: DecisionPolicy = PosteriorMeanDecisionPolicy()
    sampling_config: PosteriorSamplingConfig = PosteriorSamplingConfig()
    region_definition: RegionDefinitionConfig | None = None
    graph: object | None = None


def run_composite_historical_replay(
    dataset: HistoricalDataset,
    experiment: CompositeReplayExperiment,
    *,
    config: HistoricalRunConfig | None = None,
    run_label: str = "historical_replay",
    run_id: str | None = None,
    replay_dates: tuple[str, ...] | None = None,
    initial_policy_state: TransitionPolicyState | None = None,
    write_outputs_flag: bool = True,
) -> HistoricalReplayResult:
    config = config or HistoricalRunConfig()
    run_id = run_id or f"{run_label}_{uuid.uuid4().hex[:12]}"
    run_dir = ensure_run_directory(Path(config.outputs_dir).parent / run_label / run_id)
    posterior = experiment.initial_posterior
    policy_state = initial_policy_state or experiment.transition_policy.initial_state(layout=experiment.score_model.layout, base_dynamics=None)
    if replay_dates is None:
        replay_dates = dataset.dates[config.warm_up_days :]

    first_predictors = dataset.get_predictors(replay_dates[0])
    first_batch = experiment.batch_builder(replay_dates[0], first_predictors)
    static_scores = experiment.score_model.static_scores(first_batch)
    dataset._predicted_dates.discard(replay_dates[0])
    baselines = compute_reference_baselines(dataset.warm_up_frame(config.warm_up_days), static_scores)

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    for day_index, date in enumerate(replay_dates):
        prediction_start = perf_counter()
        predictor_batch = dataset.get_predictors(date)
        predictive, policy_state, predict_diag = experiment.transition_policy.predict(posterior, policy_state, layout=experiment.score_model.layout)
        batch = experiment.batch_builder(date, predictor_batch)
        score_mean = experiment.score_model.scores(batch, predictive.mean)
        design = experiment.score_model.explicit_design_matrix(batch)
        top_k_values = {1, 5, 10, int(np.ceil(0.01 * score_mean.size)), int(np.ceil(0.03 * score_mean.size)), int(np.ceil(0.05 * score_mean.size))}
        if experiment.region_definition is not None:
            if experiment.region_definition.top_k is not None:
                top_k_values.add(int(experiment.region_definition.top_k))
            if experiment.region_definition.top_fraction is not None:
                top_k_values.add(max(1, int(np.ceil(experiment.region_definition.top_fraction * score_mean.size))))
        decision_seed = experiment.sampling_config.seed + day_index
        prediction = build_posterior_prediction(
            date=date,
            state_mean=predictive.mean,
            state_covariance=predictive.covariance,
            score_mean=score_mean,
            design_matrix=design,
            sampling_config=experiment.sampling_config,
            top_k_values=tuple(sorted(top_k_values)),
            candidate_grid=dataset.candidate_grid,
            graph=experiment.graph,
            region_config=experiment.region_definition,
            block_score_means=experiment.score_model.block_scores(batch, predictive.mean),
            seed=decision_seed,
        )
        decision = experiment.decision_policy.decide(prediction, candidate_grid=dataset.candidate_grid, graph=experiment.graph)
        if decision.selected_index is None:
            raise ValueError("Ranking-only replay requires a concrete selected candidate.")
        selected_index = int(decision.selected_index)
        score_variance = prediction.score_variance
        prediction_time = perf_counter() - prediction_start

        target_start = perf_counter()
        outcome_batch = dataset.reveal_outcomes(date)
        observation = experiment.target_builder.build(outcome_batch.pnl, date=date)
        target_time = perf_counter() - target_start

        predictive_observation_diagnostics = dict(experiment.observation_model.diagnostics(score_mean, observation))
        update_start = perf_counter()
        update = laplace_update_composite(
            predictive,
            experiment.score_model,
            batch,
            observation,
            observation_model=experiment.observation_model,
            options=NewtonOptions(max_iterations=30),
        )
        update_time = perf_counter() - update_start
        posterior = update.posterior
        posterior_scores = experiment.score_model.scores(batch, posterior.mean)
        innovation = innovation_diagnostics(
            predictive.mean,
            predictive.covariance,
            posterior.mean,
            posterior.covariance,
            experiment.observation_model.log_factor(score_mean, observation),
            experiment.observation_model.log_factor(posterior_scores, observation),
        )
        policy_state, adaptive_diag = experiment.transition_policy.observe_update(
            predictive_posterior=predictive,
            posterior=posterior,
            observation_diagnostics={**predictive_observation_diagnostics, **innovation},
            block_diagnostics={},
            policy_state=policy_state,
            layout=experiment.score_model.layout,
            predictive_scores=score_mean,
            posterior_scores=posterior_scores,
            observation=observation,
            observation_model=experiment.observation_model,
            date=date,
        )

        selected_config_id = int(predictor_batch.config_ids[selected_index])
        selected_pnl = float(outcome_batch.pnl[selected_index])
        oracle_best_pnl = float(np.max(outcome_batch.pnl))
        selected_rank = rank_of_selected(outcome_batch.pnl, selected_index)
        selected_region = prediction.region_summary.components[decision.selected_region_id] if prediction.region_summary is not None and decision.selected_region_id is not None else None
        prediction_rows.append(
            {
                "date": date,
                "decision_policy": decision.policy_name,
                "selected_config_id": selected_config_id,
                "selected_entry_percentage": float(predictor_batch.entry_percentage[selected_index]),
                "selected_sl_trail_percentage": float(predictor_batch.sl_trail_percentage[selected_index]),
                "selected_predicted_score": float(score_mean[selected_index]),
                "selected_score_variance": float(score_variance[selected_index]),
                "selected_probability_best": np.nan if prediction.probability_best is None else float(prediction.probability_best[selected_index]),
                "selected_expected_rank": np.nan if prediction.expected_rank is None else float(prediction.expected_rank[selected_index]),
                "probability_best_entropy": np.nan if prediction.probability_best is None else probability_best_entropy(prediction.probability_best),
                "maximum_probability_best": np.nan if prediction.probability_best is None else float(np.max(prediction.probability_best)),
                "consensus_candidate_count": 0 if prediction.region_summary is None else int(prediction.region_summary.consensus_indices.size),
                "region_count": 0 if prediction.region_summary is None else len(prediction.region_summary.components),
                "selected_region_id": decision.selected_region_id,
                "selected_region_probability_best": np.nan if selected_region is None else float(selected_region.probability_best_mass),
                "selected_region_mass": np.nan if selected_region is None else float(selected_region.inclusion_mass),
                "selected_region_size": np.nan if selected_region is None else int(selected_region.candidate_count),
                "monte_carlo_sample_count": int(prediction.metadata.get("monte_carlo_sample_count", 0)),
                "monte_carlo_seed": int(decision_seed),
                "monte_carlo_tie_count": int(prediction.metadata.get("monte_carlo_tie_count", 0)),
                "selected_pnl": selected_pnl,
                "oracle_best_pnl": oracle_best_pnl,
                "regret": oracle_best_pnl - selected_pnl,
                "selected_realised_rank": selected_rank,
                "selected_in_top_1_percent": top_fraction_membership(selected_rank, outcome_batch.pnl.size, 0.01),
                "selected_in_top_3_percent": top_fraction_membership(selected_rank, outcome_batch.pnl.size, 0.03),
                "selected_in_top_5_percent": top_fraction_membership(selected_rank, outcome_batch.pnl.size, 0.05),
                "selected_positive": selected_pnl > 0.0,
            }
        )
        metric_rows.append(
            {
                "date": date,
                "state_mean_norm": float(np.linalg.norm(posterior.mean)),
                "state_update_norm": float(np.linalg.norm(posterior.mean - predictive.mean)),
                "state_prediction_norm": float(np.linalg.norm(predictive.mean)),
                "covariance_trace": float(np.trace(posterior.covariance)),
                "covariance_log_determinant": float(np.linalg.slogdet(posterior.covariance)[1]),
                "minimum_variance": float(np.min(np.diag(posterior.covariance))),
                "maximum_variance": float(np.max(np.diag(posterior.covariance))),
                "covariance_condition_estimate": posterior.summary()["condition_number"],
                "predictive_score_entropy": entropy_from_scores(score_mean),
                "target_entropy": float(getattr(observation, "metadata", {}).get("target_entropy", np.nan)),
                "target_perplexity": float(np.exp(getattr(observation, "metadata", {}).get("target_entropy", np.nan))) if "target_entropy" in getattr(observation, "metadata", {}) else np.nan,
                "maximum_target_mass": float(np.max(getattr(observation, "target_probabilities", np.array([np.nan])))),
                "probability_best_brier": np.nan if prediction.probability_best is None else probability_best_brier(prediction.probability_best, outcome_batch.pnl),
                "top_5_brier": np.nan if 5 not in prediction.probability_top_k else top_k_brier(prediction.probability_top_k[5], outcome_batch.pnl, k=5),
                "region_covers_realised_best": np.nan if prediction.region_summary is None or decision.selected_region_id is None else region_coverage(selected_region.candidate_indices, outcome_batch.pnl),
                "prediction_time": prediction_time,
                "target_time": target_time,
                "update_time": update_time,
                "checkpoint_time": 0.0,
                "policy_transition_runtime": float(0.0),
            }
        )
        posterior_rows.append({"date": date, "posterior_mean_norm": float(np.linalg.norm(posterior.mean)), "posterior_trace": float(np.trace(posterior.covariance)), "posterior_logdet": float(np.linalg.slogdet(posterior.covariance)[1])})
        target_rows.append({"date": date, **getattr(observation, "metadata", {}), "observation_type": getattr(observation, "type", getattr(observation, "metadata", {}).get("observation_type", "unknown"))})
        update_rows.append(
            {
                "date": date,
                "newton_iterations": update.newton_result.iterations,
                "final_gradient_norm": update.newton_result.gradient_norm,
                "line_search_reductions": 0,
                "jitter_used": float(posterior.diagnostics.get("posterior_precision_jitter", 0.0)),
                "update_converged": bool(update.newton_result.converged),
                **predictive_observation_diagnostics,
                **innovation,
                **{k: v for k, v in adaptive_diag.items() if k != "block_diagnostics"},
            }
        )
        baseline_rows.append({"date": date, "warm_up_global_best_config_id": baselines.warm_up_global_best_config_id, "static_surface_best_config_id": baselines.static_surface_best_config_id})

        checkpoint_start = perf_counter()
        write_checkpoint_atomic(
            HistoricalReplayCheckpoint(
                schema_version="phase_k_v1",
                run_id=run_id,
                configuration={"warm_up_days": config.warm_up_days, "sigma0": config.sigma0, "run_label": run_label},
                last_completed_date=date,
                static_alpha=experiment.static_surface.coefficients,
                posterior_mean=posterior.mean,
                posterior_covariance=posterior.covariance,
                output_row_counts={"predictions": len(prediction_rows), "daily_metrics": len(metric_rows)},
                transition_policy_family=experiment.transition_policy.metadata()["family"],
                transition_policy_config_hash=_policy_hash(experiment.transition_policy.metadata()),
                transition_policy_state=_serialize_policy_state(policy_state),
                surprise_standardizer_state=_serialize_standardizer_state(policy_state.online_standardizer_state),
                bocpd_state=_serialize_bocpd_state(policy_state.change_detector_state),
                block_multipliers=dict(policy_state.block_multipliers),
                block_discounts=dict(policy_state.block_discounts),
                pending_resets={name: {"strength": reset.strength} for name, reset in policy_state.pending_resets.items()},
                last_surprise_diagnostics=dict(policy_state.last_surprise_values),
                adaptive_schema_version=policy_state.schema_version,
                decision_policy_family=experiment.decision_policy.metadata().get("family"),
                decision_policy_config=dict(experiment.decision_policy.metadata()),
                decision_policy_config_hash=_policy_hash(experiment.decision_policy.metadata()),
                posterior_sample_count=int(experiment.sampling_config.sample_count),
                sampling_seed_state={"seed": int(experiment.sampling_config.seed), "day_index": int(day_index)},
                region_definition=None if experiment.region_definition is None else {
                    "top_k": experiment.region_definition.top_k,
                    "top_fraction": experiment.region_definition.top_fraction,
                    "inclusion_threshold": experiment.region_definition.inclusion_threshold,
                    "consensus_family": experiment.region_definition.consensus_family,
                    "edge_comembership_enabled": experiment.region_definition.edge_comembership_enabled,
                },
                decision_schema_version="phase_k_v1",
            ),
            run_dir / "checkpoints" / "latest.json",
        )
        metric_rows[-1]["checkpoint_time"] = perf_counter() - checkpoint_start

    predictions = pd.DataFrame(prediction_rows)
    daily_metrics = pd.DataFrame(metric_rows)
    posterior_diagnostics = pd.DataFrame(posterior_rows)
    target_diagnostics = pd.DataFrame(target_rows)
    update_diagnostics = pd.DataFrame(update_rows)
    baselines_frame = pd.DataFrame(baseline_rows)
    if write_outputs_flag:
        write_json(run_dir / "config.json", {"warm_up_days": config.warm_up_days, "sigma0": config.sigma0, "score_model": experiment.score_model.metadata(), "target_builder": experiment.target_builder.metadata(), "observation_model": experiment.observation_model.__class__.__name__, "transition_policy": experiment.transition_policy.metadata(), "decision_policy": experiment.decision_policy.metadata(), "sampling_config": {"sample_count": experiment.sampling_config.sample_count, "seed": experiment.sampling_config.seed, "antithetic": experiment.sampling_config.antithetic, "retain_score_samples": experiment.sampling_config.retain_score_samples}, "region_definition": None if experiment.region_definition is None else {"top_k": experiment.region_definition.top_k, "top_fraction": experiment.region_definition.top_fraction, "inclusion_threshold": experiment.region_definition.inclusion_threshold, "consensus_family": experiment.region_definition.consensus_family, "edge_comembership_enabled": experiment.region_definition.edge_comembership_enabled}})
        write_json(run_dir / "static_surface.json", {"objective": experiment.static_surface.objective, "gradient_norm": experiment.static_surface.gradient_norm, "iterations": experiment.static_surface.iterations, "converged": experiment.static_surface.converged, "regularization": experiment.static_surface.regularization})
        write_parquet(run_dir / "predictions.parquet", predictions)
        write_parquet(run_dir / "daily_metrics.parquet", daily_metrics)
        write_parquet(run_dir / "posterior_diagnostics.parquet", posterior_diagnostics)
        write_parquet(run_dir / "target_diagnostics.parquet", target_diagnostics)
        write_parquet(run_dir / "update_diagnostics.parquet", update_diagnostics)
        write_parquet(run_dir / "baselines.parquet", baselines_frame)
        (run_dir / "summary.md").write_text("\n".join([f"# {run_label}", "", f"- run_id: `{run_id}`", f"- evaluated_days: {len(predictions)}", f"- cumulative_selected_pnl: {predictions['selected_pnl'].sum():.6f}", f"- mean_selected_pnl: {predictions['selected_pnl'].mean():.6f}", f"- mean_regret: {predictions['regret'].mean():.6f}", f"- positive_day_rate: {predictions['selected_positive'].mean():.6f}", f"- maximum_drawdown: {maximum_drawdown(predictions['selected_pnl'].cumsum().to_numpy(dtype=float)):.6f}"]))
    return HistoricalReplayResult(run_id, run_dir, predictions, daily_metrics, posterior_diagnostics, target_diagnostics, update_diagnostics, baselines_frame, experiment.static_surface, posterior, policy_state)


def run_historical_replay(
    dataset: HistoricalDataset,
    candidate_basis: np.ndarray,
    *,
    target_builder: TargetBuilder,
    observation_model: ObservationModel,
    config: HistoricalRunConfig | None = None,
    run_label: str = "historical_replay",
    run_id: str | None = None,
    initial_posterior: GaussianPosterior | None = None,
    static_surface: StaticSurfaceFit | None = None,
    static_scores: np.ndarray | None = None,
    replay_dates: tuple[str, ...] | None = None,
    write_outputs_flag: bool = True,
) -> HistoricalReplayResult:
    config = config or HistoricalRunConfig()
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    if static_surface is None or static_scores is None:
        warm_up_dates = dataset.dates[: config.warm_up_days]
        warm_up_observations = [target_builder.build(dataset.day_frame(date)["pnl"].to_numpy(dtype=float), date=date) for date in warm_up_dates]
        static_surface = fit_static_surface(candidate_basis, warm_up_observations, config.static_surface, observation_model=observation_model)
        static_scores = candidate_basis @ static_surface.coefficients
    prior = make_initial_dynamic_prior(candidate_basis.shape[1], sigma0=config.sigma0) if initial_posterior is None else initial_posterior
    model = CompositeScoreModel.from_blocks([StaticBaselineBlock("baseline", candidate_basis, static_surface.coefficients, {})], [DynamicSurfaceBlock("surface", candidate_basis)], {})
    policy = FixedAdditiveTransitionPolicy(config.random_walk_variance * np.eye(candidate_basis.shape[1], dtype=float))
    experiment = CompositeReplayExperiment(
        score_model=model,
        initial_posterior=prior,
        transition_policy=policy,
        target_builder=target_builder,
        observation_model=observation_model,
        static_surface=static_surface,
        batch_builder=lambda date, predictor_batch: {},
    )
    return run_composite_historical_replay(dataset, experiment, config=config, run_label=run_label, run_id=run_id, replay_dates=replay_dates, write_outputs_flag=write_outputs_flag)


def run_historical_candidate_a(dataset: HistoricalDataset, candidate_basis: np.ndarray, config: HistoricalRunConfig | None = None) -> HistoricalReplayResult:
    config = config or HistoricalRunConfig(target=SoftTargetConfig())
    return run_historical_replay(dataset, candidate_basis, target_builder=SoftTargetBuilder(config.target), observation_model=SoftTargetObservationModel(), config=config, run_label="historical_candidate_a")


def _policy_hash(metadata: Mapping[str, object]) -> str:
    return sha256(str(sorted(metadata.items())).encode("utf-8")).hexdigest()


def _serialize_policy_state(state: TransitionPolicyState) -> dict:
    return {
        "last_completed_date": state.last_completed_date,
        "step_index": state.step_index,
        "block_multipliers": dict(state.block_multipliers),
        "block_discounts": dict(state.block_discounts),
        "last_surprise_values": dict(state.last_surprise_values),
        "schema_version": state.schema_version,
        "metadata": dict(state.metadata),
    }


def _serialize_standardizer_state(state) -> dict | None:
    if state is None:
        return None
    return {"count": state.count, "mean": state.mean, "variance": state.variance, "last_z": state.last_z}


def _serialize_bocpd_state(state) -> dict | None:
    if state is None:
        return None
    return {"log_run_length_posterior": state.log_run_length_posterior.tolist(), "mu": state.mu.tolist(), "kappa": state.kappa.tolist(), "alpha": state.alpha.tolist(), "beta": state.beta.tolist(), "step_index": state.step_index}
