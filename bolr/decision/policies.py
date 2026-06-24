from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from bolr.config.foundation import DecisionPolicyConfig
from bolr.decision.prediction import PosteriorPrediction
from bolr.decision.regions import RegionComponentSummary, weighted_graph_medoid
from bolr.model.graph import GridGraph


@dataclass(frozen=True)
class Decision:
    selected_index: int | None
    selected_config_id: object | None
    policy_name: str
    selected_score_mean: float | None
    selected_score_variance: float | None
    selected_probability_best: float | None
    selected_probability_top_k: Mapping[int, float] = field(default_factory=dict)
    selected_expected_rank: float | None = None
    selected_region_id: int | None = None
    selected_region_mass: float | None = None
    abstained: bool = False
    reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class DecisionPolicy(Protocol):
    def decide(
        self,
        prediction: PosteriorPrediction,
        *,
        candidate_grid,
        graph: GridGraph | None = None,
    ) -> Decision: ...

    def metadata(self) -> Mapping[str, object]: ...


def _break_ties(primary: np.ndarray, score_mean: np.ndarray, score_variance: np.ndarray) -> tuple[int, dict[str, object]]:
    primary = np.asarray(primary, dtype=float)
    score_mean = np.asarray(score_mean, dtype=float)
    score_variance = np.asarray(score_variance, dtype=float)
    candidates = np.flatnonzero(primary == np.max(primary))
    diagnostics = {"tie_occurred": bool(candidates.size > 1), "tie_break_stages": ["primary"]}
    if candidates.size == 1:
        return int(candidates[0]), diagnostics
    top_mean = np.max(score_mean[candidates])
    candidates = candidates[score_mean[candidates] == top_mean]
    diagnostics["tie_break_stages"].append("posterior_mean")
    if candidates.size == 1:
        return int(candidates[0]), diagnostics
    min_variance = np.min(score_variance[candidates])
    candidates = candidates[score_variance[candidates] == min_variance]
    diagnostics["tie_break_stages"].append("posterior_variance")
    return int(np.min(candidates)), diagnostics


def _decision_from_index(
    index: int,
    prediction: PosteriorPrediction,
    *,
    candidate_grid,
    policy_name: str,
    diagnostics: Mapping[str, object],
    region: RegionComponentSummary | None = None,
) -> Decision:
    top_k = {int(k): float(v[index]) for k, v in prediction.probability_top_k.items()}
    return Decision(
        selected_index=int(index),
        selected_config_id=int(candidate_grid.config_ids[index]) if candidate_grid is not None else int(index),
        policy_name=policy_name,
        selected_score_mean=float(prediction.score_mean[index]),
        selected_score_variance=float(prediction.score_variance[index]),
        selected_probability_best=None if prediction.probability_best is None else float(prediction.probability_best[index]),
        selected_probability_top_k=top_k,
        selected_expected_rank=None if prediction.expected_rank is None else float(prediction.expected_rank[index]),
        selected_region_id=None if region is None else region.region_id,
        selected_region_mass=None if region is None else region.probability_best_mass,
        diagnostics=dict(diagnostics),
    )


@dataclass(frozen=True)
class PosteriorMeanDecisionPolicy:
    config: DecisionPolicyConfig = DecisionPolicyConfig(family="posterior_mean_argmax")

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        del graph
        index, diag = _break_ties(prediction.score_mean, prediction.score_mean, prediction.score_variance)
        return _decision_from_index(index, prediction, candidate_grid=candidate_grid, policy_name=self.config.family, diagnostics=diag)

    def metadata(self) -> Mapping[str, object]:
        return {"family": self.config.family}


@dataclass(frozen=True)
class MaximumProbabilityBestDecisionPolicy:
    config: DecisionPolicyConfig = DecisionPolicyConfig(family="maximum_probability_best")

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        del graph
        if prediction.probability_best is None:
            raise ValueError("probability_best is required for maximum_probability_best.")
        index, diag = _break_ties(prediction.probability_best, prediction.score_mean, prediction.score_variance)
        return _decision_from_index(index, prediction, candidate_grid=candidate_grid, policy_name=self.config.family, diagnostics=diag)

    def metadata(self) -> Mapping[str, object]:
        return {"family": self.config.family}


@dataclass(frozen=True)
class MaximumProbabilityTopKDecisionPolicy:
    config: DecisionPolicyConfig

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        del graph
        if self.config.top_k not in prediction.probability_top_k:
            raise ValueError("Configured top_k probabilities are unavailable.")
        values = prediction.probability_top_k[int(self.config.top_k)]
        index, diag = _break_ties(values, prediction.score_mean, prediction.score_variance)
        return _decision_from_index(index, prediction, candidate_grid=candidate_grid, policy_name=self.config.family, diagnostics=diag)

    def metadata(self) -> Mapping[str, object]:
        return {"family": self.config.family, "top_k": self.config.top_k}


@dataclass(frozen=True)
class MinimumExpectedRankDecisionPolicy:
    config: DecisionPolicyConfig = DecisionPolicyConfig(family="minimum_expected_rank")

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        del graph
        if prediction.expected_rank is None:
            raise ValueError("expected_rank is required for minimum_expected_rank.")
        values = -prediction.expected_rank
        index, diag = _break_ties(values, prediction.score_mean, prediction.score_variance)
        return _decision_from_index(index, prediction, candidate_grid=candidate_grid, policy_name=self.config.family, diagnostics=diag)

    def metadata(self) -> Mapping[str, object]:
        return {"family": self.config.family}


@dataclass(frozen=True)
class ThompsonDecisionPolicy:
    config: DecisionPolicyConfig = DecisionPolicyConfig(family="thompson")

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        del graph
        if prediction.score_samples is None or prediction.score_samples.shape[0] == 0:
            raise ValueError("Thompson decision requires retained score_samples.")
        sampled_scores = prediction.score_samples[0]
        index, diag = _break_ties(sampled_scores, prediction.score_mean, prediction.score_variance)
        return _decision_from_index(index, prediction, candidate_grid=candidate_grid, policy_name=self.config.family, diagnostics=diag | {"sample_index": 0})

    def metadata(self) -> Mapping[str, object]:
        return {"family": self.config.family}


@dataclass(frozen=True)
class HighestMassRegionDecisionPolicy:
    config: DecisionPolicyConfig

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        if prediction.region_summary is None:
            raise ValueError("Region summaries are required for region-based decisions.")
        if graph is None:
            raise ValueError("Region-based decisions require a graph.")
        if not prediction.region_summary.components:
            raise ValueError("Region summary contains no components.")
        if self.config.region_selection_statistic == "probability_best":
            region_stats = np.array([component.probability_best_mass for component in prediction.region_summary.components], dtype=float)
        else:
            region_stats = np.array([component.inclusion_mass for component in prediction.region_summary.components], dtype=float)
        region_idx, diag = _break_ties(region_stats, np.array([component.maximum_score_mean for component in prediction.region_summary.components]), np.array([-component.candidate_count for component in prediction.region_summary.components], dtype=float))
        region = prediction.region_summary.components[region_idx]
        index = _select_region_representative(region, prediction, graph, self.config)
        return _decision_from_index(
            index,
            prediction,
            candidate_grid=candidate_grid,
            policy_name=self.config.family,
            diagnostics=diag | {"region_selection_statistic": self.config.region_selection_statistic, "region_representative_policy": self.config.representative_policy, "region_candidate_count": region.candidate_count},
            region=region,
        )

    def metadata(self) -> Mapping[str, object]:
        return {
            "family": self.config.family,
            "region_selection_statistic": self.config.region_selection_statistic,
            "representative_policy": self.config.representative_policy,
        }


class OutsideOptionProvider(Protocol):
    def decide(
        self,
        prediction: PosteriorPrediction,
        *,
        candidate_grid,
        graph: GridGraph | None = None,
    ) -> Decision: ...

    def metadata(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class OutsideOptionDecisionPolicy:
    provider: OutsideOptionProvider
    config: DecisionPolicyConfig = DecisionPolicyConfig(family="outside_option", outside_option_provider="provider")

    def decide(self, prediction: PosteriorPrediction, *, candidate_grid, graph: GridGraph | None = None) -> Decision:
        return self.provider.decide(prediction, candidate_grid=candidate_grid, graph=graph)

    def metadata(self) -> Mapping[str, object]:
        return {"family": self.config.family, "outside_option_provider": self.provider.metadata()}


def _select_region_representative(
    region: RegionComponentSummary,
    prediction: PosteriorPrediction,
    graph: GridGraph,
    config: DecisionPolicyConfig,
) -> int:
    indices = region.candidate_indices
    if config.representative_policy == "posterior_mean":
        values = prediction.score_mean[indices]
        chosen, _ = _break_ties(values, prediction.score_mean[indices], prediction.score_variance[indices])
        return int(indices[chosen])
    if config.representative_policy == "probability_best":
        if prediction.probability_best is None:
            raise ValueError("probability_best is required for the requested representative policy.")
        values = prediction.probability_best[indices]
        chosen, _ = _break_ties(values, prediction.score_mean[indices], prediction.score_variance[indices])
        return int(indices[chosen])
    if config.representative_policy == "probability_top_k":
        top_k = region.candidate_count if config.top_k is None else int(config.top_k)
        if top_k not in prediction.probability_top_k:
            top_k = prediction.region_summary.top_k if prediction.region_summary is not None else top_k
        values = prediction.probability_top_k[top_k][indices]
        chosen, _ = _break_ties(values, prediction.score_mean[indices], prediction.score_variance[indices])
        return int(indices[chosen])
    weights = prediction.region_summary.inclusion_probability[indices]
    medoid, _ = weighted_graph_medoid(indices, weights, graph)
    return medoid
