from __future__ import annotations

from dataclasses import dataclass, field
from math import erf
from typing import Mapping

import numpy as np

from bolr.config.foundation import PosteriorSamplingConfig, RegionDefinitionConfig
from bolr.decision.regions import RegionSummary, summarize_regions
from bolr.decision.utils import binary_entropy, effective_count, ranks_from_scores, stable_rank_order
from bolr.model.graph import GridGraph


@dataclass(frozen=True)
class PairwiseProbability:
    left_index: int
    right_index: int
    analytic_probability_left_wins: float
    monte_carlo_probability_left_wins: float | None
    mean_difference: float
    variance_difference: float


@dataclass(frozen=True)
class PosteriorPrediction:
    date: object | None
    score_mean: np.ndarray
    score_variance: np.ndarray
    state_mean: np.ndarray
    state_covariance: np.ndarray
    probability_best: np.ndarray | None = None
    probability_top_k: Mapping[int, np.ndarray] = field(default_factory=dict)
    expected_rank: np.ndarray | None = None
    rank_stddev: np.ndarray | None = None
    selected_score_covariance: np.ndarray | None = None
    selected_score_covariance_indices: np.ndarray | None = None
    score_samples: np.ndarray | None = None
    state_samples: np.ndarray | None = None
    winning_indices: np.ndarray | None = None
    pairwise_probabilities: tuple[PairwiseProbability, ...] = tuple()
    region_summary: RegionSummary | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_mean", np.asarray(self.score_mean, dtype=float).copy())
        object.__setattr__(self, "score_variance", np.asarray(self.score_variance, dtype=float).copy())
        object.__setattr__(self, "state_mean", np.asarray(self.state_mean, dtype=float).copy())
        object.__setattr__(self, "state_covariance", np.asarray(self.state_covariance, dtype=float).copy())
        if self.probability_best is not None:
            object.__setattr__(self, "probability_best", np.asarray(self.probability_best, dtype=float).copy())
        object.__setattr__(self, "probability_top_k", {int(k): np.asarray(v, dtype=float).copy() for k, v in self.probability_top_k.items()})
        if self.expected_rank is not None:
            object.__setattr__(self, "expected_rank", np.asarray(self.expected_rank, dtype=float).copy())
        if self.rank_stddev is not None:
            object.__setattr__(self, "rank_stddev", np.asarray(self.rank_stddev, dtype=float).copy())
        if self.selected_score_covariance is not None:
            object.__setattr__(self, "selected_score_covariance", np.asarray(self.selected_score_covariance, dtype=float).copy())
        if self.selected_score_covariance_indices is not None:
            object.__setattr__(self, "selected_score_covariance_indices", np.asarray(self.selected_score_covariance_indices, dtype=int).copy())
        if self.score_samples is not None:
            object.__setattr__(self, "score_samples", np.asarray(self.score_samples, dtype=float).copy())
        if self.state_samples is not None:
            object.__setattr__(self, "state_samples", np.asarray(self.state_samples, dtype=float).copy())
        if self.winning_indices is not None:
            object.__setattr__(self, "winning_indices", np.asarray(self.winning_indices, dtype=int).copy())
        for array in (self.score_mean, self.score_variance, self.state_mean, self.state_covariance):
            if not np.isfinite(array).all():
                raise ValueError("PosteriorPrediction arrays must be finite.")


def score_variance_diagonal(design_matrix: np.ndarray, state_covariance: np.ndarray) -> np.ndarray:
    design_matrix = np.asarray(design_matrix, dtype=float)
    state_covariance = np.asarray(state_covariance, dtype=float)
    return np.einsum("ij,jk,ik->i", design_matrix, state_covariance, design_matrix)


def selected_score_covariance(design_matrix: np.ndarray, state_covariance: np.ndarray, indices: np.ndarray) -> np.ndarray:
    subdesign = np.asarray(design_matrix, dtype=float)[np.asarray(indices, dtype=int)]
    return subdesign @ np.asarray(state_covariance, dtype=float) @ subdesign.T


def gaussian_state_samples(
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    *,
    config: PosteriorSamplingConfig,
    seed: int | None = None,
) -> np.ndarray:
    if config.sample_count == 0:
        return np.zeros((0, np.asarray(state_mean).size), dtype=float)
    rng = np.random.default_rng(config.seed if seed is None else seed)
    mean = np.asarray(state_mean, dtype=float)
    chol = np.linalg.cholesky(np.asarray(state_covariance, dtype=float))
    if config.antithetic:
        half = (config.sample_count + 1) // 2
        z = rng.standard_normal((half, mean.size))
        z = np.vstack([z, -z])[: config.sample_count]
    else:
        z = rng.standard_normal((config.sample_count, mean.size))
    return mean + z @ chol.T


def score_samples_from_state_samples(
    score_mean: np.ndarray,
    design_matrix: np.ndarray,
    state_samples: np.ndarray,
    state_mean: np.ndarray,
) -> np.ndarray:
    state_samples = np.asarray(state_samples, dtype=float)
    if state_samples.shape[0] == 0:
        return np.zeros((0, np.asarray(score_mean).size), dtype=float)
    centered = state_samples - np.asarray(state_mean, dtype=float)
    return np.asarray(score_mean, dtype=float) + centered @ np.asarray(design_matrix, dtype=float).T


def monte_carlo_rank_summaries(
    score_samples: np.ndarray,
    *,
    top_k_values: tuple[int, ...],
) -> dict[str, object]:
    score_samples = np.asarray(score_samples, dtype=float)
    if score_samples.shape[0] == 0:
        return {
            "probability_best": None,
            "probability_top_k": {int(k): None for k in top_k_values},
            "expected_rank": None,
            "rank_stddev": None,
            "winning_indices": None,
            "tie_count": 0,
        }

    n_samples, n_candidates = score_samples.shape
    winning_indices = np.empty(n_samples, dtype=int)
    ranks = np.empty((n_samples, n_candidates), dtype=int)
    tie_count = 0
    for idx in range(n_samples):
        order = stable_rank_order(score_samples[idx])
        winning_indices[idx] = int(order[0])
        ranks[idx] = ranks_from_scores(score_samples[idx])
        tie_count += int(np.sum(score_samples[idx] == score_samples[idx, winning_indices[idx]]) > 1)
    probability_best = np.bincount(winning_indices, minlength=n_candidates).astype(float) / float(n_samples)
    probability_top_k = {
        int(k): (ranks <= int(k)).mean(axis=0)
        for k in top_k_values
    }
    return {
        "probability_best": probability_best,
        "probability_top_k": probability_top_k,
        "expected_rank": ranks.mean(axis=0),
        "rank_stddev": ranks.std(axis=0),
        "winning_indices": winning_indices,
        "tie_count": tie_count,
    }


def pairwise_win_probabilities(
    score_mean: np.ndarray,
    design_matrix: np.ndarray,
    state_covariance: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    *,
    score_samples: np.ndarray | None = None,
) -> tuple[PairwiseProbability, ...]:
    if not pairs:
        return tuple()
    score_mean = np.asarray(score_mean, dtype=float)
    design_matrix = np.asarray(design_matrix, dtype=float)
    state_covariance = np.asarray(state_covariance, dtype=float)
    samples = None if score_samples is None else np.asarray(score_samples, dtype=float)
    results: list[PairwiseProbability] = []
    for left, right in pairs:
        diff_design = design_matrix[left] - design_matrix[right]
        mean_diff = float(score_mean[left] - score_mean[right])
        var_diff = float(diff_design @ state_covariance @ diff_design)
        if var_diff <= 0.0:
            if mean_diff > 0.0:
                analytic = 1.0
            elif mean_diff < 0.0:
                analytic = 0.0
            else:
                analytic = 1.0
        else:
            analytic = float(0.5 * (1.0 + erf(mean_diff / np.sqrt(2.0 * var_diff))))
        monte_carlo = None
        if samples is not None and samples.shape[0] > 0:
            monte_carlo = float((samples[:, left] >= samples[:, right]).mean())
        results.append(
            PairwiseProbability(
                left_index=int(left),
                right_index=int(right),
                analytic_probability_left_wins=analytic,
                monte_carlo_probability_left_wins=monte_carlo,
                mean_difference=mean_diff,
                variance_difference=max(var_diff, 0.0),
            )
        )
    return tuple(results)


def build_posterior_prediction(
    *,
    date: object | None,
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    score_mean: np.ndarray,
    design_matrix: np.ndarray,
    sampling_config: PosteriorSamplingConfig | None = None,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    selected_covariance_indices: np.ndarray | None = None,
    pairwise_pairs: tuple[tuple[int, int], ...] = tuple(),
    candidate_grid=None,
    graph: GridGraph | None = None,
    region_config: RegionDefinitionConfig | None = None,
    block_score_means: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, object] | None = None,
    seed: int | None = None,
) -> PosteriorPrediction:
    state_mean = np.asarray(state_mean, dtype=float)
    state_covariance = np.asarray(state_covariance, dtype=float)
    score_mean = np.asarray(score_mean, dtype=float)
    design_matrix = np.asarray(design_matrix, dtype=float)
    variance = score_variance_diagonal(design_matrix, state_covariance)
    sampling = sampling_config or PosteriorSamplingConfig()
    state_samples = gaussian_state_samples(state_mean, state_covariance, config=sampling, seed=seed)
    score_samples = score_samples_from_state_samples(score_mean, design_matrix, state_samples, state_mean)
    rank_stats = monte_carlo_rank_summaries(score_samples, top_k_values=tuple(sorted(set(int(k) for k in top_k_values if k > 0))))
    keep_score_samples = score_samples if sampling.retain_score_samples else None
    keep_state_samples = state_samples if sampling.retain_score_samples else None

    selected_cov = None
    selected_cov_indices = None
    if selected_covariance_indices is not None:
        selected_cov_indices = np.asarray(selected_covariance_indices, dtype=int)
        selected_cov = selected_score_covariance(design_matrix, state_covariance, selected_cov_indices)

    pairwise = pairwise_win_probabilities(
        score_mean,
        design_matrix,
        state_covariance,
        pairwise_pairs,
        score_samples=score_samples if score_samples.shape[0] > 0 else None,
    )

    region_summary = None
    if region_config is not None and graph is not None and rank_stats["probability_top_k"]:
        region_summary = summarize_regions(
            score_mean=score_mean,
            score_variance=variance,
            probability_best=rank_stats["probability_best"],
            probability_top_k=rank_stats["probability_top_k"],
            candidate_grid=candidate_grid,
            graph=graph,
            region_config=region_config,
            score_samples=score_samples,
        )

    topk = {int(k): v for k, v in rank_stats["probability_top_k"].items() if v is not None}
    diagnostics = {
        "total_score_mean_norm": float(np.linalg.norm(score_mean)),
        "score_mean_min": float(np.min(score_mean)),
        "score_mean_max": float(np.max(score_mean)),
        "score_mean_range": float(np.max(score_mean) - np.min(score_mean)),
        "selected_candidate_mean": float(score_mean[int(np.argmax(score_mean))]),
        "winner_entropy": float(binary_entropy(rank_stats["probability_best"])) if rank_stats["probability_best"] is not None else np.nan,
        "effective_winner_count": float(effective_count(rank_stats["probability_best"])) if rank_stats["probability_best"] is not None else np.nan,
        "monte_carlo_sample_count": int(score_samples.shape[0]),
        "monte_carlo_tie_count": int(rank_stats["tie_count"]),
    }
    if block_score_means is not None:
        diagnostics["block_score_mean_norms"] = {name: float(np.linalg.norm(values)) for name, values in block_score_means.items()}
    if metadata:
        diagnostics.update(metadata)

    return PosteriorPrediction(
        date=date,
        score_mean=score_mean,
        score_variance=variance,
        state_mean=state_mean,
        state_covariance=state_covariance,
        probability_best=rank_stats["probability_best"],
        probability_top_k=topk,
        expected_rank=rank_stats["expected_rank"],
        rank_stddev=rank_stats["rank_stddev"],
        selected_score_covariance=selected_cov,
        selected_score_covariance_indices=selected_cov_indices,
        score_samples=keep_score_samples,
        state_samples=keep_state_samples,
        winning_indices=rank_stats["winning_indices"],
        pairwise_probabilities=pairwise,
        region_summary=region_summary,
        metadata=diagnostics,
    )
