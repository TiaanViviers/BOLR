from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.config.foundation import CrossGroupLogisticConfig
from bolr.numerics.stable_math import softplus
from bolr.observations.base import ObservationModel
from bolr.targets.ordered_partition import OrderedPartitionObservation, deterministic_sampling_seed


@dataclass(frozen=True)
class CrossGroupLogisticObservationModel(ObservationModel):
    config: CrossGroupLogisticConfig = CrossGroupLogisticConfig()

    def log_factor(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> float:
        diagnostics = self._evaluate(scores, observation)
        return -observation.update_weight * diagnostics["loss"]

    def score_gradient(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> np.ndarray:
        diagnostics = self._evaluate(scores, observation)
        return -observation.update_weight * diagnostics["gradient"]

    def score_curvature(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> np.ndarray:
        diagnostics = self._evaluate(scores, observation)
        return observation.update_weight * diagnostics["curvature"]

    def score_curvature_hvp(
        self,
        scores: np.ndarray,
        vector: np.ndarray,
        observation: OrderedPartitionObservation,
    ) -> np.ndarray:
        diagnostics = self._evaluate(scores, observation)
        return observation.update_weight * (diagnostics["curvature"] @ np.asarray(vector, dtype=float))

    def diagnostics(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> dict[str, object]:
        diagnostics = self._evaluate(scores, observation)
        return {
            "observation_family": "candidate_b_cross_group_logistic",
            "group_count": observation.metadata["group_count"],
            "group_sizes": observation.group_sizes,
            "high_group_size": observation.metadata["high_group_size"],
            "middle_group_size": observation.metadata["middle_group_size"],
            "low_group_size": observation.metadata["low_group_size"],
            "tolerance": observation.tolerance,
            "all_irrelevant": observation.all_irrelevant,
            "update_weight": observation.update_weight,
            "possible_pair_count": diagnostics["possible_pair_count"],
            "used_pair_count": diagnostics["used_pair_count"],
            "partition_complexity_proxy": observation.metadata["partition_complexity_proxy"],
            "log_factor_at_prior_mean": -observation.update_weight * diagnostics["loss"],
            "gradient_norm_at_prior_mean": float(np.linalg.norm(-observation.update_weight * diagnostics["gradient"])),
            "curvature_trace_or_estimate": float(np.trace(observation.update_weight * diagnostics["curvature"])),
            "sampling_seed": diagnostics["sampling_seed"],
            "pair_budget": diagnostics["pair_budget"],
            "group_pair_allocations": diagnostics["group_pair_allocations"],
            "duplicate_sample_count": diagnostics["duplicate_sample_count"],
        }

    def _evaluate(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> dict[str, object]:
        scores = np.asarray(scores, dtype=float)
        n = scores.size
        gradient = np.zeros(n, dtype=float)
        curvature = np.zeros((n, n), dtype=float)
        pair_terms: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
        possible_pair_count = int(observation.metadata["possible_pair_count"])
        used_pair_count = 0
        group_pair_allocations: dict[str, int] = {}
        duplicate_sample_count = 0
        pair_budget = self.config.sampled_pair_budget
        rng = None
        sampling_seed = None
        if pair_budget is not None:
            sampling_seed = deterministic_sampling_seed(observation.metadata.get("date"), self.config.sampling_seed)
            rng = np.random.default_rng(sampling_seed)

        ordered_groups = observation.ordered_groups
        active_pairs = [(a, b) for a in range(len(ordered_groups)) for b in range(a + 1, len(ordered_groups))]
        if not active_pairs:
            return {
                "loss": 0.0,
                "gradient": gradient,
                "curvature": curvature,
                "possible_pair_count": 0,
                "used_pair_count": 0,
                "sampling_seed": sampling_seed,
                "pair_budget": pair_budget,
                "group_pair_allocations": {},
                "duplicate_sample_count": 0,
            }

        weights = np.full(len(active_pairs), 1.0 / len(active_pairs), dtype=float)
        total_loss = 0.0
        for pair_idx, (a, b) in enumerate(active_pairs):
            group_a = ordered_groups[a]
            group_b = ordered_groups[b]
            all_pairs = np.array([(int(i), int(j)) for i in group_a for j in group_b], dtype=int)
            possible = int(all_pairs.shape[0])
            if pair_budget is not None:
                alloc = max(1, pair_budget // len(active_pairs))
                alloc = min(alloc, possible) if not self.config.sampled_with_replacement else alloc
                sample_indices = rng.choice(possible, size=alloc, replace=self.config.sampled_with_replacement)
                sampled_pairs = all_pairs[sample_indices]
                used_pair_count += int(sampled_pairs.shape[0])
                duplicate_sample_count += int(sampled_pairs.shape[0] - np.unique(sample_indices).size)
            else:
                sampled_pairs = all_pairs
                used_pair_count += possible
                alloc = possible
            group_pair_allocations[f"{a}>{b}"] = alloc
            pair_values = scores[sampled_pairs[:, 1]] - scores[sampled_pairs[:, 0]]
            p = 1.0 / (1.0 + np.exp(-pair_values))
            pair_loss = softplus(pair_values)
            mean_loss = float(pair_loss.mean()) if self.config.normalize_pair_losses else float(pair_loss.sum())
            total_loss += weights[pair_idx] * mean_loss
            normalizer = sampled_pairs.shape[0] if self.config.normalize_pair_losses else 1.0
            for (i, j), prob in zip(sampled_pairs, p, strict=True):
                coeff = weights[pair_idx] / normalizer
                gradient[i] += -coeff * prob
                gradient[j] += coeff * prob
                c = coeff * prob * (1.0 - prob)
                curvature[i, i] += c
                curvature[j, j] += c
                curvature[i, j] -= c
                curvature[j, i] -= c

        return {
            "loss": total_loss,
            "gradient": gradient,
            "curvature": curvature,
            "possible_pair_count": possible_pair_count,
            "used_pair_count": used_pair_count,
            "sampling_seed": sampling_seed,
            "pair_budget": pair_budget,
            "group_pair_allocations": group_pair_allocations,
            "duplicate_sample_count": duplicate_sample_count,
        }
