from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

from bolr.numerics.stable_math import logsumexp
from bolr.observations.base import ObservationModel
from bolr.targets.ordered_partition import OrderedPartitionObservation


@dataclass(frozen=True)
class ScalarJet:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray


def strict_pl_log_probability(scores: np.ndarray, permutation: tuple[int, ...]) -> float:
    scores = np.asarray(scores, dtype=float)
    total = 0.0
    for k, item in enumerate(permutation[:-1]):
        remaining = scores[list(permutation[k:])]
        total += scores[item] - logsumexp(remaining)
    return float(total)


def brute_force_partition_log_probability(scores: np.ndarray, observation: OrderedPartitionObservation) -> float:
    groups = [tuple(map(int, group.tolist())) for group in observation.ordered_groups]
    if len(groups) <= 1:
        return 0.0
    consistent = []
    for permuted_groups in _group_internal_permutations(groups):
        permutation = tuple(idx for group in permuted_groups for idx in group)
        consistent.append(strict_pl_log_probability(scores, permutation))
    return logsumexp(np.array(consistent, dtype=float))


def _group_internal_permutations(groups: list[tuple[int, ...]]) -> list[tuple[tuple[int, ...], ...]]:
    if not groups:
        return [tuple()]
    head = groups[0]
    tail = groups[1:]
    results = []
    for perm in permutations(head):
        for suffix in _group_internal_permutations(tail):
            results.append((perm, *suffix))
    return results


@dataclass(frozen=True)
class PartitionedPlackettLuceObservationModel(ObservationModel):
    max_group_size_for_exact: int = 8

    def log_factor(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> float:
        if len(observation.ordered_groups) <= 1 or observation.update_weight == 0.0:
            return 0.0
        return observation.update_weight * self._log_probability_jet(scores, observation).value

    def score_gradient(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> np.ndarray:
        if len(observation.ordered_groups) <= 1 or observation.update_weight == 0.0:
            return np.zeros_like(scores, dtype=float)
        return observation.update_weight * self._log_probability_jet(scores, observation).gradient

    def score_curvature(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> np.ndarray:
        if len(observation.ordered_groups) <= 1 or observation.update_weight == 0.0:
            n = np.asarray(scores).size
            return np.zeros((n, n), dtype=float)
        return -observation.update_weight * self._log_probability_jet(scores, observation).hessian

    def score_curvature_hvp(self, scores: np.ndarray, vector: np.ndarray, observation: OrderedPartitionObservation) -> np.ndarray:
        return self.score_curvature(scores, observation) @ np.asarray(vector, dtype=float)

    def diagnostics(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> dict[str, object]:
        jet = self._log_probability_jet(scores, observation)
        largest_group = max((len(group) for group in observation.ordered_groups[:-1]), default=0)
        return {
            "observation_family": "candidate_b_partitioned_pl",
            "group_count": len(observation.ordered_groups),
            "group_sizes": observation.group_sizes,
            "largest_upper_partition": largest_group,
            "partition_complexity_proxy": sum((len(group) * (2 ** len(group))) for group in observation.ordered_groups[:-1]),
            "log_factor_at_prior_mean": observation.update_weight * jet.value,
            "gradient_norm_at_prior_mean": float(np.linalg.norm(observation.update_weight * jet.gradient)),
            "curvature_trace_or_estimate": float(np.trace(-observation.update_weight * jet.hessian)),
        }

    def _log_probability_jet(self, scores: np.ndarray, observation: OrderedPartitionObservation) -> ScalarJet:
        scores = np.asarray(scores, dtype=float)
        n = scores.size
        if any(len(group) > self.max_group_size_for_exact for group in observation.ordered_groups[:-1]):
            raise ValueError("Partitioned PL exact DP is configured only for reduced upper-partition sizes.")

        log_prob = ScalarJet(value=0.0, gradient=np.zeros(n, dtype=float), hessian=np.zeros((n, n), dtype=float))
        remaining_lower = np.concatenate(observation.ordered_groups[1:]) if len(observation.ordered_groups) > 1 else np.array([], dtype=int)
        for group_index, group in enumerate(observation.ordered_groups[:-1]):
            lower = np.concatenate(observation.ordered_groups[group_index + 1 :]) if group_index + 1 < len(observation.ordered_groups) else np.array([], dtype=int)
            stage = setwise_stage_probability_jet(scores, group.astype(int), lower.astype(int))
            log_stage = jet_log(stage)
            log_prob = jet_add(log_prob, log_stage)
        return log_prob


def setwise_stage_probability_jet(scores: np.ndarray, group: np.ndarray, lower: np.ndarray) -> ScalarJet:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    group = tuple(map(int, group.tolist()))
    lower = tuple(map(int, lower.tolist()))
    basis = tuple(
        ScalarJet(
            value=float(scores[idx]),
            gradient=np.eye(n, dtype=float)[idx],
            hessian=np.zeros((n, n), dtype=float),
        )
        for idx in range(n)
    )
    exp_scores = tuple(jet_exp(jet) for jet in basis)

    @lru_cache(maxsize=None)
    def recurse(active_group: tuple[int, ...]) -> ScalarJet:
        if not active_group:
            return ScalarJet(1.0, np.zeros(n, dtype=float), np.zeros((n, n), dtype=float))
        total = jet_constant(0.0, n)
        denom = jet_constant(0.0, n)
        for idx in active_group:
            denom = jet_add(denom, exp_scores[idx])
        for idx in lower:
            denom = jet_add(denom, exp_scores[idx])
        for idx in active_group:
            numerator = exp_scores[idx]
            term = jet_mul(jet_div(numerator, denom), recurse(tuple(j for j in active_group if j != idx)))
            total = jet_add(total, term)
        return total

    return recurse(group)


def jet_constant(value: float, dimension: int) -> ScalarJet:
    return ScalarJet(value=float(value), gradient=np.zeros(dimension, dtype=float), hessian=np.zeros((dimension, dimension), dtype=float))


def jet_add(left: ScalarJet, right: ScalarJet) -> ScalarJet:
    return ScalarJet(left.value + right.value, left.gradient + right.gradient, left.hessian + right.hessian)


def jet_mul(left: ScalarJet, right: ScalarJet) -> ScalarJet:
    return ScalarJet(
        left.value * right.value,
        left.gradient * right.value + right.gradient * left.value,
        left.hessian * right.value
        + right.hessian * left.value
        + np.outer(left.gradient, right.gradient)
        + np.outer(right.gradient, left.gradient),
    )


def jet_div(left: ScalarJet, right: ScalarJet) -> ScalarJet:
    return jet_mul(left, jet_inv(right))


def jet_inv(jet: ScalarJet) -> ScalarJet:
    value = 1.0 / jet.value
    grad = -jet.gradient / (jet.value**2)
    hess = (2.0 / (jet.value**3)) * np.outer(jet.gradient, jet.gradient) - jet.hessian / (jet.value**2)
    return ScalarJet(value, grad, hess)


def jet_exp(jet: ScalarJet) -> ScalarJet:
    value = float(np.exp(jet.value))
    grad = value * jet.gradient
    hess = value * (jet.hessian + np.outer(jet.gradient, jet.gradient))
    return ScalarJet(value, grad, hess)


def jet_log(jet: ScalarJet) -> ScalarJet:
    value = float(np.log(jet.value))
    grad = jet.gradient / jet.value
    hess = jet.hessian / jet.value - np.outer(jet.gradient, jet.gradient) / (jet.value**2)
    return ScalarJet(value, grad, hess)
