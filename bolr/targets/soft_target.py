from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.config.foundation import SoftTargetConfig
from bolr.numerics.stable_math import softmax
from bolr.targets.base import TargetBuilder


@dataclass(frozen=True)
class Observation:
    type: str
    utility_values: np.ndarray
    transformed_values: np.ndarray
    target_probabilities: np.ndarray
    tolerance: float
    update_weight: float
    metadata: dict[str, float | int | bool]


class SoftTargetBuilder(TargetBuilder):
    def __init__(self, config: SoftTargetConfig | None = None) -> None:
        self.config = config or SoftTargetConfig()

    def build(
        self,
        utilities: np.ndarray,
        *,
        date: object | None = None,
        candidate_metadata: object | None = None,
    ) -> Observation:
        del date, candidate_metadata
        return build_soft_target_observation(utilities, self.config)

    def metadata(self) -> dict[str, object]:
        return {
            "family": "candidate_a_soft_target",
            "kappa": self.config.kappa,
            "eta": self.config.eta,
            "clip": self.config.clip,
            "absolute_tolerance": self.config.absolute_tolerance,
            "relative_tolerance": self.config.relative_tolerance,
            "min_scale": self.config.min_scale,
        }


def build_soft_target_observation(
    utilities: np.ndarray,
    config: SoftTargetConfig | None = None,
) -> Observation:
    config = config or SoftTargetConfig()
    utilities = np.asarray(utilities, dtype=float)
    tolerance = _compute_tolerance(utilities, config)
    collapsed = _collapse_by_tolerance(utilities, tolerance)
    transformed, scale, clipping_fraction = _robust_transform(collapsed, config)
    target = softmax(config.kappa * transformed)

    tolerance_group_count = int(np.unique(collapsed).size)
    all_irrelevant = tolerance_group_count <= 1
    update_weight = 0.0 if all_irrelevant and config.no_update_if_degenerate else config.eta
    observation_type = "NO_UPDATE" if update_weight == 0.0 else "SOFT_TARGET"
    entropy = float(-(target * np.log(np.clip(target, 1e-300, 1.0))).sum())
    return Observation(
        type=observation_type,
        utility_values=utilities,
        transformed_values=transformed,
        target_probabilities=target,
        tolerance=tolerance,
        update_weight=update_weight,
        metadata={
            "positive_count": int((utilities > 0.0).sum()),
            "target_entropy": entropy,
            "tolerance_group_count": tolerance_group_count,
            "utility_scale": scale,
            "clipping_fraction": clipping_fraction,
            "all_irrelevant": all_irrelevant,
        },
    )


def _compute_tolerance(utilities: np.ndarray, config: SoftTargetConfig) -> float:
    centered = utilities - np.median(utilities)
    mad = np.median(np.abs(centered))
    robust_std = max(float(mad * 1.4826), config.min_scale)
    return max(config.absolute_tolerance, config.relative_tolerance * robust_std)


def _collapse_by_tolerance(utilities: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0.0:
        return utilities.copy()
    order = np.argsort(-utilities)
    grouped = np.empty_like(utilities)
    current_indices = [int(order[0])]
    current_anchor = float(utilities[order[0]])

    for idx in order[1:]:
        idx = int(idx)
        if abs(utilities[idx] - current_anchor) <= tolerance:
            current_indices.append(idx)
            continue
        mean_value = float(np.mean(utilities[current_indices]))
        grouped[current_indices] = mean_value
        current_indices = [idx]
        current_anchor = float(utilities[idx])

    mean_value = float(np.mean(utilities[current_indices]))
    grouped[current_indices] = mean_value
    return grouped


def _robust_transform(utilities: np.ndarray, config: SoftTargetConfig) -> tuple[np.ndarray, float, float]:
    centered = utilities - np.median(utilities)
    mad = float(np.median(np.abs(centered)))
    q75, q25 = np.percentile(utilities, [75, 25])
    iqr = float(q75 - q25)
    scale = max(mad, 0.7413 * iqr, config.min_scale)
    transformed = np.clip(centered / scale, -config.clip, config.clip)
    clipping_fraction = float(np.mean(np.abs(centered / scale) > config.clip))
    return transformed, scale, clipping_fraction
