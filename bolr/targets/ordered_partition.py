from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np

from bolr.config.foundation import OrderedPartitionConfig, OrderedPartitionToleranceConfig
from bolr.targets.base import TargetBuilder


@dataclass(frozen=True)
class OrderedPartitionObservation:
    ordered_groups: tuple[np.ndarray, ...]
    candidate_to_group: np.ndarray
    group_labels: tuple[str, ...]
    group_sizes: tuple[int, ...]
    tolerance: float
    utility_maximum: float
    utility_median: float
    utility_scale: float
    all_irrelevant: bool
    update_weight: float
    metadata: dict[str, object]


class OrderedPartitionBuilder(TargetBuilder):
    def __init__(self, config: OrderedPartitionConfig | None = None) -> None:
        self.config = config or OrderedPartitionConfig()

    def build(
        self,
        utilities: np.ndarray,
        *,
        date: object | None = None,
        candidate_metadata: object | None = None,
    ) -> OrderedPartitionObservation:
        del candidate_metadata
        utilities = np.asarray(utilities, dtype=float)
        tolerance, components, scale = compute_ordered_partition_tolerance(utilities, self.config.tolerance)
        utility_maximum = float(np.max(utilities))
        utility_median = float(np.median(utilities))
        high_mask = utilities >= utility_maximum - tolerance
        middle_mask = (utilities > self.config.positive_threshold) & ~high_mask
        low_mask = ~(high_mask | middle_mask)

        groups: list[np.ndarray] = []
        labels: list[str] = []
        for label, mask in (("high", high_mask), ("middle", middle_mask), ("low", low_mask)):
            indices = np.flatnonzero(mask).astype(int)
            if indices.size > 0:
                groups.append(indices.copy())
                labels.append(label)

        all_irrelevant = len(groups) <= 1 or np.all(utilities <= self.config.positive_threshold)
        update_weight = 1.0
        policy = self.config.all_irrelevant_policy
        observation_type = "ORDERED_PARTITION"
        if all_irrelevant:
            if policy == "no_update":
                update_weight = 0.0
                observation_type = "NO_UPDATE"
            elif policy == "reduced_weight":
                update_weight = self.config.reduced_weight
                observation_type = "REDUCED_WEIGHT"
            elif policy == "always_relative":
                update_weight = 1.0
                observation_type = "ALWAYS_RELATIVE"

        candidate_to_group = np.full(utilities.shape[0], -1, dtype=int)
        for group_idx, indices in enumerate(groups):
            candidate_to_group[indices] = group_idx
        if np.any(candidate_to_group < 0):
            raise ValueError("All candidates must belong to exactly one ordered group.")

        possible_pair_count = int(sum(len(groups[a]) * len(groups[b]) for a in range(len(groups)) for b in range(a + 1, len(groups))))
        largest_upper_partition = int(max((len(group) for group in groups[:-1]), default=len(groups[0]) if groups else 0))
        complexity_proxy = int(utilities.size + largest_upper_partition**3)
        return OrderedPartitionObservation(
            ordered_groups=tuple(group.copy() for group in groups),
            candidate_to_group=candidate_to_group,
            group_labels=tuple(labels),
            group_sizes=tuple(int(len(group)) for group in groups),
            tolerance=tolerance,
            utility_maximum=utility_maximum,
            utility_median=utility_median,
            utility_scale=scale,
            all_irrelevant=all_irrelevant,
            update_weight=update_weight,
            metadata={
                "observation_type": observation_type,
                "group_count": len(groups),
                "group_sizes": tuple(int(len(group)) for group in groups),
                "high_group_size": int(len(groups[0])) if labels and labels[0] == "high" else 0,
                "middle_group_size": int(len(groups[labels.index("middle")])) if "middle" in labels else 0,
                "low_group_size": int(len(groups[-1])) if labels and labels[-1] == "low" else 0,
                "tolerance": tolerance,
                "tolerance_absolute_component": components["absolute"],
                "tolerance_relative_component": components["relative"],
                "tolerance_execution_component": components["execution"],
                "utility_scale": scale,
                "all_irrelevant": all_irrelevant,
                "update_weight": update_weight,
                "informative_group_count": max(0, len(groups) - 1),
                "possible_pair_count": possible_pair_count,
                "largest_upper_partition": largest_upper_partition,
                "partition_complexity_proxy": complexity_proxy,
                "date": date,
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "family": "candidate_b_ordered_partition",
            "positive_threshold": self.config.positive_threshold,
            "all_irrelevant_policy": self.config.all_irrelevant_policy,
            "reduced_weight": self.config.reduced_weight,
            "tolerance": self.config.tolerance,
        }


def compute_ordered_partition_tolerance(
    utilities: np.ndarray,
    config: OrderedPartitionToleranceConfig,
) -> tuple[float, dict[str, float], float]:
    centered = utilities - np.median(utilities)
    mad_scale = 1.4826 * float(np.median(np.abs(centered)))
    q75, q25 = np.percentile(utilities, [75, 25])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    if config.robust_scale == "mad":
        scale = max(mad_scale, config.scale_floor)
    elif config.robust_scale == "iqr":
        scale = max(iqr_scale, config.scale_floor)
    else:
        scale = max(mad_scale, iqr_scale, config.scale_floor)
    components = {
        "absolute": float(config.absolute_tolerance),
        "relative": float(config.relative_tolerance * scale),
        "execution": float(config.execution_tolerance),
    }
    return max(components.values()), components, scale


def deterministic_sampling_seed(date: object | None, base_seed: int) -> int:
    payload = f"{date}|{base_seed}".encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
