from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.config.foundation import OrderedPartitionConfig, OrderedPartitionToleranceConfig
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_c_ordered_partition_matches_python_builder() -> None:
    backend = CBackend()
    config = OrderedPartitionConfig(
        tolerance=OrderedPartitionToleranceConfig(absolute_tolerance=0.2),
        positive_threshold=0.0,
        all_irrelevant_policy="no_update",
    )
    utilities = np.array([3.0, 2.5, 0.5, -1.0, -2.0])
    python_observation = OrderedPartitionBuilder(config).build(utilities, date="2024-01-01")

    with backend.ordered_partition(utilities, config) as partition:
        diagnostics = partition.diagnostics()
        assert diagnostics.group_count == len(python_observation.ordered_groups)
        assert np.isclose(diagnostics.tolerance, python_observation.tolerance)
        assert np.isclose(diagnostics.update_weight, python_observation.update_weight)
        assert diagnostics.high_group_size == python_observation.metadata["high_group_size"]
        assert diagnostics.middle_group_size == python_observation.metadata["middle_group_size"]
        assert diagnostics.low_group_size == python_observation.metadata["low_group_size"]
        assert np.array_equal(partition.candidate_to_group(), python_observation.candidate_to_group)

