import numpy as np

from bolr.config.foundation import OrderedPartitionConfig, OrderedPartitionToleranceConfig
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_ordered_partition_builder_creates_three_groups() -> None:
    utilities = np.array([3.0, 2.5, 0.5, -1.0, -2.0])
    builder = OrderedPartitionBuilder(
        OrderedPartitionConfig(
            tolerance=OrderedPartitionToleranceConfig(absolute_tolerance=0.2),
            positive_threshold=0.0,
        )
    )
    observation = builder.build(utilities, date="2024-01-01")
    assert observation.group_labels == ("high", "middle", "low")
    assert observation.group_sizes == (1, 2, 2)
    assert observation.candidate_to_group.tolist() == [0, 1, 1, 2, 2]


def test_one_group_observation_can_no_update() -> None:
    utilities = np.array([1.0, 1.0, 1.0])
    builder = OrderedPartitionBuilder(
        OrderedPartitionConfig(
            tolerance=OrderedPartitionToleranceConfig(absolute_tolerance=0.1),
            all_irrelevant_policy="no_update",
        )
    )
    observation = builder.build(utilities)
    assert observation.all_irrelevant
    assert observation.update_weight == 0.0
    assert observation.metadata["observation_type"] == "NO_UPDATE"

