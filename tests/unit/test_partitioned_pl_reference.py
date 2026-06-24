import itertools

import numpy as np

from bolr.observations.partitioned_pl import (
    PartitionedPlackettLuceObservationModel,
    brute_force_partition_log_probability,
    strict_pl_log_probability,
)
from bolr.targets.ordered_partition import OrderedPartitionObservation


def test_strict_pl_probabilities_sum_to_one() -> None:
    scores = np.array([0.2, -0.1, 0.5])
    total = 0.0
    for perm in itertools.permutations(range(3)):
        total += np.exp(strict_pl_log_probability(scores, perm))
    assert np.isclose(total, 1.0)


def test_partitioned_pl_model_matches_bruteforce() -> None:
    observation = OrderedPartitionObservation(
        ordered_groups=(np.array([0, 1]), np.array([2])),
        candidate_to_group=np.array([0, 0, 1]),
        group_labels=("high", "low"),
        group_sizes=(2, 1),
        tolerance=0.0,
        utility_maximum=1.0,
        utility_median=0.0,
        utility_scale=1.0,
        all_irrelevant=False,
        update_weight=1.0,
        metadata={"group_count": 2, "possible_pair_count": 2, "partition_complexity_proxy": 5, "high_group_size": 2, "middle_group_size": 0, "low_group_size": 1},
    )
    scores = np.array([0.3, 0.1, -0.4])
    brute = brute_force_partition_log_probability(scores, observation)
    model = PartitionedPlackettLuceObservationModel(max_group_size_for_exact=4)
    assert np.isclose(model.log_factor(scores, observation), brute)

