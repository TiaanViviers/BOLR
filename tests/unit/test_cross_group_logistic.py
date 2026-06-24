import numpy as np

from bolr.config.foundation import CrossGroupLogisticConfig
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.targets.ordered_partition import OrderedPartitionObservation


def test_cross_group_logistic_matches_manual_tiny_loss() -> None:
    observation = OrderedPartitionObservation(
        ordered_groups=(np.array([0]), np.array([1])),
        candidate_to_group=np.array([0, 1]),
        group_labels=("high", "low"),
        group_sizes=(1, 1),
        tolerance=0.0,
        utility_maximum=1.0,
        utility_median=0.0,
        utility_scale=1.0,
        all_irrelevant=False,
        update_weight=1.0,
        metadata={"group_count": 2, "possible_pair_count": 1, "partition_complexity_proxy": 3, "high_group_size": 1, "middle_group_size": 0, "low_group_size": 1},
    )
    scores = np.array([1.0, -1.0])
    model = CrossGroupLogisticObservationModel(CrossGroupLogisticConfig())
    expected = -np.log1p(np.exp(scores[1] - scores[0]))
    assert np.isclose(model.log_factor(scores, observation), expected)
    gradient = model.score_gradient(scores, observation)
    assert np.isclose(gradient.sum(), 0.0)


def test_cross_group_sampled_mode_is_deterministic() -> None:
    observation = OrderedPartitionObservation(
        ordered_groups=(np.array([0, 1]), np.array([2, 3])),
        candidate_to_group=np.array([0, 0, 1, 1]),
        group_labels=("high", "low"),
        group_sizes=(2, 2),
        tolerance=0.0,
        utility_maximum=1.0,
        utility_median=0.0,
        utility_scale=1.0,
        all_irrelevant=False,
        update_weight=1.0,
        metadata={"group_count": 2, "possible_pair_count": 4, "partition_complexity_proxy": 8, "high_group_size": 2, "middle_group_size": 0, "low_group_size": 2, "date": "2024-01-01"},
    )
    scores = np.array([0.5, 0.1, -0.2, -1.0])
    model = CrossGroupLogisticObservationModel(CrossGroupLogisticConfig(sampled_pair_budget=2, sampling_seed=7))
    first = model.log_factor(scores, observation)
    second = model.log_factor(scores, observation)
    assert np.isclose(first, second)

