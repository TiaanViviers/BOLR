import numpy as np

from bolr.config.foundation import CrossGroupLogisticConfig
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.targets.ordered_partition import OrderedPartitionObservation


def _fd_grad(func, point, step=1e-6):
    grad = np.zeros_like(point)
    for i in range(point.size):
        delta = np.zeros_like(point)
        delta[i] = step
        grad[i] = (func(point + delta) - func(point - delta)) / (2 * step)
    return grad


def test_cross_group_gradient_matches_finite_difference() -> None:
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
    scores = np.array([0.2, -0.4, 0.7])
    model = CrossGroupLogisticObservationModel(CrossGroupLogisticConfig())
    numerical = _fd_grad(lambda x: model.log_factor(x, observation), scores)
    analytic = model.score_gradient(scores, observation)
    assert np.allclose(analytic, numerical, atol=1e-6)

