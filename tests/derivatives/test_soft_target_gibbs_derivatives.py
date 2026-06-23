import numpy as np

from bolr.config.foundation import SoftTargetConfig
from bolr.numerics.derivatives import parameter_gradient, parameter_hessian, parameter_hvp
from bolr.observations.soft_target_gibbs import log_factor, score_gradient, score_hessian, score_hvp
from bolr.targets.soft_target import build_soft_target_observation


def _finite_difference_gradient(func, point, step=1e-6):
    grad = np.zeros_like(point)
    for i in range(point.size):
        delta = np.zeros_like(point)
        delta[i] = step
        grad[i] = (func(point + delta) - func(point - delta)) / (2.0 * step)
    return grad


def _finite_difference_hvp(func, point, vector, step=1e-5):
    return (_finite_difference_gradient(func, point + step * vector) - _finite_difference_gradient(func, point - step * vector)) / (2.0 * step)


def test_score_gradient_matches_finite_difference() -> None:
    scores = np.array([0.3, -0.2, 0.7, 0.1])
    observation = build_soft_target_observation(
        np.array([1.5, 0.0, 0.4, -0.3]),
        SoftTargetConfig(kappa=1.2, eta=0.8),
    )
    numerical = _finite_difference_gradient(lambda x: log_factor(x, observation), scores)
    analytic = score_gradient(scores, observation)
    assert np.allclose(analytic, numerical, atol=1e-6)


def test_score_hessian_and_hvp_match_finite_difference() -> None:
    scores = np.array([0.1, -0.4, 0.2, 1.0])
    vector = np.array([0.5, -0.1, 0.3, 0.2])
    observation = build_soft_target_observation(
        np.array([0.8, -0.2, 0.5, 1.3]),
        SoftTargetConfig(kappa=0.9, eta=1.1),
    )
    dense = score_hessian(scores, observation)
    numerical_hvp = _finite_difference_hvp(lambda x: log_factor(x, observation), scores, vector)
    analytic_hvp = score_hvp(scores, vector, observation)
    assert np.allclose(dense @ vector, analytic_hvp, atol=1e-6)
    assert np.allclose(analytic_hvp, numerical_hvp, atol=1e-5)


def test_parameter_derivatives_are_consistent() -> None:
    design = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, 1.0, 0.25],
            [-0.3, 0.2, 0.1],
            [0.0, -0.8, 0.9],
        ]
    )
    theta = np.array([0.2, -0.4, 0.7])
    observation = build_soft_target_observation(
        np.array([1.0, 0.2, -0.7, 0.4]),
        SoftTargetConfig(kappa=1.4, eta=0.6),
    )

    def objective(param):
        return log_factor(design @ param, observation)

    scores = design @ theta
    numerical_grad = _finite_difference_gradient(objective, theta)
    analytic_grad = -observation.update_weight * parameter_gradient(design, scores, observation)
    assert np.allclose(analytic_grad, numerical_grad, atol=1e-6)

    hessian = -parameter_hessian(design, scores, observation)
    vector = np.array([0.4, -0.2, 0.5])
    analytic_hvp = hessian @ vector
    fast_hvp = -parameter_hvp(design, scores, vector, observation)
    numerical_hvp = _finite_difference_hvp(objective, theta, vector)
    assert np.allclose(analytic_hvp, fast_hvp, atol=1e-6)
    assert np.allclose(fast_hvp, numerical_hvp, atol=1e-5)
