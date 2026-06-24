import numpy as np

from bolr.dynamics.discount import CovarianceDiscountDynamics
from bolr.dynamics.random_walk import IsotropicRandomWalkDynamics
from bolr.posterior.state import GaussianPosterior


def test_isotropic_random_walk_adds_process_covariance() -> None:
    posterior = GaussianPosterior(mean=np.zeros(3), covariance=np.eye(3))
    predicted = IsotropicRandomWalkDynamics(process_variance=0.25).predict(posterior, dt=2.0)
    assert np.allclose(predicted.mean, posterior.mean)
    assert np.allclose(predicted.covariance, np.eye(3) * 1.5)


def test_covariance_discount_scales_covariance() -> None:
    posterior = GaussianPosterior(mean=np.zeros(2), covariance=np.array([[2.0, 0.1], [0.1, 1.0]]))
    predicted = CovarianceDiscountDynamics(discount_factor=0.8).predict(posterior)
    assert np.allclose(predicted.mean, posterior.mean)
    assert np.allclose(predicted.covariance, posterior.covariance / 0.8)

