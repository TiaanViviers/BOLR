import numpy as np

from bolr.model.diagnostics import gaussian_kl_divergence, innovation_diagnostics
from bolr.model.penalties import difference_penalty
from bolr.model.structured import diagonal_process_noise, isotropic_process_noise, penalty_shaped_process_noise, prior_from_penalty


def test_structured_prior_is_proper_and_spd() -> None:
    penalty = difference_penalty(5, 2)
    prior = prior_from_penalty(penalty, smooth_weight=2.0, ridge=0.3)
    eig = np.linalg.eigvalsh(prior.precision)
    assert np.min(eig) > 0.0
    assert np.allclose(prior.covariance @ prior.precision, np.eye(5), atol=1e-8)


def test_penalty_shaped_process_noise_is_psd() -> None:
    penalty = difference_penalty(6, 2)
    q = penalty_shaped_process_noise(penalty, scale=0.5, properization=0.2)
    eig = np.linalg.eigvalsh(q.covariance)
    assert np.min(eig) > 0.0
    iso = isotropic_process_noise(3, 0.4)
    diag = diagonal_process_noise(np.array([0.1, 0.2, 0.3]))
    assert iso.covariance.shape == (3, 3)
    assert diag.covariance.shape == (3, 3)


def test_innovation_diagnostics_and_gaussian_kl_are_finite() -> None:
    mean0 = np.zeros(2)
    cov0 = np.array([[2.0, 0.2], [0.2, 1.5]])
    mean1 = np.array([0.1, -0.2])
    cov1 = np.array([[1.5, 0.1], [0.1, 1.0]])
    kl = gaussian_kl_divergence(mean1, cov1, mean0, cov0)
    assert kl >= 0.0
    diag = innovation_diagnostics(mean0, cov0, mean1, cov1, -3.0, -2.5)
    assert diag["state_update_l2"] > 0.0
    assert diag["gaussian_kl"] >= 0.0

