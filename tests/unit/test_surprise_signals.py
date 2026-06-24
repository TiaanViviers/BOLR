import numpy as np

from bolr.adaptation.surprise import GeneralizedPredictiveLossSurprise, PosteriorKLSurprise, PosteriorMahalanobisSurprise
from bolr.posterior.state import GaussianPosterior


class _ObsModel:
    def log_factor(self, scores, observation):
        return -float(np.sum(scores**2)) * observation.update_weight


class _Observation:
    def __init__(self, update_weight=1.0, metadata=None):
        self.update_weight = update_weight
        self.metadata = metadata or {}


def test_generalized_predictive_loss_handles_zero_update_weight() -> None:
    signal = GeneralizedPredictiveLossSurprise()
    value, diagnostics = signal.compute(
        predictive_posterior=GaussianPosterior(np.zeros(1), np.eye(1)),
        posterior=GaussianPosterior(np.zeros(1), np.eye(1)),
        predictive_scores=np.array([0.3]),
        posterior_scores=np.array([0.1]),
        observation=_Observation(update_weight=0.0),
        observation_model=_ObsModel(),
        update_diagnostics={},
    )
    assert value is None
    assert diagnostics["missing"] is True


def test_posterior_surprise_signals_are_finite() -> None:
    predictive = GaussianPosterior(np.zeros(2), np.array([[2.0, 0.1], [0.1, 1.5]]))
    posterior = GaussianPosterior(np.array([0.3, -0.2]), np.array([[1.2, 0.05], [0.05, 1.0]]))
    mahal_value, _ = PosteriorMahalanobisSurprise().compute(
        predictive_posterior=predictive,
        posterior=posterior,
        predictive_scores=np.zeros(2),
        posterior_scores=np.zeros(2),
        observation=_Observation(),
        observation_model=_ObsModel(),
        update_diagnostics={},
    )
    kl_value, _ = PosteriorKLSurprise().compute(
        predictive_posterior=predictive,
        posterior=posterior,
        predictive_scores=np.zeros(2),
        posterior_scores=np.zeros(2),
        observation=_Observation(),
        observation_model=_ObsModel(),
        update_diagnostics={},
    )
    assert mahal_value > 0.0
    assert kl_value > 0.0
