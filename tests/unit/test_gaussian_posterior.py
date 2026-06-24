import numpy as np

from bolr.posterior.sampling import posterior_score_prediction
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign


def test_gaussian_posterior_prediction_and_sampling_shapes() -> None:
    posterior = GaussianPosterior(
        mean=np.array([0.1, -0.2, 0.3, 0.4]),
        covariance=np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.1, 1.2, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.05],
                [0.0, 0.0, 0.05, 0.7],
            ]
        ),
    )
    design = DailyDesign(
        candidate_basis=np.array([[1.0, 0.5], [0.2, -0.3], [-0.4, 0.7]]),
        context_vector=np.array([1.0, -0.2]),
        static_scores=np.array([0.1, 0.0, -0.1]),
    )
    prediction = posterior_score_prediction(posterior, design, n_samples=5, seed=7)
    assert prediction["score_mean"].shape == (3,)
    assert prediction["score_variance"].shape == (3,)
    assert prediction["score_samples"].shape == (5, 3)
    assert np.all(prediction["score_variance"] >= 0.0)

