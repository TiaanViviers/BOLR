from __future__ import annotations

import numpy as np

from bolr.config.foundation import PosteriorSamplingConfig
from bolr.decision.prediction import build_posterior_prediction
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign


def posterior_score_prediction(
    posterior: GaussianPosterior,
    design: DailyDesign,
    n_samples: int = 0,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    prediction = build_posterior_prediction(
        date=None,
        state_mean=posterior.mean,
        state_covariance=posterior.covariance,
        score_mean=posterior.score_mean(design),
        design_matrix=design.explicit_matrix(),
        sampling_config=PosteriorSamplingConfig(sample_count=n_samples, seed=0 if seed is None else seed, retain_score_samples=n_samples > 0),
        top_k_values=(1,),
        seed=seed,
    )
    result: dict[str, np.ndarray] = {"score_mean": prediction.score_mean, "score_variance": prediction.score_variance}
    if prediction.score_samples is not None:
        result["score_samples"] = prediction.score_samples
    return result
