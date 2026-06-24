from __future__ import annotations

from typing import Mapping

import numpy as np

from bolr.model.diagnostics import gaussian_kl_divergence
from bolr.observations.base import ObservationModel
from bolr.posterior.diagnostics import jittered_cholesky
from bolr.posterior.state import GaussianPosterior


class GeneralizedPredictiveLossSurprise:
    name = "generalized_predictive_loss"

    def compute(
        self,
        *,
        predictive_posterior: GaussianPosterior,
        posterior: GaussianPosterior,
        predictive_scores: np.ndarray,
        posterior_scores: np.ndarray,
        observation: object,
        observation_model: ObservationModel,
        update_diagnostics: Mapping[str, object],
    ) -> tuple[float | None, Mapping[str, object]]:
        del predictive_posterior, posterior, posterior_scores, update_diagnostics
        raw = float(-observation_model.log_factor(np.asarray(predictive_scores, dtype=float), observation))
        update_weight = float(getattr(observation, "update_weight", getattr(observation, "metadata", {}).get("update_weight", 1.0)))
        if update_weight == 0.0:
            return None, {"surprise_signal_name": self.name, "raw_surprise": None, "normalised_surprise": None, "missing": True}
        metadata = getattr(observation, "metadata", {})
        info_size = float(max(metadata.get("possible_pair_count", metadata.get("tolerance_group_count", 1)), 1))
        normalized = raw / max(update_weight, 1e-12)
        return normalized, {
            "surprise_signal_name": self.name,
            "raw_surprise": raw,
            "normalised_surprise": normalized,
            "information_size_normalised_surprise": normalized / info_size,
            "effective_strength": update_weight,
            "missing": False,
        }


class PosteriorMahalanobisSurprise:
    name = "posterior_mahalanobis"

    def compute(
        self,
        *,
        predictive_posterior: GaussianPosterior,
        posterior: GaussianPosterior,
        predictive_scores: np.ndarray,
        posterior_scores: np.ndarray,
        observation: object,
        observation_model: ObservationModel,
        update_diagnostics: Mapping[str, object],
    ) -> tuple[float | None, Mapping[str, object]]:
        del predictive_scores, posterior_scores, observation, observation_model, update_diagnostics
        delta = posterior.mean - predictive_posterior.mean
        chol = jittered_cholesky(predictive_posterior.covariance).factor
        whitened = np.linalg.solve(chol, delta)
        value = float(whitened @ whitened)
        return value, {"surprise_signal_name": self.name, "raw_surprise": value, "normalised_surprise": value, "missing": False}


class PosteriorKLSurprise:
    name = "posterior_kl"

    def compute(
        self,
        *,
        predictive_posterior: GaussianPosterior,
        posterior: GaussianPosterior,
        predictive_scores: np.ndarray,
        posterior_scores: np.ndarray,
        observation: object,
        observation_model: ObservationModel,
        update_diagnostics: Mapping[str, object],
    ) -> tuple[float | None, Mapping[str, object]]:
        del predictive_scores, posterior_scores, observation, observation_model, update_diagnostics
        value = gaussian_kl_divergence(posterior.mean, posterior.covariance, predictive_posterior.mean, predictive_posterior.covariance)
        return value, {"surprise_signal_name": self.name, "raw_surprise": value, "normalised_surprise": value, "missing": False}
