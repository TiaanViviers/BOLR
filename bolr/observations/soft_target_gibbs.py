from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.observations.base import ObservationModel
from bolr.numerics.stable_math import cross_entropy, log_softmax, softmax
from bolr.targets.soft_target import Observation


def log_factor(scores: np.ndarray, observation: Observation) -> float:
    if observation.update_weight == 0.0:
        return 0.0
    loss = cross_entropy(observation.target_probabilities, log_softmax(scores))
    return -observation.update_weight * loss


def score_gradient(scores: np.ndarray, observation: Observation) -> np.ndarray:
    if observation.update_weight == 0.0:
        return np.zeros_like(scores, dtype=float)
    q = softmax(scores)
    return observation.update_weight * (observation.target_probabilities - q)


def score_observed_information(scores: np.ndarray, observation: Observation) -> np.ndarray:
    if observation.update_weight == 0.0:
        n = np.asarray(scores).shape[0]
        return np.zeros((n, n), dtype=float)
    q = softmax(scores)
    covariance = np.diag(q) - np.outer(q, q)
    return observation.update_weight * covariance


def score_hessian(scores: np.ndarray, observation: Observation) -> np.ndarray:
    return -score_observed_information(scores, observation)


def score_hvp(scores: np.ndarray, vector: np.ndarray, observation: Observation) -> np.ndarray:
    if observation.update_weight == 0.0:
        return np.zeros_like(vector, dtype=float)
    q = softmax(scores)
    vector = np.asarray(vector, dtype=float)
    x = q * vector
    covariance_vector = x - q * np.sum(x)
    return -observation.update_weight * covariance_vector


@dataclass(frozen=True)
class SoftTargetObservationModel(ObservationModel):
    def log_factor(self, scores: np.ndarray, observation: Observation) -> float:
        return log_factor(scores, observation)

    def score_gradient(self, scores: np.ndarray, observation: Observation) -> np.ndarray:
        return score_gradient(scores, observation)

    def score_curvature(self, scores: np.ndarray, observation: Observation) -> np.ndarray:
        return score_observed_information(scores, observation)

    def score_curvature_hvp(self, scores: np.ndarray, vector: np.ndarray, observation: Observation) -> np.ndarray:
        return -score_hvp(scores, vector, observation)

    def diagnostics(self, scores: np.ndarray, observation: Observation) -> dict[str, object]:
        curvature = self.score_curvature(scores, observation)
        return {
            "observation_family": "candidate_a_soft_target",
            "log_factor_at_scores": self.log_factor(scores, observation),
            "gradient_norm_at_scores": float(np.linalg.norm(self.score_gradient(scores, observation))),
            "curvature_trace_or_estimate": float(np.trace(curvature)),
        }
