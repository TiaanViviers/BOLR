from __future__ import annotations

import numpy as np

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
