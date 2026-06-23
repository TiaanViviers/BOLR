from __future__ import annotations

import numpy as np

from bolr.numerics.stable_math import softmax
from bolr.targets.soft_target import Observation


def loss_score_gradient(scores: np.ndarray, observation: Observation) -> np.ndarray:
    q = softmax(scores)
    return q - observation.target_probabilities


def loss_score_observed_information(scores: np.ndarray) -> np.ndarray:
    q = softmax(scores)
    return np.diag(q) - np.outer(q, q)


def parameter_gradient(design: np.ndarray, scores: np.ndarray, observation: Observation) -> np.ndarray:
    return design.T @ loss_score_gradient(scores, observation)


def parameter_hessian(design: np.ndarray, scores: np.ndarray, observation: Observation) -> np.ndarray:
    curvature = loss_score_observed_information(scores)
    return observation.update_weight * (design.T @ curvature @ design)


def parameter_hvp(
    design: np.ndarray,
    scores: np.ndarray,
    vector: np.ndarray,
    observation: Observation,
) -> np.ndarray:
    q = softmax(scores)
    design_vector = design @ np.asarray(vector, dtype=float)
    weighted = q * design_vector
    covariance_vector = weighted - q * np.sum(weighted)
    return observation.update_weight * (design.T @ covariance_vector)
