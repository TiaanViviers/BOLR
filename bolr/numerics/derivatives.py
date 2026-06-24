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


def structured_parameter_gradient(
    candidate_basis: np.ndarray,
    context_vector: np.ndarray,
    scores: np.ndarray,
    observation: Observation,
) -> np.ndarray:
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    context_vector = np.asarray(context_vector, dtype=float)
    candidate_component = candidate_basis.T @ loss_score_gradient(scores, observation)
    return np.kron(context_vector, candidate_component)


def structured_candidate_observed_information(
    candidate_basis: np.ndarray,
    scores: np.ndarray,
    observation: Observation,
) -> np.ndarray:
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    curvature = loss_score_observed_information(scores)
    return observation.update_weight * (candidate_basis.T @ curvature @ candidate_basis)


def structured_parameter_hessian(
    candidate_basis: np.ndarray,
    context_vector: np.ndarray,
    scores: np.ndarray,
    observation: Observation,
) -> np.ndarray:
    candidate_information = structured_candidate_observed_information(candidate_basis, scores, observation)
    context_outer = np.outer(np.asarray(context_vector, dtype=float), np.asarray(context_vector, dtype=float))
    return np.kron(context_outer, candidate_information)


def structured_parameter_hvp(
    candidate_basis: np.ndarray,
    context_vector: np.ndarray,
    scores: np.ndarray,
    vector: np.ndarray,
    observation: Observation,
) -> np.ndarray:
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    context_vector = np.asarray(context_vector, dtype=float)
    vector = np.asarray(vector, dtype=float)
    candidate_dim = candidate_basis.shape[1]
    context_dim = context_vector.size
    vector_matrix = vector.reshape((candidate_dim, context_dim), order="F")
    projected = candidate_basis @ vector_matrix @ context_vector
    weighted = softmax(scores) * projected
    covariance_vector = weighted - softmax(scores) * np.sum(weighted)
    candidate_term = candidate_basis.T @ covariance_vector
    return observation.update_weight * np.kron(context_vector, candidate_term)
