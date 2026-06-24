from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.observations.base import ObservationModel
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.config.foundation import StaticSurfaceConfig
from bolr.inference.newton import NewtonOptions, damped_newton_solve
from bolr.model.penalties import QuadraticPenalty


@dataclass(frozen=True)
class StaticSurfaceFit:
    coefficients: np.ndarray
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    regularization: float
    diagnostics: dict[str, float | int | bool | str]


@dataclass(frozen=True)
class CompositeStaticFit:
    coefficients: np.ndarray
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    diagnostics: dict[str, float | int | bool | str]


def fit_static_surface(
    candidate_basis: np.ndarray,
    observations: list[object],
    config: StaticSurfaceConfig | None = None,
    observation_model: ObservationModel | None = None,
    prior_penalty: QuadraticPenalty | None = None,
    prior_mean: np.ndarray | None = None,
) -> StaticSurfaceFit:
    config = config or StaticSurfaceConfig()
    observation_model = observation_model or SoftTargetObservationModel()
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    dimension = candidate_basis.shape[1]
    if prior_penalty is None:
        prior_precision = config.regularization * np.eye(dimension, dtype=float)
        penalty_name = "ridge"
    else:
        if prior_penalty.dimension != dimension:
            raise ValueError("prior_penalty dimension mismatch.")
        prior_precision = prior_penalty.matrix
        penalty_name = prior_penalty.name
    prior_mean = np.zeros(dimension, dtype=float) if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(-1)

    def objective(alpha: np.ndarray) -> float:
        scores = candidate_basis @ alpha
        total = 0.0
        for observation in observations:
            total += observation_model.log_factor(scores, observation)
        delta = alpha - prior_mean
        total -= 0.5 * float(delta @ prior_precision @ delta)
        return total

    def gradient(alpha: np.ndarray) -> np.ndarray:
        scores = candidate_basis @ alpha
        total = np.zeros(dimension, dtype=float)
        for observation in observations:
            total += candidate_basis.T @ observation_model.score_gradient(scores, observation)
        total -= prior_precision @ (alpha - prior_mean)
        return total

    def information(alpha: np.ndarray) -> np.ndarray:
        scores = candidate_basis @ alpha
        total = np.zeros((dimension, dimension), dtype=float)
        for observation in observations:
            total += candidate_basis.T @ observation_model.score_curvature(scores, observation) @ candidate_basis
        total += prior_precision
        return total

    result = damped_newton_solve(
        start=np.zeros(dimension, dtype=float),
        objective_fn=objective,
        gradient_fn=gradient,
        information_fn=information,
        options=NewtonOptions(max_iterations=config.max_iterations, gradient_tolerance=config.gradient_tolerance),
    )
    return StaticSurfaceFit(
        coefficients=result.point,
        objective=result.objective_value,
        gradient_norm=result.gradient_norm,
        iterations=result.iterations,
        converged=result.converged,
        regularization=config.regularization,
        diagnostics={
            "message": result.message,
            "step_norm": result.step_norm,
            "damping": result.damping,
            "penalty_name": penalty_name,
            "penalty_trace": float(np.trace(prior_precision)),
        },
    )


def fit_static_composite(
    model: object,
    batches: list[object],
    observations: list[object],
    *,
    prior_precision: np.ndarray,
    prior_mean: np.ndarray | None = None,
    config: StaticSurfaceConfig | None = None,
    observation_model: ObservationModel | None = None,
) -> CompositeStaticFit:
    from bolr.inference.laplace import laplace_update_composite
    from bolr.model.composite import CompositeScoreModel

    if not isinstance(model, CompositeScoreModel):
        raise TypeError("model must be a CompositeScoreModel.")
    if len(batches) != len(observations):
        raise ValueError("batches and observations must have the same length.")
    config = config or StaticSurfaceConfig()
    observation_model = observation_model or SoftTargetObservationModel()
    prior_precision = np.asarray(prior_precision, dtype=float)
    prior_mean = np.zeros(model.layout.total_dimension, dtype=float) if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(-1)
    if prior_precision.shape != (model.layout.total_dimension, model.layout.total_dimension):
        raise ValueError("prior_precision shape mismatch.")

    def objective(theta: np.ndarray) -> float:
        delta = theta - prior_mean
        total = -0.5 * float(delta @ prior_precision @ delta)
        for batch, observation in zip(batches, observations, strict=True):
            total += float(observation_model.log_factor(model.scores(batch, theta), observation))
        return total

    def gradient(theta: np.ndarray) -> np.ndarray:
        delta = theta - prior_mean
        total = -prior_precision @ delta
        for batch, observation in zip(batches, observations, strict=True):
            total += model.transpose_multiply(batch, observation_model.score_gradient(model.scores(batch, theta), observation))
        return total

    def information(theta: np.ndarray) -> np.ndarray:
        total = prior_precision.copy()
        identity = np.eye(model.layout.total_dimension, dtype=float)
        for batch, observation in zip(batches, observations, strict=True):
            score_curvature = observation_model.score_curvature(model.scores(batch, theta), observation)
            cols = [model.parameter_hvp_from_score_curvature(batch, score_curvature, identity[:, idx]) for idx in range(identity.shape[1])]
            total += np.column_stack(cols)
        return total

    result = damped_newton_solve(
        start=prior_mean.copy(),
        objective_fn=objective,
        gradient_fn=gradient,
        information_fn=information,
        options=NewtonOptions(max_iterations=config.max_iterations, gradient_tolerance=config.gradient_tolerance),
    )
    return CompositeStaticFit(
        coefficients=result.point,
        objective=result.objective_value,
        gradient_norm=result.gradient_norm,
        iterations=result.iterations,
        converged=result.converged,
        diagnostics={
            "message": result.message,
            "step_norm": result.step_norm,
            "damping": result.damping,
            "block_count": len(model.dynamic_blocks),
            "prior_trace": float(np.trace(prior_precision)),
        },
    )
