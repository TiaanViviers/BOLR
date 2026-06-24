from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.model.composite import CompositeScoreModel
from bolr.observations.base import ObservationModel
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.inference.newton import NewtonOptions, NewtonResult, damped_newton_solve
from bolr.numerics.derivatives import (
    parameter_gradient,
    parameter_hessian,
    structured_parameter_gradient,
    structured_parameter_hessian,
)
from bolr.observations.soft_target_gibbs import log_factor
from bolr.posterior.diagnostics import jittered_cholesky
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign
from bolr.targets.soft_target import Observation


@dataclass(frozen=True)
class LaplaceUpdateResult:
    posterior: GaussianPosterior
    mode: np.ndarray
    newton_result: NewtonResult
    used_structured_curvature: bool
    log_posterior: float


@dataclass(frozen=True)
class CompositeLaplaceUpdateResult:
    posterior: GaussianPosterior
    mode: np.ndarray
    newton_result: NewtonResult
    log_posterior: float


def laplace_update(
    prior: GaussianPosterior,
    design: DailyDesign,
    observation: object,
    options: NewtonOptions | None = None,
    use_structured_curvature: bool = True,
    observation_model: ObservationModel | None = None,
) -> LaplaceUpdateResult:
    observation_model = observation_model or SoftTargetObservationModel()
    prior_precision = np.linalg.inv(prior.covariance)
    prior_mean = prior.mean

    def objective(theta: np.ndarray) -> float:
        delta = theta - prior_mean
        scores = design.scores(theta)
        prior_term = -0.5 * delta @ prior_precision @ delta
        return float(prior_term + observation_model.log_factor(scores, observation))

    def gradient(theta: np.ndarray) -> np.ndarray:
        delta = theta - prior_mean
        scores = design.scores(theta)
        if use_structured_curvature:
            observation_gradient = structured_gradient_of_log_factor(design, scores, observation, observation_model)
        else:
            observation_gradient = explicit_gradient_of_log_factor(design, scores, observation, observation_model)
        return -prior_precision @ delta + observation_gradient

    def information(theta: np.ndarray) -> np.ndarray:
        scores = design.scores(theta)
        if use_structured_curvature:
            observation_information = structured_observed_information_of_log_factor(design, scores, observation, observation_model)
        else:
            observation_information = explicit_observed_information_of_log_factor(design, scores, observation, observation_model)
        return prior_precision + observation_information

    newton_result = damped_newton_solve(
        start=prior_mean,
        objective_fn=objective,
        gradient_fn=gradient,
        information_fn=information,
        options=options,
    )
    if not newton_result.converged:
        posterior = prior.with_diagnostics(
            laplace_update_status="fallback_to_prior",
            newton_message=newton_result.message,
        )
        return LaplaceUpdateResult(
            posterior=posterior,
            mode=prior.mean.copy(),
            newton_result=newton_result,
            used_structured_curvature=use_structured_curvature,
            log_posterior=objective(prior.mean),
        )

    posterior_information = information(newton_result.point)
    chol = jittered_cholesky(posterior_information)
    identity = np.eye(prior.dimension, dtype=float)
    inverse = np.linalg.solve(chol.factor.T, np.linalg.solve(chol.factor, identity))
    posterior = GaussianPosterior(
        mean=newton_result.point,
        covariance=inverse,
        state_layout=prior.state_layout,
        timestamp=prior.timestamp,
        version=prior.version,
        diagnostics={
            **prior.diagnostics,
            "laplace_update_status": "updated",
            "newton_iterations": newton_result.iterations,
            "newton_gradient_norm": newton_result.gradient_norm,
            "newton_step_norm": newton_result.step_norm,
            "newton_damping": newton_result.damping,
            "posterior_precision_jitter": chol.jitter,
            "used_structured_curvature": use_structured_curvature,
        },
    )
    return LaplaceUpdateResult(
        posterior=posterior,
        mode=newton_result.point,
        newton_result=newton_result,
        used_structured_curvature=use_structured_curvature,
        log_posterior=objective(newton_result.point),
    )


def explicit_gradient_of_log_factor(
    design: DailyDesign,
    scores: np.ndarray,
    observation: object,
    observation_model: ObservationModel,
) -> np.ndarray:
    return design.explicit_matrix().T @ observation_model.score_gradient(scores, observation)


def explicit_observed_information_of_log_factor(
    design: DailyDesign,
    scores: np.ndarray,
    observation: object,
    observation_model: ObservationModel,
) -> np.ndarray:
    return design.explicit_matrix().T @ observation_model.score_curvature(scores, observation) @ design.explicit_matrix()


def structured_gradient_of_log_factor(
    design: DailyDesign,
    scores: np.ndarray,
    observation: object,
    observation_model: ObservationModel,
) -> np.ndarray:
    score_gradient = observation_model.score_gradient(scores, observation)
    candidate_component = design.candidate_basis.T @ score_gradient
    return np.kron(design.context_vector, candidate_component)


def structured_observed_information_of_log_factor(
    design: DailyDesign,
    scores: np.ndarray,
    observation: object,
    observation_model: ObservationModel,
) -> np.ndarray:
    candidate_information = design.candidate_basis.T @ observation_model.score_curvature(scores, observation) @ design.candidate_basis
    return np.kron(np.outer(design.context_vector, design.context_vector), candidate_information)


def laplace_update_composite(
    prior: GaussianPosterior,
    model: CompositeScoreModel,
    batch: object,
    observation: object,
    observation_model: ObservationModel | None = None,
    options: NewtonOptions | None = None,
) -> CompositeLaplaceUpdateResult:
    observation_model = observation_model or SoftTargetObservationModel()
    prior_precision = np.linalg.inv(prior.covariance)
    prior_mean = prior.mean

    def objective(theta: np.ndarray) -> float:
        delta = theta - prior_mean
        scores = model.scores(batch, theta)
        return float(-0.5 * delta @ prior_precision @ delta + observation_model.log_factor(scores, observation))

    def gradient(theta: np.ndarray) -> np.ndarray:
        delta = theta - prior_mean
        scores = model.scores(batch, theta)
        return -prior_precision @ delta + model.transpose_multiply(batch, observation_model.score_gradient(scores, observation))

    def information(theta: np.ndarray) -> np.ndarray:
        scores = model.scores(batch, theta)
        score_curvature = observation_model.score_curvature(scores, observation)
        identity = np.eye(prior.dimension, dtype=float)
        cols = []
        for idx in range(prior.dimension):
            cols.append(model.parameter_hvp_from_score_curvature(batch, score_curvature, identity[:, idx]))
        return prior_precision + np.column_stack(cols)

    newton_result = damped_newton_solve(
        start=prior_mean,
        objective_fn=objective,
        gradient_fn=gradient,
        information_fn=information,
        options=options,
    )
    if not newton_result.converged:
        posterior = prior.with_diagnostics(laplace_update_status="fallback_to_prior", newton_message=newton_result.message)
        return CompositeLaplaceUpdateResult(
            posterior=posterior,
            mode=prior.mean.copy(),
            newton_result=newton_result,
            log_posterior=objective(prior.mean),
        )
    posterior_information = information(newton_result.point)
    chol = jittered_cholesky(posterior_information)
    identity = np.eye(prior.dimension, dtype=float)
    inverse = np.linalg.solve(chol.factor.T, np.linalg.solve(chol.factor, identity))
    posterior = GaussianPosterior(
        mean=newton_result.point,
        covariance=inverse,
        state_layout=prior.state_layout,
        timestamp=prior.timestamp,
        version=prior.version,
        diagnostics={
            **prior.diagnostics,
            "laplace_update_status": "updated",
            "newton_iterations": newton_result.iterations,
            "newton_gradient_norm": newton_result.gradient_norm,
            "newton_step_norm": newton_result.step_norm,
            "newton_damping": newton_result.damping,
            "posterior_precision_jitter": chol.jitter,
        },
    )
    return CompositeLaplaceUpdateResult(
        posterior=posterior,
        mode=newton_result.point,
        newton_result=newton_result,
        log_posterior=objective(newton_result.point),
    )
