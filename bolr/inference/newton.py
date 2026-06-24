from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bolr.posterior.diagnostics import jittered_cholesky


ObjectiveFn = Callable[[np.ndarray], float]
GradientFn = Callable[[np.ndarray], np.ndarray]
InformationFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NewtonOptions:
    max_iterations: int = 25
    gradient_tolerance: float = 1e-8
    step_tolerance: float = 1e-8
    initial_damping: float = 0.0
    max_backtracking_steps: int = 12
    line_search_shrinkage: float = 0.5
    initial_jitter: float = 1e-10
    max_jitter_attempts: int = 8


@dataclass(frozen=True)
class NewtonResult:
    point: np.ndarray
    objective_value: float
    converged: bool
    iterations: int
    gradient_norm: float
    step_norm: float
    damping: float
    jitter: float
    fallback_used: bool
    message: str


def damped_newton_solve(
    start: np.ndarray,
    objective_fn: ObjectiveFn,
    gradient_fn: GradientFn,
    information_fn: InformationFn,
    options: NewtonOptions | None = None,
) -> NewtonResult:
    options = options or NewtonOptions()
    point = np.asarray(start, dtype=float).copy()
    objective = float(objective_fn(point))
    damping = options.initial_damping
    eye = np.eye(point.size, dtype=float)
    last_step_norm = 0.0

    for iteration in range(1, options.max_iterations + 1):
        gradient = np.asarray(gradient_fn(point), dtype=float)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= options.gradient_tolerance:
            return NewtonResult(
                point=point,
                objective_value=objective,
                converged=True,
                iterations=iteration - 1,
                gradient_norm=gradient_norm,
                step_norm=last_step_norm,
                damping=damping,
                jitter=0.0,
                fallback_used=False,
                message="Gradient tolerance reached.",
            )

        information = np.asarray(information_fn(point), dtype=float) + damping * eye
        chol = jittered_cholesky(
            information,
            initial_jitter=options.initial_jitter,
            max_attempts=options.max_jitter_attempts,
        )
        step = _cholesky_solve(chol.factor, gradient)
        step_norm = float(np.linalg.norm(step))
        if step_norm <= options.step_tolerance:
            return NewtonResult(
                point=point,
                objective_value=objective,
                converged=True,
                iterations=iteration,
                gradient_norm=gradient_norm,
                step_norm=step_norm,
                damping=damping,
                jitter=chol.jitter,
                fallback_used=False,
                message="Step tolerance reached.",
            )

        accepted = False
        step_scale = 1.0
        candidate_point = point
        candidate_objective = objective
        for _ in range(options.max_backtracking_steps):
            trial_point = point + step_scale * step
            trial_objective = float(objective_fn(trial_point))
            if trial_objective >= objective:
                accepted = True
                candidate_point = trial_point
                candidate_objective = trial_objective
                break
            step_scale *= options.line_search_shrinkage

        if not accepted:
            damping = max(1e-8, 10.0 * (damping if damping > 0.0 else 1.0))
            continue

        point = candidate_point
        objective = candidate_objective
        last_step_norm = step_scale * step_norm

    gradient = np.asarray(gradient_fn(point), dtype=float)
    return NewtonResult(
        point=point,
        objective_value=float(objective_fn(point)),
        converged=False,
        iterations=options.max_iterations,
        gradient_norm=float(np.linalg.norm(gradient)),
        step_norm=last_step_norm,
        damping=damping,
        jitter=0.0,
        fallback_used=True,
        message="Maximum iterations reached without convergence.",
    )


def _cholesky_solve(cholesky_factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(cholesky_factor, rhs)
    return np.linalg.solve(cholesky_factor.T, y)

