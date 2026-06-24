from __future__ import annotations

import numpy as np

from bolr.model.penalties import QuadraticPenalty
from bolr.posterior.diagnostics import jittered_cholesky


def gaussian_kl_divergence(
    mean_p: np.ndarray,
    cov_p: np.ndarray,
    mean_q: np.ndarray,
    cov_q: np.ndarray,
) -> float:
    mean_p = np.asarray(mean_p, dtype=float)
    mean_q = np.asarray(mean_q, dtype=float)
    cov_p = np.asarray(cov_p, dtype=float)
    cov_q = np.asarray(cov_q, dtype=float)
    chol_q = jittered_cholesky(cov_q).factor
    solve_q = np.linalg.solve(chol_q, cov_p)
    trace_term = float(np.trace(np.linalg.solve(chol_q.T, solve_q)))
    delta = mean_q - mean_p
    whitened = np.linalg.solve(chol_q, delta)
    mahal = float(whitened @ whitened)
    logdet_q = 2.0 * float(np.sum(np.log(np.diag(chol_q))))
    chol_p = jittered_cholesky(cov_p).factor
    logdet_p = 2.0 * float(np.sum(np.log(np.diag(chol_p))))
    logdet_ratio = float(logdet_q - logdet_p)
    p = mean_p.size
    return 0.5 * (trace_term + mahal - p + logdet_ratio)


def innovation_diagnostics(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    log_factor_prior: float,
    log_factor_posterior: float,
) -> dict[str, float]:
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_cov = np.asarray(prior_cov, dtype=float)
    posterior_mean = np.asarray(posterior_mean, dtype=float)
    posterior_cov = np.asarray(posterior_cov, dtype=float)
    delta = posterior_mean - prior_mean
    chol = jittered_cholesky(prior_cov).factor
    whitened = np.linalg.solve(chol, delta)
    mahal = float(whitened @ whitened)
    return {
        "state_update_l2": float(np.linalg.norm(delta)),
        "state_update_mahalanobis": mahal,
        "log_factor_prior": float(log_factor_prior),
        "log_factor_posterior": float(log_factor_posterior),
        "objective_improvement": float(log_factor_posterior - log_factor_prior),
        "gaussian_kl": gaussian_kl_divergence(posterior_mean, posterior_cov, prior_mean, prior_cov),
        "prior_trace": float(np.trace(prior_cov)),
        "posterior_trace": float(np.trace(posterior_cov)),
        "variance_trace_change": float(np.trace(posterior_cov) - np.trace(prior_cov)),
    }


def block_innovation_diagnostics(
    prior_mean: np.ndarray,
    prior_cov_block: np.ndarray,
    posterior_mean: np.ndarray,
) -> dict[str, float]:
    delta = np.asarray(posterior_mean, dtype=float) - np.asarray(prior_mean, dtype=float)
    prior_cov_block = np.asarray(prior_cov_block, dtype=float)
    chol = jittered_cholesky(prior_cov_block).factor
    whitened = np.linalg.solve(chol, delta)
    return {
        "update_l2": float(np.linalg.norm(delta)),
        "update_mahalanobis": float(whitened @ whitened),
        "variance_trace": float(np.trace(prior_cov_block)),
    }


def roughness_diagnostics(
    state: np.ndarray,
    penalty: QuadraticPenalty,
    *,
    prior_mean: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
) -> dict[str, float]:
    state = np.asarray(state, dtype=float).reshape(-1)
    prior_mean = np.zeros_like(state) if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(-1)
    if prior_mean.size != state.size:
        raise ValueError("prior_mean dimension mismatch.")
    precision = penalty.matrix if prior_precision is None else np.asarray(prior_precision, dtype=float)
    if precision.shape != (state.size, state.size):
        raise ValueError("prior_precision dimension mismatch.")
    centered = state - prior_mean
    return {
        "roughness": float(state @ penalty.matrix @ state),
        "ridge_energy": float(state @ state),
        "prior_standardized_norm": float(centered @ precision @ centered),
    }
