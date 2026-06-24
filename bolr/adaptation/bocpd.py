from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Mapping

import numpy as np
from scipy.special import gammaln, logsumexp

from bolr.config.foundation import BOCPDConfig


@dataclass(frozen=True)
class BOCPDState:
    log_run_length_posterior: np.ndarray
    mu: np.ndarray
    kappa: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    step_index: int


class BOCPDDetector:
    def __init__(self, config: BOCPDConfig | None = None) -> None:
        self.config = config or BOCPDConfig()

    def initial_state(self) -> BOCPDState:
        size = self.config.max_run_length + 1
        log_probs = np.full(size, -np.inf, dtype=float)
        log_probs[0] = 0.0
        mu = np.full(size, self.config.prior_mean, dtype=float)
        kappa = np.full(size, self.config.prior_kappa, dtype=float)
        alpha = np.full(size, self.config.prior_alpha, dtype=float)
        beta = np.full(size, self.config.prior_beta, dtype=float)
        return BOCPDState(log_probs, mu, kappa, alpha, beta, 0)

    def step(self, value: float | None, state: BOCPDState) -> tuple[BOCPDState, dict[str, object]]:
        if value is None:
            if self.config.missing_policy == "hold":
                return state, self._diagnostics(state, None, 0.0, float(np.exp(state.log_run_length_posterior[0]))) | {"missing_policy": "hold"}
            return self._hazard_only_step(state)

        current_max = min(state.step_index, self.config.max_run_length)
        log_pred = np.array(
            [_student_t_log_pdf(value, state.mu[r], state.kappa[r], state.alpha[r], state.beta[r]) for r in range(current_max + 1)],
            dtype=float,
        )
        hazard = self.config.hazard
        new_log = np.full(self.config.max_run_length + 1, -np.inf, dtype=float)
        cp_terms = state.log_run_length_posterior[: current_max + 1] + log(hazard) + log_pred
        new_log[0] = logsumexp(cp_terms)
        truncation_terms: list[float] = []
        for r in range(current_max + 1):
            target = r + 1
            term = state.log_run_length_posterior[r] + log(1.0 - hazard) + log_pred[r]
            if target <= self.config.max_run_length:
                new_log[target] = term
            else:
                truncation_terms.append(term)
        normalizer = logsumexp(new_log[np.isfinite(new_log)])
        new_log -= normalizer
        truncation_mass = 0.0 if not truncation_terms else float(np.exp(logsumexp(np.asarray(truncation_terms, dtype=float)) - normalizer))

        size = self.config.max_run_length + 1
        new_mu = np.full(size, self.config.prior_mean, dtype=float)
        new_kappa = np.full(size, self.config.prior_kappa, dtype=float)
        new_alpha = np.full(size, self.config.prior_alpha, dtype=float)
        new_beta = np.full(size, self.config.prior_beta, dtype=float)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = _posterior_update(
            value, self.config.prior_mean, self.config.prior_kappa, self.config.prior_alpha, self.config.prior_beta
        )
        for r in range(current_max + 1):
            target = r + 1
            if target > self.config.max_run_length:
                continue
            new_mu[target], new_kappa[target], new_alpha[target], new_beta[target] = _posterior_update(
                value, state.mu[r], state.kappa[r], state.alpha[r], state.beta[r]
            )
        new_state = BOCPDState(new_log, new_mu, new_kappa, new_alpha, new_beta, state.step_index + 1)
        predictive_log_density = float(logsumexp(state.log_run_length_posterior[: current_max + 1] + log_pred))
        return new_state, self._diagnostics(new_state, predictive_log_density, truncation_mass, float(np.exp(new_log[0]))) | {"missing_policy": "observed"}

    def _hazard_only_step(self, state: BOCPDState) -> tuple[BOCPDState, dict[str, object]]:
        current_max = min(state.step_index, self.config.max_run_length)
        hazard = self.config.hazard
        new_log = np.full(self.config.max_run_length + 1, -np.inf, dtype=float)
        new_log[0] = logsumexp(state.log_run_length_posterior[: current_max + 1] + log(hazard))
        truncation_terms: list[float] = []
        for r in range(current_max + 1):
            target = r + 1
            term = state.log_run_length_posterior[r] + log(1.0 - hazard)
            if target <= self.config.max_run_length:
                new_log[target] = term
            else:
                truncation_terms.append(term)
        normalizer = logsumexp(new_log[np.isfinite(new_log)])
        new_log -= normalizer
        truncation_mass = 0.0 if not truncation_terms else float(np.exp(logsumexp(np.asarray(truncation_terms, dtype=float)) - normalizer))
        new_mu = np.full(self.config.max_run_length + 1, self.config.prior_mean, dtype=float)
        new_kappa = np.full(self.config.max_run_length + 1, self.config.prior_kappa, dtype=float)
        new_alpha = np.full(self.config.max_run_length + 1, self.config.prior_alpha, dtype=float)
        new_beta = np.full(self.config.max_run_length + 1, self.config.prior_beta, dtype=float)
        for r in range(current_max + 1):
            target = r + 1
            if target > self.config.max_run_length:
                continue
            new_mu[target] = state.mu[r]
            new_kappa[target] = state.kappa[r]
            new_alpha[target] = state.alpha[r]
            new_beta[target] = state.beta[r]
        new_state = BOCPDState(new_log, new_mu, new_kappa, new_alpha, new_beta, state.step_index + 1)
        return new_state, self._diagnostics(new_state, None, truncation_mass, float(np.exp(new_log[0]))) | {"missing_policy": "hazard_only"}

    def _diagnostics(self, state: BOCPDState, predictive_log_density: float | None, truncation_mass: float, change_probability: float) -> dict[str, object]:
        probs = np.exp(state.log_run_length_posterior)
        probs /= probs.sum()
        idx = np.arange(probs.size, dtype=float)
        positive = probs > 0.0
        return {
            "change_probability": float(change_probability),
            "run_length_posterior": probs,
            "map_run_length": int(np.argmax(probs)),
            "expected_run_length": float(np.sum(idx * probs)),
            "run_length_entropy": float(-np.sum(probs[positive] * np.log(probs[positive]))),
            "predictive_log_density": predictive_log_density,
            "hazard": self.config.hazard,
            "truncation_mass": float(truncation_mass),
        }

    def metadata(self) -> Mapping[str, object]:
        return {"family": "bocpd_gaussian_nig", "config": self.config}


def _posterior_update(x: float, mu: float, kappa: float, alpha: float, beta: float) -> tuple[float, float, float, float]:
    kappa_n = kappa + 1.0
    mu_n = (kappa * mu + x) / kappa_n
    alpha_n = alpha + 0.5
    beta_n = beta + 0.5 * kappa * (x - mu) ** 2 / kappa_n
    return float(mu_n), float(kappa_n), float(alpha_n), float(beta_n)


def _student_t_log_pdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    y = (x - mu) ** 2 / scale2
    return float(
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (log(nu) + log(pi) + log(scale2))
        - 0.5 * (nu + 1.0) * log(1.0 + y / nu)
    )
