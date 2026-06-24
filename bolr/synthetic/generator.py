from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.representation.score_design import DailyDesign, theta_from_matrix
from bolr.synthetic.scenarios import SyntheticDay, SyntheticScenario
from bolr.synthetic.surfaces import drifting_interaction_sequence, stationary_interaction_matrix


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    n_days: int
    candidate_basis: np.ndarray
    context_dim: int
    observation_noise: float = 0.05
    seed: int = 0


class SyntheticContextualSurfaceGenerator:
    def __init__(self, config: SyntheticGeneratorConfig) -> None:
        self.config = config
        self.candidate_basis = np.asarray(config.candidate_basis, dtype=float)
        self.rng = np.random.default_rng(config.seed)

    def stationary_scenario(self) -> SyntheticScenario:
        interaction = stationary_interaction_matrix(
            self.candidate_basis.shape[1],
            self.config.context_dim,
            seed=self.config.seed,
        )
        thetas = np.repeat(theta_from_matrix(interaction)[None, :], self.config.n_days, axis=0)
        contexts = self._context_matrix()
        return self._build_scenario(thetas=thetas, contexts=contexts)

    def drifting_scenario(self) -> SyntheticScenario:
        interactions = drifting_interaction_sequence(
            self.candidate_basis.shape[1],
            self.config.context_dim,
            n_steps=self.config.n_days,
            seed=self.config.seed,
        )
        thetas = np.stack([theta_from_matrix(interaction) for interaction in interactions], axis=0)
        contexts = self._context_matrix()
        return self._build_scenario(thetas=thetas, contexts=contexts)

    def _context_matrix(self) -> np.ndarray:
        raw = self.rng.normal(size=(self.config.n_days, self.config.context_dim))
        raw[:, 0] = 1.0
        return raw

    def _build_scenario(self, thetas: np.ndarray, contexts: np.ndarray) -> SyntheticScenario:
        days: list[SyntheticDay] = []
        for theta, context in zip(thetas, contexts, strict=True):
            design = DailyDesign(candidate_basis=self.candidate_basis, context_vector=context)
            scores = design.scores(theta)
            utilities = scores + self.rng.normal(scale=self.config.observation_noise, size=scores.shape)
            days.append(
                SyntheticDay(
                    context_vector=context,
                    theta=theta,
                    scores=scores,
                    utilities=utilities,
                )
            )
        return SyntheticScenario(
            candidate_basis=self.candidate_basis,
            days=tuple(days),
            context_dim=self.config.context_dim,
        )
