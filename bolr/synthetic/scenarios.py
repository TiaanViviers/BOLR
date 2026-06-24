from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDay:
    context_vector: np.ndarray
    theta: np.ndarray
    scores: np.ndarray
    utilities: np.ndarray


@dataclass(frozen=True)
class SyntheticScenario:
    candidate_basis: np.ndarray
    days: tuple[SyntheticDay, ...]
    context_dim: int

