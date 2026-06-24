from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np


class ObservationModel(Protocol):
    def log_factor(self, scores: np.ndarray, observation: object) -> float:
        ...

    def score_gradient(self, scores: np.ndarray, observation: object) -> np.ndarray:
        ...

    def score_curvature(self, scores: np.ndarray, observation: object) -> np.ndarray:
        ...

    def score_curvature_hvp(self, scores: np.ndarray, vector: np.ndarray, observation: object) -> np.ndarray:
        ...

    def diagnostics(self, scores: np.ndarray, observation: object) -> Mapping[str, object]:
        ...
