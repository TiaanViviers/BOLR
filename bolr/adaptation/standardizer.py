from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from bolr.config.foundation import SurpriseStandardizerConfig


@dataclass(frozen=True)
class EWStandardizerState:
    count: int
    mean: float
    variance: float
    last_z: float | None


class EWStandardizer:
    def __init__(self, config: SurpriseStandardizerConfig | None = None) -> None:
        self.config = config or SurpriseStandardizerConfig()

    def initial_state(self) -> EWStandardizerState:
        return EWStandardizerState(count=0, mean=0.0, variance=self.config.variance_floor, last_z=None)

    def step(self, value: float | None, state: EWStandardizerState) -> tuple[EWStandardizerState, dict[str, float | int | bool | None]]:
        if value is None:
            return state, {
                "value": None,
                "z_score": None,
                "mean_before": state.mean,
                "scale_before": float(np.sqrt(max(state.variance, self.config.variance_floor))),
                "missing": True,
            }
        mean_before = state.mean
        scale_before = float(np.sqrt(max(state.variance, self.config.variance_floor)))
        z = 0.0 if state.count < self.config.warmup_count else (value - mean_before) / scale_before
        if self.config.clip_z is not None:
            z = float(np.clip(z, -self.config.clip_z, self.config.clip_z))
        alpha = self.config.decay
        delta = value - state.mean
        mean_after = (1.0 - alpha) * state.mean + alpha * value
        variance_after = (1.0 - alpha) * (state.variance + alpha * delta * delta)
        variance_after = float(max(variance_after, self.config.variance_floor))
        new_state = EWStandardizerState(
            count=state.count + 1,
            mean=float(mean_after),
            variance=variance_after,
            last_z=float(z),
        )
        return new_state, {
            "value": float(value),
            "z_score": float(z),
            "mean_before": float(mean_before),
            "scale_before": float(scale_before),
            "missing": False,
        }

    def metadata(self) -> Mapping[str, object]:
        return {"family": "ew_standardizer", "config": self.config}
