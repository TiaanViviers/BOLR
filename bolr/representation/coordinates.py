from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.config.foundation import CoordinateTransformConfig


@dataclass(frozen=True)
class CoordinateState:
    entry_mean: float
    entry_scale: float
    stop_mean: float
    stop_scale: float


class LogCoordinateTransform:
    def __init__(self, config: CoordinateTransformConfig | None = None) -> None:
        self.config = config or CoordinateTransformConfig()
        self.state: CoordinateState | None = None

    def fit(self, entry_values: np.ndarray, stop_values: np.ndarray) -> "LogCoordinateTransform":
        entry_logs = np.log(np.asarray(entry_values, dtype=float))
        stop_logs = np.log(np.asarray(stop_values, dtype=float))
        entry_scale = float(entry_logs.std())
        stop_scale = float(stop_logs.std())
        self.state = CoordinateState(
            entry_mean=float(entry_logs.mean()),
            entry_scale=max(entry_scale, self.config.eps),
            stop_mean=float(stop_logs.mean()),
            stop_scale=max(stop_scale, self.config.eps),
        )
        return self

    def transform(self, entry_values: np.ndarray, stop_values: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("LogCoordinateTransform must be fitted before transform().")
        entry_logs = np.log(np.asarray(entry_values, dtype=float))
        stop_logs = np.log(np.asarray(stop_values, dtype=float))
        entry = (entry_logs - self.state.entry_mean) / self.state.entry_scale
        stop = (stop_logs - self.state.stop_mean) / self.state.stop_scale
        return np.column_stack([entry, stop])
