from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HistoricalBaselines:
    warm_up_global_best_config_id: int
    static_surface_best_config_id: int


def compute_reference_baselines(
    warm_up_frame: pd.DataFrame,
    static_scores: np.ndarray,
) -> HistoricalBaselines:
    warm_up_means = warm_up_frame.groupby("config_id", sort=True)["pnl"].mean()
    warm_up_global_best = int(warm_up_means.idxmax())
    static_surface_best = int(np.argmax(static_scores))
    return HistoricalBaselines(
        warm_up_global_best_config_id=warm_up_global_best,
        static_surface_best_config_id=static_surface_best,
    )
