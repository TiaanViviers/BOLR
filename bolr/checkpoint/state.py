from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistoricalReplayCheckpoint:
    schema_version: str
    run_id: str
    configuration: dict
    last_completed_date: str
    static_alpha: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    output_row_counts: dict[str, int]
    graph_metadata: dict | None = None
    transition_policy_family: str | None = None
    transition_policy_config_hash: str | None = None
    transition_policy_state: dict | None = None
    surprise_standardizer_state: dict | None = None
    bocpd_state: dict | None = None
    block_multipliers: dict | None = None
    block_discounts: dict | None = None
    pending_resets: dict | None = None
    last_surprise_diagnostics: dict | None = None
    adaptive_schema_version: str | None = None
    decision_policy_family: str | None = None
    decision_policy_config: dict | None = None
    decision_policy_config_hash: str | None = None
    posterior_sample_count: int | None = None
    sampling_seed_state: dict | None = None
    region_definition: dict | None = None
    decision_schema_version: str | None = None
