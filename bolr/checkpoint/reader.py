from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bolr.checkpoint.state import HistoricalReplayCheckpoint


def read_checkpoint(path: str | Path) -> HistoricalReplayCheckpoint:
    payload = json.loads(Path(path).read_text())
    return HistoricalReplayCheckpoint(
        schema_version=payload["schema_version"],
        run_id=payload["run_id"],
        configuration=payload["configuration"],
        last_completed_date=payload["last_completed_date"],
        static_alpha=np.asarray(payload["static_alpha"], dtype=float),
        posterior_mean=np.asarray(payload["posterior_mean"], dtype=float),
        posterior_covariance=np.asarray(payload["posterior_covariance"], dtype=float),
        output_row_counts=payload["output_row_counts"],
        graph_metadata=payload.get("graph_metadata"),
        transition_policy_family=payload.get("transition_policy_family"),
        transition_policy_config_hash=payload.get("transition_policy_config_hash"),
        transition_policy_state=payload.get("transition_policy_state"),
        surprise_standardizer_state=payload.get("surprise_standardizer_state"),
        bocpd_state=payload.get("bocpd_state"),
        block_multipliers=payload.get("block_multipliers"),
        block_discounts=payload.get("block_discounts"),
        pending_resets=payload.get("pending_resets"),
        last_surprise_diagnostics=payload.get("last_surprise_diagnostics"),
        adaptive_schema_version=payload.get("adaptive_schema_version"),
        decision_policy_family=payload.get("decision_policy_family"),
        decision_policy_config=payload.get("decision_policy_config"),
        decision_policy_config_hash=payload.get("decision_policy_config_hash"),
        posterior_sample_count=payload.get("posterior_sample_count"),
        sampling_seed_state=payload.get("sampling_seed_state"),
        region_definition=payload.get("region_definition"),
        decision_schema_version=payload.get("decision_schema_version"),
    )
