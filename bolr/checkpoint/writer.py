from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bolr.checkpoint.state import HistoricalReplayCheckpoint


def write_checkpoint_atomic(checkpoint: HistoricalReplayCheckpoint, path: str | Path) -> None:
    path = Path(path)
    payload = asdict(checkpoint)
    payload["static_alpha"] = checkpoint.static_alpha.tolist()
    payload["posterior_mean"] = checkpoint.posterior_mean.tolist()
    payload["posterior_covariance"] = checkpoint.posterior_covariance.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)
