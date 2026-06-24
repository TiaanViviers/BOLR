from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_run_directory(path: str | Path) -> Path:
    run_dir = Path(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
