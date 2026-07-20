"""Canonical candidate realised-value extraction for L5 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bolr.data.candidate_grid import CandidateGrid
from bolr.data.historical_dataset import HistoricalDataset


@dataclass(frozen=True)
class CandidateIdentity:
    candidate_index: int
    config_id: int
    entry_percentage: float
    sl_trail_percentage: float
    entry_index: int
    stop_index: int


def candidate_identity(grid: CandidateGrid, candidate_index: int) -> CandidateIdentity:
    idx = int(candidate_index)
    if idx < 0 or idx >= grid.n_candidates:
        raise ValueError(f"candidate_index out of range: {idx}")
    entries = np.unique(grid.entry_values)
    stops = np.unique(grid.stop_values)
    entry = float(grid.entry_values[idx])
    stop = float(grid.stop_values[idx])
    config_id = int(grid.config_ids[idx])
    if config_id != idx:
        raise ValueError(f"Canonical config_id mismatch: index={idx}, config_id={config_id}")
    return CandidateIdentity(
        candidate_index=idx,
        config_id=config_id,
        entry_percentage=entry,
        sl_trail_percentage=stop,
        entry_index=int(np.where(entries == entry)[0][0]),
        stop_index=int(np.where(stops == stop)[0][0]),
    )


def build_pnl_matrix(
    dataset: HistoricalDataset,
    dates: Sequence[str],
) -> np.ndarray:
    """Return shape (n_dates, n_candidates) realised PnL in canonical order."""
    rows = [dataset.day_frame(date)["pnl"].to_numpy(dtype=float) for date in dates]
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("PnL matrix must be 2-D.")
    if matrix.shape[1] != dataset.candidate_grid.n_candidates:
        raise ValueError("PnL matrix candidate dimension mismatch.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PnL matrix contains non-finite values.")
    return matrix


def get_candidate_replay_values(
    dataset: HistoricalDataset,
    *,
    candidate_index: int,
    replay_dates: Sequence[str],
) -> np.ndarray:
    """Extract one candidate's realised PnL over replay dates in order."""
    identity = candidate_identity(dataset.candidate_grid, candidate_index)
    values = np.empty(len(replay_dates), dtype=float)
    for i, date in enumerate(replay_dates):
        day = dataset.day_frame(date)
        config_ids = day["config_id"].to_numpy(dtype=int)
        if not np.array_equal(config_ids, dataset.candidate_grid.config_ids):
            raise RuntimeError(f"Non-canonical candidate order on {date}.")
        row = day.iloc[identity.candidate_index]
        if int(row["config_id"]) != identity.config_id:
            raise RuntimeError(f"config_id mismatch on {date} for candidate {candidate_index}.")
        if not np.isclose(float(row["entry_percentage"]), identity.entry_percentage):
            raise RuntimeError(f"entry_percentage mismatch on {date} for candidate {candidate_index}.")
        if not np.isclose(float(row["sl_trail_percentage"]), identity.sl_trail_percentage):
            raise RuntimeError(f"sl_trail_percentage mismatch on {date} for candidate {candidate_index}.")
        values[i] = float(row["pnl"])
    return values


def argmax_with_lowest_index(scores: np.ndarray) -> int:
    """Argmax with lowest-index tie-break (NumPy first-max semantics)."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot argmax an empty score vector.")
    return int(np.argmax(scores))
