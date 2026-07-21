"""Robustness metrics for L5.4 Candidate B Thompson audit."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def parse_int_spec(spec: str) -> tuple[int, ...]:
    """Parse '1:30' or '1,2,5' or '1:5,10,20' into a sorted unique tuple."""
    values: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid range {part}: end < start.")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError(f"Empty integer specification: {spec!r}")
    return tuple(sorted(values))


def max_drawdown(series: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_start": -1,
            "max_drawdown_end": -1,
            "max_drawdown_duration_days": 0,
            "longest_underwater_period": 0,
            "final_cumulative": 0.0,
        }
    cum = np.cumsum(values)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    end = int(np.argmax(dd))
    start = int(np.argmax(cum[: end + 1])) if end >= 0 else 0
    underwater = 0
    longest = 0
    for i in range(len(cum)):
        if cum[i] < peak[i]:
            underwater += 1
            longest = max(longest, underwater)
        else:
            underwater = 0
    return {
        "max_drawdown": float(dd[end]),
        "max_drawdown_start": start,
        "max_drawdown_end": end,
        "max_drawdown_duration_days": int(end - start),
        "longest_underwater_period": int(longest),
        "final_cumulative": float(cum[-1]),
    }


def daily_delta_summary(deltas: Sequence[float]) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=float)
    if d.size == 0:
        return {
            "mean_daily_delta": float("nan"),
            "median_daily_delta": float("nan"),
            "std_daily_delta": float("nan"),
            "total_delta": 0.0,
            "positive_delta_day_rate": float("nan"),
            "positive_delta_days": 0,
            "negative_delta_days": 0,
            "zero_delta_days": 0,
            "worst_daily_delta": float("nan"),
            "best_daily_delta": float("nan"),
        }
    pos = d > 0
    neg = d < 0
    zero = d == 0
    dd = max_drawdown(d)
    return {
        "mean_daily_delta": float(d.mean()),
        "median_daily_delta": float(np.median(d)),
        "std_daily_delta": float(d.std(ddof=0)),
        "total_delta": float(d.sum()),
        "positive_delta_day_rate": float(pos.mean()),
        "positive_delta_days": int(pos.sum()),
        "negative_delta_days": int(neg.sum()),
        "zero_delta_days": int(zero.sum()),
        "worst_daily_delta": float(d.min()),
        "best_daily_delta": float(d.max()),
        "max_drawdown_of_cumulative_delta": dd["max_drawdown"],
        "max_drawdown_start": dd["max_drawdown_start"],
        "max_drawdown_end": dd["max_drawdown_end"],
        "max_drawdown_duration_days": dd["max_drawdown_duration_days"],
        "longest_underwater_period": dd["longest_underwater_period"],
        "final_cumulative_delta": dd["final_cumulative"],
    }


def block_bootstrap_totals(
    daily_deltas: Sequence[float],
    *,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    d = np.asarray(daily_deltas, dtype=float)
    n = int(d.size)
    if n == 0:
        raise ValueError("daily_deltas must be non-empty.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    rng = np.random.default_rng(int(seed))
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(0, max(n - block_size + 1, 1))
    samples = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        pieces: list[np.ndarray] = []
        for s in chosen:
            pieces.append(d[int(s) : int(s) + block_size])
        path = np.concatenate(pieces)[:n]
        samples[b] = float(path.sum())
    mean_daily = samples / float(n)
    return {
        "block_size": int(block_size),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "observed_total_delta": float(d.sum()),
        "observed_mean_daily_delta": float(d.mean()),
        "mean_total_delta": float(samples.mean()),
        "median_total_delta": float(np.median(samples)),
        "ci_025_total_delta": float(np.quantile(samples, 0.025)),
        "ci_975_total_delta": float(np.quantile(samples, 0.975)),
        "prob_total_delta_gt_0": float(np.mean(samples > 0.0)),
        "ci_025_mean_daily_delta": float(np.quantile(mean_daily, 0.025)),
        "ci_975_mean_daily_delta": float(np.quantile(mean_daily, 0.975)),
        "samples": samples,
    }


def cost_adjusted_totals(
    *,
    strategy_pnl: Sequence[float],
    candidate_41_pnl: Sequence[float],
    selected: Sequence[int],
    cost_per_day: float = 0.0,
    cost_per_non_41_switch: float = 0.0,
    cost_per_turnover: float = 0.0,
    baseline_index: int = 41,
) -> dict[str, float]:
    pnl = np.asarray(strategy_pnl, dtype=float)
    c41 = np.asarray(candidate_41_pnl, dtype=float)
    sel = np.asarray(selected, dtype=int)
    if pnl.shape != c41.shape or pnl.shape != sel.shape:
        raise ValueError("cost arrays must share shape.")
    n = int(pnl.size)
    non_41 = sel != int(baseline_index)
    if n <= 1:
        turnover = 0
    else:
        turnover = int(np.sum(sel[1:] != sel[:-1]))
    strategy_cost = (
        float(cost_per_day) * n
        + float(cost_per_non_41_switch) * float(non_41.sum())
        + float(cost_per_turnover) * float(turnover)
    )
    # Always-41 trades every day but never switches away from 41 and has zero turnover.
    baseline_cost = float(cost_per_day) * n
    raw_total = float(pnl.sum())
    baseline_raw = float(c41.sum())
    return {
        "cost_per_day": float(cost_per_day),
        "cost_per_non_41_switch": float(cost_per_non_41_switch),
        "cost_per_turnover": float(cost_per_turnover),
        "raw_total_pnl": raw_total,
        "candidate_41_raw_total_pnl": baseline_raw,
        "strategy_total_cost": strategy_cost,
        "candidate_41_total_cost": baseline_cost,
        "cost_adjusted_total_pnl": raw_total - strategy_cost,
        "candidate_41_cost_adjusted_total_pnl": baseline_raw - baseline_cost,
        "cost_adjusted_delta_vs_41": (raw_total - strategy_cost) - (baseline_raw - baseline_cost),
        "non_41_days": float(non_41.sum()),
        "turnover_count": float(turnover),
    }


def break_even_costs(
    *,
    raw_total_pnl: float,
    candidate_41_total_pnl: float,
    day_count: int,
    non_41_days: int,
    turnover_count: int,
) -> dict[str, float]:
    edge = float(raw_total_pnl) - float(candidate_41_total_pnl)
    # Flat daily costs cancel vs always-41; break-even on differential costs only.
    be_day = float("inf") if day_count <= 0 else float("nan")  # cancels; not meaningful alone
    be_non41 = float(edge / non_41_days) if non_41_days > 0 else float("inf")
    be_turn = float(edge / turnover_count) if turnover_count > 0 else float("inf")
    return {
        "raw_edge_vs_41": edge,
        "break_even_cost_per_day_vs_41": be_day,
        "break_even_cost_per_non_41_switch_vs_41": be_non41,
        "break_even_cost_per_turnover_vs_41": be_turn,
    }


def assign_split_labels(dates: Sequence[str]) -> list[dict[str, Any]]:
    """Return named contiguous splits covering the replay dates."""
    frame = pd.DataFrame({"date": list(dates)})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["day_index"] = np.arange(len(frame))
    splits: list[dict[str, Any]] = []

    def add(name: str, mask: pd.Series) -> None:
        idx = frame.loc[mask, "day_index"].to_numpy(dtype=int)
        if idx.size == 0:
            return
        splits.append(
            {
                "split_name": name,
                "split_start": str(frame.loc[mask, "date"].iloc[0].date()),
                "split_end": str(frame.loc[mask, "date"].iloc[-1].date()),
                "day_indices": idx.tolist(),
            }
        )

    add("full", pd.Series(True, index=frame.index))
    add("2023", frame["date"].dt.year == 2023)
    add("2024", frame["date"].dt.year == 2024)
    mid = len(frame) // 2
    add("H1", frame["day_index"] < mid)
    add("H2", frame["day_index"] >= mid)
    for year in (2023, 2024):
        for q in (1, 2, 3, 4):
            mask = (frame["date"].dt.year == year) & (frame["date"].dt.quarter == q)
            label = f"Q{q}_{year}" if not (year == 2024 and q == 4) else f"Q{q}_{year}_partial"
            add(label, mask)
    window = 100
    step = 25
    for start in range(0, max(len(frame) - window + 1, 1), step):
        end = start + window
        if end > len(frame):
            break
        add(f"rolling_100_start_{start}", (frame["day_index"] >= start) & (frame["day_index"] < end))
    return splits


def summarize_split(
    daily: pd.DataFrame,
    *,
    day_indices: Sequence[int],
    split_name: str,
    split_start: str,
    split_end: str,
    candidate_41_pnl: Sequence[float],
    baseline_index: int = 41,
) -> dict[str, Any]:
    from bolr.evaluation.policy_matrix_metrics import bad_switch_diagnostics, selection_entropy

    idx = list(day_indices)
    part = daily.iloc[idx]
    selected = part["selected_candidate_index"].astype(int).tolist()
    pnl = part["realised_selected_value"].astype(float).to_numpy()
    c41 = np.asarray(candidate_41_pnl, dtype=float)[idx]
    deltas = pnl - c41
    switches = bad_switch_diagnostics(selected, pnl.tolist(), c41.tolist(), baseline_index=baseline_index)
    _, n_eff = selection_entropy(selected)
    dd = max_drawdown(deltas)
    return {
        "split_name": split_name,
        "split_start": split_start,
        "split_end": split_end,
        "day_count": int(len(idx)),
        "total_pnl": float(pnl.sum()),
        "candidate_41_total_pnl": float(c41.sum()),
        "delta_vs_41_total": float(deltas.sum()),
        "mean_delta_vs_41": float(deltas.mean()) if len(idx) else float("nan"),
        "positive_delta_day_rate": float(np.mean(deltas > 0)) if len(idx) else float("nan"),
        "unique_selected_candidates": int(len(set(selected))),
        "effective_selected_candidates": float(n_eff),
        "bad_switch_count": switches["bad_switch_count"],
        "bad_switch_total_cost": switches["bad_switch_total_cost"],
        "good_switch_count": switches["good_switch_count"],
        "good_switch_total_gain": switches["good_switch_total_gain"],
        "max_drawdown_delta": dd["max_drawdown"],
    }
