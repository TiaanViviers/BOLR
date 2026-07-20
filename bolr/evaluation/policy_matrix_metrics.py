"""Cross-strategy comparison metrics for L5.2 Candidate A policy matrix."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def selection_entropy(selected: Sequence[int]) -> tuple[float, float]:
    counts = Counter(int(x) for x in selected)
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    probs = np.asarray([c / total for c in counts.values()], dtype=float)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-300, 1.0))).sum())
    return entropy, float(math.exp(entropy))


def turnover_stats(selected: Sequence[int]) -> tuple[int, float]:
    selected = [int(x) for x in selected]
    if len(selected) <= 1:
        return 0, 0.0
    changes = sum(1 for a, b in zip(selected, selected[1:]) if a != b)
    return int(changes), float(changes / (len(selected) - 1))


def bad_switch_diagnostics(
    selected: Sequence[int],
    selected_pnl: Sequence[float],
    candidate_41_pnl: Sequence[float],
    *,
    baseline_index: int = 41,
) -> dict[str, float | int]:
    selected = np.asarray(selected, dtype=int)
    selected_pnl = np.asarray(selected_pnl, dtype=float)
    candidate_41_pnl = np.asarray(candidate_41_pnl, dtype=float)
    if selected.shape != selected_pnl.shape or selected.shape != candidate_41_pnl.shape:
        raise ValueError("bad-switch arrays must share shape.")
    mask = selected != int(baseline_index)
    days_not = int(mask.sum())
    if days_not == 0:
        return {
            "days_not_candidate_41": 0,
            "non_41_mean_pnl": float("nan"),
            "candidate_41_pnl_on_non_41_days": float("nan"),
            "delta_vs_41_on_non_41_days": 0.0,
            "bad_switch_count": 0,
            "bad_switch_total_cost": 0.0,
            "good_switch_count": 0,
            "good_switch_total_gain": 0.0,
        }
    deltas = selected_pnl[mask] - candidate_41_pnl[mask]
    bad = deltas < 0.0
    good = deltas > 0.0
    return {
        "days_not_candidate_41": days_not,
        "non_41_mean_pnl": float(selected_pnl[mask].mean()),
        "candidate_41_pnl_on_non_41_days": float(candidate_41_pnl[mask].mean()),
        "delta_vs_41_on_non_41_days": float(deltas.sum()),
        "bad_switch_count": int(bad.sum()),
        "bad_switch_total_cost": float((-deltas[bad]).sum()) if bad.any() else 0.0,
        "good_switch_count": int(good.sum()),
        "good_switch_total_gain": float(deltas[good].sum()) if good.any() else 0.0,
    }


def probability_best_bins(daily: pd.DataFrame) -> list[dict[str, Any]]:
    if "selected_probability_best" not in daily.columns:
        return []
    p = pd.to_numeric(daily["selected_probability_best"], errors="coerce")
    valid = p.notna()
    if not valid.any():
        return []
    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
    labels = ["0.00-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", "0.30+"]
    frame = daily.loc[valid].copy()
    frame["_pbest"] = p.loc[valid]
    frame["_bin"] = pd.cut(frame["_pbest"], bins=edges, labels=labels, right=False, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for label in labels:
        part = frame[frame["_bin"] == label]
        if part.empty:
            rows.append(
                {
                    "bin": label,
                    "count": 0,
                    "mean_selected_probability_best": float("nan"),
                    "empirical_top_1_hit_rate": float("nan"),
                    "empirical_top_10_hit_rate": float("nan"),
                    "mean_realised_rank": float("nan"),
                    "mean_pnl": float("nan"),
                }
            )
            continue
        ranks = part["realised_rank"].to_numpy(dtype=float)
        rows.append(
            {
                "bin": label,
                "count": int(len(part)),
                "mean_selected_probability_best": float(part["_pbest"].mean()),
                "empirical_top_1_hit_rate": float(np.mean(ranks <= 1.0)),
                "empirical_top_10_hit_rate": float(np.mean(ranks <= 10.0)),
                "mean_realised_rank": float(np.mean(ranks)),
                "mean_pnl": float(part["realised_selected_value"].mean()),
            }
        )
    return rows


def summarize_strategy_daily(
    daily: pd.DataFrame,
    *,
    strategy_name: str,
    strategy_family: str,
    transition_family: str,
    decision_policy: str,
    always_41_total_pnl: float,
    always_41_mean_pnl: float,
    candidate_41_pnl: Sequence[float] | None = None,
    laplace_failure_count: int | None = None,
    checkpoint_count: int | None = None,
    total_elapsed_seconds: float | None = None,
    mean_day_ms: float | None = None,
) -> dict[str, Any]:
    selected = daily["selected_candidate_index"].astype(int).tolist()
    pnl = daily["realised_selected_value"].astype(float).to_numpy()
    regret = daily["regret"].astype(float).to_numpy()
    ranks = daily["realised_rank"].astype(float).to_numpy()
    entropy, n_eff = selection_entropy(selected)
    turnover_count, turnover_rate = turnover_stats(selected)
    counts = Counter(selected)
    most_selected, most_count = counts.most_common(1)[0]
    total_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean())
    c41_days = int(sum(1 for x in selected if x == 41))
    summary: dict[str, Any] = {
        "strategy_name": strategy_name,
        "strategy_family": strategy_family,
        "transition_family": transition_family,
        "decision_policy": decision_policy,
        "day_count": int(len(daily)),
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "median_pnl": float(np.median(pnl)),
        "std_pnl": float(np.std(pnl, ddof=0)),
        "min_daily_pnl": float(pnl.min()),
        "max_daily_pnl": float(pnl.max()),
        "positive_day_rate": float(np.mean(pnl > 0.0)),
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
        "total_regret": float(np.sum(regret)),
        "mean_realised_rank": float(np.mean(ranks)),
        "median_realised_rank": float(np.median(ranks)),
        "top_1_hit_rate": float(np.mean(ranks <= 1.0)),
        "top_5_hit_rate": float(np.mean(ranks <= 5.0)),
        "top_10_hit_rate": float(np.mean(ranks <= 10.0)),
        "unique_selected_candidates": int(len(counts)),
        "most_selected_candidate": int(most_selected),
        "most_selected_candidate_share": float(most_count / len(selected)),
        "selection_entropy": entropy,
        "effective_selected_candidates": n_eff,
        "turnover_count": turnover_count,
        "turnover_rate": turnover_rate,
        "candidate_41_selected_days": c41_days,
        "candidate_41_selected_share": float(c41_days / len(selected)),
        "candidate_41_delta_total_pnl": total_pnl - float(always_41_total_pnl),
        "candidate_41_delta_mean_pnl": mean_pnl - float(always_41_mean_pnl),
        "laplace_failure_count": int(laplace_failure_count) if laplace_failure_count is not None else (
            int((~daily["laplace_converged"].astype(bool)).sum()) if "laplace_converged" in daily.columns else ""
        ),
        "checkpoint_count": checkpoint_count if checkpoint_count is not None else "",
        "total_elapsed_seconds": total_elapsed_seconds if total_elapsed_seconds is not None else "",
        "mean_day_ms": mean_day_ms if mean_day_ms is not None else (
            float(daily["elapsed_day_ms"].mean()) if "elapsed_day_ms" in daily.columns else ""
        ),
    }
    if candidate_41_pnl is not None:
        summary.update(
            bad_switch_diagnostics(
                selected,
                pnl.tolist(),
                list(candidate_41_pnl),
            )
        )
    return summary


def normalize_native_daily(
    daily: pd.DataFrame,
    *,
    strategy_name: str,
    strategy_family: str,
    grid_config_ids: np.ndarray,
) -> pd.DataFrame:
    frame = daily.copy()
    frame["strategy_name"] = strategy_name
    frame["strategy_family"] = strategy_family
    if "selected_config_id" not in frame.columns:
        frame["selected_config_id"] = frame["selected_candidate_index"].map(lambda i: int(grid_config_ids[int(i)]))
    return frame
