"""Phase L5.4 Candidate B fixed Thompson robustness audit."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bolr.config.foundation import (
    CandidateGridConfig,
    CrossGroupLogisticConfig,
    OrderedPartitionConfig,
    OrderedPartitionToleranceConfig,
    SoftTargetConfig,
    StaticSurfaceConfig,
)
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.candidate_b_policy_matrix import load_l5_2_comparison
from bolr.evaluation.candidate_replay_values import build_pnl_matrix
from bolr.evaluation.native_candidate_a_replay import NativeCandidateAReplayConfig, build_candidate_basis, resolve_date_windows
from bolr.evaluation.native_candidate_b_replay import (
    DEFAULT_SAMPLED_PAIR_BUDGET,
    NativeCandidateBReplayConfig,
    run_native_candidate_b_replay,
    warmup_candidate_b_state,
)
from bolr.evaluation.outputs import ensure_run_directory
from bolr.evaluation.policy_matrix_metrics import bad_switch_diagnostics, selection_entropy, turnover_stats
from bolr.evaluation.robustness_metrics import (
    assign_split_labels,
    block_bootstrap_totals,
    break_even_costs,
    cost_adjusted_totals,
    daily_delta_summary,
    max_drawdown,
    parse_int_spec,
    summarize_split,
)

COST_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
BOOTSTRAP_BLOCK_SIZES: tuple[int, ...] = (5, 10, 20)
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260721


@dataclass(frozen=True)
class ThompsonRobustnessConfig:
    data_path: str = "data/YM_full.parquet"
    grid_csv: str = "data/YM_grid.csv"
    output_dir: str = "outputs/l5_candidate_b_thompson_robustness"
    l5_2_comparison_dir: str | None = "outputs/l5_candidate_a_policy_matrix/comparison"
    l5_3_comparison_dir: str | None = "outputs/l5_candidate_b_native_replay/comparison"
    warmup_start: str | None = "2021-01-29"
    warmup_end: str | None = "2023-01-11"
    replay_start: str | None = "2023-01-12"
    replay_end: str | None = "2024-10-08"
    warm_up_days: int = 504
    maximum_days: int | None = None
    candidate_count: int = 1428
    rng_seed: int = 20260720
    rng_streams: tuple[int, ...] = tuple(range(1, 31))
    pair_budget: int = DEFAULT_SAMPLED_PAIR_BUDGET
    pair_sampling_seed: int = 0
    pair_budgets: tuple[int, ...] = (2048, 4096, 8192)
    pair_sampling_seeds: tuple[int, ...] = (0, 1, 2)
    monte_carlo_sample_count: int = 512
    monte_carlo_chunk_size: int = 64
    top_k: int = 10
    antithetic: bool = True
    checkpoint_every_n_days: int = 25
    overwrite_outputs: bool = False
    resume_existing: bool = False
    dry_run: bool = False
    mode: str = "seed_robustness"  # seed_robustness|pair_sampling|cost_analysis|split_analysis|analyse_existing|all
    relative_tolerance: float = 0.1
    normalize_pair_losses: bool = True
    sigma0: float = 1.0
    random_walk_variance: float = 0.05
    command_line: tuple[str, ...] = ()
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        allowed = {
            "seed_robustness",
            "pair_sampling",
            "cost_analysis",
            "split_analysis",
            "analyse_existing",
            "all",
        }
        if self.mode not in allowed:
            raise ValueError(f"Unsupported mode: {self.mode}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        temp = path.with_suffix(path.suffix + ".tmp")
        rows.to_csv(temp, index=False)
        temp.replace(path)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp.replace(path)


def run_name_for(*, stream: int, pair_budget: int, pair_seed: int, maximum_days: int | None) -> str:
    suffix = f"_d{maximum_days}" if maximum_days is not None else ""
    return f"cb_fixed_thompson_s{stream}_pb{pair_budget}_ps{pair_seed}{suffix}"


def load_baseline_series(
    *,
    project_root: Path,
    config: ThompsonRobustnessConfig,
    replay_dates: Sequence[str],
    dataset: HistoricalDataset,
) -> dict[str, Any]:
    replay_pnl = build_pnl_matrix(dataset, replay_dates)
    candidate_41_pnl = replay_pnl[:, 41].astype(float)
    always_41_total = float(candidate_41_pnl.sum())
    always_41_mean = float(candidate_41_pnl.mean())

    a_pnl = None
    a_total = None
    a_mean = None
    oracle_pnl = None
    oracle_total = None
    b_pbest_note = "not imported"

    l52_summary, l52_daily, l52_note = load_l5_2_comparison(
        Path(config.l5_2_comparison_dir) if config.l5_2_comparison_dir else None
    )
    if l52_daily is not None and not l52_daily.empty:
        a_frame = l52_daily.loc[l52_daily["strategy_name"] == "candidate_a_fixed_probability_best"].sort_values("day_index")
        if not a_frame.empty:
            a_pnl = a_frame["realised_selected_value"].astype(float).to_numpy()
            if len(a_pnl) != len(replay_dates):
                a_pnl = a_pnl[: len(replay_dates)]
            a_total = float(a_pnl.sum())
            a_mean = float(a_pnl.mean())
        o_frame = l52_daily.loc[l52_daily["strategy_name"] == "oracle_static_best_replay"].sort_values("day_index")
        if not o_frame.empty:
            oracle_pnl = o_frame["realised_selected_value"].astype(float).to_numpy()[: len(replay_dates)]
            oracle_total = float(oracle_pnl.sum())
        row_41 = l52_summary.loc[l52_summary["strategy_name"] == "baseline_always_41"] if l52_summary is not None else None
        if row_41 is not None and not row_41.empty:
            always_41_total = float(row_41.iloc[0]["total_pnl"])
            always_41_mean = float(row_41.iloc[0]["mean_pnl"])

    l53_dir = Path(config.l5_3_comparison_dir) if config.l5_3_comparison_dir else None
    if l53_dir is not None and not l53_dir.is_absolute():
        l53_dir = project_root / l53_dir
    if l53_dir is not None and (l53_dir / "strategy_summary.csv").exists():
        s = pd.read_csv(l53_dir / "strategy_summary.csv")
        row = s.loc[s["strategy_name"] == "candidate_b_sampled_fixed_probability_best"]
        if not row.empty:
            b_pbest_note = f"L5.3 total_pnl={float(row.iloc[0]['total_pnl']):.4f} (collapsed to 41)"

    return {
        "candidate_41_pnl": candidate_41_pnl,
        "always_41_total": always_41_total,
        "always_41_mean": always_41_mean,
        "candidate_a_pnl": a_pnl,
        "candidate_a_total": a_total,
        "candidate_a_mean": a_mean,
        "oracle_pnl": oracle_pnl,
        "oracle_total": oracle_total,
        "l5_2_note": l52_note,
        "b_pbest_note": b_pbest_note,
    }


def build_daily_delta_rows(
    daily: pd.DataFrame,
    *,
    run_id: str,
    rng_seed: int,
    rng_stream: int,
    pair_budget: int,
    pair_sampling_seed: int,
    candidate_41_pnl: np.ndarray,
    candidate_a_pnl: np.ndarray | None,
    oracle_pnl: np.ndarray | None,
) -> list[dict[str, Any]]:
    selected = daily["selected_candidate_index"].astype(int).to_numpy()
    pnl = daily["realised_selected_value"].astype(float).to_numpy()
    n = len(daily)
    c41 = np.asarray(candidate_41_pnl[:n], dtype=float)
    a = np.asarray(candidate_a_pnl[:n], dtype=float) if candidate_a_pnl is not None else np.full(n, np.nan)
    o = np.asarray(oracle_pnl[:n], dtype=float) if oracle_pnl is not None else np.full(n, np.nan)
    rows: list[dict[str, Any]] = []
    prev = None
    for i in range(n):
        delta_41 = float(pnl[i] - c41[i])
        is_switch = bool(prev is not None and selected[i] != prev)
        is_non_41 = bool(selected[i] != 41)
        rows.append(
            {
                "run_id": run_id,
                "rng_seed": rng_seed,
                "rng_stream": rng_stream,
                "pair_budget": pair_budget,
                "pair_sampling_seed": pair_sampling_seed,
                "day_index": int(daily.iloc[i]["day_index"]) if "day_index" in daily.columns else i,
                "date": str(daily.iloc[i]["date"]),
                "selected_candidate_index": int(selected[i]),
                "selected_config_id": int(daily.iloc[i]["selected_config_id"]) if "selected_config_id" in daily.columns else "",
                "selected_entry_index": daily.iloc[i].get("selected_entry_index", ""),
                "selected_stop_index": daily.iloc[i].get("selected_stop_index", ""),
                "strategy_pnl": float(pnl[i]),
                "candidate_41_pnl": float(c41[i]),
                "candidate_a_fixed_probability_best_pnl": float(a[i]) if np.isfinite(a[i]) else "",
                "oracle_static_best_replay_pnl": float(o[i]) if np.isfinite(o[i]) else "",
                "delta_vs_41": delta_41,
                "delta_vs_candidate_a": float(pnl[i] - a[i]) if np.isfinite(a[i]) else "",
                "delta_vs_oracle": float(pnl[i] - o[i]) if np.isfinite(o[i]) else "",
                "selected_is_41": is_non_41 is False,
                "is_switch_from_previous_day": is_switch,
                "is_non_41_switch": is_non_41,
                "good_switch_vs_41": bool(is_non_41 and delta_41 > 0),
                "bad_switch_vs_41": bool(is_non_41 and delta_41 < 0),
                "realised_rank": daily.iloc[i].get("realised_rank", ""),
                "regret": daily.iloc[i].get("regret", ""),
                "selected_probability_best": daily.iloc[i].get("selected_probability_best", ""),
                "selected_expected_rank": daily.iloc[i].get("selected_expected_rank", ""),
            }
        )
        prev = int(selected[i])
    return rows


def registry_row_from_result(
    *,
    run_id: str,
    run_name: str,
    status: str,
    mode: str,
    output_path: Path,
    config: ThompsonRobustnessConfig,
    stream: int,
    pair_budget: int,
    pair_seed: int,
    daily: pd.DataFrame | None,
    summary: dict[str, Any] | None,
    baselines: dict[str, Any],
    error_message: str = "",
) -> dict[str, Any]:
    always_41_total = float(baselines["always_41_total"])
    always_41_mean = float(baselines["always_41_mean"])
    a_total = baselines.get("candidate_a_total")
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "status": status,
        "mode": mode,
        "output_path": str(output_path),
        "observation_model": "candidate_b",
        "candidate_b_variant": "sampled",
        "transition_family": "fixed",
        "decision_policy": "thompson",
        "rng_seed": config.rng_seed,
        "rng_stream": stream,
        "pair_budget": pair_budget,
        "pair_sampling_seed": pair_seed,
        "normalize_pair_losses": config.normalize_pair_losses,
        "relative_tolerance": config.relative_tolerance,
        "mc_samples": config.monte_carlo_sample_count,
        "mc_chunk_size": config.monte_carlo_chunk_size,
        "top_k": config.top_k,
        "warmup_start": config.warmup_start,
        "warmup_end": config.warmup_end,
        "replay_start": config.replay_start,
        "replay_end": config.replay_end,
        "maximum_days": config.maximum_days if config.maximum_days is not None else "",
        "day_count": "",
        "total_pnl": "",
        "mean_pnl": "",
        "delta_vs_41_total": "",
        "delta_vs_41_mean": "",
        "delta_vs_candidate_a_total": "",
        "unique_selected_candidates": "",
        "effective_selected_candidates": "",
        "most_selected_candidate": "",
        "most_selected_candidate_share": "",
        "turnover_rate": "",
        "net_switch_value_vs_41": "",
        "bad_switch_count_vs_41": "",
        "bad_switch_total_cost_vs_41": "",
        "good_switch_count_vs_41": "",
        "good_switch_total_gain_vs_41": "",
        "checkpoint_count": "",
        "laplace_failure_count": "",
        "total_elapsed_seconds": "",
        "error_message": error_message,
    }
    if daily is None or summary is None or status != "completed":
        return row
    selected = daily["selected_candidate_index"].astype(int).tolist()
    pnl = daily["realised_selected_value"].astype(float).to_numpy()
    c41 = np.asarray(baselines["candidate_41_pnl"][: len(pnl)], dtype=float)
    entropy, n_eff = selection_entropy(selected)
    to_c, to_r = turnover_stats(selected)
    switches = bad_switch_diagnostics(selected, pnl.tolist(), c41.tolist())
    counts = pd.Series(selected).value_counts()
    total_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean())
    row.update(
        {
            "day_count": int(len(daily)),
            "total_pnl": total_pnl,
            "mean_pnl": mean_pnl,
            "delta_vs_41_total": total_pnl - always_41_total * (len(pnl) / max(len(baselines["candidate_41_pnl"]), 1))
            if config.maximum_days is not None
            else total_pnl - always_41_total,
            "delta_vs_41_mean": mean_pnl - float(c41.mean()),
            "delta_vs_candidate_a_total": (total_pnl - float(a_total)) if a_total is not None and config.maximum_days is None else "",
            "unique_selected_candidates": int(counts.size),
            "effective_selected_candidates": float(n_eff),
            "most_selected_candidate": int(counts.index[0]),
            "most_selected_candidate_share": float(counts.iloc[0] / len(selected)),
            "turnover_rate": float(to_r),
            "net_switch_value_vs_41": float(switches["good_switch_total_gain"]) - float(switches["bad_switch_total_cost"]),
            "bad_switch_count_vs_41": switches["bad_switch_count"],
            "bad_switch_total_cost_vs_41": switches["bad_switch_total_cost"],
            "good_switch_count_vs_41": switches["good_switch_count"],
            "good_switch_total_gain_vs_41": switches["good_switch_total_gain"],
            "checkpoint_count": summary.get("checkpoint_count", ""),
            "laplace_failure_count": summary.get("laplace_failure_count", ""),
            "total_elapsed_seconds": summary.get("total_elapsed_seconds", ""),
            "selection_entropy": entropy,
            "turnover_count": to_c,
        }
    )
    # Prefer exact same-length delta when maximum_days truncates.
    if config.maximum_days is not None:
        row["delta_vs_41_total"] = float(pnl.sum() - c41.sum())
        if baselines.get("candidate_a_pnl") is not None:
            a = np.asarray(baselines["candidate_a_pnl"][: len(pnl)], dtype=float)
            row["delta_vs_candidate_a_total"] = float(pnl.sum() - a.sum())
    return row


def analyse_runs(
    *,
    registry: pd.DataFrame,
    daily_panel: pd.DataFrame,
    baselines: dict[str, Any],
    output_root: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    completed = registry.loc[registry["status"] == "completed"].copy()
    comparison = output_root / "seed_robustness" / "comparison"
    ensure_run_directory(comparison)
    stats_dir = output_root / "statistical_tests"
    cost_dir = output_root / "cost_analysis"
    split_dir = output_root / "split_analysis"
    for d in (stats_dir, cost_dir, split_dir):
        ensure_run_directory(d)

    # Seed robustness summary
    if completed.empty:
        seed_summary = {
            "run_count": int(len(registry)),
            "completed_run_count": 0,
            "failed_run_count": int((registry["status"] == "failed").sum()),
        }
        _write_csv(comparison / "seed_robustness_summary.csv", [seed_summary])
        return {"seed_summary": seed_summary}

    totals = completed["total_pnl"].astype(float)
    deltas = completed["delta_vs_41_total"].astype(float)
    a_deltas = pd.to_numeric(completed.get("delta_vs_candidate_a_total"), errors="coerce")
    seed_summary = {
        "run_count": int(len(registry)),
        "completed_run_count": int(len(completed)),
        "failed_run_count": int((registry["status"] == "failed").sum()),
        "mean_total_pnl": float(totals.mean()),
        "median_total_pnl": float(totals.median()),
        "std_total_pnl": float(totals.std(ddof=0)),
        "min_total_pnl": float(totals.min()),
        "max_total_pnl": float(totals.max()),
        "p05_total_pnl": float(totals.quantile(0.05)),
        "p25_total_pnl": float(totals.quantile(0.25)),
        "p75_total_pnl": float(totals.quantile(0.75)),
        "p95_total_pnl": float(totals.quantile(0.95)),
        "mean_delta_vs_41": float(deltas.mean()),
        "median_delta_vs_41": float(deltas.median()),
        "p05_delta_vs_41": float(deltas.quantile(0.05)),
        "p95_delta_vs_41": float(deltas.quantile(0.95)),
        "share_beating_41": float((deltas > 0).mean()),
        "share_beating_candidate_a": float((a_deltas > 0).mean()) if a_deltas.notna().any() else "",
        "share_positive_total_pnl": float((totals > 0).mean()),
        "mean_effective_selected_candidates": float(completed["effective_selected_candidates"].astype(float).mean()),
        "median_effective_selected_candidates": float(completed["effective_selected_candidates"].astype(float).median()),
        "mean_turnover_rate": float(completed["turnover_rate"].astype(float).mean()),
        "mean_bad_switch_cost": float(completed["bad_switch_total_cost_vs_41"].astype(float).mean()),
        "mean_good_switch_gain": float(completed["good_switch_total_gain_vs_41"].astype(float).mean()),
        "mean_net_switch_value": float(completed["net_switch_value_vs_41"].astype(float).mean()),
        "best_stream": int(completed.loc[deltas.idxmax(), "rng_stream"]),
        "worst_stream": int(completed.loc[deltas.idxmin(), "rng_stream"]),
        "best_delta_vs_41": float(deltas.max()),
        "worst_delta_vs_41": float(deltas.min()),
        "always_41_total_pnl": float(baselines["always_41_total"]),
        "candidate_a_fixed_probability_best_total_pnl": baselines.get("candidate_a_total") or "",
    }
    _write_csv(comparison / "seed_robustness_summary.csv", [seed_summary])
    _atomic_write_text(comparison / "seed_robustness_summary.json", json.dumps(seed_summary, indent=2, default=str))

    # Selection / bad-switch robustness
    sel_rows = []
    bad_rows = []
    split_rows = []
    cost_rows = []
    per_run_delta_rows = []
    bootstrap_rows = []
    bootstrap_sample_rows = []

    for _, reg in completed.iterrows():
        run_id = str(reg["run_id"])
        part = daily_panel.loc[daily_panel["run_id"] == run_id].sort_values("day_index")
        if part.empty:
            continue
        selected = part["selected_candidate_index"].astype(int).tolist()
        pnl = part["strategy_pnl"].astype(float).tolist()
        c41 = part["candidate_41_pnl"].astype(float).tolist()
        deltas_arr = part["delta_vs_41"].astype(float).to_numpy()
        entropy, n_eff = selection_entropy(selected)
        to_c, to_r = turnover_stats(selected)
        counts = pd.Series(selected).value_counts()
        top5 = counts.head(5)
        switches = bad_switch_diagnostics(selected, pnl, c41)
        neutral = int(((pd.Series(selected) != 41) & (pd.Series(pnl) - pd.Series(c41) == 0)).sum())
        non_41 = int((pd.Series(selected) != 41).sum())
        sel_rows.append(
            {
                "run_id": run_id,
                "rng_stream": int(reg["rng_stream"]),
                "unique_selected_candidates": int(counts.size),
                "effective_selected_candidates": float(n_eff),
                "selection_entropy": float(entropy),
                "most_selected_candidate": int(counts.index[0]),
                "most_selected_candidate_share": float(counts.iloc[0] / len(selected)),
                "candidate_41_selected_days": int((pd.Series(selected) == 41).sum()),
                "candidate_41_selected_share": float((pd.Series(selected) == 41).mean()),
                "top_5_selected_candidates": ",".join(str(int(i)) for i in top5.index.tolist()),
                "top_5_selected_candidate_shares": ",".join(f"{float(v / len(selected)):.6f}" for v in top5.tolist()),
                "turnover_count": to_c,
                "turnover_rate": to_r,
                "non_41_days": non_41,
                "non_41_day_share": float(non_41 / len(selected)),
            }
        )
        good_gains = part.loc[part["good_switch_vs_41"].astype(bool), "delta_vs_41"].astype(float)
        bad_costs = (-part.loc[part["bad_switch_vs_41"].astype(bool), "delta_vs_41"].astype(float))
        bad_rows.append(
            {
                "run_id": run_id,
                "rng_stream": int(reg["rng_stream"]),
                "days_not_candidate_41": switches["days_not_candidate_41"],
                "good_switch_count": switches["good_switch_count"],
                "bad_switch_count": switches["bad_switch_count"],
                "neutral_switch_count": neutral,
                "good_switch_total_gain": switches["good_switch_total_gain"],
                "bad_switch_total_cost": switches["bad_switch_total_cost"],
                "net_switch_value": float(switches["good_switch_total_gain"]) - float(switches["bad_switch_total_cost"]),
                "mean_good_switch_gain": float(good_gains.mean()) if len(good_gains) else float("nan"),
                "mean_bad_switch_cost": float(bad_costs.mean()) if len(bad_costs) else float("nan"),
                "largest_good_switch": float(good_gains.max()) if len(good_gains) else float("nan"),
                "largest_bad_switch": float(bad_costs.max()) if len(bad_costs) else float("nan"),
            }
        )
        delta_stats = daily_delta_summary(deltas_arr)
        delta_stats["run_id"] = run_id
        delta_stats["rng_stream"] = int(reg["rng_stream"])
        switch_mask = part["is_non_41_switch"].astype(bool)
        switch_deltas = part.loc[switch_mask, "delta_vs_41"].astype(float)
        delta_stats["positive_switch_days"] = int((switch_deltas > 0).sum()) if len(switch_deltas) else 0
        delta_stats["negative_switch_days"] = int((switch_deltas < 0).sum()) if len(switch_deltas) else 0
        delta_stats["positive_switch_day_rate"] = float((switch_deltas > 0).mean()) if len(switch_deltas) else float("nan")
        raw_dd = max_drawdown(pnl)
        delta_stats["max_drawdown_raw_pnl"] = raw_dd["max_drawdown"]
        per_run_delta_rows.append(delta_stats)

        # Splits
        splits = assign_split_labels(part["date"].astype(str).tolist())
        # Need a daily frame with realised_selected_value for summarize_split
        daily_for_split = part.rename(columns={"strategy_pnl": "realised_selected_value"})
        for sp in splits:
            row = summarize_split(
                daily_for_split,
                day_indices=sp["day_indices"],
                split_name=sp["split_name"],
                split_start=sp["split_start"],
                split_end=sp["split_end"],
                candidate_41_pnl=c41,
            )
            row["run_id"] = run_id
            row["rng_stream"] = int(reg["rng_stream"])
            split_rows.append(row)

        # Costs
        for c_day in COST_LEVELS:
            for c_switch in COST_LEVELS:
                costs = cost_adjusted_totals(
                    strategy_pnl=pnl,
                    candidate_41_pnl=c41,
                    selected=selected,
                    cost_per_day=c_day,
                    cost_per_non_41_switch=c_switch,
                    cost_per_turnover=0.0,
                )
                be = break_even_costs(
                    raw_total_pnl=float(np.sum(pnl)),
                    candidate_41_total_pnl=float(np.sum(c41)),
                    day_count=len(pnl),
                    non_41_days=int(non_41),
                    turnover_count=int(to_c),
                )
                cost_rows.append(
                    {
                        "run_id": run_id,
                        "rng_stream": int(reg["rng_stream"]),
                        **costs,
                        **be,
                    }
                )
            for c_turn in COST_LEVELS:
                if c_turn == 0.0 and c_day == 0.0:
                    continue
                costs = cost_adjusted_totals(
                    strategy_pnl=pnl,
                    candidate_41_pnl=c41,
                    selected=selected,
                    cost_per_day=0.0,
                    cost_per_non_41_switch=0.0,
                    cost_per_turnover=c_turn,
                )
                be = break_even_costs(
                    raw_total_pnl=float(np.sum(pnl)),
                    candidate_41_total_pnl=float(np.sum(c41)),
                    day_count=len(pnl),
                    non_41_days=int(non_41),
                    turnover_count=int(to_c),
                )
                cost_rows.append({"run_id": run_id, "rng_stream": int(reg["rng_stream"]), **costs, **be})

        # Per-run bootstrap on best/worst and all runs (store compact summary only for all)
        for block in BOOTSTRAP_BLOCK_SIZES:
            boot = block_bootstrap_totals(
                deltas_arr,
                block_size=block,
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed + int(reg["rng_stream"]) * 100 + block,
            )
            samples = boot.pop("samples")
            boot["run_id"] = run_id
            boot["rng_stream"] = int(reg["rng_stream"])
            boot["mode"] = "per_run"
            bootstrap_rows.append(boot)
            # store only stream summaries' sample quantiles, not all 5000 for every run
            if int(reg["rng_stream"]) in {
                int(seed_summary["best_stream"]),
                int(seed_summary["worst_stream"]),
                1,
            }:
                for i, val in enumerate(samples):
                    if i % 50 == 0:  # thin samples for file size
                        bootstrap_sample_rows.append(
                            {
                                "run_id": run_id,
                                "rng_stream": int(reg["rng_stream"]),
                                "block_size": block,
                                "sample_index": i,
                                "total_delta": float(val),
                            }
                        )

    # Mean-daily-delta path across streams (average delta path)
    if not daily_panel.empty:
        pivot = daily_panel.pivot_table(index="day_index", columns="run_id", values="delta_vs_41", aggfunc="first")
        mean_path = pivot.mean(axis=1).to_numpy(dtype=float)
        for block in BOOTSTRAP_BLOCK_SIZES:
            boot = block_bootstrap_totals(
                mean_path,
                block_size=block,
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed + block,
            )
            samples = boot.pop("samples")
            boot["run_id"] = "mean_across_streams"
            boot["rng_stream"] = ""
            boot["mode"] = "mean_path_across_streams"
            bootstrap_rows.append(boot)
            for i, val in enumerate(samples):
                if i % 50 == 0:
                    bootstrap_sample_rows.append(
                        {
                            "run_id": "mean_across_streams",
                            "rng_stream": "",
                            "block_size": block,
                            "sample_index": i,
                            "total_delta": float(val),
                        }
                    )

    _write_csv(comparison / "selection_robustness.csv", sel_rows)
    _write_csv(comparison / "bad_switch_robustness.csv", bad_rows)
    _write_csv(split_dir / "split_summary.csv", split_rows)
    _write_csv(cost_dir / "cost_sensitivity.csv", cost_rows)
    _write_csv(stats_dir / "per_run_delta_summary.csv", per_run_delta_rows)
    _write_csv(stats_dir / "block_bootstrap_summary.csv", [{k: v for k, v in r.items()} for r in bootstrap_rows])
    _atomic_write_text(stats_dir / "block_bootstrap_summary.json", json.dumps(bootstrap_rows, indent=2, default=str))
    _write_csv(stats_dir / "block_bootstrap_samples.csv", bootstrap_sample_rows)

    # Aggregate candidate discovery across streams
    cand_rows = []
    if not daily_panel.empty:
        for cand, group in daily_panel.groupby("selected_candidate_index"):
            deltas_g = group["delta_vs_41"].astype(float)
            cand_rows.append(
                {
                    "candidate_index": int(cand),
                    "selected_day_count_total": int(len(group)),
                    "selected_run_count": int(group["run_id"].nunique()),
                    "mean_pnl_when_selected": float(group["strategy_pnl"].astype(float).mean()),
                    "mean_delta_vs_41_when_selected": float(deltas_g.mean()),
                    "good_switch_count": int((deltas_g > 0).sum()) if int(cand) != 41 else 0,
                    "bad_switch_count": int((deltas_g < 0).sum()) if int(cand) != 41 else 0,
                    "net_switch_value": float(deltas_g.sum()) if int(cand) != 41 else 0.0,
                }
            )
        _write_csv(comparison / "candidate_selection_across_streams.csv", sorted(cand_rows, key=lambda r: -r["selected_day_count_total"]))

    return {
        "seed_summary": seed_summary,
        "selection_rows": sel_rows,
        "bad_switch_rows": bad_rows,
        "bootstrap_rows": bootstrap_rows,
    }


def run_thompson_robustness(config: ThompsonRobustnessConfig, *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or logging.getLogger("bolr.l5_thompson_robustness")
    project_root = Path(__file__).resolve().parents[2]
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    ensure_run_directory(output_root)

    modes = []
    if config.mode == "all":
        modes = ["seed_robustness", "split_analysis", "cost_analysis"]
    elif config.mode == "analyse_existing":
        modes = ["analyse_existing"]
    else:
        modes = [config.mode]

    planned: list[dict[str, Any]] = []
    if "seed_robustness" in modes or config.mode == "all":
        for stream in config.rng_streams:
            planned.append(
                {
                    "mode": "seed_robustness",
                    "stream": int(stream),
                    "pair_budget": int(config.pair_budget),
                    "pair_seed": int(config.pair_sampling_seed),
                }
            )
    if "pair_sampling" in modes:
        for pb in config.pair_budgets:
            for ps in config.pair_sampling_seeds:
                for stream in config.rng_streams:
                    planned.append({"mode": "pair_sampling", "stream": int(stream), "pair_budget": int(pb), "pair_seed": int(ps)})

    if config.dry_run:
        print(json.dumps({"planned_runs": planned, "mode": config.mode}, indent=2))
        return {"planned_runs": planned, "dry_run": True, "output_root": str(output_root)}

    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    grid_csv = Path(config.grid_csv)
    if not grid_csv.is_absolute():
        grid_csv = project_root / grid_csv

    grid = load_candidate_grid(grid_csv, CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet(data_path, candidate_grid=grid)
    candidate_basis = build_candidate_basis(grid)
    date_cfg = NativeCandidateAReplayConfig(
        warmup_start=config.warmup_start,
        warmup_end=config.warmup_end,
        replay_start=config.replay_start,
        replay_end=config.replay_end,
        warm_up_days=config.warm_up_days,
        maximum_days=config.maximum_days,
        candidate_count=config.candidate_count,
    )
    warmup_dates, replay_dates = resolve_date_windows(dataset.dates, date_cfg)
    baselines = load_baseline_series(project_root=project_root, config=config, replay_dates=replay_dates, dataset=dataset)

    # Shared warm-up once
    partition = OrderedPartitionConfig(tolerance=OrderedPartitionToleranceConfig(relative_tolerance=config.relative_tolerance))
    need_runs = [p for p in planned if p["mode"] in {"seed_robustness", "pair_sampling"}]
    model = prior = static_surface = warmup_diag = None
    if need_runs and config.mode != "analyse_existing":
        cross0 = CrossGroupLogisticConfig(
            sampled_pair_budget=int(config.pair_budget),
            sampling_seed=int(config.pair_sampling_seed),
            normalize_pair_losses=config.normalize_pair_losses,
        )
        model, prior, static_surface, warmup_diag = warmup_candidate_b_state(
            dataset,
            candidate_basis,
            warmup_dates,
            partition_config=partition,
            cross_group_config=cross0,
            static_surface_config=StaticSurfaceConfig(),
            sigma0=config.sigma0,
            soft_target=SoftTargetConfig(),
        )

    registry_rows: list[dict[str, Any]] = []
    daily_panel_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    seed_runs_dir = output_root / "seed_robustness" / "runs"
    pair_runs_dir = output_root / "pair_sampling_robustness" / "runs"
    seed_runs_dir.mkdir(parents=True, exist_ok=True)
    pair_runs_dir.mkdir(parents=True, exist_ok=True)

    for plan in planned:
        stream = int(plan["stream"])
        pair_budget = int(plan["pair_budget"])
        pair_seed = int(plan["pair_seed"])
        mode = str(plan["mode"])
        name = run_name_for(stream=stream, pair_budget=pair_budget, pair_seed=pair_seed, maximum_days=config.maximum_days)
        run_id = name
        parent = seed_runs_dir if mode == "seed_robustness" else pair_runs_dir
        run_dir = parent / name
        summary_path = run_dir / "summary.json"
        daily_path = run_dir / "daily_results.csv"

        if config.resume_existing and summary_path.exists() and daily_path.exists():
            log.info("Resuming existing run %s", name)
            daily = pd.read_csv(daily_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = "completed"
            error = ""
        else:
            if config.mode == "analyse_existing":
                continue
            log.info("Starting %s", name)
            started = perf_counter()
            try:
                cross = CrossGroupLogisticConfig(
                    sampled_pair_budget=pair_budget,
                    sampling_seed=pair_seed,
                    normalize_pair_losses=config.normalize_pair_losses,
                )
                run_cfg = NativeCandidateBReplayConfig(
                    run_name=name,
                    strategy_name="candidate_b_sampled_fixed_thompson",
                    data_path=str(data_path),
                    grid_csv=str(grid_csv),
                    output_dir=str(parent),
                    warmup_start=config.warmup_start,
                    warmup_end=config.warmup_end,
                    replay_start=config.replay_start,
                    replay_end=config.replay_end,
                    warm_up_days=config.warm_up_days,
                    maximum_days=config.maximum_days,
                    candidate_count=config.candidate_count,
                    variant="sampled",
                    transition_family="fixed",
                    adaptive_enabled=False,
                    random_walk_variance=config.random_walk_variance,
                    sigma0=config.sigma0,
                    decision_policy="thompson",
                    monte_carlo_sample_count=config.monte_carlo_sample_count,
                    monte_carlo_chunk_size=config.monte_carlo_chunk_size,
                    top_k=config.top_k,
                    antithetic=config.antithetic,
                    rng_seed=config.rng_seed,
                    rng_stream=stream,
                    checkpoint_every_n_days=config.checkpoint_every_n_days,
                    overwrite_outputs=config.overwrite_outputs or config.resume_existing,
                    partition=partition,
                    cross_group=cross,
                    command_line=config.command_line,
                    write_outputs=True,
                )
                result = run_native_candidate_b_replay(
                    run_cfg,
                    dataset=dataset,
                    candidate_basis=candidate_basis,
                    grid=grid,
                    model=model,
                    initial_posterior=prior,
                    static_surface=static_surface,
                    warmup_diagnostics=warmup_diag,
                    logger=log,
                )
                daily = pd.DataFrame(result.daily_results)
                summary = result.summary
                status = "completed"
                error = ""
                print(f"run={name} elapsed={perf_counter()-started:.1f}s total_pnl={summary.get('total_pnl')}")
            except Exception as exc:  # noqa: BLE001
                log.exception("Run %s failed", name)
                daily = None
                summary = None
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                print(f"run={name} FAILED: {error}")

        reg = registry_row_from_result(
            run_id=run_id,
            run_name=name,
            status=status,
            mode=mode,
            output_path=run_dir,
            config=config,
            stream=stream,
            pair_budget=pair_budget,
            pair_seed=pair_seed,
            daily=daily,
            summary=summary,
            baselines=baselines,
            error_message=error,
        )
        registry_rows.append(reg)
        if daily is not None and status == "completed":
            # Fix delta vs 41 using same-length arrays
            pnl = daily["realised_selected_value"].astype(float).to_numpy()
            c41 = np.asarray(baselines["candidate_41_pnl"][: len(pnl)], dtype=float)
            reg["delta_vs_41_total"] = float(pnl.sum() - c41.sum())
            reg["delta_vs_41_mean"] = float(pnl.mean() - c41.mean())
            if baselines.get("candidate_a_pnl") is not None:
                a = np.asarray(baselines["candidate_a_pnl"][: len(pnl)], dtype=float)
                reg["delta_vs_candidate_a_total"] = float(pnl.sum() - a.sum())
            daily_panel_rows.extend(
                build_daily_delta_rows(
                    daily,
                    run_id=run_id,
                    rng_seed=config.rng_seed,
                    rng_stream=stream,
                    pair_budget=pair_budget,
                    pair_sampling_seed=pair_seed,
                    candidate_41_pnl=np.asarray(baselines["candidate_41_pnl"], dtype=float),
                    candidate_a_pnl=baselines.get("candidate_a_pnl"),
                    oracle_pnl=baselines.get("oracle_pnl"),
                )
            )
            if mode == "pair_sampling":
                pair_rows.append(
                    {
                        "pair_budget": pair_budget,
                        "pair_sampling_seed": pair_seed,
                        "rng_stream": stream,
                        "total_pnl": reg["total_pnl"],
                        "delta_vs_41_total": reg["delta_vs_41_total"],
                        "unique_selected_candidates": reg["unique_selected_candidates"],
                        "effective_selected_candidates": reg["effective_selected_candidates"],
                        "mean_sampled_pairs_per_day": summary.get("mean_sampled_pair_count", ""),
                        "max_sampled_pairs_per_day": summary.get("max_sampled_pair_count", ""),
                        "mean_pair_sample_rate": float(pd.to_numeric(daily.get("candidate_b_pair_sample_rate"), errors="coerce").mean())
                        if "candidate_b_pair_sample_rate" in daily.columns
                        else "",
                        "mean_possible_pair_count": float(pd.to_numeric(daily.get("candidate_b_possible_pair_count"), errors="coerce").mean())
                        if "candidate_b_possible_pair_count" in daily.columns
                        else "",
                        "laplace_failure_count": summary.get("laplace_failure_count", ""),
                        "total_elapsed_seconds": summary.get("total_elapsed_seconds", ""),
                    }
                )

    # Load existing runs for analyse_existing / post-analysis
    if config.mode == "analyse_existing" or "analyse_existing" in modes or not planned:
        for parent, mode_name in ((seed_runs_dir, "seed_robustness"), (pair_runs_dir, "pair_sampling")):
            if not parent.exists():
                continue
            for run_dir in sorted(parent.glob("cb_fixed_thompson_*")):
                if not (run_dir / "summary.json").exists():
                    continue
                if any(r["run_id"] == run_dir.name for r in registry_rows):
                    continue
                daily = pd.read_csv(run_dir / "daily_results.csv")
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                # parse stream/budget/seed from name
                parts = run_dir.name.split("_")
                stream = int([p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()][0])
                pair_budget = int([p[2:] for p in parts if p.startswith("pb")][0])
                pair_seed = int([p[2:] for p in parts if p.startswith("ps")][0].split("d")[0])
                reg = registry_row_from_result(
                    run_id=run_dir.name,
                    run_name=run_dir.name,
                    status="completed",
                    mode=mode_name,
                    output_path=run_dir,
                    config=config,
                    stream=stream,
                    pair_budget=pair_budget,
                    pair_seed=pair_seed,
                    daily=daily,
                    summary=summary,
                    baselines=baselines,
                )
                pnl = daily["realised_selected_value"].astype(float).to_numpy()
                c41 = np.asarray(baselines["candidate_41_pnl"][: len(pnl)], dtype=float)
                reg["delta_vs_41_total"] = float(pnl.sum() - c41.sum())
                registry_rows.append(reg)
                daily_panel_rows.extend(
                    build_daily_delta_rows(
                        daily,
                        run_id=run_dir.name,
                        rng_seed=config.rng_seed,
                        rng_stream=stream,
                        pair_budget=pair_budget,
                        pair_sampling_seed=pair_seed,
                        candidate_41_pnl=np.asarray(baselines["candidate_41_pnl"], dtype=float),
                        candidate_a_pnl=baselines.get("candidate_a_pnl"),
                        oracle_pnl=baselines.get("oracle_pnl"),
                    )
                )

    registry = pd.DataFrame(registry_rows)
    daily_panel = pd.DataFrame(daily_panel_rows)
    _write_csv(output_root / "run_registry.csv", registry)
    seed_comp = output_root / "seed_robustness" / "comparison"
    ensure_run_directory(seed_comp)
    if not daily_panel.empty:
        seed_panel = daily_panel.loc[daily_panel["pair_budget"] == config.pair_budget]
        # Prefer default pair seed panel for seed robustness
        seed_panel = seed_panel.loc[seed_panel["pair_sampling_seed"] == config.pair_sampling_seed] if not seed_panel.empty else seed_panel
        _write_csv(seed_comp / "seed_daily_delta_panel.csv", seed_panel if not seed_panel.empty else daily_panel)

    analysis = {}
    if not registry.empty and (
        config.mode in {"analyse_existing", "all", "seed_robustness", "cost_analysis", "split_analysis"} or True
    ):
        seed_reg = registry.loc[registry["mode"] == "seed_robustness"] if "mode" in registry.columns else registry
        seed_panel = daily_panel
        if not seed_panel.empty and "pair_budget" in seed_panel.columns:
            seed_panel = seed_panel.loc[
                (seed_panel["pair_budget"] == config.pair_budget) & (seed_panel["pair_sampling_seed"] == config.pair_sampling_seed)
            ]
        if seed_reg.empty:
            seed_reg = registry
        analysis = analyse_runs(
            registry=seed_reg,
            daily_panel=seed_panel if not seed_panel.empty else daily_panel,
            baselines=baselines,
            output_root=output_root,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        )

    if pair_rows:
        pair_comp = output_root / "pair_sampling_robustness" / "comparison"
        ensure_run_directory(pair_comp)
        pair_df = pd.DataFrame(pair_rows)
        _write_csv(pair_comp / "pair_sampling_summary.csv", pair_df)
        agg = (
            pair_df.groupby("pair_budget", as_index=False)
            .agg(
                mean_delta_vs_41=("delta_vs_41_total", "mean"),
                median_delta_vs_41=("delta_vs_41_total", "median"),
                std_delta_vs_41=("delta_vs_41_total", "std"),
                share_beating_41=("delta_vs_41_total", lambda s: float((s.astype(float) > 0).mean())),
                mean_runtime=("total_elapsed_seconds", "mean"),
            )
        )
        _write_csv(pair_comp / "pair_budget_summary.csv", agg)

    readme = "\n".join(
        [
            "# L5.4 Candidate B Fixed Thompson Robustness",
            "",
            f"- Mode: {config.mode}",
            f"- Replay days: {len(replay_dates)} ({replay_dates[0]} → {replay_dates[-1]})",
            f"- Streams planned: {config.rng_streams[:5]}{'...' if len(config.rng_streams)>5 else ''} ({len(config.rng_streams)} total)",
            f"- Pair budget/seed default: {config.pair_budget}/{config.pair_sampling_seed}",
            f"- L5.2 import: {baselines['l5_2_note']}",
            f"- L5.3 B pbest: {baselines['b_pbest_note']}",
            f"- Completed runs: {int((registry['status']=='completed').sum()) if not registry.empty else 0}",
            f"- Failed runs: {int((registry['status']=='failed').sum()) if not registry.empty else 0}",
            "",
            "PnL is observational evidence only.",
            "",
        ]
    )
    if analysis.get("seed_summary"):
        ss = analysis["seed_summary"]
        readme += (
            f"\n## Seed summary\n\n"
            f"- share_beating_41: {ss.get('share_beating_41')}\n"
            f"- median_delta_vs_41: {ss.get('median_delta_vs_41')}\n"
            f"- mean_delta_vs_41: {ss.get('mean_delta_vs_41')}\n"
            f"- p05/p95 delta: {ss.get('p05_delta_vs_41')} / {ss.get('p95_delta_vs_41')}\n"
            f"- best/worst stream: {ss.get('best_stream')} / {ss.get('worst_stream')}\n"
        )
    _atomic_write_text(output_root / "README.md", readme)

    return {
        "output_root": str(output_root),
        "registry": registry_rows,
        "analysis": analysis,
        "planned_runs": planned,
        "day_count": len(replay_dates),
        "baselines": {
            "always_41_total": baselines["always_41_total"],
            "candidate_a_total": baselines.get("candidate_a_total"),
            "oracle_total": baselines.get("oracle_total"),
            "l5_2_note": baselines["l5_2_note"],
            "b_pbest_note": baselines["b_pbest_note"],
        },
    }
