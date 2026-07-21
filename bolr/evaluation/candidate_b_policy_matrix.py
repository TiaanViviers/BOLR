"""Phase L5.3 Candidate B native replay matrix and L5.2 comparison import."""

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
    StaticSurfaceConfig,
)
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.candidate_replay_values import build_pnl_matrix
from bolr.evaluation.native_candidate_a_replay import build_candidate_basis, resolve_date_windows, NativeCandidateAReplayConfig
from bolr.evaluation.native_candidate_b_replay import (
    DEFAULT_SAMPLED_PAIR_BUDGET,
    NativeCandidateBReplayConfig,
    run_native_candidate_b_replay,
    warmup_candidate_b_state,
)
from bolr.evaluation.outputs import ensure_run_directory
from bolr.evaluation.policy_matrix_metrics import (
    bad_switch_diagnostics,
    normalize_native_daily,
    probability_best_bins,
    selection_entropy,
    summarize_strategy_daily,
    turnover_stats,
)

REQUIRED_STRATEGIES: tuple[str, ...] = (
    "candidate_b_sampled_fixed_probability_best",
    "candidate_b_sampled_fixed_posterior_mean",
    "candidate_b_sampled_fixed_thompson",
    "candidate_b_sampled_adaptive_probability_best",
    "candidate_b_sampled_adaptive_posterior_mean",
    "candidate_b_sampled_adaptive_thompson",
)

L52_IMPORT_STRATEGIES: tuple[str, ...] = (
    "baseline_always_41",
    "oracle_static_best_replay",
    "candidate_a_fixed_probability_best",
    "candidate_a_adaptive_probability_best",
    "candidate_a_fixed_thompson",
    "candidate_a_adaptive_thompson",
)

_STREAM_OFFSETS = {
    "candidate_b_sampled_fixed_probability_best": 0,
    "candidate_b_sampled_fixed_posterior_mean": 1,
    "candidate_b_sampled_fixed_thompson": 2,
    "candidate_b_sampled_adaptive_probability_best": 3,
    "candidate_b_sampled_adaptive_posterior_mean": 4,
    "candidate_b_sampled_adaptive_thompson": 5,
}

_DECISION_BY_NAME = {
    "probability_best": "probability_best",
    "posterior_mean": "posterior_mean",
    "thompson": "thompson",
}


@dataclass(frozen=True)
class CandidateBMatrixConfig:
    data_path: str = "data/YM_full.parquet"
    grid_csv: str = "data/YM_grid.csv"
    output_dir: str = "outputs/l5_candidate_b_native_replay"
    l5_2_comparison_dir: str | None = "outputs/l5_candidate_a_policy_matrix/comparison"
    warmup_start: str | None = "2021-01-29"
    warmup_end: str | None = "2023-01-11"
    replay_start: str | None = "2023-01-12"
    replay_end: str | None = "2024-10-08"
    warm_up_days: int = 504
    maximum_days: int | None = None
    candidate_count: int = 1428
    variant: str = "sampled"
    sampled_pair_budget: int = DEFAULT_SAMPLED_PAIR_BUDGET
    monte_carlo_sample_count: int = 512
    monte_carlo_chunk_size: int = 64
    top_k: int = 10
    antithetic: bool = True
    rng_seed: int = 20260720
    rng_stream: int = 1
    checkpoint_every_n_days: int = 25
    checkpoint_at_end: bool = True
    overwrite_outputs: bool = False
    strategies: tuple[str, ...] = REQUIRED_STRATEGIES
    include_fixed: bool = True
    include_adaptive: bool = True
    partition: OrderedPartitionConfig = field(
        default_factory=lambda: OrderedPartitionConfig(
            tolerance=OrderedPartitionToleranceConfig(relative_tolerance=0.1),
        )
    )
    static_surface: StaticSurfaceConfig = field(default_factory=StaticSurfaceConfig)
    random_walk_variance: float = 0.05
    sigma0: float = 1.0
    command_line: tuple[str, ...] = ()

    def resolved_strategies(self) -> tuple[str, ...]:
        selected = list(self.strategies)
        if not self.include_fixed:
            selected = [s for s in selected if "_fixed_" not in s]
        if not self.include_adaptive:
            selected = [s for s in selected if "_adaptive_" not in s]
        unknown = [name for name in selected if name not in REQUIRED_STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown Candidate B strategies: {unknown}")
        return tuple(selected)


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


def parse_strategy_name(name: str) -> tuple[str, str, str]:
    # candidate_b_{variant}_{transition}_{decision...}
    prefix = "candidate_b_"
    if not name.startswith(prefix):
        raise ValueError(f"Cannot parse strategy name: {name}")
    rest = name[len(prefix) :]
    for transition in ("fixed", "adaptive"):
        marker = f"_{transition}_"
        if marker not in rest:
            continue
        variant, decision = rest.split(marker, 1)
        if decision not in _DECISION_BY_NAME:
            raise ValueError(f"Unknown decision policy in strategy name: {name}")
        return variant, transition, _DECISION_BY_NAME[decision]
    raise ValueError(f"Cannot parse strategy name: {name}")


def load_l5_2_comparison(comparison_dir: Path | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    if comparison_dir is None:
        return None, None, "l5_2_comparison_dir not provided"
    if not comparison_dir.is_absolute():
        comparison_dir = Path(__file__).resolve().parents[2] / comparison_dir
    summary_path = comparison_dir / "strategy_summary.csv"
    daily_path = comparison_dir / "strategy_daily_panel.csv"
    if not summary_path.exists() or not daily_path.exists():
        return None, None, f"L5.2 comparison files missing under {comparison_dir}"
    summary = pd.read_csv(summary_path)
    daily = pd.read_csv(daily_path)
    keep = summary["strategy_name"].isin(L52_IMPORT_STRATEGIES)
    summary = summary.loc[keep].copy()
    daily = daily.loc[daily["strategy_name"].isin(L52_IMPORT_STRATEGIES)].copy()
    return summary, daily, f"imported from {comparison_dir}"


def candidate_b_pair_diagnostics(daily: pd.DataFrame) -> dict[str, Any]:
    possible = pd.to_numeric(daily.get("candidate_b_possible_pair_count"), errors="coerce")
    sampled = pd.to_numeric(daily.get("candidate_b_sampled_pair_count"), errors="coerce")
    rate = pd.to_numeric(daily.get("candidate_b_pair_sample_rate"), errors="coerce")
    weight = pd.to_numeric(daily.get("candidate_b_total_pair_weight"), errors="coerce")
    informative = daily.get("candidate_b_partition_informative")
    fallback = daily.get("candidate_b_fallback_used")
    return {
        "mean_possible_pair_count": float(possible.mean()) if possible.notna().any() else float("nan"),
        "median_possible_pair_count": float(possible.median()) if possible.notna().any() else float("nan"),
        "p95_possible_pair_count": float(possible.quantile(0.95)) if possible.notna().any() else float("nan"),
        "max_possible_pair_count": float(possible.max()) if possible.notna().any() else float("nan"),
        "mean_sampled_pair_count": float(sampled.mean()) if sampled.notna().any() else float("nan"),
        "median_sampled_pair_count": float(sampled.median()) if sampled.notna().any() else float("nan"),
        "p95_sampled_pair_count": float(sampled.quantile(0.95)) if sampled.notna().any() else float("nan"),
        "max_sampled_pair_count": float(sampled.max()) if sampled.notna().any() else float("nan"),
        "mean_pair_sample_rate": float(rate.mean()) if rate.notna().any() else float("nan"),
        "mean_total_pair_weight": float(weight.mean()) if weight.notna().any() else float("nan"),
        "mean_partition_entropy": "",
        "mean_r3_count": float(pd.to_numeric(daily.get("candidate_b_r3_count"), errors="coerce").mean()),
        "mean_r2_count": float(pd.to_numeric(daily.get("candidate_b_r2_count"), errors="coerce").mean()),
        "mean_r1_count": float(pd.to_numeric(daily.get("candidate_b_r1_count"), errors="coerce").mean()),
        "mean_r0_count": float(pd.to_numeric(daily.get("candidate_b_r0_count"), errors="coerce").mean()),
        "informative_day_count": int(informative.astype(bool).sum()) if informative is not None else "",
        "fallback_day_count": int(fallback.astype(bool).sum()) if fallback is not None else "",
    }


def selection_diagnostics_row(daily: pd.DataFrame, *, strategy_name: str) -> dict[str, Any]:
    selected = daily["selected_candidate_index"].astype(int).tolist()
    entropy, n_eff = selection_entropy(selected)
    turnover_count, turnover_rate = turnover_stats(selected)
    counts = pd.Series(selected).value_counts()
    top5 = counts.head(5)
    return {
        "strategy_name": strategy_name,
        "unique_selected_candidates": int(counts.size),
        "selection_entropy": entropy,
        "effective_selected_candidates": n_eff,
        "turnover_rate": turnover_rate,
        "turnover_count": turnover_count,
        "most_selected_candidate": int(counts.index[0]),
        "most_selected_candidate_share": float(counts.iloc[0] / len(selected)),
        "candidate_41_selected_share": float((pd.Series(selected) == 41).mean()),
        "top_5_selected_candidates": ",".join(str(int(i)) for i in top5.index.tolist()),
        "top_5_selected_candidate_shares": ",".join(f"{float(v / len(selected)):.6f}" for v in top5.tolist()),
    }


def enrich_candidate_b_summary(
    summary: dict[str, Any],
    daily: pd.DataFrame,
    *,
    always_41_total: float,
    always_41_mean: float,
    candidate_a_fixed_pbest_total: float | None,
    candidate_a_fixed_pbest_mean: float | None,
    oracle_total: float | None,
    oracle_mean: float | None,
    candidate_41_pnl: Sequence[float],
) -> dict[str, Any]:
    out = dict(summary)
    selected = daily["selected_candidate_index"].astype(int).tolist()
    pnl = daily["realised_selected_value"].astype(float).to_numpy()
    total_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean())
    out["candidate_41_delta_total_pnl"] = total_pnl - float(always_41_total)
    out["candidate_41_delta_mean_pnl"] = mean_pnl - float(always_41_mean)
    if candidate_a_fixed_pbest_total is not None:
        out["candidate_a_fixed_probability_best_delta_total_pnl"] = total_pnl - float(candidate_a_fixed_pbest_total)
        out["candidate_a_fixed_probability_best_delta_mean_pnl"] = mean_pnl - float(candidate_a_fixed_pbest_mean or 0.0)
    else:
        out["candidate_a_fixed_probability_best_delta_total_pnl"] = ""
        out["candidate_a_fixed_probability_best_delta_mean_pnl"] = ""
    if oracle_total is not None:
        out["oracle_gap_total_pnl"] = total_pnl - float(oracle_total)
        out["oracle_gap_mean_pnl"] = mean_pnl - float(oracle_mean or 0.0)
    else:
        out["oracle_gap_total_pnl"] = ""
        out["oracle_gap_mean_pnl"] = ""
    switches = bad_switch_diagnostics(selected, pnl.tolist(), list(candidate_41_pnl))
    out.update(switches)
    out["bad_switch_count_vs_41"] = switches["bad_switch_count"]
    out["bad_switch_total_cost_vs_41"] = switches["bad_switch_total_cost"]
    out["good_switch_count_vs_41"] = switches["good_switch_count"]
    out["good_switch_total_gain_vs_41"] = switches["good_switch_total_gain"]
    out["net_switch_value"] = float(switches["good_switch_total_gain"]) - float(switches["bad_switch_total_cost"])
    return out


def run_candidate_b_matrix(config: CandidateBMatrixConfig, *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or logging.getLogger("bolr.l5_candidate_b_matrix")
    project_root = Path(__file__).resolve().parents[2]
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    comparison_dir = output_root / "comparison"
    ensure_run_directory(comparison_dir)

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
    replay_pnl = build_pnl_matrix(dataset, replay_dates)
    candidate_41_pnl = replay_pnl[:, 41].tolist()
    always_41_total = float(np.sum(candidate_41_pnl))
    always_41_mean = float(np.mean(candidate_41_pnl))

    cross_group = CrossGroupLogisticConfig(
        sampled_pair_budget=None if config.variant == "exact" else int(config.sampled_pair_budget),
        sampling_seed=0,
        normalize_pair_losses=True,
    )
    model, prior, static_surface, warmup_diag = warmup_candidate_b_state(
        dataset,
        candidate_basis,
        warmup_dates,
        partition_config=config.partition,
        cross_group_config=cross_group,
        static_surface_config=config.static_surface,
        sigma0=config.sigma0,
    )

    l52_summary, l52_daily, l52_note = load_l5_2_comparison(
        Path(config.l5_2_comparison_dir) if config.l5_2_comparison_dir else None
    )
    candidate_a_fixed_pbest_total = None
    candidate_a_fixed_pbest_mean = None
    oracle_total = None
    oracle_mean = None
    if l52_summary is not None and not l52_summary.empty:
        row_a = l52_summary.loc[l52_summary["strategy_name"] == "candidate_a_fixed_probability_best"]
        if not row_a.empty:
            candidate_a_fixed_pbest_total = float(row_a.iloc[0]["total_pnl"])
            candidate_a_fixed_pbest_mean = float(row_a.iloc[0]["mean_pnl"])
        row_o = l52_summary.loc[l52_summary["strategy_name"] == "oracle_static_best_replay"]
        if not row_o.empty:
            oracle_total = float(row_o.iloc[0]["total_pnl"])
            oracle_mean = float(row_o.iloc[0]["mean_pnl"])
        row_41 = l52_summary.loc[l52_summary["strategy_name"] == "baseline_always_41"]
        if not row_41.empty:
            always_41_total = float(row_41.iloc[0]["total_pnl"])
            always_41_mean = float(row_41.iloc[0]["mean_pnl"])

    summary_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []

    if l52_summary is not None:
        for _, row in l52_summary.iterrows():
            payload = row.to_dict()
            payload["observation_model"] = "candidate_a" if str(payload["strategy_name"]).startswith("candidate_a") else "baseline"
            payload["candidate_b_variant"] = ""
            summary_rows.append(payload)
        if l52_daily is not None:
            daily_frames.append(l52_daily)

    strategies = config.resolved_strategies()
    for name in strategies:
        started = perf_counter()
        log.info("Starting strategy %s", name)
        try:
            variant, transition, decision = parse_strategy_name(name)
            if variant != config.variant:
                raise ValueError(f"Strategy {name} variant mismatch with config.variant={config.variant}")
            run_cfg = NativeCandidateBReplayConfig(
                run_name=name,
                strategy_name=name,
                data_path=str(data_path),
                grid_csv=str(grid_csv),
                output_dir=str(output_root),
                warmup_start=config.warmup_start,
                warmup_end=config.warmup_end,
                replay_start=config.replay_start,
                replay_end=config.replay_end,
                warm_up_days=config.warm_up_days,
                maximum_days=config.maximum_days,
                candidate_count=config.candidate_count,
                variant=variant,
                transition_family="adaptive" if transition == "adaptive" else "fixed",
                adaptive_enabled=transition == "adaptive",
                random_walk_variance=config.random_walk_variance,
                sigma0=config.sigma0,
                decision_policy=decision,
                monte_carlo_sample_count=config.monte_carlo_sample_count,
                monte_carlo_chunk_size=config.monte_carlo_chunk_size,
                top_k=config.top_k,
                antithetic=config.antithetic,
                rng_seed=config.rng_seed,
                rng_stream=int(config.rng_stream) + int(_STREAM_OFFSETS.get(name, 0)),
                checkpoint_every_n_days=config.checkpoint_every_n_days,
                checkpoint_at_end=config.checkpoint_at_end,
                overwrite_outputs=config.overwrite_outputs,
                partition=config.partition,
                cross_group=cross_group,
                static_surface=config.static_surface,
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
            daily = normalize_native_daily(
                daily,
                strategy_name=name,
                strategy_family="candidate_b_native",
                grid_config_ids=grid.config_ids,
            )
            base_summary = summarize_strategy_daily(
                daily,
                strategy_name=name,
                strategy_family="candidate_b_native",
                transition_family=run_cfg.transition_family,
                decision_policy=decision,
                always_41_total_pnl=always_41_total,
                always_41_mean_pnl=always_41_mean,
                candidate_41_pnl=candidate_41_pnl,
                laplace_failure_count=int(result.summary.get("laplace_failure_count", 0)),
                checkpoint_count=int(result.summary.get("checkpoint_count", 0)),
                total_elapsed_seconds=float(result.summary.get("total_elapsed_seconds", 0.0)),
                mean_day_ms=float(result.summary.get("mean_day_ms", float("nan"))),
            )
            base_summary["observation_model"] = "candidate_b"
            base_summary["candidate_b_variant"] = variant
            base_summary["max_day_ms"] = result.summary.get("max_day_ms")
            base_summary["mean_sampled_pair_count"] = result.summary.get("mean_sampled_pair_count")
            base_summary["max_sampled_pair_count"] = result.summary.get("max_sampled_pair_count")
            enriched = enrich_candidate_b_summary(
                base_summary,
                daily,
                always_41_total=always_41_total,
                always_41_mean=always_41_mean,
                candidate_a_fixed_pbest_total=candidate_a_fixed_pbest_total,
                candidate_a_fixed_pbest_mean=candidate_a_fixed_pbest_mean,
                oracle_total=oracle_total,
                oracle_mean=oracle_mean,
                candidate_41_pnl=candidate_41_pnl,
            )
            summary_rows.append(enriched)
            daily_frames.append(daily)
            selection_rows.append(selection_diagnostics_row(daily, strategy_name=name))
            pair_diag = candidate_b_pair_diagnostics(daily)
            pair_diag["strategy_name"] = name
            pair_rows.append(pair_diag)
            if decision == "probability_best":
                for row in probability_best_bins(daily):
                    row = dict(row)
                    row["strategy_name"] = name
                    if "empirical_top_5_hit_rate" not in row and "realised_rank" in daily.columns:
                        # policy_matrix_metrics currently omits top-5; compute if needed later
                        pass
                    calibration_rows.append(row)
            elapsed = perf_counter() - started
            log.info("Finished %s in %.1fs total_pnl=%.4f", name, elapsed, enriched["total_pnl"])
            print(f"strategy={name} elapsed_seconds={elapsed:.1f} total_pnl={enriched['total_pnl']:.4f}")
        except Exception as exc:  # noqa: BLE001 - matrix continues/reporting
            elapsed = perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            blocked.append({"strategy_name": name, "reason": reason, "elapsed_seconds": f"{elapsed:.1f}"})
            log.exception("Strategy %s blocked after %.1fs", name, elapsed)
            print(f"strategy={name} BLOCKED after {elapsed:.1f}s: {reason}")

    summary_df = pd.DataFrame(summary_rows)
    daily_panel = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    _write_csv(comparison_dir / "strategy_summary.csv", summary_df)
    _atomic_write_text(
        comparison_dir / "strategy_summary.json",
        json.dumps(summary_rows, indent=2, sort_keys=True, default=str),
    )
    _write_csv(comparison_dir / "strategy_daily_panel.csv", daily_panel)
    _write_csv(comparison_dir / "selection_diagnostics.csv", selection_rows)
    _write_csv(comparison_dir / "candidate_b_pair_diagnostics.csv", pair_rows)
    _write_csv(comparison_dir / "probability_best_calibration.csv", calibration_rows)
    if blocked:
        _write_csv(comparison_dir / "blocked_strategies.csv", blocked)

    readme = "\n".join(
        [
            "# L5.3 Candidate B Native Replay Comparison",
            "",
            f"- Replay days: {len(replay_dates)}",
            f"- Range: {replay_dates[0]} → {replay_dates[-1]}",
            f"- Variant: {config.variant}",
            f"- Pair budget: {config.sampled_pair_budget if config.variant == 'sampled' else 'exact'}",
            f"- L5.2 import: {l52_note}",
            f"- Strategies completed: {len(summary_rows) - (0 if l52_summary is None else len(l52_summary))}",
            f"- Strategies blocked: {len(blocked)}",
            f"- always-41 total PnL: {always_41_total:.4f}",
            (
                f"- Candidate A fixed probability-best total PnL: {candidate_a_fixed_pbest_total:.4f}"
                if candidate_a_fixed_pbest_total is not None
                else "- Candidate A fixed probability-best: not imported"
            ),
            "",
            "PnL is observational evidence only.",
            "",
        ]
    )
    _atomic_write_text(comparison_dir / "README.md", readme)
    return {
        "output_root": str(output_root),
        "comparison_dir": str(comparison_dir),
        "summary_rows": summary_rows,
        "blocked": blocked,
        "l5_2_note": l52_note,
        "always_41_total_pnl": always_41_total,
        "candidate_a_fixed_probability_best_total_pnl": candidate_a_fixed_pbest_total,
        "oracle_static_best_replay_total_pnl": oracle_total,
        "day_count": len(replay_dates),
        "replay_range": [replay_dates[0], replay_dates[-1]],
    }
