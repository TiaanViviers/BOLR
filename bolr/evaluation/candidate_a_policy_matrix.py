"""Phase L5.2 Candidate A policy/static-baseline comparison matrix."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bolr.config.foundation import CandidateGridConfig, SoftTargetConfig, StaticSurfaceConfig
from bolr.data.candidate_grid import CandidateGrid, load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.candidate_replay_values import (
    argmax_with_lowest_index,
    build_pnl_matrix,
    candidate_identity,
    get_candidate_replay_values,
)
from bolr.evaluation.metrics import rank_of_selected
from bolr.evaluation.native_candidate_a_replay import (
    NativeCandidateAReplayConfig,
    build_candidate_basis,
    resolve_date_windows,
    run_native_candidate_a_replay,
    warmup_candidate_a_state,
)
from bolr.evaluation.outputs import ensure_run_directory, write_json
from bolr.evaluation.policy_matrix_metrics import (
    normalize_native_daily,
    probability_best_bins,
    summarize_strategy_daily,
)

REQUIRED_STRATEGIES: tuple[str, ...] = (
    "baseline_always_41",
    "baseline_best_warmup_static",
    "baseline_trailing_mean_5",
    "baseline_trailing_mean_20",
    "oracle_static_best_replay",
    "candidate_a_fixed_posterior_mean",
    "candidate_a_fixed_probability_best",
    "candidate_a_fixed_thompson",
    "candidate_a_adaptive_posterior_mean",
    "candidate_a_adaptive_probability_best",
    "candidate_a_adaptive_thompson",
)

OPTIONAL_STRATEGIES: tuple[str, ...] = (
    "baseline_best_warmup_sharpe",
    "baseline_trailing_positive_5",
    "baseline_trailing_positive_20",
)

ALL_STRATEGIES: tuple[str, ...] = REQUIRED_STRATEGIES + OPTIONAL_STRATEGIES

_STREAM_OFFSETS = {
    "candidate_a_fixed_posterior_mean": 0,
    "candidate_a_fixed_probability_best": 1,
    "candidate_a_fixed_thompson": 2,
    "candidate_a_adaptive_posterior_mean": 3,
    "candidate_a_adaptive_probability_best": 4,
    "candidate_a_adaptive_thompson": 5,
}


@dataclass(frozen=True)
class PolicyMatrixConfig:
    data_path: str = "data/YM_full.parquet"
    grid_csv: str = "data/YM_grid.csv"
    output_dir: str = "outputs/l5_candidate_a_policy_matrix"
    warmup_start: str | None = "2021-01-29"
    warmup_end: str | None = "2023-01-11"
    replay_start: str | None = "2023-01-12"
    replay_end: str | None = "2024-10-08"
    warm_up_days: int = 504
    maximum_days: int | None = None
    candidate_count: int = 1428
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
    include_optional: bool = False
    soft_target: SoftTargetConfig = field(default_factory=SoftTargetConfig)
    static_surface: StaticSurfaceConfig = field(default_factory=StaticSurfaceConfig)
    random_walk_variance: float = 0.05
    sigma0: float = 1.0
    command_line: tuple[str, ...] = ()

    def resolved_strategies(self) -> tuple[str, ...]:
        selected = list(self.strategies)
        if self.include_optional:
            for name in OPTIONAL_STRATEGIES:
                if name not in selected:
                    selected.append(name)
        unknown = [name for name in selected if name not in ALL_STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}")
        return tuple(selected)


@dataclass(frozen=True)
class StrategyRunResult:
    strategy_name: str
    strategy_family: str
    transition_family: str
    decision_policy: str
    run_dir: Path
    daily: pd.DataFrame
    summary: dict[str, Any]
    elapsed_seconds: float
    skipped: bool = False
    skip_reason: str = ""


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


def _day_row(
    *,
    strategy_name: str,
    strategy_family: str,
    day_index: int,
    date: str,
    selected: int,
    grid: CandidateGrid,
    pnl_row: np.ndarray,
) -> dict[str, Any]:
    identity = candidate_identity(grid, selected)
    best_index = int(np.argmax(pnl_row))
    realised_selected = float(pnl_row[selected])
    realised_best = float(pnl_row[best_index])
    return {
        "strategy_name": strategy_name,
        "strategy_family": strategy_family,
        "day_index": day_index,
        "date": date,
        "selected_candidate_index": selected,
        "selected_entry_index": identity.entry_index,
        "selected_stop_index": identity.stop_index,
        "selected_config_id": identity.config_id,
        "realised_selected_value": realised_selected,
        "realised_best_value": realised_best,
        "best_candidate_index": best_index,
        "realised_rank": rank_of_selected(pnl_row, selected),
        "regret": realised_best - realised_selected,
        "selected_positive": bool(realised_selected > 0.0),
        "candidate_count": int(grid.n_candidates),
    }


def _finalize_baseline(
    *,
    strategy_name: str,
    strategy_family: str,
    transition_family: str,
    decision_policy: str,
    rows: list[dict[str, Any]],
    output_root: Path,
    always_41_total: float,
    always_41_mean: float,
    candidate_41_pnl: np.ndarray,
    elapsed: float,
    manifest_extra: Mapping[str, Any],
    overwrite: bool,
) -> StrategyRunResult:
    run_dir = output_root / strategy_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists: {run_dir}")
    ensure_run_directory(run_dir)
    (run_dir / "logs").mkdir(exist_ok=True)
    daily = pd.DataFrame(rows)
    summary = summarize_strategy_daily(
        daily,
        strategy_name=strategy_name,
        strategy_family=strategy_family,
        transition_family=transition_family,
        decision_policy=decision_policy,
        always_41_total_pnl=always_41_total,
        always_41_mean_pnl=always_41_mean,
        candidate_41_pnl=candidate_41_pnl,
        laplace_failure_count=0,
        checkpoint_count=0,
        total_elapsed_seconds=elapsed,
        mean_day_ms=float(1000.0 * elapsed / max(len(rows), 1)),
    )
    _write_csv(run_dir / "daily_results.csv", daily)
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))
    manifest = {
        "strategy_name": strategy_name,
        "strategy_family": strategy_family,
        "transition_family": transition_family,
        "decision_policy": decision_policy,
        "checkpoint_format_version": None,
        "backend": "baseline",
        **dict(manifest_extra),
        "summary": summary,
    }
    _atomic_write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
    (run_dir / "logs" / "replay.log").write_text(f"{strategy_name} completed days={len(rows)}\n", encoding="utf-8")
    return StrategyRunResult(
        strategy_name=strategy_name,
        strategy_family=strategy_family,
        transition_family=transition_family,
        decision_policy=decision_policy,
        run_dir=run_dir,
        daily=daily,
        summary=summary,
        elapsed_seconds=elapsed,
    )


def run_always_candidate_baseline(
    *,
    strategy_name: str,
    candidate_index: int,
    grid: CandidateGrid,
    replay_dates: Sequence[str],
    replay_pnl: np.ndarray,
    output_root: Path,
    always_41_total: float,
    always_41_mean: float,
    candidate_41_pnl: np.ndarray,
    overwrite: bool,
    manifest_extra: Mapping[str, Any],
) -> StrategyRunResult:
    started = perf_counter()
    identity = candidate_identity(grid, candidate_index)
    rows = [
        _day_row(
            strategy_name=strategy_name,
            strategy_family="static_baseline",
            day_index=i,
            date=date,
            selected=candidate_index,
            grid=grid,
            pnl_row=replay_pnl[i],
        )
        for i, date in enumerate(replay_dates)
    ]
    return _finalize_baseline(
        strategy_name=strategy_name,
        strategy_family="static_baseline",
        transition_family="none",
        decision_policy="static",
        rows=rows,
        output_root=output_root,
        always_41_total=always_41_total,
        always_41_mean=always_41_mean,
        candidate_41_pnl=candidate_41_pnl,
        elapsed=perf_counter() - started,
        overwrite=overwrite,
        manifest_extra={
            **dict(manifest_extra),
            "selected_identity": asdict(identity),
            "leakage": strategy_name.startswith("oracle_"),
        },
    )


def run_trailing_baseline(
    *,
    strategy_name: str,
    metric: str,
    window: int,
    grid: CandidateGrid,
    warmup_pnl: np.ndarray,
    replay_pnl: np.ndarray,
    replay_dates: Sequence[str],
    output_root: Path,
    always_41_total: float,
    always_41_mean: float,
    candidate_41_pnl: np.ndarray,
    overwrite: bool,
    manifest_extra: Mapping[str, Any],
) -> StrategyRunResult:
    started = perf_counter()
    history = [row.copy() for row in warmup_pnl]
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(replay_dates):
        if len(history) < window:
            raise RuntimeError(f"Trailing window {window} exceeds available history before {date}.")
        window_stack = np.stack(history[-window:], axis=0)
        if metric == "mean":
            scores = window_stack.mean(axis=0)
        elif metric == "positive_rate":
            scores = (window_stack > 0.0).mean(axis=0)
        else:
            raise ValueError(f"Unsupported trailing metric: {metric}")
        selected = argmax_with_lowest_index(scores)
        rows.append(
            _day_row(
                strategy_name=strategy_name,
                strategy_family="trailing_baseline",
                day_index=i,
                date=date,
                selected=selected,
                grid=grid,
                pnl_row=replay_pnl[i],
            )
        )
        # Current-day PnL becomes history only after the decision.
        history.append(replay_pnl[i].copy())
    return _finalize_baseline(
        strategy_name=strategy_name,
        strategy_family="trailing_baseline",
        transition_family="none",
        decision_policy=f"trailing_{metric}_{window}",
        rows=rows,
        output_root=output_root,
        always_41_total=always_41_total,
        always_41_mean=always_41_mean,
        candidate_41_pnl=candidate_41_pnl,
        elapsed=perf_counter() - started,
        overwrite=overwrite,
        manifest_extra={**dict(manifest_extra), "window": window, "metric": metric, "leakage": False},
    )


def _native_strategy_spec(name: str) -> tuple[str, str]:
    mapping = {
        "candidate_a_fixed_posterior_mean": ("fixed", "posterior_mean"),
        "candidate_a_fixed_probability_best": ("fixed", "probability_best"),
        "candidate_a_fixed_thompson": ("fixed", "thompson"),
        "candidate_a_adaptive_posterior_mean": ("adaptive", "posterior_mean"),
        "candidate_a_adaptive_probability_best": ("adaptive", "probability_best"),
        "candidate_a_adaptive_thompson": ("adaptive", "thompson"),
    }
    return mapping[name]


def run_native_strategy(
    *,
    strategy_name: str,
    config: PolicyMatrixConfig,
    dataset: HistoricalDataset,
    candidate_basis: np.ndarray,
    output_root: Path,
    always_41_total: float,
    always_41_mean: float,
    candidate_41_pnl: np.ndarray,
    model,
    initial_posterior,
    static_surface,
    warmup_diagnostics: Mapping[str, Any],
) -> StrategyRunResult:
    transition, decision = _native_strategy_spec(strategy_name)
    stream = int(config.rng_stream) + int(_STREAM_OFFSETS[strategy_name])
    started = perf_counter()
    native_cfg = NativeCandidateAReplayConfig(
        run_name=strategy_name,
        data_path=config.data_path,
        grid_csv=config.grid_csv,
        output_dir=str(output_root),
        warmup_start=config.warmup_start,
        warmup_end=config.warmup_end,
        replay_start=config.replay_start,
        replay_end=config.replay_end,
        warm_up_days=config.warm_up_days,
        maximum_days=config.maximum_days,
        candidate_count=config.candidate_count,
        transition_family=transition,
        adaptive_enabled=transition == "adaptive",
        random_walk_variance=config.random_walk_variance,
        sigma0=config.sigma0,
        decision_policy=decision,
        monte_carlo_sample_count=config.monte_carlo_sample_count,
        monte_carlo_chunk_size=config.monte_carlo_chunk_size,
        top_k=config.top_k,
        antithetic=config.antithetic,
        rng_seed=config.rng_seed,
        rng_stream=stream,
        checkpoint_every_n_days=config.checkpoint_every_n_days,
        checkpoint_at_end=config.checkpoint_at_end,
        overwrite_outputs=config.overwrite_outputs,
        soft_target=config.soft_target,
        static_surface=config.static_surface,
        command_line=config.command_line,
        write_outputs=True,
    )
    result = run_native_candidate_a_replay(
        native_cfg,
        dataset=dataset,
        candidate_basis=candidate_basis,
        grid=dataset.candidate_grid,
        model=model,
        initial_posterior=initial_posterior,
        static_surface=static_surface,
        warmup_diagnostics=dict(warmup_diagnostics),
    )
    daily = normalize_native_daily(
        pd.DataFrame(result.daily_results),
        strategy_name=strategy_name,
        strategy_family="candidate_a_native",
        grid_config_ids=dataset.candidate_grid.config_ids,
    )
    # Rewrite daily with strategy columns for panel consistency.
    _write_csv(result.run_dir / "daily_results.csv", daily)
    elapsed = perf_counter() - started
    summary = summarize_strategy_daily(
        daily,
        strategy_name=strategy_name,
        strategy_family="candidate_a_native",
        transition_family=transition,
        decision_policy=decision,
        always_41_total_pnl=always_41_total,
        always_41_mean_pnl=always_41_mean,
        candidate_41_pnl=candidate_41_pnl,
        laplace_failure_count=int(result.summary.get("laplace_failure_count", 0) or 0),
        checkpoint_count=int(result.summary.get("checkpoint_count", 0) or 0),
        total_elapsed_seconds=elapsed,
        mean_day_ms=float(result.summary.get("mean_day_ms") or 0.0),
    )
    summary["rng_seed"] = config.rng_seed
    summary["rng_stream"] = stream
    summary["thompson_sample_zero"] = decision == "thompson"
    _atomic_write_text(result.run_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))
    manifest = dict(result.manifest)
    manifest.update(
        {
            "strategy_name": strategy_name,
            "strategy_family": "candidate_a_native",
            "transition_family": transition,
            "decision_policy": decision,
            "rng_seed": config.rng_seed,
            "rng_stream": stream,
            "rng_stream_offset_policy": "base_stream_plus_strategy_offset",
            "matrix_summary": summary,
        }
    )
    _atomic_write_text(result.run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return StrategyRunResult(
        strategy_name=strategy_name,
        strategy_family="candidate_a_native",
        transition_family=transition,
        decision_policy=decision,
        run_dir=result.run_dir,
        daily=daily,
        summary=summary,
        elapsed_seconds=elapsed,
    )


def _print_terminal_summary(summaries: Sequence[Mapping[str, Any]]) -> None:
    deployable = [s for s in summaries if s.get("strategy_family") != "oracle_baseline"]
    oracle = [s for s in summaries if s.get("strategy_family") == "oracle_baseline"]
    ordered = sorted(deployable, key=lambda s: float(s["total_pnl"]), reverse=True) + oracle
    header = (
        f"{'strategy':<38} {'total_pnl':>10} {'mean_pnl':>9} {'mean_regret':>11} "
        f"{'top10':>6} {'unique':>6} {'max_share':>9} {'vs_41':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in ordered:
        mark = " [oracle]" if s.get("strategy_family") == "oracle_baseline" else ""
        print(
            f"{str(s['strategy_name']) + mark:<38} "
            f"{float(s['total_pnl']):>10.2f} "
            f"{float(s['mean_pnl']):>9.4f} "
            f"{float(s['mean_regret']):>11.4f} "
            f"{float(s['top_10_hit_rate']):>6.3f} "
            f"{int(s['unique_selected_candidates']):>6d} "
            f"{float(s['most_selected_candidate_share']):>9.3f} "
            f"{float(s['candidate_41_delta_total_pnl']):>10.2f}"
        )


def run_candidate_a_policy_matrix(config: PolicyMatrixConfig) -> dict[str, Any]:
    log = logging.getLogger("bolr.l5_policy_matrix")
    project_root = Path(__file__).resolve().parents[2]
    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    grid_csv = Path(config.grid_csv)
    if not grid_csv.is_absolute():
        grid_csv = project_root / grid_csv
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    strategies = config.resolved_strategies()
    grid = load_candidate_grid(grid_csv, CandidateGridConfig())
    if grid.n_candidates != config.candidate_count:
        raise ValueError(f"Expected candidate_count={config.candidate_count}, found {grid.n_candidates}.")
    dataset = HistoricalDataset.from_parquet(data_path, candidate_grid=grid)
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
    warmup_pnl = build_pnl_matrix(dataset, warmup_dates)
    replay_pnl = build_pnl_matrix(dataset, replay_dates)
    candidate_41_pnl = get_candidate_replay_values(dataset, candidate_index=41, replay_dates=replay_dates)
    always_41_total = float(candidate_41_pnl.sum())
    always_41_mean = float(candidate_41_pnl.mean())
    identity_41 = candidate_identity(grid, 41)

    # Warm-up static selectors (no leakage).
    warmup_mean = warmup_pnl.mean(axis=0)
    best_warmup_idx = argmax_with_lowest_index(warmup_mean)
    warmup_std = warmup_pnl.std(axis=0, ddof=0)
    sharpe_scores = warmup_mean / (warmup_std + 1e-8)
    best_sharpe_idx = argmax_with_lowest_index(sharpe_scores)
    oracle_idx = argmax_with_lowest_index(replay_pnl.mean(axis=0))

    shared_manifest = {
        "warmup_start": warmup_dates[0],
        "warmup_end": warmup_dates[-1],
        "replay_start": replay_dates[0],
        "replay_end": replay_dates[-1],
        "replay_day_count": len(replay_dates),
        "candidate_41_identity": asdict(identity_41),
        "always_41_total_pnl": always_41_total,
        "always_41_mean_pnl": always_41_mean,
    }

    candidate_basis = None
    model = None
    initial_posterior = None
    static_surface = None
    warmup_diagnostics = None
    needs_native = any(name.startswith("candidate_a_") for name in strategies)
    if needs_native:
        candidate_basis = build_candidate_basis(grid)
        model, initial_posterior, static_surface, warmup_diagnostics = warmup_candidate_a_state(
            dataset,
            candidate_basis,
            warmup_dates,
            soft_target=config.soft_target,
            static_surface_config=config.static_surface,
            sigma0=config.sigma0,
        )

    results: list[StrategyRunResult] = []
    for name in strategies:
        log.info("Running strategy %s", name)
        if name == "baseline_always_41":
            results.append(
                run_always_candidate_baseline(
                    strategy_name=name,
                    candidate_index=41,
                    grid=grid,
                    replay_dates=replay_dates,
                    replay_pnl=replay_pnl,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra=shared_manifest,
                )
            )
        elif name == "baseline_best_warmup_static":
            results.append(
                run_always_candidate_baseline(
                    strategy_name=name,
                    candidate_index=best_warmup_idx,
                    grid=grid,
                    replay_dates=replay_dates,
                    replay_pnl=replay_pnl,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra={**shared_manifest, "selected_from": "warmup_mean_pnl", "selected_index": best_warmup_idx},
                )
            )
        elif name == "baseline_best_warmup_sharpe":
            results.append(
                run_always_candidate_baseline(
                    strategy_name=name,
                    candidate_index=best_sharpe_idx,
                    grid=grid,
                    replay_dates=replay_dates,
                    replay_pnl=replay_pnl,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra={**shared_manifest, "selected_from": "warmup_sharpe", "selected_index": best_sharpe_idx},
                )
            )
        elif name == "oracle_static_best_replay":
            result = run_always_candidate_baseline(
                strategy_name=name,
                candidate_index=oracle_idx,
                grid=grid,
                replay_dates=replay_dates,
                replay_pnl=replay_pnl,
                output_root=output_root,
                always_41_total=always_41_total,
                always_41_mean=always_41_mean,
                candidate_41_pnl=candidate_41_pnl,
                overwrite=config.overwrite_outputs,
                manifest_extra={
                    **shared_manifest,
                    "selected_from": "replay_mean_pnl_oracle",
                    "selected_index": oracle_idx,
                    "leakage": True,
                    "warning": "oracle_static_best_replay uses replay-period outcomes; not deployable",
                },
            )
            # Relabel family for sorting/separation.
            summary = dict(result.summary)
            summary["strategy_family"] = "oracle_baseline"
            _atomic_write_text(result.run_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))
            results.append(
                StrategyRunResult(
                    strategy_name=result.strategy_name,
                    strategy_family="oracle_baseline",
                    transition_family=result.transition_family,
                    decision_policy=result.decision_policy,
                    run_dir=result.run_dir,
                    daily=result.daily.assign(strategy_family="oracle_baseline"),
                    summary=summary,
                    elapsed_seconds=result.elapsed_seconds,
                )
            )
        elif name == "baseline_trailing_mean_5":
            results.append(
                run_trailing_baseline(
                    strategy_name=name,
                    metric="mean",
                    window=5,
                    grid=grid,
                    warmup_pnl=warmup_pnl,
                    replay_pnl=replay_pnl,
                    replay_dates=replay_dates,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra=shared_manifest,
                )
            )
        elif name == "baseline_trailing_mean_20":
            results.append(
                run_trailing_baseline(
                    strategy_name=name,
                    metric="mean",
                    window=20,
                    grid=grid,
                    warmup_pnl=warmup_pnl,
                    replay_pnl=replay_pnl,
                    replay_dates=replay_dates,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra=shared_manifest,
                )
            )
        elif name == "baseline_trailing_positive_5":
            results.append(
                run_trailing_baseline(
                    strategy_name=name,
                    metric="positive_rate",
                    window=5,
                    grid=grid,
                    warmup_pnl=warmup_pnl,
                    replay_pnl=replay_pnl,
                    replay_dates=replay_dates,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra=shared_manifest,
                )
            )
        elif name == "baseline_trailing_positive_20":
            results.append(
                run_trailing_baseline(
                    strategy_name=name,
                    metric="positive_rate",
                    window=20,
                    grid=grid,
                    warmup_pnl=warmup_pnl,
                    replay_pnl=replay_pnl,
                    replay_dates=replay_dates,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    overwrite=config.overwrite_outputs,
                    manifest_extra=shared_manifest,
                )
            )
        elif name.startswith("candidate_a_"):
            assert candidate_basis is not None
            results.append(
                run_native_strategy(
                    strategy_name=name,
                    config=config,
                    dataset=dataset,
                    candidate_basis=candidate_basis,
                    output_root=output_root,
                    always_41_total=always_41_total,
                    always_41_mean=always_41_mean,
                    candidate_41_pnl=candidate_41_pnl,
                    model=model,
                    initial_posterior=initial_posterior,
                    static_surface=static_surface,
                    warmup_diagnostics=warmup_diagnostics or {},
                )
            )
        else:
            raise ValueError(f"Unhandled strategy: {name}")

    summaries = [r.summary for r in results]
    panel = pd.concat([r.daily for r in results], ignore_index=True)
    selection_diag_rows = []
    for r in results:
        selection_diag_rows.append(
            {
                "strategy_name": r.strategy_name,
                "unique_selected_candidates": r.summary["unique_selected_candidates"],
                "selection_entropy": r.summary["selection_entropy"],
                "effective_selected_candidates": r.summary["effective_selected_candidates"],
                "turnover_count": r.summary["turnover_count"],
                "turnover_rate": r.summary["turnover_rate"],
                "most_selected_candidate": r.summary["most_selected_candidate"],
                "most_selected_candidate_share": r.summary["most_selected_candidate_share"],
                "candidate_41_selected_share": r.summary["candidate_41_selected_share"],
                "candidate_41_delta_total_pnl": r.summary["candidate_41_delta_total_pnl"],
                "bad_switch_count": r.summary.get("bad_switch_count", ""),
                "bad_switch_total_cost": r.summary.get("bad_switch_total_cost", ""),
                "good_switch_count": r.summary.get("good_switch_count", ""),
                "good_switch_total_gain": r.summary.get("good_switch_total_gain", ""),
            }
        )

    calib_rows: list[dict[str, Any]] = []
    for r in results:
        if r.decision_policy in {"probability_best", "maximum_probability_best"} or "probability_best" in r.strategy_name:
            for row in probability_best_bins(r.daily):
                calib_rows.append({"strategy_name": r.strategy_name, **row})

    _write_csv(comparison_dir / "strategy_summary.csv", pd.DataFrame(summaries))
    _atomic_write_text(
        comparison_dir / "strategy_summary.json",
        json.dumps(summaries, indent=2, sort_keys=True, default=str),
    )
    _write_csv(comparison_dir / "strategy_daily_panel.csv", panel)
    _write_csv(comparison_dir / "selection_diagnostics.csv", selection_diag_rows)
    if calib_rows:
        _write_csv(comparison_dir / "probability_best_bins.csv", calib_rows)

    best_non_oracle = max(
        (s for s in summaries if s.get("strategy_family") != "oracle_baseline"),
        key=lambda s: float(s["total_pnl"]),
    )
    readme = "\n".join(
        [
            "# L5.2 Candidate A Policy Matrix",
            "",
            f"- warmup: `{warmup_dates[0]}` → `{warmup_dates[-1]}`",
            f"- replay: `{replay_dates[0]}` → `{replay_dates[-1]}` ({len(replay_dates)} days)",
            f"- always-41 identity: entry={identity_41.entry_percentage}, stop={identity_41.sl_trail_percentage}, config_id={identity_41.config_id}",
            f"- always-41 total PnL: {always_41_total:.4f}",
            f"- best non-oracle strategy: `{best_non_oracle['strategy_name']}` total_pnl={best_non_oracle['total_pnl']:.4f}",
            "",
            "Oracle baselines are leakage diagnostics and are not deployable.",
            "",
        ]
    )
    (comparison_dir / "README.md").write_text(readme, encoding="utf-8")
    _print_terminal_summary(summaries)
    return {
        "output_dir": str(output_root),
        "comparison_dir": str(comparison_dir),
        "warmup_dates": [warmup_dates[0], warmup_dates[-1]],
        "replay_dates": [replay_dates[0], replay_dates[-1]],
        "day_count": len(replay_dates),
        "strategies": [r.strategy_name for r in results],
        "summaries": summaries,
        "best_non_oracle": best_non_oracle,
        "always_41_total_pnl": always_41_total,
        "always_41_mean_pnl": always_41_mean,
        "results": results,
    }
