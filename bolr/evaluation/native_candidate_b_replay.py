"""Native Candidate B historical replay harness (Phase L5.3).

Orchestrates sampled (or exact) Candidate B replay on the C11 replay engine.
Model mathematics are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from bolr.backend.c_backend import CBackend, CCandidateBSampledObservation, CReplayEngine, make_restore_context
from bolr.config.foundation import (
    CandidateGridConfig,
    CrossGroupLogisticConfig,
    OrderedPartitionConfig,
    OrderedPartitionToleranceConfig,
    SoftTargetConfig,
    StaticSurfaceConfig,
)
from bolr.data.candidate_grid import CandidateGrid, load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.candidate_replay_values import candidate_identity
from bolr.evaluation.metrics import rank_of_selected
from bolr.evaluation.native_candidate_a_replay import (
    REPLAY_PHASE_AWAITING_OUTCOME,
    REPLAY_PHASE_READY,
    _assert_finite_symmetric,
    _default_adaptive_policy,
    _entry_stop_indices,
    _git_commit,
    _linux_rss_kb,
    _stable_hash,
    build_candidate_basis,
    resolve_date_windows,
    resolve_decision_policy,
    resolve_retention,
)
from bolr.evaluation.outputs import ensure_run_directory
from bolr.initialization.static_surface import StaticSurfaceFit
from bolr.model.composite import CompositeScoreModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.ordered_partition import OrderedPartitionBuilder

# Default sampled budget for full historical replay (exact is too dense for 1428 candidates).
DEFAULT_SAMPLED_PAIR_BUDGET = 4096


@dataclass(frozen=True)
class NativeCandidateBReplayConfig:
    run_name: str = "candidate_b_sampled_fixed_probability_best"
    data_path: str = "data/YM_full.parquet"
    grid_csv: str = "data/YM_grid.csv"
    output_dir: str = "outputs/l5_candidate_b_native_replay"
    checkpoint_dir: str | None = None
    warmup_start: str | None = "2021-01-29"
    warmup_end: str | None = "2023-01-11"
    replay_start: str | None = "2023-01-12"
    replay_end: str | None = "2024-10-08"
    warm_up_days: int = 504
    maximum_days: int | None = None
    candidate_count: int = 1428
    variant: str = "sampled"  # sampled | exact
    transition_family: str = "fixed"
    adaptive_enabled: bool = False
    random_walk_variance: float = 0.05
    sigma0: float = 1.0
    decision_policy: str = "probability_best"
    monte_carlo_sample_count: int = 512
    monte_carlo_chunk_size: int = 64
    top_k: int = 10
    antithetic: bool = True
    rng_seed: int = 20260720
    rng_stream: int = 1
    checkpoint_every_n_days: int = 25
    checkpoint_at_end: bool = True
    checkpoint_at_pending_day: bool = False
    force_restart_after_day: int | None = None
    force_restart_pending_day: int | None = None
    overwrite_outputs: bool = False
    partition: OrderedPartitionConfig = field(
        default_factory=lambda: OrderedPartitionConfig(
            tolerance=OrderedPartitionToleranceConfig(relative_tolerance=0.1),
        )
    )
    cross_group: CrossGroupLogisticConfig = field(
        default_factory=lambda: CrossGroupLogisticConfig(
            sampled_pair_budget=DEFAULT_SAMPLED_PAIR_BUDGET,
            sampling_seed=0,
            normalize_pair_losses=True,
        )
    )
    static_surface: StaticSurfaceConfig = field(default_factory=StaticSurfaceConfig)
    soft_target: SoftTargetConfig = field(default_factory=SoftTargetConfig)
    command_line: tuple[str, ...] = ()
    write_outputs: bool = True
    strategy_name: str | None = None
    strategy_family: str = "candidate_b_native"

    def __post_init__(self) -> None:
        if self.variant not in {"sampled", "exact"}:
            raise ValueError("variant must be 'sampled' or 'exact'.")
        if self.variant == "sampled" and self.cross_group.sampled_pair_budget is None:
            raise ValueError("sampled variant requires cross_group.sampled_pair_budget.")
        if self.variant == "exact" and self.cross_group.sampled_pair_budget is not None:
            raise ValueError("exact variant requires sampled_pair_budget=None.")
        if self.transition_family not in {"fixed", "adaptive"}:
            raise ValueError("transition_family must be fixed or adaptive.")


@dataclass(frozen=True)
class NativeCandidateBReplayResult:
    run_name: str
    run_dir: Path
    daily_results: list[dict[str, Any]]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    checkpoint_events: list[dict[str, Any]]
    forced_ready_restart_passed: bool | None
    forced_pending_restart_passed: bool | None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _data_fingerprint(path: Path, dataset: HistoricalDataset) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(dataset.n_days).encode("utf-8"))
    digest.update(dataset.dates[0].encode("utf-8"))
    digest.update(dataset.dates[-1].encode("utf-8"))
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "n_rows": int(dataset.metadata.n_rows),
        "n_dates": int(dataset.n_days),
        "start_date": dataset.dates[0],
        "end_date": dataset.dates[-1],
        "fingerprint": digest.hexdigest(),
    }


def _group_label_counts(labels: Sequence[str], sizes: Sequence[int]) -> dict[str, int]:
    mapping = {"high": "r3", "middle": "r2", "low": "r1"}
    out = {"r3": 0, "r2": 0, "r1": 0, "r0": 0}
    for label, size in zip(labels, sizes):
        key = mapping.get(str(label), "r0")
        out[key] = int(size)
    return out


def warmup_candidate_b_state(
    dataset: HistoricalDataset,
    candidate_basis: np.ndarray,
    warmup_dates: Sequence[str],
    *,
    partition_config: OrderedPartitionConfig,
    cross_group_config: CrossGroupLogisticConfig,
    static_surface_config: StaticSurfaceConfig,
    sigma0: float,
    soft_target: SoftTargetConfig | None = None,
) -> tuple[CompositeScoreModel, GaussianPosterior, StaticSurfaceFit, dict[str, Any]]:
    """Warm-start with the same SoftTarget static surface used in L5.1/L5.2.

    Candidate B observations are used only during replay updates. Sharing the
    SoftTarget warm-up keeps A/B comparisons on a common prior surface and avoids
    the prohibitive cost of fitting a static surface under dense pairwise B losses.
    """
    del partition_config, cross_group_config  # replay-only; preserved in call signature for API clarity
    from bolr.evaluation.native_candidate_a_replay import warmup_candidate_a_state

    model, prior, static_surface, diagnostics = warmup_candidate_a_state(
        dataset,
        candidate_basis,
        warmup_dates,
        soft_target=soft_target or SoftTargetConfig(),
        static_surface_config=static_surface_config,
        sigma0=sigma0,
    )
    diagnostics = dict(diagnostics)
    diagnostics["warmup_backend"] = "python_static_surface_soft_target_shared_with_l5_1"
    diagnostics["warmup_note"] = (
        "L5.3 uses SoftTarget warm-up identical to L5.1/L5.2; Candidate B applies only at finish_day."
    )
    return model, prior, static_surface, diagnostics


def run_native_candidate_b_replay(
    config: NativeCandidateBReplayConfig,
    *,
    dataset: HistoricalDataset | None = None,
    candidate_basis: np.ndarray | None = None,
    grid: CandidateGrid | None = None,
    model: CompositeScoreModel | None = None,
    initial_posterior: GaussianPosterior | None = None,
    static_surface: StaticSurfaceFit | None = None,
    warmup_diagnostics: dict[str, Any] | None = None,
    synthetic_days: Sequence[tuple[str, np.ndarray]] | None = None,
    logger: logging.Logger | None = None,
) -> NativeCandidateBReplayResult:
    log = logger or logging.getLogger("bolr.native_candidate_b_replay")
    project_root = Path(__file__).resolve().parents[2]
    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    grid_csv = Path(config.grid_csv)
    if not grid_csv.is_absolute():
        grid_csv = project_root / grid_csv

    run_started = perf_counter()
    timing: dict[str, float] = {
        "partition_construction_seconds": 0.0,
        "pair_sampling_seconds": 0.0,
        "observation_build_seconds": 0.0,
    }
    checkpoint_events: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    forced_ready_ok: bool | None = None
    forced_pending_ok: bool | None = None
    pnl_by_date: dict[str, np.ndarray] | None

    prep_started = perf_counter()
    if grid is None:
        grid = load_candidate_grid(grid_csv, CandidateGridConfig())
    if grid.n_candidates != config.candidate_count:
        raise ValueError(f"Expected candidate_count={config.candidate_count}, found {grid.n_candidates}.")

    from bolr.evaluation.native_candidate_a_replay import NativeCandidateAReplayConfig

    if synthetic_days is not None:
        replay_dates = tuple(date for date, _ in synthetic_days)
        warmup_dates = ("synthetic-warmup-start", "synthetic-warmup-end")
        pnl_by_date = {date: np.asarray(values, dtype=float) for date, values in synthetic_days}
        for date, values in pnl_by_date.items():
            if values.size != config.candidate_count:
                raise ValueError(f"Synthetic day {date} has unexpected candidate count.")
        if candidate_basis is None or model is None or initial_posterior is None or static_surface is None:
            raise ValueError("synthetic_days requires model, initial_posterior, static_surface, and candidate_basis.")
        prior = initial_posterior
        warmup_diag = dict(warmup_diagnostics or {})
        warmup_diag.setdefault("warmup_backend", "injected")
        warmup_diag.setdefault("warmup_dates", list(warmup_dates))
        warmup_diag.setdefault("warmup_day_count", 0)
        warmup_diag.setdefault("state_dimension", int(candidate_basis.shape[1]))
        warmup_diag.setdefault("initial_posterior_hash", "injected")
        warmup_diag.setdefault("initial_covariance_trace", float(np.trace(prior.covariance)))
        warmup_diag.setdefault("warmup_elapsed_seconds", 0.0)
        data_fingerprint = {
            "path": "synthetic",
            "size_bytes": 0,
            "n_rows": int(config.candidate_count * len(replay_dates)),
            "n_dates": len(replay_dates),
            "start_date": replay_dates[0],
            "end_date": replay_dates[-1],
            "fingerprint": _stable_hash({"synthetic": True, "days": list(replay_dates)}),
        }
        timing["data_preparation_seconds"] = perf_counter() - prep_started
    else:
        if dataset is None:
            dataset = HistoricalDataset.from_parquet(data_path, candidate_grid=grid)
        if candidate_basis is None:
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
        pnl_by_date = None
        data_fingerprint = _data_fingerprint(data_path, dataset)
        timing["data_preparation_seconds"] = perf_counter() - prep_started
        if model is None or initial_posterior is None or static_surface is None:
            model, prior, static_surface, warmup_diag = warmup_candidate_b_state(
                dataset,
                candidate_basis,
                warmup_dates,
                partition_config=config.partition,
                cross_group_config=config.cross_group,
                static_surface_config=config.static_surface,
                sigma0=config.sigma0,
                soft_target=config.soft_target,
            )
        else:
            prior = initial_posterior
            warmup_diag = dict(warmup_diagnostics or {})
            warmup_diag.setdefault("warmup_backend", "injected")
            warmup_diag.setdefault("warmup_elapsed_seconds", 0.0)
            warmup_diag.setdefault("state_dimension", int(candidate_basis.shape[1]))
            warmup_diag.setdefault("initial_posterior_hash", "injected")
            warmup_diag.setdefault("initial_covariance_trace", float(np.trace(prior.covariance)))
            warmup_diag.setdefault("warmup_dates", [warmup_dates[0], warmup_dates[-1]])
            warmup_diag.setdefault("warmup_day_count", len(warmup_dates))
    timing["warmup_seconds"] = float(warmup_diag.get("warmup_elapsed_seconds", 0.0))

    strategy_name = config.strategy_name or config.run_name
    run_dir: Path | None = None
    checkpoints_jsonl: Path | None = None
    needs_checkpoints = bool(
        config.checkpoint_every_n_days > 0
        or config.checkpoint_at_end
        or config.checkpoint_at_pending_day
        or config.force_restart_after_day is not None
        or config.force_restart_pending_day is not None
    )
    if config.write_outputs:
        run_dir = Path(config.output_dir) / config.run_name
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
        if run_dir.exists() and any(run_dir.iterdir()) and not config.overwrite_outputs:
            raise FileExistsError(f"Output directory exists: {run_dir}")
        ensure_run_directory(run_dir)
        (run_dir / "logs").mkdir(exist_ok=True)
        checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "logs" / "replay.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(file_handler)
        log.setLevel(logging.INFO)
        checkpoints_jsonl = run_dir / "checkpoints.jsonl"
        if checkpoints_jsonl.exists():
            checkpoints_jsonl.unlink()
    elif needs_checkpoints:
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="bolr_l53_ckpt_"))
    else:
        checkpoint_dir = Path(".")

    decision_config = resolve_decision_policy(config.decision_policy)
    retention = resolve_retention(config.decision_policy)
    top_k_values = tuple(sorted(k for k in {1, 5, 10, int(config.top_k)} if 1 <= k <= int(config.candidate_count))) or (1,)
    process_noise = np.eye(candidate_basis.shape[1], dtype=float) * float(config.random_walk_variance)
    adaptive_policy_py = _default_adaptive_policy(process_noise) if config.transition_family == "adaptive" else None
    fixed_transition = SimpleNamespace(
        family="additive",
        process_noise=process_noise,
        global_discount=0.0,
        block_discount_scales=None,
    )
    partition_builder = OrderedPartitionBuilder(config.partition)

    backend = CBackend()
    abi = f"{backend.library.lib.bolr_abi_version_major()}.{backend.library.lib.bolr_abi_version_minor()}.{backend.library.lib.bolr_abi_version_patch()}"
    begin_ms_total = 0.0
    finish_ms_total = 0.0
    checkpoint_ms_total = 0.0

    def write_checkpoint_event(engine: CReplayEngine, *, path: Path, phase: int, day_index: int, date: str, decision_id: int | None, kind: str) -> None:
        nonlocal checkpoint_ms_total
        started = perf_counter()
        size_report = engine.checkpoint_size_report()
        engine.write_checkpoint(path, durable=True, replace_existing=True)
        elapsed_ms = (perf_counter() - started) * 1000.0
        checkpoint_ms_total += elapsed_ms
        event = {
            "checkpoint_path": str(path),
            "kind": kind,
            "phase": int(phase),
            "day_index": int(day_index),
            "date": date,
            "encoded_size_bytes": int(size_report.total_bytes),
            "write_elapsed_ms": float(elapsed_ms),
            "rng_state_metadata": {"rng_seed": config.rng_seed, "rng_stream": config.rng_stream},
            "decision_id": decision_id,
        }
        checkpoint_events.append(event)
        if checkpoints_jsonl is not None:
            _append_jsonl(checkpoints_jsonl, event)

    try:
        with backend.model_artifacts(model, {}) as artifacts:
            model_hash = int(artifacts.model_schema_hash)
            state_layout_hash = int(artifacts.state_layout_hash)
            c_posterior = artifacts.state_from_posterior(prior)
            c_policy = None
            c_adaptive_state = None
            adaptive_policy_hash = None
            try:
                if adaptive_policy_py is not None:
                    c_policy = backend.adaptive_policy(adaptive_policy_py, model.layout)
                    c_adaptive_state = backend.adaptive_state(c_policy)
                    adaptive_policy_hash = str(backend.library.lib.bolr_adaptive_policy_configuration_hash(c_policy._require_open()))
                with backend.rng(seed=config.rng_seed, stream=config.rng_stream) as rng:
                    with backend.decision_policy(decision_config) as decision_policy_handle:
                        if adaptive_policy_py is None:
                            engine = backend.replay_engine_fixed(c_posterior, fixed_transition, rng)
                        else:
                            engine = backend.replay_engine_adaptive(c_posterior, c_policy, c_adaptive_state, rng)
                        restore_context = make_restore_context(artifacts, adaptive_policy=c_policy)

                        def reopen_from_checkpoint(path: Path) -> CReplayEngine:
                            return CReplayEngine.read_checkpoint(path, restore_context, adaptive_policy=c_policy, library=backend.library)

                        try:
                            latest_ready: Path | None = None
                            for day_index, date in enumerate(replay_dates):
                                day_started = perf_counter()
                                phase_before = int(engine.phase)
                                if phase_before != REPLAY_PHASE_READY:
                                    raise RuntimeError(f"Engine not READY before {date}: phase={phase_before}")
                                predictors = None
                                if pnl_by_date is None:
                                    assert dataset is not None
                                    predictors = dataset.get_predictors(date)
                                    if predictors.config_ids.size != config.candidate_count:
                                        raise RuntimeError(f"Unexpected candidate count on {date}.")
                                    if not np.array_equal(predictors.config_ids, grid.config_ids):
                                        raise RuntimeError(f"Non-canonical candidate order on {date}.")

                                begin_started = perf_counter()
                                decision, begin_diag = engine.begin_day(
                                    artifacts,
                                    decision_policy_handle,
                                    ranking_sample_count=int(config.monte_carlo_sample_count),
                                    chunk_size=int(config.monte_carlo_chunk_size),
                                    top_k_values=top_k_values,
                                    antithetic=bool(config.antithetic),
                                    retention=retention,
                                )
                                begin_ms = (perf_counter() - begin_started) * 1000.0
                                begin_ms_total += begin_ms
                                selected_index = int(decision.selected_index)
                                if selected_index < 0 or selected_index >= config.candidate_count:
                                    raise RuntimeError(f"Selected index out of range on {date}: {selected_index}")
                                if int(engine.phase) != REPLAY_PHASE_AWAITING_OUTCOME:
                                    raise RuntimeError(f"Expected AWAITING_OUTCOME after begin_day on {date}.")
                                if decision_config.family == "thompson" and config.monte_carlo_sample_count > 0:
                                    if int(begin_diag.retained_score_sample_count) < 1:
                                        raise RuntimeError("Thompson retention did not keep sample zero.")

                                pending_path = None
                                if config.checkpoint_at_pending_day or config.force_restart_pending_day == day_index:
                                    pending_path = checkpoint_dir / f"pending_day_{day_index:04d}.bolrcp"
                                    write_checkpoint_event(
                                        engine,
                                        path=pending_path,
                                        phase=REPLAY_PHASE_AWAITING_OUTCOME,
                                        day_index=day_index,
                                        date=date,
                                        decision_id=selected_index,
                                        kind="pending",
                                    )
                                if config.force_restart_pending_day == day_index:
                                    assert pending_path is not None
                                    selected_before = selected_index
                                    engine.close()
                                    engine = reopen_from_checkpoint(pending_path)
                                    if int(engine.phase) != REPLAY_PHASE_AWAITING_OUTCOME:
                                        raise RuntimeError("Pending restore phase invalid.")
                                    if int(engine.pending_selected_index) != selected_before:
                                        raise RuntimeError("Pending restore changed selected candidate.")
                                    forced_pending_ok = True

                                if pnl_by_date is None:
                                    assert dataset is not None
                                    outcomes = dataset.reveal_outcomes(date)
                                    pnl = np.asarray(outcomes.pnl, dtype=float)
                                else:
                                    pnl = np.asarray(pnl_by_date[date], dtype=float)
                                if pnl.size != config.candidate_count or not np.all(np.isfinite(pnl)):
                                    raise RuntimeError(f"Invalid realised vector on {date}.")

                                partition_started = perf_counter()
                                partition_obs = partition_builder.build(pnl, date=date)
                                timing["partition_construction_seconds"] += perf_counter() - partition_started
                                group_counts = _group_label_counts(partition_obs.group_labels, partition_obs.group_sizes)
                                if sum(partition_obs.group_sizes) != config.candidate_count:
                                    raise RuntimeError(f"Partition group sizes do not sum to candidate count on {date}.")

                                obs_started = perf_counter()
                                if config.variant == "exact":
                                    obs_handle = backend.candidate_b_exact_observation(
                                        partition_obs,
                                        normalize_pair_losses=bool(config.cross_group.normalize_pair_losses),
                                    )
                                    pair_meta = {
                                        "possible_pair_count": int(partition_obs.metadata["possible_pair_count"]),
                                        "used_pair_count": int(partition_obs.metadata["possible_pair_count"]),
                                        "duplicate_sample_count": 0,
                                        "update_weight": float(partition_obs.update_weight),
                                        "normalize_pair_losses": bool(config.cross_group.normalize_pair_losses),
                                        "sampling_seed": None,
                                        "pair_budget": None,
                                        "pair_weights": np.empty(0),
                                    }
                                else:
                                    sample_started = perf_counter()
                                    pair_meta = backend.materialize_candidate_b_pairs(partition_obs, config.cross_group)
                                    timing["pair_sampling_seconds"] += perf_counter() - sample_started
                                    used = int(pair_meta["used_pair_count"])
                                    possible = int(pair_meta["possible_pair_count"])
                                    if used < 0 or used > possible:
                                        raise RuntimeError(f"Invalid sampled pair count on {date}: {used}/{possible}")
                                    winners = np.asarray(pair_meta["winner_indices"], dtype=np.int64)
                                    losers = np.asarray(pair_meta["loser_indices"], dtype=np.int64)
                                    if winners.size and (int(winners.min()) < 0 or int(winners.max()) >= config.candidate_count):
                                        raise RuntimeError(f"Winner indices out of range on {date}.")
                                    if losers.size and (int(losers.min()) < 0 or int(losers.max()) >= config.candidate_count):
                                        raise RuntimeError(f"Loser indices out of range on {date}.")
                                    obs_handle = CCandidateBSampledObservation(
                                        int(pair_meta["candidate_count"]),
                                        pair_meta["winner_indices"],
                                        pair_meta["loser_indices"],
                                        pair_meta["pair_weights"],
                                        update_weight=float(pair_meta["update_weight"]),
                                        possible_pair_count=int(pair_meta["possible_pair_count"]),
                                        duplicate_sample_count=int(pair_meta["duplicate_sample_count"]),
                                        normalize_pair_losses=bool(pair_meta["normalize_pair_losses"]),
                                        library=backend.library,
                                    )
                                timing["observation_build_seconds"] += perf_counter() - obs_started

                                finish_started = perf_counter()
                                with obs_handle:
                                    c_diag = obs_handle.diagnostics()
                                    laplace_diag, finish_diag = engine.finish_day(
                                        artifacts,
                                        obs_handle,
                                        effective_strength=float(partition_obs.update_weight),
                                        information_size=float(max(int(pair_meta["possible_pair_count"]), 1)),
                                        informative=not bool(partition_obs.all_irrelevant),
                                    )
                                finish_ms = (perf_counter() - finish_started) * 1000.0
                                finish_ms_total += finish_ms
                                if int(finish_diag.phase_after) != REPLAY_PHASE_READY or int(engine.phase) != REPLAY_PHASE_READY:
                                    raise RuntimeError(f"finish_day did not return READY on {date}.")
                                if int(finish_diag.selected_index) != selected_index:
                                    raise RuntimeError("finish_day altered already-issued decision.")

                                posterior_mean = engine.posterior_mean(artifacts.state_dimension)
                                posterior_cov = engine.posterior_covariance(artifacts.state_dimension)
                                _assert_finite_symmetric(posterior_mean, posterior_cov)

                                entry_index, stop_index = _entry_stop_indices(grid, selected_index)
                                identity = candidate_identity(grid, selected_index)
                                realised_selected = float(pnl[selected_index])
                                realised_best = float(np.max(pnl))
                                best_index = int(np.argmax(pnl))
                                realised_rank = rank_of_selected(pnl, selected_index)
                                day_ms = (perf_counter() - day_started) * 1000.0

                                checkpoint_written = False
                                checkpoint_path_str = ""
                                should_ready = False
                                if config.checkpoint_every_n_days > 0 and (day_index + 1) % config.checkpoint_every_n_days == 0:
                                    should_ready = True
                                if config.force_restart_after_day == day_index:
                                    should_ready = True
                                if config.checkpoint_at_end and day_index == len(replay_dates) - 1:
                                    should_ready = True
                                if should_ready:
                                    ready_path = checkpoint_dir / f"ready_day_{day_index:04d}.bolrcp"
                                    write_checkpoint_event(
                                        engine,
                                        path=ready_path,
                                        phase=REPLAY_PHASE_READY,
                                        day_index=day_index,
                                        date=date,
                                        decision_id=None,
                                        kind="ready",
                                    )
                                    checkpoint_written = True
                                    checkpoint_path_str = str(ready_path)
                                    latest_ready = ready_path
                                if config.force_restart_after_day == day_index:
                                    assert latest_ready is not None
                                    mean_before = posterior_mean.copy()
                                    engine.close()
                                    engine = reopen_from_checkpoint(latest_ready)
                                    if int(engine.phase) != REPLAY_PHASE_READY:
                                        raise RuntimeError("Ready restore phase invalid.")
                                    if not np.allclose(mean_before, engine.posterior_mean(artifacts.state_dimension), atol=1e-12, rtol=1e-12):
                                        raise RuntimeError("Ready restore changed posterior mean.")
                                    forced_ready_ok = True

                                pair_weights = np.asarray(pair_meta.get("pair_weights", np.empty(0)), dtype=float)
                                used_pairs = int(c_diag.used_pair_count)
                                possible_pairs = int(c_diag.possible_pair_count)
                                sample_rate = float(used_pairs / possible_pairs) if possible_pairs > 0 else 0.0
                                daily_rows.append(
                                    {
                                        "strategy_name": strategy_name,
                                        "strategy_family": config.strategy_family,
                                        "observation_model": "candidate_b",
                                        "candidate_b_variant": config.variant,
                                        "transition_family": config.transition_family,
                                        "decision_policy": decision_config.family,
                                        "day_index": day_index,
                                        "date": date,
                                        "phase_before_begin": phase_before,
                                        "phase_after_finish": int(finish_diag.phase_after),
                                        "selected_candidate_index": selected_index,
                                        "selected_entry_index": entry_index,
                                        "selected_stop_index": stop_index,
                                        "selected_config_id": identity.config_id,
                                        "realised_selected_value": realised_selected,
                                        "realised_best_value": realised_best,
                                        "best_candidate_index": best_index,
                                        "realised_rank": realised_rank,
                                        "regret": realised_best - realised_selected,
                                        "selected_positive": bool(realised_selected > 0.0),
                                        "candidate_count": int(config.candidate_count),
                                        "selected_score_mean": float(decision.selected_score_mean),
                                        "selected_score_variance": float(decision.selected_score_variance),
                                        "selected_probability_best": float(decision.selected_probability_best),
                                        "selected_probability_top_k": "",
                                        "selected_expected_rank": float(decision.selected_expected_rank),
                                        "posterior_cov_trace": float(laplace_diag.posterior_covariance_trace),
                                        "predictive_cov_trace": float(laplace_diag.prior_covariance_trace),
                                        "laplace_converged": bool(laplace_diag.newton.converged),
                                        "laplace_iterations": int(laplace_diag.newton.iterations),
                                        "rng_seed": int(config.rng_seed),
                                        "rng_stream": int(config.rng_stream),
                                        "checkpoint_written": checkpoint_written,
                                        "checkpoint_path": checkpoint_path_str,
                                        "elapsed_begin_ms": begin_ms,
                                        "elapsed_finish_ms": finish_ms,
                                        "elapsed_day_ms": day_ms,
                                        "candidate_b_partition_informative": not bool(partition_obs.all_irrelevant),
                                        "candidate_b_group_count": int(partition_obs.metadata["group_count"]),
                                        "candidate_b_r3_count": group_counts["r3"],
                                        "candidate_b_r2_count": group_counts["r2"],
                                        "candidate_b_r1_count": group_counts["r1"],
                                        "candidate_b_r0_count": group_counts["r0"],
                                        "candidate_b_possible_pair_count": possible_pairs,
                                        "candidate_b_sampled_pair_count": used_pairs,
                                        "candidate_b_pair_sample_rate": sample_rate,
                                        "candidate_b_total_pair_weight": float(pair_weights.sum()) if pair_weights.size else float("nan"),
                                        "candidate_b_mean_pair_weight": float(pair_weights.mean()) if pair_weights.size else float("nan"),
                                        "candidate_b_max_pair_weight": float(pair_weights.max()) if pair_weights.size else float("nan"),
                                        "candidate_b_temperature": "",
                                        "candidate_b_effective_strength": float(partition_obs.update_weight),
                                        "candidate_b_sampling_seed": pair_meta.get("sampling_seed", ""),
                                        "candidate_b_sampling_deterministic": config.variant == "sampled",
                                        "candidate_b_fallback_used": bool(partition_obs.all_irrelevant),
                                        "candidate_b_duplicate_sample_count": int(c_diag.duplicate_sample_count),
                                        "adaptive_applied": bool(finish_diag.adaptive_applied),
                                        "surprise": "",
                                        "standardised_surprise": "",
                                        "change_probability": "",
                                        "expected_run_length": "",
                                        "max_multiplier": "",
                                        "mean_multiplier": "",
                                        "reset_scheduled": "",
                                        "reset_applied": "",
                                    }
                                )
                                log.info(
                                    "day=%s index=%s selected=%s pnl=%.4f pairs=%s/%s begin_ms=%.1f finish_ms=%.1f",
                                    date,
                                    day_index,
                                    selected_index,
                                    realised_selected,
                                    used_pairs,
                                    possible_pairs,
                                    begin_ms,
                                    finish_ms,
                                )
                        finally:
                            engine.close()
            finally:
                if c_adaptive_state is not None:
                    c_adaptive_state.close()
                if c_policy is not None:
                    c_policy.close()
                c_posterior.close()
    except Exception:
        log.exception("Native Candidate B replay failed.")
        raise

    total_elapsed = perf_counter() - run_started
    timing.update(
        {
            "begin_day_seconds": begin_ms_total / 1000.0,
            "finish_day_seconds": finish_ms_total / 1000.0,
            "checkpoint_write_seconds": checkpoint_ms_total / 1000.0,
            "total_run_seconds": total_elapsed,
        }
    )
    memory = _linux_rss_kb()
    pnl = np.asarray([r["realised_selected_value"] for r in daily_rows], dtype=float)
    ranks = np.asarray([r["realised_rank"] for r in daily_rows], dtype=float)
    sampled_pairs = np.asarray([r["candidate_b_sampled_pair_count"] for r in daily_rows], dtype=float)
    summary = {
        "strategy_name": strategy_name,
        "strategy_family": config.strategy_family,
        "observation_model": "candidate_b",
        "candidate_b_variant": config.variant,
        "transition_family": config.transition_family,
        "decision_policy": decision_config.family,
        "day_count": len(daily_rows),
        "total_pnl": float(pnl.sum()) if daily_rows else 0.0,
        "mean_pnl": float(pnl.mean()) if daily_rows else float("nan"),
        "median_pnl": float(np.median(pnl)) if daily_rows else float("nan"),
        "mean_regret": float(np.mean([r["regret"] for r in daily_rows])) if daily_rows else float("nan"),
        "mean_realised_rank": float(ranks.mean()) if daily_rows else float("nan"),
        "top_1_hit_rate": float(np.mean(ranks <= 1)) if daily_rows else float("nan"),
        "top_5_hit_rate": float(np.mean(ranks <= 5)) if daily_rows else float("nan"),
        "top_10_hit_rate": float(np.mean(ranks <= 10)) if daily_rows else float("nan"),
        "laplace_failure_count": int(sum(1 for r in daily_rows if not r["laplace_converged"])),
        "mean_laplace_iterations": float(np.mean([r["laplace_iterations"] for r in daily_rows])) if daily_rows else float("nan"),
        "checkpoint_count": len(checkpoint_events),
        "total_elapsed_seconds": total_elapsed,
        "mean_day_ms": float(np.mean([r["elapsed_day_ms"] for r in daily_rows])) if daily_rows else float("nan"),
        "max_day_ms": float(np.max([r["elapsed_day_ms"] for r in daily_rows])) if daily_rows else float("nan"),
        "mean_sampled_pair_count": float(sampled_pairs.mean()) if daily_rows else float("nan"),
        "median_sampled_pair_count": float(np.median(sampled_pairs)) if daily_rows else float("nan"),
        "max_sampled_pair_count": float(sampled_pairs.max()) if daily_rows else float("nan"),
        "forced_ready_restart_passed": forced_ready_ok,
        "forced_pending_restart_passed": forced_pending_ok,
        "timing": timing,
        "memory": memory,
    }
    cross_group_payload = {
        "normalize_pair_losses": config.cross_group.normalize_pair_losses,
        "sampled_pair_budget": config.cross_group.sampled_pair_budget,
        "sampled_with_replacement": config.cross_group.sampled_with_replacement,
        "sampling_seed": config.cross_group.sampling_seed,
    }
    partition_payload = {
        "positive_threshold": config.partition.positive_threshold,
        "all_irrelevant_policy": config.partition.all_irrelevant_policy,
        "reduced_weight": config.partition.reduced_weight,
        "tolerance": asdict(config.partition.tolerance),
    }
    manifest = {
        "run_name": config.run_name,
        "strategy_name": strategy_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "abi_version": abi,
        "checkpoint_format_version": "1.0",
        "data_path": str(data_path),
        "data_fingerprint": data_fingerprint,
        "warmup_range": [warmup_dates[0], warmup_dates[-1]],
        "replay_range": [replay_dates[0], replay_dates[-1]],
        "candidate_count": config.candidate_count,
        "state_dimension": int(candidate_basis.shape[1]),
        "model_schema_hash": model_hash,
        "state_layout_hash": state_layout_hash,
        "candidate_grid_hash": hashlib.sha256(np.ascontiguousarray(grid.config_ids).tobytes()).hexdigest(),
        "observation_model": "candidate_b",
        "candidate_b_variant": config.variant,
        "candidate_b_config_hash": _stable_hash(cross_group_payload),
        "partition_config_hash": _stable_hash(partition_payload),
        "pair_sampling_config_hash": _stable_hash(cross_group_payload),
        "partition_config": partition_payload,
        "pair_sampling_config": cross_group_payload,
        "transition_family": config.transition_family,
        "decision_policy": decision_config.family,
        "monte_carlo_config_hash": _stable_hash(
            {
                "sample_count": config.monte_carlo_sample_count,
                "chunk_size": config.monte_carlo_chunk_size,
                "top_k": config.top_k,
                "antithetic": config.antithetic,
                "retention": retention,
            }
        ),
        "rng_seed": config.rng_seed,
        "rng_stream": config.rng_stream,
        "checkpoint_schedule": {
            "every_n_days": config.checkpoint_every_n_days,
            "at_end": config.checkpoint_at_end,
            "force_restart_after_day": config.force_restart_after_day,
            "force_restart_pending_day": config.force_restart_pending_day,
        },
        "backend_provenance": {
            "warmup_backend": warmup_diag.get("warmup_backend"),
            "replay_backend": "c11_replay_engine",
            "adaptive_policy_hash": adaptive_policy_hash,
            "static_surface_objective": float(static_surface.objective),
        },
        "warmup": warmup_diag,
        "command_line": list(config.command_line) if config.command_line else list(sys.argv),
        "timing": timing,
        "memory": memory,
    }
    if config.write_outputs and run_dir is not None:
        _write_csv(run_dir / "daily_results.csv", daily_rows)
        _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))
        _atomic_write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return NativeCandidateBReplayResult(
        run_name=config.run_name,
        run_dir=run_dir or Path("."),
        daily_results=daily_rows,
        manifest=manifest,
        summary=summary,
        checkpoint_events=checkpoint_events,
        forced_ready_restart_passed=forced_ready_ok,
        forced_pending_restart_passed=forced_pending_ok,
    )
