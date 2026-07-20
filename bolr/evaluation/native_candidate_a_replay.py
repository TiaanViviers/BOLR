"""Native Candidate A historical replay harness (Phase L5.1).

Orchestrates full-period Candidate A replay on the C11 replay engine with
durable BOLRCP01 checkpoints. Model mathematics are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy
from bolr.backend.c_backend import (
    CBackend,
    CCandidateAObservation,
    CReplayEngine,
    make_restore_context,
)
from bolr.config.foundation import (
    AdaptiveTransitionConfig,
    BlockAdaptationConfig,
    BOCPDConfig,
    CandidateGridConfig,
    DecisionPolicyConfig,
    SoftTargetConfig,
    SplineAxisConfig,
    StaticSurfaceConfig,
    SurpriseStandardizerConfig,
    TensorBasisConfig,
)
from bolr.data.candidate_grid import CandidateGrid, load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.metrics import rank_of_selected
from bolr.evaluation.outputs import ensure_run_directory, write_json
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.initialization.static_surface import StaticSurfaceFit, fit_static_surface
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.targets.soft_target import SoftTargetBuilder

REPLAY_PHASE_READY = 1
REPLAY_PHASE_AWAITING_OUTCOME = 2

_DECISION_POLICY_ALIASES = {
    "posterior_mean": "posterior_mean_argmax",
    "posterior_mean_argmax": "posterior_mean_argmax",
    "probability_best": "maximum_probability_best",
    "maximum_probability_best": "maximum_probability_best",
    "thompson": "thompson",
    "thompson_sample_zero": "thompson",
    "region_medoid": "highest_mass_region",
    "highest_mass_region": "highest_mass_region",
}


@dataclass(frozen=True)
class NativeCandidateAReplayConfig:
    run_name: str = "candidate_a_native_fixed_v1"
    data_path: str = "data/YM_full.parquet"
    grid_csv: str = "data/YM_grid.csv"
    output_dir: str = "outputs/l5_candidate_a_native_fixed"
    checkpoint_dir: str | None = None
    warmup_start: str | None = None
    warmup_end: str | None = None
    replay_start: str | None = None
    replay_end: str | None = None
    warm_up_days: int = 504
    maximum_days: int | None = None
    candidate_count: int = 1428
    transition_family: str = "fixed"
    adaptive_enabled: bool = False
    random_walk_variance: float = 0.05
    sigma0: float = 1.0
    decision_policy: str = "posterior_mean"
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
    soft_target: SoftTargetConfig = field(default_factory=SoftTargetConfig)
    static_surface: StaticSurfaceConfig = field(default_factory=StaticSurfaceConfig)
    command_line: tuple[str, ...] = ()
    write_outputs: bool = True
    resume_from_latest: bool = False

    def __post_init__(self) -> None:
        if self.warm_up_days <= 0:
            raise ValueError("warm_up_days must be positive.")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        if self.monte_carlo_sample_count < 0:
            raise ValueError("monte_carlo_sample_count must be non-negative.")
        if self.monte_carlo_chunk_size <= 0:
            raise ValueError("monte_carlo_chunk_size must be positive.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.checkpoint_every_n_days < 0:
            raise ValueError("checkpoint_every_n_days must be non-negative.")
        if self.transition_family not in {"fixed", "adaptive"}:
            raise ValueError("transition_family must be 'fixed' or 'adaptive'.")
        if self.adaptive_enabled and self.transition_family != "adaptive":
            raise ValueError("adaptive_enabled requires transition_family='adaptive'.")
        if self.decision_policy not in _DECISION_POLICY_ALIASES:
            raise ValueError(f"Unsupported decision_policy: {self.decision_policy}")


@dataclass(frozen=True)
class NativeCandidateAReplayResult:
    run_name: str
    run_dir: Path
    daily_results: list[dict[str, Any]]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    checkpoint_events: list[dict[str, Any]]
    forced_ready_restart_passed: bool | None
    forced_pending_restart_passed: bool | None


def resolve_decision_policy(name: str) -> DecisionPolicyConfig:
    family = _DECISION_POLICY_ALIASES[name]
    if family == "highest_mass_region":
        return DecisionPolicyConfig(
            family=family,
            region_selection_statistic="probability_best",
            representative_policy="weighted_medoid",
        )
    if family == "maximum_probability_top_k":
        return DecisionPolicyConfig(family=family, top_k=10)
    return DecisionPolicyConfig(family=family)


def resolve_retention(decision_policy: str) -> str:
    family = _DECISION_POLICY_ALIASES[decision_policy]
    if family == "thompson":
        return "sample_zero"
    return "none"


def build_candidate_basis(grid: CandidateGrid) -> np.ndarray:
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(
        grid.entry_values,
        grid.stop_values,
    )
    return (
        TensorProductBasis(
            TensorBasisConfig(
                entry_basis=SplineAxisConfig(n_basis=6, degree=3),
                stop_basis=SplineAxisConfig(n_basis=8, degree=3),
            )
        )
        .fit_transform(coordinates)
        .reduced_basis
    )


def resolve_date_windows(
    dates: Sequence[str],
    config: NativeCandidateAReplayConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    date_list = list(dates)
    date_index = {date: idx for idx, date in enumerate(date_list)}

    if config.warmup_start is not None or config.warmup_end is not None:
        if config.warmup_start is None or config.warmup_end is None:
            raise ValueError("warmup_start and warmup_end must both be provided.")
        if config.warmup_start not in date_index or config.warmup_end not in date_index:
            raise ValueError("Warm-up date window is outside the historical calendar.")
        warm_start = date_index[config.warmup_start]
        warm_end = date_index[config.warmup_end]
        if warm_end < warm_start:
            raise ValueError("warmup_end precedes warmup_start.")
        warmup_dates = tuple(date_list[warm_start : warm_end + 1])
    else:
        warmup_dates = tuple(date_list[: config.warm_up_days])

    first_replay_idx = date_index[warmup_dates[-1]] + 1
    if config.replay_start is not None:
        if config.replay_start not in date_index:
            raise ValueError("replay_start is outside the historical calendar.")
        first_replay_idx = max(first_replay_idx, date_index[config.replay_start])
    last_replay_idx = len(date_list) - 1
    if config.replay_end is not None:
        if config.replay_end not in date_index:
            raise ValueError("replay_end is outside the historical calendar.")
        last_replay_idx = min(last_replay_idx, date_index[config.replay_end])
    if last_replay_idx < first_replay_idx:
        raise ValueError("Resolved replay date window is empty.")
    replay_dates = date_list[first_replay_idx : last_replay_idx + 1]
    if config.maximum_days is not None:
        replay_dates = replay_dates[: int(config.maximum_days)]
    if not replay_dates:
        raise ValueError("Resolved replay date window is empty.")
    return warmup_dates, tuple(replay_dates)


def _stable_hash(payload: Mapping[str, Any] | Sequence[Any] | str | int | float | bool | None) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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


def _linux_rss_kb() -> dict[str, int | None]:
    current: int | None = None
    peak: int | None = None
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return {"current_rss_kb": None, "max_rss_kb": None}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            peak = int(line.split()[1])
    return {"current_rss_kb": current, "max_rss_kb": peak}


def _entry_stop_indices(grid: CandidateGrid, candidate_index: int) -> tuple[int, int]:
    entry = float(grid.entry_values[candidate_index])
    stop = float(grid.stop_values[candidate_index])
    entries = np.unique(grid.entry_values)
    stops = np.unique(grid.stop_values)
    return int(np.where(entries == entry)[0][0]), int(np.where(stops == stop)[0][0])


def _assert_finite_symmetric(mean: np.ndarray, covariance: np.ndarray) -> None:
    if not np.all(np.isfinite(mean)):
        raise RuntimeError("Posterior mean contains non-finite values.")
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError("Posterior covariance contains non-finite values.")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-8):
        raise RuntimeError("Posterior covariance is not symmetric.")


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


def _summarize_daily(rows: Sequence[Mapping[str, Any]], *, checkpoint_count: int, total_elapsed_seconds: float) -> dict[str, Any]:
    if not rows:
        return {
            "day_count": 0,
            "checkpoint_count": checkpoint_count,
            "total_elapsed_seconds": total_elapsed_seconds,
        }
    selected = np.asarray([float(r["realised_selected_value"]) for r in rows], dtype=float)
    regret = np.asarray([float(r["regret"]) for r in rows], dtype=float)
    ranks = np.asarray([float(r["realised_rank"]) for r in rows], dtype=float)
    day_ms = np.asarray([float(r["elapsed_day_ms"]) for r in rows], dtype=float)
    laplace_iters = np.asarray([float(r["laplace_iterations"]) for r in rows], dtype=float)
    laplace_failures = sum(1 for r in rows if not bool(r["laplace_converged"]))
    return {
        "day_count": len(rows),
        "mean_selected_value_pnl": float(np.mean(selected)),
        "median_selected_value_pnl": float(np.median(selected)),
        "total_selected_value_pnl": float(np.sum(selected)),
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
        "positive_day_rate": float(np.mean(selected > 0.0)),
        "mean_realised_rank": float(np.mean(ranks)),
        "median_realised_rank": float(np.median(ranks)),
        "top_1_hit_rate": float(np.mean(ranks <= 1.0)),
        "top_5_hit_rate": float(np.mean(ranks <= 5.0)),
        "top_10_hit_rate": float(np.mean(ranks <= 10.0)),
        "mean_laplace_iterations": float(np.mean(laplace_iters)),
        "laplace_failure_count": int(laplace_failures),
        "checkpoint_count": int(checkpoint_count),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "mean_day_ms": float(np.mean(day_ms)),
        "max_day_ms": float(np.max(day_ms)),
    }


def _default_adaptive_policy(process_noise: np.ndarray) -> AdaptiveAdditiveTransitionPolicy:
    return AdaptiveAdditiveTransitionPolicy(
        process_noise,
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.05, max_run_length=64),
            blocks=(
                BlockAdaptationConfig(
                    block_name="surface",
                    transition_family="additive",
                    amplitude=1.0,
                    decay=0.0,
                ),
            ),
        ),
    )


def _prepare_run_dir(config: NativeCandidateAReplayConfig) -> Path:
    run_dir = Path(config.output_dir) / config.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not config.overwrite_outputs:
        raise FileExistsError(f"Output directory already exists: {run_dir}. Pass overwrite_outputs=True to replace.")
    ensure_run_directory(run_dir)
    (run_dir / "logs").mkdir(exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def warmup_candidate_a_state(
    dataset: HistoricalDataset,
    candidate_basis: np.ndarray,
    warmup_dates: Sequence[str],
    *,
    soft_target: SoftTargetConfig,
    static_surface_config: StaticSurfaceConfig,
    sigma0: float,
) -> tuple[CompositeScoreModel, GaussianPosterior, StaticSurfaceFit, dict[str, Any]]:
    started = perf_counter()
    builder = SoftTargetBuilder(soft_target)
    observations = [
        builder.build(dataset.day_frame(date)["pnl"].to_numpy(dtype=float), date=date) for date in warmup_dates
    ]
    static_surface = fit_static_surface(
        candidate_basis,
        observations,
        static_surface_config,
        observation_model=SoftTargetObservationModel(),
    )
    prior = make_initial_dynamic_prior(candidate_basis.shape[1], sigma0=sigma0)
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", candidate_basis, static_surface.coefficients, {"fit": "l5_1_warmup"})],
        [DynamicSurfaceBlock("surface", candidate_basis)],
        {},
    )
    mean_hash = hashlib.sha256(np.ascontiguousarray(prior.mean).tobytes()).hexdigest()
    cov_hash = hashlib.sha256(np.ascontiguousarray(prior.covariance).tobytes()).hexdigest()
    diagnostics = {
        "warmup_backend": "python_static_surface",
        "warmup_dates": [warmup_dates[0], warmup_dates[-1]],
        "warmup_day_count": len(warmup_dates),
        "state_dimension": int(candidate_basis.shape[1]),
        "initial_posterior_hash": hashlib.sha256(f"{mean_hash}:{cov_hash}".encode("utf-8")).hexdigest(),
        "initial_covariance_trace": float(np.trace(prior.covariance)),
        "static_surface_converged": bool(static_surface.converged),
        "static_surface_iterations": int(static_surface.iterations),
        "warmup_elapsed_seconds": float(perf_counter() - started),
    }
    return model, prior, static_surface, diagnostics


def run_native_candidate_a_replay(
    config: NativeCandidateAReplayConfig,
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
) -> NativeCandidateAReplayResult:
    """Run a full native Candidate A historical replay with durable checkpoints."""
    log = logger or logging.getLogger("bolr.native_candidate_a_replay")
    project_root = Path(__file__).resolve().parents[2]
    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    grid_csv = Path(config.grid_csv)
    if not grid_csv.is_absolute():
        grid_csv = project_root / grid_csv

    run_started = perf_counter()
    timing: dict[str, float] = {}
    checkpoint_events: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    forced_ready_ok: bool | None = None
    forced_pending_ok: bool | None = None

    prep_started = perf_counter()
    if grid is None:
        grid = load_candidate_grid(grid_csv, CandidateGridConfig())
    if grid.n_candidates != config.candidate_count:
        raise ValueError(f"Expected candidate_count={config.candidate_count}, found {grid.n_candidates}.")

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
    else:
        if dataset is None:
            dataset = HistoricalDataset.from_parquet(data_path, candidate_grid=grid)
        if candidate_basis is None:
            candidate_basis = build_candidate_basis(grid)
        warmup_dates, replay_dates = resolve_date_windows(dataset.dates, config)
        pnl_by_date = None
        data_fingerprint = _data_fingerprint(data_path, dataset)
        if model is None or initial_posterior is None or static_surface is None:
            timing["data_preparation_seconds"] = perf_counter() - prep_started
            model, prior, static_surface, warmup_diag = warmup_candidate_a_state(
                dataset,
                candidate_basis,
                warmup_dates,
                soft_target=config.soft_target,
                static_surface_config=config.static_surface,
                sigma0=config.sigma0,
            )
        else:
            prior = initial_posterior
            warmup_diag = dict(warmup_diagnostics or {})
            warmup_diag.setdefault("warmup_backend", "injected")
            warmup_diag.setdefault("warmup_dates", [warmup_dates[0], warmup_dates[-1]])
            warmup_diag.setdefault("warmup_day_count", len(warmup_dates))
            warmup_diag.setdefault("state_dimension", int(candidate_basis.shape[1]))
            warmup_diag.setdefault("initial_posterior_hash", "injected")
            warmup_diag.setdefault("initial_covariance_trace", float(np.trace(prior.covariance)))
            warmup_diag.setdefault("warmup_elapsed_seconds", 0.0)
            timing["data_preparation_seconds"] = perf_counter() - prep_started
    if "data_preparation_seconds" not in timing:
        timing["data_preparation_seconds"] = perf_counter() - prep_started
    timing["warmup_seconds"] = float(warmup_diag.get("warmup_elapsed_seconds", 0.0))

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
        run_dir = _prepare_run_dir(config)
        checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "logs" / "replay.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(file_handler)
        log.setLevel(logging.INFO)
        checkpoints_jsonl = run_dir / "checkpoints.jsonl"
        if checkpoints_jsonl.exists():
            checkpoints_jsonl.unlink()
    elif needs_checkpoints:
        import tempfile

        checkpoint_dir = Path(tempfile.mkdtemp(prefix="bolr_l5_ckpt_"))
    else:
        checkpoint_dir = Path(".")

    decision_config = resolve_decision_policy(config.decision_policy)
    retention = resolve_retention(config.decision_policy)
    top_k_values = tuple(sorted(k for k in {1, 5, 10, int(config.top_k)} if 1 <= k <= int(config.candidate_count)))
    if not top_k_values:
        top_k_values = (1,)
    process_noise = np.eye(candidate_basis.shape[1], dtype=float) * float(config.random_walk_variance)
    adaptive_policy_py = _default_adaptive_policy(process_noise) if config.transition_family == "adaptive" else None
    fixed_transition = SimpleNamespace(
        family="additive",
        process_noise=process_noise,
        global_discount=0.0,
        block_discount_scales=None,
    )

    backend = CBackend()
    abi = f"{backend.library.lib.bolr_abi_version_major()}.{backend.library.lib.bolr_abi_version_minor()}.{backend.library.lib.bolr_abi_version_patch()}"
    soft_hash = _stable_hash(asdict(config.soft_target))
    decision_hash = _stable_hash(asdict(decision_config))
    transition_hash = _stable_hash(
        {
            "family": config.transition_family,
            "adaptive_enabled": bool(config.adaptive_enabled or config.transition_family == "adaptive"),
            "random_walk_variance": config.random_walk_variance,
            "adaptive": None if adaptive_policy_py is None else adaptive_policy_py.metadata(),
        }
    )
    mc_hash = _stable_hash(
        {
            "sample_count": config.monte_carlo_sample_count,
            "chunk_size": config.monte_carlo_chunk_size,
            "top_k": config.top_k,
            "antithetic": config.antithetic,
            "retention": retention,
        }
    )
    model_hash = int(0)
    state_layout_hash = int(0)
    adaptive_policy_hash: str | None = None

    begin_ms_total = 0.0
    finish_ms_total = 0.0
    checkpoint_ms_total = 0.0

    def write_checkpoint_event(
        engine: CReplayEngine,
        *,
        path: Path,
        phase: int,
        day_index: int,
        date: str,
        decision_id: int | None,
        kind: str,
        rng_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
            "rng_state_metadata": rng_meta,
            "decision_id": decision_id,
        }
        checkpoint_events.append(event)
        if checkpoints_jsonl is not None:
            _append_jsonl(checkpoints_jsonl, event)
        return event

    try:
        with backend.model_artifacts(model, {}) as artifacts:
            model_hash = int(artifacts.model_schema_hash)
            state_layout_hash = int(artifacts.state_layout_hash)
            c_posterior = artifacts.state_from_posterior(prior)
            c_policy = None
            c_adaptive_state = None
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

                        restore_context = make_restore_context(
                            artifacts,
                            adaptive_policy=c_policy,
                        )

                        def reopen_from_checkpoint(path: Path) -> CReplayEngine:
                            return CReplayEngine.read_checkpoint(path, restore_context, adaptive_policy=c_policy, library=backend.library)

                        try:
                            latest_ready: Path | None = None
                            for day_index, date in enumerate(replay_dates):
                                day_started = perf_counter()
                                phase_before = int(engine.phase)
                                if phase_before != REPLAY_PHASE_READY:
                                    raise RuntimeError(f"Engine not READY before day {date}: phase={phase_before}")

                                predictors = None
                                if pnl_by_date is None:
                                    assert dataset is not None
                                    predictors = dataset.get_predictors(date)
                                    if predictors.config_ids.size != config.candidate_count:
                                        raise RuntimeError(f"Unexpected candidate count on {date}.")
                                    if not np.array_equal(predictors.config_ids, grid.config_ids):
                                        raise RuntimeError(f"Non-canonical candidate order on {date}.")
                                else:
                                    if grid.config_ids.size != config.candidate_count:
                                        raise RuntimeError(f"Unexpected candidate count on {date}.")

                                rng_start = ""
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
                                        raise RuntimeError("Thompson policy did not retain sample-zero scores.")

                                pending_checkpoint_path = None
                                if config.checkpoint_at_pending_day or config.force_restart_pending_day == day_index:
                                    pending_checkpoint_path = checkpoint_dir / f"pending_day_{day_index:04d}.bolrcp"
                                    write_checkpoint_event(
                                        engine,
                                        path=pending_checkpoint_path,
                                        phase=REPLAY_PHASE_AWAITING_OUTCOME,
                                        day_index=day_index,
                                        date=date,
                                        decision_id=selected_index,
                                        kind="pending",
                                        rng_meta={"rng_seed": config.rng_seed, "rng_stream": config.rng_stream},
                                    )

                                if config.force_restart_pending_day == day_index:
                                    assert pending_checkpoint_path is not None
                                    selected_before = selected_index
                                    engine.close()
                                    engine = reopen_from_checkpoint(pending_checkpoint_path)
                                    if int(engine.phase) != REPLAY_PHASE_AWAITING_OUTCOME:
                                        raise RuntimeError("Pending restore did not yield AWAITING_OUTCOME.")
                                    if int(engine.pending_selected_index) != selected_before:
                                        raise RuntimeError("Pending restore changed selected candidate.")
                                    forced_pending_ok = True
                                    log.info("Forced pending restart succeeded on day_index=%s date=%s", day_index, date)

                                if pnl_by_date is None:
                                    assert dataset is not None
                                    outcomes = dataset.reveal_outcomes(date)
                                    pnl = np.asarray(outcomes.pnl, dtype=float)
                                else:
                                    pnl = np.asarray(pnl_by_date[date], dtype=float)
                                if pnl.size != config.candidate_count:
                                    raise RuntimeError(f"Realised vector length mismatch on {date}.")
                                if not np.all(np.isfinite(pnl)):
                                    raise RuntimeError(f"Missing/non-finite realised outcomes on {date}.")

                                target, update_weight, target_diag = backend.build_candidate_a_target(pnl, config.soft_target)
                                effective_strength = float(config.soft_target.eta) * float(update_weight)
                                target_effective_mass = float(math.exp(target_diag.target_entropy)) if target_diag.informative else 0.0

                                finish_started = perf_counter()
                                with CCandidateAObservation(
                                    target,
                                    eta=float(config.soft_target.eta),
                                    update_weight=float(update_weight),
                                    library=backend.library,
                                ) as observation:
                                    laplace_diag, finish_diag = engine.finish_day(
                                        artifacts,
                                        observation,
                                        effective_strength=effective_strength,
                                        information_size=float(target.size),
                                        informative=bool(target_diag.informative),
                                    )
                                finish_ms = (perf_counter() - finish_started) * 1000.0
                                finish_ms_total += finish_ms
                                if int(finish_diag.phase_after) != REPLAY_PHASE_READY or int(engine.phase) != REPLAY_PHASE_READY:
                                    raise RuntimeError(f"finish_day did not return READY on {date}.")
                                if int(finish_diag.selected_index) != selected_index:
                                    raise RuntimeError("finish_day altered the already-issued decision.")

                                posterior_mean = engine.posterior_mean(artifacts.state_dimension)
                                posterior_cov = engine.posterior_covariance(artifacts.state_dimension)
                                _assert_finite_symmetric(posterior_mean, posterior_cov)
                                rng_end = ""
                                entry_index, stop_index = _entry_stop_indices(grid, selected_index)
                                realised_selected = float(pnl[selected_index])
                                realised_best = float(np.max(pnl))
                                best_index = int(np.argmax(pnl))
                                realised_rank = rank_of_selected(pnl, selected_index)
                                day_ms = (perf_counter() - day_started) * 1000.0

                                checkpoint_written = False
                                checkpoint_path_str = ""
                                should_ready_ckpt = False
                                if config.checkpoint_every_n_days > 0 and (day_index + 1) % config.checkpoint_every_n_days == 0:
                                    should_ready_ckpt = True
                                if config.force_restart_after_day == day_index:
                                    should_ready_ckpt = True
                                if config.checkpoint_at_end and day_index == len(replay_dates) - 1:
                                    should_ready_ckpt = True

                                if should_ready_ckpt:
                                    ready_path = checkpoint_dir / f"ready_day_{day_index:04d}.bolrcp"
                                    write_checkpoint_event(
                                        engine,
                                        path=ready_path,
                                        phase=REPLAY_PHASE_READY,
                                        day_index=day_index,
                                        date=date,
                                        decision_id=None,
                                        kind="ready",
                                        rng_meta={"rng_seed": config.rng_seed, "rng_stream": config.rng_stream},
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
                                        raise RuntimeError("Ready restore did not yield READY.")
                                    mean_after = engine.posterior_mean(artifacts.state_dimension)
                                    if not np.allclose(mean_before, mean_after, atol=1e-12, rtol=1e-12):
                                        raise RuntimeError("Ready restore changed posterior mean.")
                                    forced_ready_ok = True
                                    log.info("Forced ready restart succeeded after day_index=%s date=%s", day_index, date)

                                row = {
                                    "run_name": config.run_name,
                                    "day_index": day_index,
                                    "date": date,
                                    "phase_before_begin": phase_before,
                                    "phase_after_finish": int(finish_diag.phase_after),
                                    "selected_candidate_index": selected_index,
                                    "selected_entry_index": entry_index,
                                    "selected_stop_index": stop_index,
                                    "decision_policy": decision_config.family,
                                    "selected_score_mean": float(decision.selected_score_mean),
                                    "selected_score_variance": float(decision.selected_score_variance),
                                    "selected_probability_best": float(decision.selected_probability_best),
                                    "selected_probability_top_k": "",
                                    "selected_expected_rank": float(decision.selected_expected_rank),
                                    "selected_region_id": int(decision.selected_region_id),
                                    "realised_selected_value": realised_selected,
                                    "realised_best_value": realised_best,
                                    "realised_rank": realised_rank,
                                    "regret": realised_best - realised_selected,
                                    "selected_positive": bool(realised_selected > 0.0),
                                    "best_candidate_index": best_index,
                                    "candidate_count": int(config.candidate_count),
                                    "candidate_a_informative": bool(target_diag.informative),
                                    "candidate_a_target_entropy": float(target_diag.target_entropy),
                                    "candidate_a_target_max": float(target_diag.target_maximum),
                                    "candidate_a_target_min": float(target_diag.target_minimum),
                                    "candidate_a_target_effective_mass": target_effective_mass,
                                    "candidate_a_effective_strength": effective_strength,
                                    "candidate_a_fallback_used": bool(target_diag.fallback_used),
                                    "laplace_converged": bool(laplace_diag.newton.converged),
                                    "laplace_iterations": int(laplace_diag.newton.iterations),
                                    "posterior_cov_trace": float(laplace_diag.posterior_covariance_trace),
                                    "predictive_cov_trace": float(laplace_diag.prior_covariance_trace),
                                    "rng_draw_count_start": rng_start,
                                    "rng_draw_count_end": rng_end,
                                    "rng_seed": int(config.rng_seed),
                                    "rng_stream": int(config.rng_stream),
                                    "checkpoint_written": checkpoint_written,
                                    "checkpoint_path": checkpoint_path_str,
                                    "elapsed_begin_ms": begin_ms,
                                    "elapsed_finish_ms": finish_ms,
                                    "elapsed_day_ms": day_ms,
                                    "begin_retained_score_sample_count": int(begin_diag.retained_score_sample_count),
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
                                daily_rows.append(row)
                                log.info(
                                    "day=%s index=%s selected=%s pnl=%.6f rank=%s begin_ms=%.1f finish_ms=%.1f",
                                    date,
                                    day_index,
                                    selected_index,
                                    realised_selected,
                                    realised_rank,
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
        log.exception("Native Candidate A replay failed; preserving latest valid checkpoint if any.")
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
    summary = _summarize_daily(daily_rows, checkpoint_count=len(checkpoint_events), total_elapsed_seconds=total_elapsed)
    summary["timing"] = timing
    summary["memory"] = memory
    summary["forced_ready_restart_passed"] = forced_ready_ok
    summary["forced_pending_restart_passed"] = forced_pending_ok
    summary["transition_family"] = config.transition_family
    summary["decision_policy"] = decision_config.family
    summary["mean_selected_value"] = summary.get("mean_selected_value_pnl")
    summary["median_selected_value"] = summary.get("median_selected_value_pnl")
    summary["total_selected_value"] = summary.get("total_selected_value_pnl")

    manifest = {
        "run_name": config.run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "abi_version": abi,
        "checkpoint_format_version": "1.0",
        "checkpoint_magic": "BOLRCP01",
        "data_path": str(data_path),
        "data_fingerprint": data_fingerprint,
        "date_range": {
            "warmup_start": warmup_dates[0],
            "warmup_end": warmup_dates[-1],
            "replay_start": replay_dates[0],
            "replay_end": replay_dates[-1],
            "replay_day_count": len(replay_dates),
        },
        "warmup": warmup_diag,
        "candidate_count": config.candidate_count,
        "state_dimension": int(candidate_basis.shape[1]),
        "model_schema_hash": model_hash,
        "state_layout_hash": state_layout_hash,
        "candidate_grid_hash": hashlib.sha256(np.ascontiguousarray(grid.config_ids).tobytes()).hexdigest(),
        "candidate_a_config_hash": soft_hash,
        "transition_config_hash": transition_hash,
        "decision_config_hash": decision_hash,
        "monte_carlo_config_hash": mc_hash,
        "rng_seed": config.rng_seed,
        "rng_stream": config.rng_stream,
        "checkpoint_schedule": {
            "every_n_days": config.checkpoint_every_n_days,
            "at_end": config.checkpoint_at_end,
            "at_pending_day": config.checkpoint_at_pending_day,
            "force_restart_after_day": config.force_restart_after_day,
            "force_restart_pending_day": config.force_restart_pending_day,
        },
        "backend_provenance": {
            "warmup_backend": warmup_diag["warmup_backend"],
            "replay_backend": "c11_replay_engine",
            "transition_family": config.transition_family,
            "adaptive_enabled": bool(config.adaptive_enabled or config.transition_family == "adaptive"),
            "adaptive_policy_hash": adaptive_policy_hash,
            "decision_policy": decision_config.family,
            "static_surface_objective": float(static_surface.objective),
        },
        "command_line": list(config.command_line) if config.command_line else list(sys.argv),
        "timing": timing,
        "memory": memory,
    }

    if config.write_outputs and run_dir is not None:
        _write_csv(run_dir / "daily_results.csv", daily_rows)
        _atomic_write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
        _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))
        write_json(run_dir / "config.json", asdict(replace(config, soft_target=config.soft_target)))

    return NativeCandidateAReplayResult(
        run_name=config.run_name,
        run_dir=run_dir or Path("."),
        daily_results=daily_rows,
        manifest=manifest,
        summary=summary,
        checkpoint_events=checkpoint_events,
        forced_ready_restart_passed=forced_ready_ok,
        forced_pending_restart_passed=forced_pending_ok,
    )
