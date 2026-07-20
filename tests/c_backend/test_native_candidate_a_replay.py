from __future__ import annotations

from pathlib import Path

import numpy as np

from bolr.config.foundation import SoftTargetConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.evaluation.native_candidate_a_replay import (
    NativeCandidateAReplayConfig,
    resolve_date_windows,
    run_native_candidate_a_replay,
)
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.initialization.static_surface import StaticSurfaceFit
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock


def _tiny_grid(n: int = 3) -> CandidateGrid:
    entries = np.linspace(1.0, float(n), n)
    stops = np.linspace(0.1, 0.1 * n, n)
    pairs = [(float(e), float(s)) for e, s in zip(entries, stops)]
    return CandidateGrid(
        config_ids=np.arange(n, dtype=int),
        entry_values=np.asarray([p[0] for p in pairs], dtype=float),
        stop_values=np.asarray([p[1] for p in pairs], dtype=float),
        pair_to_id={p: i for i, p in enumerate(pairs)},
        grid_shape=(n, 1),
    )


def _tiny_model(n: int = 3) -> tuple[CompositeScoreModel, np.ndarray, StaticSurfaceFit]:
    phi = np.array([[1.0, 0.0], [0.2, 0.8], [-0.3, 0.4]], dtype=float)[:n]
    coeffs = np.array([0.1, -0.2], dtype=float)
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, coeffs, {})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    static = StaticSurfaceFit(
        coefficients=coeffs,
        objective=0.0,
        gradient_norm=0.0,
        iterations=1,
        converged=True,
        regularization=1.0,
        diagnostics={"source": "fixture"},
    )
    return model, phi, static


def _base_config(tmp_path: Path, **overrides) -> NativeCandidateAReplayConfig:
    payload = dict(
        run_name="native_a_fixture",
        output_dir=str(tmp_path),
        candidate_count=3,
        warm_up_days=1,
        decision_policy="posterior_mean",
        monte_carlo_sample_count=16,
        monte_carlo_chunk_size=4,
        top_k=2,
        antithetic=True,
        rng_seed=11,
        rng_stream=2,
        checkpoint_every_n_days=1,
        checkpoint_at_end=True,
        overwrite_outputs=True,
        soft_target=SoftTargetConfig(),
    )
    payload.update(overrides)
    return NativeCandidateAReplayConfig(**payload)


def test_resolve_date_windows_maximum_days() -> None:
    dates = tuple(f"2021-01-{i:02d}" for i in range(1, 11))
    config = NativeCandidateAReplayConfig(warm_up_days=3, maximum_days=2)
    warmup, replay = resolve_date_windows(dates, config)
    assert warmup == dates[:3]
    assert replay == dates[3:5]


def test_native_candidate_a_bounded_replay_writes_outputs(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    days = [
        ("2026-01-01", np.array([1.0, 0.2, -0.3])),
        ("2026-01-02", np.array([0.1, 0.8, -0.2])),
        ("2026-01-03", np.array([-0.4, 0.0, 0.9])),
    ]
    result = run_native_candidate_a_replay(
        _base_config(tmp_path),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    run_dir = result.run_dir
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "daily_results.csv").exists()
    assert (run_dir / "checkpoints.jsonl").exists()
    assert len(result.daily_results) == 3
    assert result.summary["day_count"] == 3
    assert "candidate_a_config_hash" in result.manifest
    assert "model_schema_hash" in result.manifest
    assert result.manifest["abi_version"].startswith("1.8")
    assert len(list((run_dir / "checkpoints").glob("*.bolrcp"))) >= 3


def test_native_candidate_a_ready_restart_matches_uninterrupted(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    days = [
        ("2026-01-01", np.array([1.0, 0.2, -0.3])),
        ("2026-01-02", np.array([0.1, 0.8, -0.2])),
        ("2026-01-03", np.array([-0.4, 0.0, 0.9])),
        ("2026-01-04", np.array([0.5, -0.1, 0.2])),
    ]
    direct = run_native_candidate_a_replay(
        _base_config(tmp_path / "direct", run_name="direct", force_restart_after_day=None),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    restarted = run_native_candidate_a_replay(
        _base_config(tmp_path / "restart", run_name="restart", force_restart_after_day=1),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    assert restarted.forced_ready_restart_passed is True
    assert [row["selected_candidate_index"] for row in restarted.daily_results] == [
        row["selected_candidate_index"] for row in direct.daily_results
    ]
    assert [row["realised_selected_value"] for row in restarted.daily_results] == [
        row["realised_selected_value"] for row in direct.daily_results
    ]


def test_native_candidate_a_pending_restart_preserves_decision(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    days = [
        ("2026-01-01", np.array([1.0, 0.2, -0.3])),
        ("2026-01-02", np.array([0.1, 0.8, -0.2])),
        ("2026-01-03", np.array([-0.4, 0.0, 0.9])),
    ]
    direct = run_native_candidate_a_replay(
        _base_config(tmp_path / "direct", run_name="direct"),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    pending = run_native_candidate_a_replay(
        _base_config(tmp_path / "pending", run_name="pending", force_restart_pending_day=1),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    assert pending.forced_pending_restart_passed is True
    assert [row["selected_candidate_index"] for row in pending.daily_results] == [
        row["selected_candidate_index"] for row in direct.daily_results
    ]
    # Decision on the interrupted day must match the uninterrupted twin.
    assert pending.daily_results[1]["selected_candidate_index"] == direct.daily_results[1]["selected_candidate_index"]


def test_native_candidate_a_adaptive_bounded_fixture(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    days = [
        ("2026-01-01", np.array([1.0, 0.2, -0.3])),
        ("2026-01-02", np.array([0.1, 0.8, -0.2])),
    ]
    result = run_native_candidate_a_replay(
        _base_config(
            tmp_path,
            run_name="adaptive",
            transition_family="adaptive",
            adaptive_enabled=True,
            decision_policy="probability_best",
        ),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    assert result.summary["day_count"] == 2
    assert result.manifest["backend_provenance"]["adaptive_enabled"] is True
    assert all(row["phase_after_finish"] == 1 for row in result.daily_results)
