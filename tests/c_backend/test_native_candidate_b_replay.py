from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bolr.config.foundation import CrossGroupLogisticConfig, OrderedPartitionConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.evaluation.native_candidate_b_replay import (
    NativeCandidateBReplayConfig,
    run_native_candidate_b_replay,
)
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.initialization.static_surface import StaticSurfaceFit
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock


def _tiny_grid(n: int = 4) -> CandidateGrid:
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


def _tiny_model(n: int = 4) -> tuple[CompositeScoreModel, np.ndarray, StaticSurfaceFit]:
    phi = np.array(
        [
            [1.0, 0.0],
            [0.2, 0.8],
            [-0.3, 0.4],
            [0.5, -0.1],
        ],
        dtype=float,
    )[:n]
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


def _base_config(tmp_path: Path, **overrides) -> NativeCandidateBReplayConfig:
    payload = dict(
        run_name="native_b_fixture",
        output_dir=str(tmp_path),
        candidate_count=4,
        warm_up_days=1,
        variant="sampled",
        decision_policy="probability_best",
        monte_carlo_sample_count=16,
        monte_carlo_chunk_size=4,
        top_k=2,
        antithetic=True,
        rng_seed=11,
        rng_stream=2,
        checkpoint_every_n_days=1,
        checkpoint_at_end=True,
        overwrite_outputs=True,
        partition=OrderedPartitionConfig(),
        cross_group=CrossGroupLogisticConfig(sampled_pair_budget=4, sampling_seed=17),
    )
    payload.update(overrides)
    return NativeCandidateBReplayConfig(**payload)


def _synthetic_days() -> list[tuple[str, np.ndarray]]:
    return [
        ("2026-01-01", np.array([1.0, 0.2, -0.3, 0.05])),
        ("2026-01-02", np.array([0.1, 0.8, -0.2, 0.15])),
        ("2026-01-03", np.array([-0.4, 0.0, 0.9, 0.2])),
    ]


def test_native_candidate_b_sampled_replay_writes_outputs(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    result = run_native_candidate_b_replay(
        _base_config(tmp_path),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=_synthetic_days(),
    )
    assert result.summary["day_count"] == 3
    assert (result.run_dir / "daily_results.csv").exists()
    assert (result.run_dir / "summary.json").exists()
    assert (result.run_dir / "manifest.json").exists()
    daily = pd.read_csv(result.run_dir / "daily_results.csv")
    assert "candidate_b_sampled_pair_count" in daily.columns
    assert "candidate_b_possible_pair_count" in daily.columns
    assert np.isfinite(daily["candidate_b_sampled_pair_count"]).all()
    assert (daily["candidate_b_sampled_pair_count"] <= daily["candidate_b_possible_pair_count"]).all()
    assert np.isfinite(result.summary["total_pnl"])


def test_native_candidate_b_exact_replay_completes(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    result = run_native_candidate_b_replay(
        _base_config(
            tmp_path,
            run_name="native_b_exact",
            variant="exact",
            cross_group=CrossGroupLogisticConfig(sampled_pair_budget=None),
        ),
        grid=_tiny_grid(),
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=_synthetic_days(),
    )
    assert result.summary["day_count"] == 3
    daily = pd.DataFrame(result.daily_results)
    assert (daily["candidate_b_sampled_pair_count"] >= 0).all()
    assert np.isfinite(daily["realised_selected_value"]).all()
