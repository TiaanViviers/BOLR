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


def _days() -> list[tuple[str, np.ndarray]]:
    return [
        ("2026-02-01", np.array([1.2, 0.1, -0.5, 0.05])),
        ("2026-02-02", np.array([0.0, 0.9, -0.1, 0.2])),
        ("2026-02-03", np.array([-0.2, 0.3, 1.0, 0.15])),
        ("2026-02-04", np.array([0.4, -0.3, 0.2, 0.8])),
    ]


def _cfg(tmp_path: Path, **overrides) -> NativeCandidateBReplayConfig:
    payload = dict(
        run_name="native_b_restart",
        output_dir=str(tmp_path),
        candidate_count=4,
        variant="sampled",
        decision_policy="posterior_mean",
        monte_carlo_sample_count=16,
        monte_carlo_chunk_size=4,
        top_k=2,
        rng_seed=29,
        rng_stream=4,
        checkpoint_every_n_days=1,
        checkpoint_at_end=True,
        overwrite_outputs=True,
        partition=OrderedPartitionConfig(),
        cross_group=CrossGroupLogisticConfig(sampled_pair_budget=4, sampling_seed=17),
    )
    payload.update(overrides)
    return NativeCandidateBReplayConfig(**payload)


def test_candidate_b_ready_and_pending_restart_match_uninterrupted(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    grid = _tiny_grid()
    days = _days()

    baseline = run_native_candidate_b_replay(
        _cfg(tmp_path / "base", run_name="base"),
        grid=grid,
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    restarted = run_native_candidate_b_replay(
        _cfg(
            tmp_path / "restart",
            run_name="restart",
            force_restart_after_day=1,
            force_restart_pending_day=2,
        ),
        grid=grid,
        candidate_basis=basis,
        model=model,
        initial_posterior=prior,
        static_surface=static,
        synthetic_days=days,
    )
    assert restarted.forced_ready_restart_passed is True
    assert restarted.forced_pending_restart_passed is True
    base_daily = pd.DataFrame(baseline.daily_results)
    rest_daily = pd.DataFrame(restarted.daily_results)
    assert base_daily["selected_candidate_index"].tolist() == rest_daily["selected_candidate_index"].tolist()
    assert base_daily["candidate_b_sampled_pair_count"].tolist() == rest_daily["candidate_b_sampled_pair_count"].tolist()
    assert base_daily["candidate_b_possible_pair_count"].tolist() == rest_daily["candidate_b_possible_pair_count"].tolist()
    assert np.allclose(
        base_daily["realised_selected_value"].to_numpy(dtype=float),
        rest_daily["realised_selected_value"].to_numpy(dtype=float),
    )
