from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bolr.config.foundation import CrossGroupLogisticConfig, OrderedPartitionConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.evaluation.candidate_b_thompson_robustness import (
    ThompsonRobustnessConfig,
    build_daily_delta_rows,
    run_name_for,
    run_thompson_robustness,
)
from bolr.evaluation.native_candidate_b_replay import NativeCandidateBReplayConfig, run_native_candidate_b_replay
from bolr.evaluation.robustness_metrics import parse_int_spec
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
    phi = np.array([[1.0, 0.0], [0.2, 0.8], [-0.3, 0.4], [0.5, -0.1]], dtype=float)[:n]
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


def test_run_name_and_stream_parser() -> None:
    assert run_name_for(stream=2, pair_budget=4096, pair_seed=0, maximum_days=20).endswith("_d20")
    assert parse_int_spec("1:3") == (1, 2, 3)


def test_daily_delta_panel_vs_41() -> None:
    daily = pd.DataFrame(
        {
            "day_index": [0, 1, 2],
            "date": ["2023-01-12", "2023-01-13", "2023-01-16"],
            "selected_candidate_index": [41, 7, 41],
            "selected_config_id": [41, 7, 41],
            "realised_selected_value": [1.0, 3.0, 0.5],
            "realised_rank": [2, 1, 3],
            "regret": [0.1, 0.0, 0.2],
            "selected_probability_best": [0.2, 0.1, 0.3],
            "selected_expected_rank": [5, 2, 4],
        }
    )
    rows = build_daily_delta_rows(
        daily,
        run_id="demo",
        rng_seed=1,
        rng_stream=1,
        pair_budget=4096,
        pair_sampling_seed=0,
        candidate_41_pnl=np.array([1.0, 1.0, 0.5]),
        candidate_a_pnl=np.array([1.0, 1.1, 0.4]),
        oracle_pnl=None,
    )
    assert rows[0]["delta_vs_41"] == 0.0
    assert rows[1]["delta_vs_41"] == 2.0
    assert rows[1]["good_switch_vs_41"] is True
    assert rows[1]["is_non_41_switch"] is True


def test_bounded_thompson_fixture_and_resume(tmp_path: Path) -> None:
    model, basis, static = _tiny_model()
    prior = make_initial_dynamic_prior(basis.shape[1], sigma0=1.0)
    grid = _tiny_grid()
    days = [
        ("2026-01-01", np.array([1.0, 0.2, -0.3, 0.05])),
        ("2026-01-02", np.array([0.1, 0.8, -0.2, 0.15])),
        ("2026-01-03", np.array([-0.4, 0.0, 0.9, 0.2])),
    ]
    # Direct fixture replay for two streams via harness
    for stream in (1, 2):
        run_native_candidate_b_replay(
            NativeCandidateBReplayConfig(
                run_name=f"cb_fixed_thompson_s{stream}_pb4_ps0",
                output_dir=str(tmp_path / "seed_robustness" / "runs"),
                candidate_count=4,
                variant="sampled",
                transition_family="fixed",
                decision_policy="thompson",
                monte_carlo_sample_count=16,
                monte_carlo_chunk_size=4,
                top_k=2,
                rng_seed=11,
                rng_stream=stream,
                checkpoint_every_n_days=1,
                overwrite_outputs=True,
                partition=OrderedPartitionConfig(),
                cross_group=CrossGroupLogisticConfig(sampled_pair_budget=4, sampling_seed=0),
            ),
            grid=grid,
            candidate_basis=basis,
            model=model,
            initial_posterior=prior,
            static_surface=static,
            synthetic_days=days,
        )
    # Analyse existing should load them (uses real YM data path only if running full; here we just check names)
    assert (tmp_path / "seed_robustness" / "runs" / "cb_fixed_thompson_s1_pb4_ps0" / "summary.json").exists()
    assert (tmp_path / "seed_robustness" / "runs" / "cb_fixed_thompson_s2_pb4_ps0" / "daily_results.csv").exists()


def test_dry_run_lists_planned(tmp_path: Path) -> None:
    result = run_thompson_robustness(
        ThompsonRobustnessConfig(
            output_dir=str(tmp_path),
            rng_streams=(1, 2),
            mode="seed_robustness",
            dry_run=True,
            maximum_days=5,
        )
    )
    assert result["dry_run"] is True
    assert len(result["planned_runs"]) == 2
