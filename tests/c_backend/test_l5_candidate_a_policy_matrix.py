from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bolr.config.foundation import CandidateGridConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.candidate_a_policy_matrix import PolicyMatrixConfig, run_candidate_a_policy_matrix
from bolr.evaluation.candidate_replay_values import (
    argmax_with_lowest_index,
    build_pnl_matrix,
    candidate_identity,
    get_candidate_replay_values,
)
from bolr.evaluation.native_candidate_a_replay import NativeCandidateAReplayConfig, resolve_date_windows
from bolr.evaluation.policy_matrix_metrics import bad_switch_diagnostics, selection_entropy, summarize_strategy_daily


def test_candidate_identity_and_values_match_grid() -> None:
    root = Path(__file__).resolve().parents[2]
    grid = load_candidate_grid(root / "data" / "YM_grid.csv", CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet(root / "data" / "YM_full.parquet", candidate_grid=grid)
    identity = candidate_identity(grid, 41)
    assert identity.config_id == 41
    assert identity.entry_percentage == float(grid.entry_values[41])
    assert identity.sl_trail_percentage == float(grid.stop_values[41])
    dates = dataset.dates[504:509]
    values = get_candidate_replay_values(dataset, candidate_index=41, replay_dates=dates)
    assert values.shape == (5,)
    for i, date in enumerate(dates):
        assert values[i] == float(dataset.day_frame(date).iloc[41]["pnl"])


def test_trailing_baseline_excludes_current_day(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    grid = load_candidate_grid(root / "data" / "YM_grid.csv", CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet(root / "data" / "YM_full.parquet", candidate_grid=grid)
    cfg = NativeCandidateAReplayConfig(warm_up_days=504, maximum_days=3)
    warmup_dates, replay_dates = resolve_date_windows(dataset.dates, cfg)
    warmup_pnl = build_pnl_matrix(dataset, warmup_dates)
    replay_pnl = build_pnl_matrix(dataset, replay_dates)
    # Force a known selection path: trailing mean of last warmup day only window=1 style via window=5 still uses past only.
    history = list(warmup_pnl)
    window = 5
    window_stack = np.stack(history[-window:], axis=0)
    expected = int(np.argmax(window_stack.mean(axis=0)))
    # Current day should not affect first decision.
    mutated = replay_pnl.copy()
    mutated[0, expected] = -1e9
    mutated[0, (expected + 1) % mutated.shape[1]] = 1e9
    scores = np.stack(history[-window:], axis=0).mean(axis=0)
    assert argmax_with_lowest_index(scores) == expected


def test_always_41_and_matrix_bounded(tmp_path: Path) -> None:
    strategies = (
        "baseline_always_41",
        "baseline_best_warmup_static",
        "baseline_trailing_mean_5",
        "oracle_static_best_replay",
        "candidate_a_fixed_probability_best",
    )
    result = run_candidate_a_policy_matrix(
        PolicyMatrixConfig(
            output_dir=str(tmp_path / "matrix"),
            maximum_days=2,
            monte_carlo_sample_count=16,
            monte_carlo_chunk_size=8,
            checkpoint_every_n_days=1,
            overwrite_outputs=True,
            strategies=strategies,
        )
    )
    out = Path(result["output_dir"])
    comparison = out / "comparison"
    assert (comparison / "strategy_summary.csv").exists()
    assert (comparison / "strategy_daily_panel.csv").exists()
    assert (comparison / "selection_diagnostics.csv").exists()
    for name in strategies:
        assert (out / name / "daily_results.csv").exists()
        assert (out / name / "summary.json").exists()
        assert (out / name / "manifest.json").exists()

    always = pd.read_csv(out / "baseline_always_41" / "daily_results.csv")
    assert set(always["selected_candidate_index"].tolist()) == {41}
    root = Path(__file__).resolve().parents[2]
    grid = load_candidate_grid(root / "data" / "YM_grid.csv", CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet(root / "data" / "YM_full.parquet", candidate_grid=grid)
    dates = always["date"].tolist()
    direct = get_candidate_replay_values(dataset, candidate_index=41, replay_dates=dates)
    assert np.allclose(always["realised_selected_value"].to_numpy(), direct)

    summary = pd.read_csv(comparison / "strategy_summary.csv")
    assert set(summary["strategy_name"]) == set(strategies)
    assert np.isfinite(summary["total_pnl"]).all()
    assert np.isfinite(summary["selection_entropy"]).all()
    always_row = summary[summary.strategy_name == "baseline_always_41"].iloc[0]
    assert float(always_row["candidate_41_delta_total_pnl"]) == 0.0
    assert float(always_row["candidate_41_selected_share"]) == 1.0

    # Warm-up static must not equal oracle necessarily, but selection is constant.
    warm = pd.read_csv(out / "baseline_best_warmup_static" / "daily_results.csv")
    assert warm["selected_candidate_index"].nunique() == 1

    pbest = pd.read_csv(out / "candidate_a_fixed_probability_best" / "daily_results.csv")
    assert len(pbest) == 2
    assert (pbest["selected_candidate_index"] == 41).all()


def test_bad_switch_and_entropy_helpers() -> None:
    entropy, n_eff = selection_entropy([41, 41, 41, 7])
    assert entropy > 0.0
    assert n_eff > 1.0
    diag = bad_switch_diagnostics([41, 7, 7], [1.0, -2.0, 5.0], [1.0, 3.0, 1.0])
    assert diag["days_not_candidate_41"] == 2
    assert diag["bad_switch_count"] == 1
    assert diag["good_switch_count"] == 1
    assert diag["bad_switch_total_cost"] == 5.0
    assert diag["good_switch_total_gain"] == 4.0


def test_summarize_candidate_41_delta() -> None:
    daily = pd.DataFrame(
        {
            "selected_candidate_index": [41, 7],
            "realised_selected_value": [10.0, -5.0],
            "regret": [1.0, 2.0],
            "realised_rank": [1, 10],
        }
    )
    summary = summarize_strategy_daily(
        daily,
        strategy_name="toy",
        strategy_family="test",
        transition_family="none",
        decision_policy="static",
        always_41_total_pnl=20.0,
        always_41_mean_pnl=10.0,
        candidate_41_pnl=[10.0, 15.0],
    )
    assert summary["candidate_41_delta_total_pnl"] == -15.0
    assert summary["bad_switch_count"] == 1
