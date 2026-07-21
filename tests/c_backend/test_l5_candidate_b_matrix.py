from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bolr.evaluation.candidate_b_policy_matrix import (
    candidate_b_pair_diagnostics,
    enrich_candidate_b_summary,
    load_l5_2_comparison,
    parse_strategy_name,
    selection_diagnostics_row,
)
from bolr.evaluation.policy_matrix_metrics import bad_switch_diagnostics


def test_parse_strategy_name() -> None:
    variant, transition, decision = parse_strategy_name("candidate_b_sampled_fixed_probability_best")
    assert variant == "sampled"
    assert transition == "fixed"
    assert decision == "probability_best"
    variant, transition, decision = parse_strategy_name("candidate_b_sampled_adaptive_thompson")
    assert variant == "sampled"
    assert transition == "adaptive"
    assert decision == "thompson"


def test_bad_switch_diagnostics_vs_41() -> None:
    selected = [41, 10, 41, 7]
    selected_pnl = [1.0, -2.0, 0.5, 3.0]
    c41 = [1.0, 1.0, 0.5, 1.0]
    out = bad_switch_diagnostics(selected, selected_pnl, c41)
    assert out["days_not_candidate_41"] == 2
    assert out["bad_switch_count"] == 1
    assert out["good_switch_count"] == 1
    assert out["bad_switch_total_cost"] == 3.0
    assert out["good_switch_total_gain"] == 2.0


def test_load_l5_2_comparison_missing(tmp_path: Path) -> None:
    summary, daily, note = load_l5_2_comparison(tmp_path / "missing")
    assert summary is None
    assert daily is None
    assert "missing" in note


def test_load_l5_2_comparison_present(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    pd.DataFrame(
        [
            {
                "strategy_name": "baseline_always_41",
                "total_pnl": 10.0,
                "mean_pnl": 1.0,
                "strategy_family": "baseline",
                "transition_family": "none",
                "decision_policy": "static",
            },
            {
                "strategy_name": "candidate_a_fixed_probability_best",
                "total_pnl": 12.0,
                "mean_pnl": 1.2,
                "strategy_family": "candidate_a",
                "transition_family": "fixed",
                "decision_policy": "probability_best",
            },
            {
                "strategy_name": "candidate_a_fixed_posterior_mean",
                "total_pnl": 0.0,
                "mean_pnl": 0.0,
                "strategy_family": "candidate_a",
                "transition_family": "fixed",
                "decision_policy": "posterior_mean",
            },
        ]
    ).to_csv(comparison / "strategy_summary.csv", index=False)
    pd.DataFrame(
        [
            {"strategy_name": "baseline_always_41", "day_index": 0, "selected_candidate_index": 41},
            {"strategy_name": "candidate_a_fixed_probability_best", "day_index": 0, "selected_candidate_index": 41},
            {"strategy_name": "candidate_a_fixed_posterior_mean", "day_index": 0, "selected_candidate_index": 2},
        ]
    ).to_csv(comparison / "strategy_daily_panel.csv", index=False)
    summary, daily, note = load_l5_2_comparison(comparison)
    assert summary is not None and daily is not None
    assert set(summary["strategy_name"]) == {"baseline_always_41", "candidate_a_fixed_probability_best"}
    assert "imported" in note


def test_pair_and_selection_diagnostics() -> None:
    daily = pd.DataFrame(
        {
            "selected_candidate_index": [41, 41, 10, 41],
            "realised_selected_value": [1.0, 0.5, -1.0, 0.2],
            "regret": [0.1, 0.2, 1.5, 0.0],
            "realised_rank": [2, 3, 20, 1],
            "candidate_b_possible_pair_count": [100, 120, 80, 90],
            "candidate_b_sampled_pair_count": [40, 40, 40, 40],
            "candidate_b_pair_sample_rate": [0.4, 0.33, 0.5, 0.44],
            "candidate_b_total_pair_weight": [1.0, 1.0, 1.0, 1.0],
            "candidate_b_r3_count": [2, 1, 3, 2],
            "candidate_b_r2_count": [3, 4, 2, 3],
            "candidate_b_r1_count": [5, 5, 5, 5],
            "candidate_b_r0_count": [0, 0, 0, 0],
            "candidate_b_partition_informative": [True, True, True, False],
            "candidate_b_fallback_used": [False, False, False, True],
        }
    )
    pair = candidate_b_pair_diagnostics(daily)
    assert pair["mean_sampled_pair_count"] == 40.0
    assert pair["fallback_day_count"] == 1
    sel = selection_diagnostics_row(daily, strategy_name="demo")
    assert sel["unique_selected_candidates"] == 2
    enriched = enrich_candidate_b_summary(
        {"total_pnl": 0.7, "mean_pnl": 0.175},
        daily,
        always_41_total=1.0,
        always_41_mean=0.25,
        candidate_a_fixed_pbest_total=2.0,
        candidate_a_fixed_pbest_mean=0.5,
        oracle_total=5.0,
        oracle_mean=1.25,
        candidate_41_pnl=[1.0, 0.5, 0.0, 0.2],
    )
    assert enriched["candidate_41_delta_total_pnl"] == pytest.approx(-0.3)
    assert enriched["candidate_a_fixed_probability_best_delta_total_pnl"] == pytest.approx(-1.3)
    assert enriched["oracle_gap_total_pnl"] == pytest.approx(-4.3)
