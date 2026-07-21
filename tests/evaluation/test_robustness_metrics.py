from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bolr.evaluation.robustness_metrics import (
    block_bootstrap_totals,
    break_even_costs,
    cost_adjusted_totals,
    daily_delta_summary,
    max_drawdown,
    parse_int_spec,
)


def test_parse_int_spec_range_and_list() -> None:
    assert parse_int_spec("1:5") == (1, 2, 3, 4, 5)
    assert parse_int_spec("1,3,5") == (1, 3, 5)
    assert parse_int_spec("1:3,10,20") == (1, 2, 3, 10, 20)


def test_cost_sensitivity_formulas() -> None:
    pnl = [10.0, -2.0, 5.0]
    c41 = [1.0, 1.0, 1.0]
    selected = [41, 7, 7]
    out = cost_adjusted_totals(
        strategy_pnl=pnl,
        candidate_41_pnl=c41,
        selected=selected,
        cost_per_day=1.0,
        cost_per_non_41_switch=2.0,
        cost_per_turnover=0.5,
    )
    # strategy costs: 3*1 + 2*2 + 1*0.5 = 3+4+0.5 = 7.5
    assert out["strategy_total_cost"] == pytest.approx(7.5)
    assert out["cost_adjusted_total_pnl"] == pytest.approx(13.0 - 7.5)
    assert out["candidate_41_cost_adjusted_total_pnl"] == pytest.approx(3.0 - 3.0)
    assert out["cost_adjusted_delta_vs_41"] == pytest.approx(5.5 - 0.0)


def test_break_even_non_41_switch() -> None:
    be = break_even_costs(
        raw_total_pnl=100.0,
        candidate_41_total_pnl=80.0,
        day_count=10,
        non_41_days=4,
        turnover_count=5,
    )
    assert be["break_even_cost_per_non_41_switch_vs_41"] == pytest.approx(5.0)
    assert be["break_even_cost_per_turnover_vs_41"] == pytest.approx(4.0)


def test_block_bootstrap_deterministic() -> None:
    deltas = np.linspace(-1.0, 1.5, 40)
    a = block_bootstrap_totals(deltas, block_size=5, n_bootstrap=200, seed=11)
    b = block_bootstrap_totals(deltas, block_size=5, n_bootstrap=200, seed=11)
    assert a["mean_total_delta"] == pytest.approx(b["mean_total_delta"])
    assert a["ci_025_total_delta"] == pytest.approx(b["ci_025_total_delta"])
    assert 0.0 <= a["prob_total_delta_gt_0"] <= 1.0


def test_max_drawdown_and_delta_summary() -> None:
    deltas = [1.0, 1.0, -3.0, 0.5, 0.5]
    dd = max_drawdown(deltas)
    assert dd["max_drawdown"] == pytest.approx(3.0)
    summary = daily_delta_summary(deltas)
    assert summary["total_delta"] == pytest.approx(0.0)
    assert summary["positive_delta_days"] == 4
    assert summary["negative_delta_days"] == 1
