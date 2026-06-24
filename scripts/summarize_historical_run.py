from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    daily_metrics = pd.read_parquet(run_dir / "daily_metrics.parquet")
    print(f"evaluated_days={len(predictions)}")
    print(f"cumulative_selected_pnl={predictions['selected_pnl'].sum():.6f}")
    print(f"mean_selected_pnl={predictions['selected_pnl'].mean():.6f}")
    print(f"positive_day_rate={predictions['selected_positive'].mean():.6f}")
    print(f"mean_regret={predictions['regret'].mean():.6f}")
    print(f"avg_newton_iterations={daily_metrics['target_time'].count():.6f}")


if __name__ == "__main__":
    main()
