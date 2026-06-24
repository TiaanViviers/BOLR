from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import OrderedPartitionConfig, OrderedPartitionToleranceConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-parquet", default="data/YM_full.parquet")
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--output-dir", default="outputs/candidate_b_tolerance_audit")
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HistoricalDataset.from_parquet(args.historical_parquet, load_candidate_grid(args.grid_csv))
    builder = OrderedPartitionBuilder(
        OrderedPartitionConfig(
            tolerance=OrderedPartitionToleranceConfig(
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
            )
        )
    )
    rows = []
    for date in dataset.dates:
        utilities = dataset.day_frame(date)["pnl"].to_numpy(dtype=float)
        observation = builder.build(utilities, date=date)
        rows.append({"date": date, **observation.metadata})
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_dir / "daily_group_diagnostics.parquet", index=False)
    summary = {}
    for column in ["high_group_size", "middle_group_size", "low_group_size", "possible_pair_count", "largest_upper_partition", "partition_complexity_proxy"]:
        series = frame[column].to_numpy(dtype=float)
        summary[column] = {
            "min": float(np.min(series)),
            "median": float(np.median(series)),
            "mean": float(np.mean(series)),
            "p90": float(np.percentile(series, 90)),
            "p95": float(np.percentile(series, 95)),
            "p99": float(np.percentile(series, 99)),
            "max": float(np.max(series)),
        }
    summary.update(
        {
            "empty_middle_frequency": float(np.mean(frame["middle_group_size"] == 0)),
            "one_group_frequency": float(np.mean(frame["group_count"] == 1)),
            "all_irrelevant_frequency": float(np.mean(frame["all_irrelevant"])),
            "reduced_weight_frequency": float(np.mean(frame["observation_type"] == "REDUCED_WEIGHT")),
            "no_update_frequency": float(np.mean(frame["observation_type"] == "NO_UPDATE")),
        }
    )
    (output_dir / "config.json").write_text(json.dumps({
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "days": len(dataset.dates),
    }, indent=2, sort_keys=True))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = ["# Candidate B Tolerance Audit", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    (output_dir / "summary.md").write_text("\n".join(lines))
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
