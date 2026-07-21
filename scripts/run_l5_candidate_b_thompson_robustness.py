#!/usr/bin/env python3
"""CLI for Phase L5.4 Candidate B fixed Thompson robustness audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.evaluation.candidate_b_thompson_robustness import ThompsonRobustnessConfig, run_thompson_robustness
from bolr.evaluation.native_candidate_b_replay import DEFAULT_SAMPLED_PAIR_BUDGET
from bolr.evaluation.robustness_metrics import parse_int_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate B fixed Thompson robustness audit (L5.4).")
    parser.add_argument("--data", "--data-path", dest="data_path", default="data/YM_full.parquet")
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--output-dir", default="outputs/l5_candidate_b_thompson_robustness")
    parser.add_argument("--l5-2-comparison-dir", default="outputs/l5_candidate_a_policy_matrix/comparison")
    parser.add_argument("--l5-3-comparison-dir", default="outputs/l5_candidate_b_native_replay/comparison")
    parser.add_argument("--warmup-start", default="2021-01-29")
    parser.add_argument("--warmup-end", default="2023-01-11")
    parser.add_argument("--replay-start", default="2023-01-12")
    parser.add_argument("--replay-end", default="2024-10-08")
    parser.add_argument("--maximum-days", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=20260720)
    parser.add_argument("--rng-streams", default="1:30")
    parser.add_argument("--pair-budget", type=int, default=DEFAULT_SAMPLED_PAIR_BUDGET)
    parser.add_argument("--pair-sampling-seed", type=int, default=0)
    parser.add_argument("--pair-budgets", default="2048,4096,8192")
    parser.add_argument("--pair-sampling-seeds", default="0,1,2")
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--mc-chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=25)
    parser.add_argument(
        "--mode",
        default="seed_robustness",
        choices=("seed_robustness", "pair_sampling", "cost_analysis", "split_analysis", "analyse_existing", "all"),
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ThompsonRobustnessConfig(
        data_path=args.data_path,
        grid_csv=args.grid_csv,
        output_dir=args.output_dir,
        l5_2_comparison_dir=args.l5_2_comparison_dir,
        l5_3_comparison_dir=args.l5_3_comparison_dir,
        warmup_start=args.warmup_start,
        warmup_end=args.warmup_end,
        replay_start=args.replay_start,
        replay_end=args.replay_end,
        maximum_days=args.maximum_days,
        rng_seed=args.rng_seed,
        rng_streams=parse_int_spec(args.rng_streams),
        pair_budget=args.pair_budget,
        pair_sampling_seed=args.pair_sampling_seed,
        pair_budgets=parse_int_spec(args.pair_budgets),
        pair_sampling_seeds=parse_int_spec(args.pair_sampling_seeds),
        monte_carlo_sample_count=args.mc_samples,
        monte_carlo_chunk_size=args.mc_chunk_size,
        top_k=args.top_k,
        checkpoint_every_n_days=args.checkpoint_every_n_days,
        mode=args.mode,
        resume_existing=args.resume_existing,
        overwrite_outputs=args.overwrite_outputs,
        dry_run=args.dry_run,
        command_line=tuple(sys.argv if argv is None else ["run_l5_candidate_b_thompson_robustness.py", *argv]),
    )
    result = run_thompson_robustness(config)
    if result.get("dry_run"):
        return 0
    print(f"output_root={result['output_root']}")
    print(f"day_count={result['day_count']}")
    print(f"planned_runs={len(result.get('planned_runs', []))}")
    completed = sum(1 for r in result.get("registry", []) if r.get("status") == "completed")
    failed = sum(1 for r in result.get("registry", []) if r.get("status") == "failed")
    print(f"completed={completed} failed={failed}")
    ss = (result.get("analysis") or {}).get("seed_summary") or {}
    if ss:
        print(f"share_beating_41={ss.get('share_beating_41')}")
        print(f"median_delta_vs_41={ss.get('median_delta_vs_41')}")
        print(f"mean_delta_vs_41={ss.get('mean_delta_vs_41')}")
        print(f"best_stream={ss.get('best_stream')} worst_stream={ss.get('worst_stream')}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
