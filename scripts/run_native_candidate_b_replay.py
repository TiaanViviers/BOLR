#!/usr/bin/env python3
"""CLI for Phase L5.3 native Candidate B historical replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import (
    CrossGroupLogisticConfig,
    OrderedPartitionConfig,
    OrderedPartitionToleranceConfig,
    StaticSurfaceConfig,
)
from bolr.evaluation.native_candidate_b_replay import (
    DEFAULT_SAMPLED_PAIR_BUDGET,
    NativeCandidateBReplayConfig,
    run_native_candidate_b_replay,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native Candidate B historical replay with durable checkpoints.")
    parser.add_argument("--run-name", default="candidate_b_sampled_fixed_probability_best")
    parser.add_argument("--data", "--data-path", dest="data_path", default="data/YM_full.parquet")
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--output-dir", default="outputs/l5_candidate_b_native_replay")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--warmup-start", default=None)
    parser.add_argument("--warmup-end", default=None)
    parser.add_argument("--replay-start", default=None)
    parser.add_argument("--replay-end", default=None)
    parser.add_argument("--warm-up-days", type=int, default=504)
    parser.add_argument("--maximum-days", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=1428)
    parser.add_argument("--variant", choices=("sampled", "exact"), default="sampled")
    parser.add_argument("--sampled-pair-budget", type=int, default=DEFAULT_SAMPLED_PAIR_BUDGET)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--transition", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--decision-policy", default="probability_best", choices=(
        "posterior_mean",
        "probability_best",
        "thompson",
        "thompson_sample_zero",
        "region_medoid",
    ))
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--mc-chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--antithetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rng-seed", type=int, default=20260720)
    parser.add_argument("--rng-stream", type=int, default=1)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=25)
    parser.add_argument("--checkpoint-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-at-pending-day", action="store_true")
    parser.add_argument("--force-restart-after-day", type=int, default=None)
    parser.add_argument("--force-pending-restart-on-day", type=int, default=None)
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--random-walk-variance", type=float, default=0.05)
    parser.add_argument("--relative-tolerance", type=float, default=0.1)
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--alpha-regularization", type=float, default=1.0)
    parser.add_argument("--l5-2-comparison-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cross_group = CrossGroupLogisticConfig(
        sampled_pair_budget=None if args.variant == "exact" else int(args.sampled_pair_budget),
        sampling_seed=int(args.sampling_seed),
        normalize_pair_losses=True,
    )
    partition = OrderedPartitionConfig(
        positive_threshold=float(args.positive_threshold),
        tolerance=OrderedPartitionToleranceConfig(relative_tolerance=float(args.relative_tolerance)),
    )
    config = NativeCandidateBReplayConfig(
        run_name=args.run_name,
        data_path=args.data_path,
        grid_csv=args.grid_csv,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        warmup_start=args.warmup_start,
        warmup_end=args.warmup_end,
        replay_start=args.replay_start,
        replay_end=args.replay_end,
        warm_up_days=args.warm_up_days,
        maximum_days=args.maximum_days,
        candidate_count=args.candidate_count,
        variant=args.variant,
        transition_family=args.transition,
        adaptive_enabled=args.transition == "adaptive",
        random_walk_variance=args.random_walk_variance,
        sigma0=args.sigma0,
        decision_policy=args.decision_policy,
        monte_carlo_sample_count=args.mc_samples,
        monte_carlo_chunk_size=args.mc_chunk_size,
        top_k=args.top_k,
        antithetic=args.antithetic,
        rng_seed=args.rng_seed,
        rng_stream=args.rng_stream,
        checkpoint_every_n_days=args.checkpoint_every_n_days,
        checkpoint_at_end=args.checkpoint_at_end,
        checkpoint_at_pending_day=args.checkpoint_at_pending_day,
        force_restart_after_day=args.force_restart_after_day,
        force_restart_pending_day=args.force_pending_restart_on_day,
        overwrite_outputs=args.overwrite_outputs,
        partition=partition,
        cross_group=cross_group,
        static_surface=StaticSurfaceConfig(regularization=args.alpha_regularization),
        command_line=tuple(sys.argv if argv is None else ["run_native_candidate_b_replay.py", *argv]),
    )
    result = run_native_candidate_b_replay(config)
    if args.l5_2_comparison_dir:
        manifest_path = result.run_dir / "manifest.json"
        if manifest_path.exists():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["l5_2_summary_path"] = args.l5_2_comparison_dir
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = result.summary
    print(f"run_name={result.run_name}")
    print(f"run_dir={result.run_dir}")
    print(f"evaluated_days={summary.get('day_count')}")
    print(f"total_pnl={summary.get('total_pnl')}")
    print(f"mean_pnl={summary.get('mean_pnl')}")
    print(f"mean_regret={summary.get('mean_regret')}")
    print(f"mean_sampled_pair_count={summary.get('mean_sampled_pair_count')}")
    print(f"max_sampled_pair_count={summary.get('max_sampled_pair_count')}")
    print(f"checkpoint_count={summary.get('checkpoint_count')}")
    print(f"forced_ready_restart_passed={result.forced_ready_restart_passed}")
    print(f"forced_pending_restart_passed={result.forced_pending_restart_passed}")
    print(f"total_elapsed_seconds={summary.get('total_elapsed_seconds')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
