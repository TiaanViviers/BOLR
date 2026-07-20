#!/usr/bin/env python3
"""CLI for Phase L5.1 native Candidate A historical replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import SoftTargetConfig, StaticSurfaceConfig
from bolr.evaluation.native_candidate_a_replay import NativeCandidateAReplayConfig, run_native_candidate_a_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native Candidate A historical replay with durable checkpoints.")
    parser.add_argument("--run-name", default="candidate_a_native_fixed_v1")
    parser.add_argument("--data", "--data-path", dest="data_path", default="data/YM_full.parquet")
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--output-dir", default="outputs/l5_candidate_a_native_fixed")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--warmup-start", default=None)
    parser.add_argument("--warmup-end", default=None)
    parser.add_argument("--replay-start", default=None)
    parser.add_argument("--replay-end", default=None)
    parser.add_argument("--warm-up-days", type=int, default=504)
    parser.add_argument("--maximum-days", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=1428)
    parser.add_argument("--transition", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--decision-policy", default="posterior_mean", choices=(
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
    parser.add_argument("--resume-from-latest", action="store_true")
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--random-walk-variance", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=4.0)
    parser.add_argument("--min-scale", type=float, default=1e-6)
    parser.add_argument("--alpha-regularization", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adaptive_enabled = bool(args.adaptive or args.transition == "adaptive")
    config = NativeCandidateAReplayConfig(
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
        transition_family="adaptive" if adaptive_enabled else "fixed",
        adaptive_enabled=adaptive_enabled,
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
        soft_target=SoftTargetConfig(kappa=args.kappa, eta=args.eta, clip=args.clip, min_scale=args.min_scale),
        static_surface=StaticSurfaceConfig(regularization=args.alpha_regularization),
        command_line=tuple(sys.argv if argv is None else ["run_native_candidate_a_replay.py", *argv]),
        resume_from_latest=args.resume_from_latest,
    )
    result = run_native_candidate_a_replay(config)
    summary = result.summary
    print(f"run_name={result.run_name}")
    print(f"run_dir={result.run_dir}")
    print(f"evaluated_days={summary.get('day_count')}")
    print(f"mean_selected_pnl={summary.get('mean_selected_value_pnl')}")
    print(f"mean_regret={summary.get('mean_regret')}")
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
