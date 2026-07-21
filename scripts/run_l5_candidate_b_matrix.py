#!/usr/bin/env python3
"""CLI for Phase L5.3 Candidate B strategy matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.evaluation.candidate_b_policy_matrix import (
    REQUIRED_STRATEGIES,
    CandidateBMatrixConfig,
    run_candidate_b_matrix,
)
from bolr.evaluation.native_candidate_b_replay import DEFAULT_SAMPLED_PAIR_BUDGET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Candidate B native replay matrix and L5.2 comparison.")
    parser.add_argument("--data", "--data-path", dest="data_path", default="data/YM_full.parquet")
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--output-dir", default="outputs/l5_candidate_b_native_replay")
    parser.add_argument("--l5-2-comparison-dir", default="outputs/l5_candidate_a_policy_matrix/comparison")
    parser.add_argument("--variant", choices=("sampled", "exact"), default="sampled")
    parser.add_argument("--sampled-pair-budget", type=int, default=DEFAULT_SAMPLED_PAIR_BUDGET)
    parser.add_argument("--maximum-days", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--mc-chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rng-seed", type=int, default=20260720)
    parser.add_argument("--rng-stream", type=int, default=1)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=25)
    parser.add_argument("--include-fixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adaptive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strategies", default=None, help="Comma-separated strategy names.")
    parser.add_argument("--overwrite-outputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.strategies:
        strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    else:
        strategies = REQUIRED_STRATEGIES
    config = CandidateBMatrixConfig(
        data_path=args.data_path,
        grid_csv=args.grid_csv,
        output_dir=args.output_dir,
        l5_2_comparison_dir=args.l5_2_comparison_dir,
        maximum_days=args.maximum_days,
        variant=args.variant,
        sampled_pair_budget=args.sampled_pair_budget,
        monte_carlo_sample_count=args.mc_samples,
        monte_carlo_chunk_size=args.mc_chunk_size,
        top_k=args.top_k,
        rng_seed=args.rng_seed,
        rng_stream=args.rng_stream,
        checkpoint_every_n_days=args.checkpoint_every_n_days,
        include_fixed=args.include_fixed,
        include_adaptive=args.include_adaptive,
        strategies=strategies,
        overwrite_outputs=args.overwrite_outputs,
        command_line=tuple(sys.argv if argv is None else ["run_l5_candidate_b_matrix.py", *argv]),
    )
    result = run_candidate_b_matrix(config)
    print(f"comparison_dir={result['comparison_dir']}")
    print(f"day_count={result['day_count']}")
    print(f"l5_2_note={result['l5_2_note']}")
    print(f"blocked={len(result['blocked'])}")
    return 0 if not result["blocked"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
