#!/usr/bin/env python3
"""CLI for Phase L5.2 Candidate A policy/static-baseline matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import SoftTargetConfig, StaticSurfaceConfig
from bolr.evaluation.candidate_a_policy_matrix import (
    ALL_STRATEGIES,
    OPTIONAL_STRATEGIES,
    REQUIRED_STRATEGIES,
    PolicyMatrixConfig,
    run_candidate_a_policy_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run L5.2 Candidate A policy and baseline matrix.")
    parser.add_argument("--data", "--data-path", dest="data_path", default="data/YM_full.parquet")
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--output-dir", default="outputs/l5_candidate_a_policy_matrix")
    parser.add_argument("--warmup-start", default="2021-01-29")
    parser.add_argument("--warmup-end", default="2023-01-11")
    parser.add_argument("--replay-start", default="2023-01-12")
    parser.add_argument("--replay-end", default="2024-10-08")
    parser.add_argument("--warm-up-days", type=int, default=504)
    parser.add_argument("--maximum-days", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=1428)
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--mc-chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--antithetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rng-seed", type=int, default=20260720)
    parser.add_argument("--rng-stream", type=int, default=1)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=25)
    parser.add_argument("--checkpoint-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument(
        "--strategies",
        default=",".join(REQUIRED_STRATEGIES),
        help="Comma-separated strategy names. Use 'all' or 'required'.",
    )
    parser.add_argument("--include-optional", action="store_true", help="Append optional baselines.")
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--random-walk-variance", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=4.0)
    parser.add_argument("--min-scale", type=float, default=1e-6)
    parser.add_argument("--alpha-regularization", type=float, default=1.0)
    return parser


def _parse_strategies(raw: str, include_optional: bool) -> tuple[str, ...]:
    text = raw.strip()
    if text == "required":
        selected = list(REQUIRED_STRATEGIES)
    elif text == "all":
        selected = list(ALL_STRATEGIES)
    else:
        selected = [part.strip() for part in text.split(",") if part.strip()]
    if include_optional:
        for name in OPTIONAL_STRATEGIES:
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PolicyMatrixConfig(
        data_path=args.data_path,
        grid_csv=args.grid_csv,
        output_dir=args.output_dir,
        warmup_start=args.warmup_start,
        warmup_end=args.warmup_end,
        replay_start=args.replay_start,
        replay_end=args.replay_end,
        warm_up_days=args.warm_up_days,
        maximum_days=args.maximum_days,
        candidate_count=args.candidate_count,
        monte_carlo_sample_count=args.mc_samples,
        monte_carlo_chunk_size=args.mc_chunk_size,
        top_k=args.top_k,
        antithetic=args.antithetic,
        rng_seed=args.rng_seed,
        rng_stream=args.rng_stream,
        checkpoint_every_n_days=args.checkpoint_every_n_days,
        checkpoint_at_end=args.checkpoint_at_end,
        overwrite_outputs=args.overwrite_outputs,
        strategies=_parse_strategies(args.strategies, args.include_optional),
        include_optional=False,
        soft_target=SoftTargetConfig(kappa=args.kappa, eta=args.eta, clip=args.clip, min_scale=args.min_scale),
        static_surface=StaticSurfaceConfig(regularization=args.alpha_regularization),
        random_walk_variance=args.random_walk_variance,
        sigma0=args.sigma0,
        command_line=tuple(sys.argv if argv is None else ["run_l5_candidate_a_policy_matrix.py", *argv]),
    )
    result = run_candidate_a_policy_matrix(config)
    print(f"output_dir={result['output_dir']}")
    print(f"day_count={result['day_count']}")
    print(f"strategies={','.join(result['strategies'])}")
    print(f"best_non_oracle={result['best_non_oracle']['strategy_name']}")
    print(f"always_41_total_pnl={result['always_41_total_pnl']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
