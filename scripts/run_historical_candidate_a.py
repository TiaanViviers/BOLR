from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import CandidateGridConfig, HistoricalRunConfig, SoftTargetConfig, SplineAxisConfig, StaticSurfaceConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.prequential_runner import run_historical_candidate_a
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.tensor_basis import TensorProductBasis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--historical-parquet", default="data/YM_full.parquet")
    parser.add_argument("--warm-up-days", type=int, default=504)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--random-walk-variance", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=4.0)
    parser.add_argument("--min-scale", type=float, default=1e-6)
    parser.add_argument("--alpha-regularization", type=float, default=1.0)
    args = parser.parse_args()

    grid = load_candidate_grid(args.grid_csv, CandidateGridConfig())
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(
        grid.entry_values,
        grid.stop_values,
    )
    candidate_basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=6, degree=3),
            stop_basis=SplineAxisConfig(n_basis=8, degree=3),
        )
    ).fit_transform(coordinates).reduced_basis
    dataset = HistoricalDataset.from_parquet(args.historical_parquet, candidate_grid=grid)
    result = run_historical_candidate_a(
        dataset=dataset,
        candidate_basis=candidate_basis,
        config=HistoricalRunConfig(
            warm_up_days=args.warm_up_days,
            sigma0=args.sigma0,
            random_walk_variance=args.random_walk_variance,
            target=SoftTargetConfig(kappa=args.kappa, eta=args.eta, clip=args.clip, min_scale=args.min_scale),
            static_surface=StaticSurfaceConfig(regularization=args.alpha_regularization),
        ),
    )
    print(f"run_id={result.run_id}")
    print(f"run_dir={result.run_dir}")
    print(f"evaluated_days={len(result.predictions)}")
    print(f"mean_selected_pnl={result.predictions['selected_pnl'].mean():.6f}")
    print(f"mean_regret={result.predictions['regret'].mean():.6f}")


if __name__ == "__main__":
    main()
