from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import CandidateGridConfig, SelectedColumnsContextConfig, SoftTargetConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid, load_day_utility_vector
from bolr.numerics.derivatives import parameter_gradient, parameter_hessian, parameter_hvp
from bolr.observations.soft_target_gibbs import log_factor, score_gradient
from bolr.representation.context_basis import SelectedColumnsContextBasis
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.score_design import build_explicit_design, matrix_from_theta, structured_scores, theta_from_matrix
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.targets.soft_target import build_soft_target_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-csv", default="data/YM_grid.csv")
    parser.add_argument("--day", default="2008.01.02")
    args = parser.parse_args()

    grid = load_candidate_grid(args.grid_csv, CandidateGridConfig())
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(
        grid.entry_values,
        grid.stop_values,
    )
    basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=6, degree=3),
            stop_basis=SplineAxisConfig(n_basis=8, degree=3),
        )
    ).fit_transform(coordinates)

    context_rows = [
        {
            "volatility": 1.5,
            "trend": -0.3,
            "momentum": 0.8,
        },
        {
            "volatility": 1.8,
            "trend": 0.2,
            "momentum": 0.4,
        },
    ]
    context_basis = SelectedColumnsContextBasis(
        SelectedColumnsContextConfig(columns=("volatility", "trend", "momentum"))
    ).fit(context_rows)
    psi = context_basis.transform(context_rows[:1])[0]

    candidate_basis = basis.reduced_basis
    interaction = np.arange(candidate_basis.shape[1] * psi.size, dtype=float).reshape(
        candidate_basis.shape[1],
        psi.size,
        order="F",
    )
    theta = theta_from_matrix(interaction)
    explicit_design = build_explicit_design(candidate_basis, psi)
    explicit_scores = explicit_design @ theta
    structured = structured_scores(candidate_basis, interaction, psi)
    max_design_error = float(np.max(np.abs(explicit_scores - structured)))

    utilities = load_day_utility_vector(args.grid_csv, args.day, grid)
    observation = build_soft_target_observation(
        utilities,
        SoftTargetConfig(kappa=1.0, eta=1.0, absolute_tolerance=0.0, relative_tolerance=0.0),
    )
    scores = explicit_scores
    grad = score_gradient(scores, observation)
    param_grad = parameter_gradient(explicit_design, scores, observation)
    param_hess = parameter_hessian(explicit_design, scores, observation)
    hvp = parameter_hvp(explicit_design, scores, np.ones_like(theta), observation)

    print(f"grid_shape={grid.grid_shape} n_candidates={grid.n_candidates}")
    print(f"candidate_basis_shape={candidate_basis.shape}")
    print(f"context_dim={psi.size} design_shape={explicit_design.shape}")
    print(f"max_design_identity_error={max_design_error:.3e}")
    print(f"observation_type={observation.type} update_weight={observation.update_weight:.3f}")
    print(f"log_factor={log_factor(scores, observation):.6f}")
    print(f"score_grad_norm={np.linalg.norm(grad):.6f}")
    print(f"param_grad_norm={np.linalg.norm(param_grad):.6f}")
    print(f"param_hessian_trace={np.trace(param_hess):.6f}")
    print(f"hvp_norm={np.linalg.norm(hvp):.6f}")
    reconstructed = matrix_from_theta(theta, candidate_basis.shape[1], psi.size)
    print(f"theta_roundtrip_error={np.max(np.abs(reconstructed - interaction)):.3e}")


if __name__ == "__main__":
    main()
