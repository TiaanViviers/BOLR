from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.config.foundation import CandidateGridConfig, SelectedColumnsContextConfig, SoftTargetConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid, load_day_utility_vector
from bolr.numerics.derivatives import parameter_gradient, parameter_hessian, parameter_hvp
from bolr.observations.soft_target_gibbs import log_factor, score_gradient, score_hessian, score_hvp
from bolr.representation.context_basis import SelectedColumnsContextBasis
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.score_design import build_explicit_design, matrix_from_theta, structured_scores, theta_from_matrix
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.targets.soft_target import build_soft_target_observation


def _finite_difference_gradient(func, point: np.ndarray, step: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(point, dtype=float)
    for idx in range(point.size):
        delta = np.zeros_like(point, dtype=float)
        delta[idx] = step
        grad[idx] = (func(point + delta) - func(point - delta)) / (2.0 * step)
    return grad


def _finite_difference_hvp(func, point: np.ndarray, vector: np.ndarray, step: float = 1e-5) -> np.ndarray:
    return (
        _finite_difference_gradient(func, point + step * vector)
        - _finite_difference_gradient(func, point - step * vector)
    ) / (2.0 * step)


def _load_day_utilities(grid_csv: str) -> dict[str, np.ndarray]:
    by_day: dict[str, list[float]] = {}
    with open(grid_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_day.setdefault(row["date"], []).append(float(row["pnl"]))
    return {
        day: np.asarray(values, dtype=float)
        for day, values in sorted(by_day.items())
    }


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
    singular_values = np.linalg.svd(basis.centered_basis, compute_uv=False)
    retained_singular_values = singular_values[singular_values > 1e-10]

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
    score_vector = np.array([0.5, -0.1, 0.3, 0.2])
    score_probe = np.array([0.1, -0.4, 0.2, 1.0])
    score_observation = build_soft_target_observation(
        np.array([0.8, -0.2, 0.5, 1.3]),
        SoftTargetConfig(kappa=0.9, eta=1.1),
    )
    score_fd_grad = _finite_difference_gradient(lambda x: log_factor(x, score_observation), score_probe)
    score_grad_error = float(
        np.max(np.abs(score_gradient(score_probe, score_observation) - score_fd_grad))
    )
    score_fd_hvp = _finite_difference_hvp(lambda x: log_factor(x, score_observation), score_probe, score_vector)
    score_hvp_error = float(
        np.max(np.abs(score_hvp(score_probe, score_vector, score_observation) - score_fd_hvp))
    )
    score_curvature_error = float(
        np.max(np.abs(score_hessian(score_probe, score_observation) @ score_vector - score_fd_hvp))
    )
    parameter_design = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, 1.0, 0.25],
            [-0.3, 0.2, 0.1],
            [0.0, -0.8, 0.9],
        ]
    )
    parameter_theta = np.array([0.2, -0.4, 0.7])
    parameter_vector = np.array([0.4, -0.2, 0.5])
    parameter_observation = build_soft_target_observation(
        np.array([1.0, 0.2, -0.7, 0.4]),
        SoftTargetConfig(kappa=1.4, eta=0.6),
    )
    parameter_scores = parameter_design @ parameter_theta
    parameter_objective = lambda param: log_factor(parameter_design @ param, parameter_observation)
    parameter_fd_grad = _finite_difference_gradient(parameter_objective, parameter_theta)
    parameter_grad_error = float(
        np.max(
            np.abs(
                -parameter_observation.update_weight
                * parameter_gradient(parameter_design, parameter_scores, parameter_observation)
                - parameter_fd_grad
            )
        )
    )
    parameter_fd_hvp = _finite_difference_hvp(parameter_objective, parameter_theta, parameter_vector)
    parameter_hvp_error = float(
        np.max(
            np.abs(
                -parameter_hvp(
                    parameter_design,
                    parameter_scores,
                    parameter_vector,
                    parameter_observation,
                )
                - parameter_fd_hvp
            )
        )
    )
    all_day_utilities = _load_day_utilities(args.grid_csv)
    soft_target_summaries: list[dict[str, float | str | bool]] = []
    for day, day_utilities in all_day_utilities.items():
        day_observation = build_soft_target_observation(
            day_utilities,
            SoftTargetConfig(kappa=1.0, eta=1.0, absolute_tolerance=0.0, relative_tolerance=0.0),
        )
        entropy = float(day_observation.metadata["target_entropy"])
        soft_target_summaries.append(
            {
                "day": day,
                "entropy": entropy,
                "perplexity": math.exp(entropy),
                "max_mass": float(day_observation.target_probabilities.max()),
                "clipping_fraction": float(day_observation.metadata["clipping_fraction"]),
                "utility_scale": float(day_observation.metadata["utility_scale"]),
                "all_irrelevant": bool(day_observation.metadata["all_irrelevant"]),
            }
        )
    soft_target_summaries.sort(key=lambda row: float(row["entropy"]))
    representative_days = {
        "lowest_entropy": soft_target_summaries[0],
        "median_entropy": soft_target_summaries[len(soft_target_summaries) // 2],
        "highest_entropy": soft_target_summaries[-1],
    }

    print(f"grid_shape={grid.grid_shape} n_candidates={grid.n_candidates}")
    print(f"candidate_basis_shape={candidate_basis.shape}")
    print(
        "candidate_basis_rank="
        f"{candidate_basis.shape[1]} raw_width={basis.original_basis.shape[1]}"
    )
    print(f"candidate_basis_largest_singular_value={retained_singular_values[0]:.12f}")
    print(f"candidate_basis_smallest_retained_singular_value={retained_singular_values[-1]:.12f}")
    print(
        "candidate_basis_condition_number="
        f"{retained_singular_values[0] / retained_singular_values[-1]:.12f}"
    )
    print(
        "candidate_basis_max_abs_centered_column_mean="
        f"{np.max(np.abs(basis.centered_basis.mean(axis=0))):.3e}"
    )
    print(f"context_dim={psi.size} design_shape={explicit_design.shape}")
    print(f"max_design_identity_error={max_design_error:.3e}")
    print(f"observation_type={observation.type} update_weight={observation.update_weight:.3f}")
    print(f"log_factor={log_factor(scores, observation):.6f}")
    print(f"score_grad_norm={np.linalg.norm(grad):.6f}")
    print(f"param_grad_norm={np.linalg.norm(param_grad):.6f}")
    print(f"param_hessian_trace={np.trace(param_hess):.6f}")
    print(f"hvp_norm={np.linalg.norm(hvp):.6f}")
    print(f"score_gradient_fd_max_abs_error={score_grad_error:.12e}")
    print(f"score_curvature_fd_max_abs_error={score_curvature_error:.12e}")
    print(f"score_hvp_fd_max_abs_error={score_hvp_error:.12e}")
    print(f"parameter_gradient_fd_max_abs_error={parameter_grad_error:.12e}")
    print(f"parameter_hvp_fd_max_abs_error={parameter_hvp_error:.12e}")
    reconstructed = matrix_from_theta(theta, candidate_basis.shape[1], psi.size)
    print(f"theta_roundtrip_error={np.max(np.abs(reconstructed - interaction)):.3e}")
    for label, summary in representative_days.items():
        print(
            f"{label}_day={summary['day']} entropy={summary['entropy']:.12f} "
            f"perplexity={summary['perplexity']:.12f} max_mass={summary['max_mass']:.12f} "
            f"clipping_fraction={summary['clipping_fraction']:.12f} "
            f"utility_scale={summary['utility_scale']:.12f} "
            f"all_irrelevant={summary['all_irrelevant']}"
        )


if __name__ == "__main__":
    main()
