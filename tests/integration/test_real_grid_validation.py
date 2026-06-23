import numpy as np

from bolr.config.foundation import SelectedColumnsContextConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.representation.context_basis import SelectedColumnsContextBasis
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.score_design import build_explicit_design, structured_scores, theta_from_matrix
from bolr.representation.tensor_basis import TensorProductBasis


def test_real_grid_supports_design_identity() -> None:
    grid = load_candidate_grid("data/YM_grid.csv")
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

    context_rows = [
        {"volatility": 0.5, "trend": -1.0, "momentum": 0.2},
        {"volatility": 1.5, "trend": 1.0, "momentum": -0.4},
    ]
    psi = SelectedColumnsContextBasis(
        SelectedColumnsContextConfig(columns=("volatility", "trend", "momentum"))
    ).fit(context_rows).transform(context_rows[:1])[0]

    matrix = np.arange(candidate_basis.shape[1] * psi.size, dtype=float).reshape(
        candidate_basis.shape[1],
        psi.size,
        order="F",
    )
    theta = theta_from_matrix(matrix)
    explicit = build_explicit_design(candidate_basis, psi) @ theta
    structured = structured_scores(candidate_basis, matrix, psi)
    assert candidate_basis.shape[0] == 1428
    assert np.allclose(explicit, structured)


def test_real_grid_centered_basis_has_zero_column_sums() -> None:
    grid = load_candidate_grid("data/YM_grid.csv")
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
    assert np.allclose(basis.centered_basis.sum(axis=0), 0.0, atol=1e-10)
