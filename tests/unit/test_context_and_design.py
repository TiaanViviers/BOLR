import numpy as np

from bolr.config.foundation import SelectedColumnsContextConfig
from bolr.representation.context_basis import SelectedColumnsContextBasis
from bolr.representation.score_design import build_explicit_design, structured_scores, theta_from_matrix


def test_selected_columns_context_basis_standardizes_features() -> None:
    rows = [
        {"a": 1.0, "b": 2.0},
        {"a": 3.0, "b": 4.0},
        {"a": 5.0, "b": 6.0},
    ]
    basis = SelectedColumnsContextBasis(SelectedColumnsContextConfig(columns=("a", "b"))).fit(rows)
    transformed = basis.transform(rows)
    assert transformed.shape == (3, 3)
    assert np.allclose(transformed[:, 0], 1.0)
    assert np.allclose(transformed[:, 1:].mean(axis=0), 0.0)


def test_explicit_design_matches_structured_scores() -> None:
    phi = np.array([[1.0, -1.0], [0.5, 0.25]])
    psi = np.array([1.0, -2.0, 0.5])
    matrix = np.array([[0.2, -0.3, 0.7], [1.1, 0.4, -0.2]])
    theta = theta_from_matrix(matrix)
    explicit = build_explicit_design(phi, psi)
    assert np.allclose(explicit @ theta, structured_scores(phi, matrix, psi))
