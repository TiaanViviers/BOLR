import numpy as np

from bolr.model.penalties import (
    context_matrix_penalty,
    difference_matrix,
    difference_penalty,
    project_penalty,
    tensor_product_penalty,
)


def test_first_and_second_difference_penalties_have_expected_nullspaces() -> None:
    r1 = difference_penalty(5, 1)
    r2 = difference_penalty(5, 2)
    constant = np.ones(5)
    affine = np.arange(5, dtype=float)
    assert np.allclose(r1.matrix @ constant, 0.0)
    assert np.allclose(r2.matrix @ constant, 0.0)
    assert np.allclose(r2.matrix @ affine, 0.0)
    assert r1.nullity() == 1
    assert r2.nullity() == 2


def test_tensor_penalty_and_projection_preserve_quadratic_form() -> None:
    raw = tensor_product_penalty(3, 4, entry_order=2, stop_order=2)
    lift = np.eye(12)[:, :11]
    projected = project_penalty(raw, lift)
    beta = np.linspace(-0.3, 0.7, 11)
    assert np.isclose(beta @ projected.matrix @ beta, (lift @ beta) @ raw.matrix @ (lift @ beta))
    assert projected.is_psd()


def test_context_matrix_penalty_matches_matrix_form() -> None:
    candidate = difference_penalty(3, 1)
    rm = np.diag([2.0, 0.5])
    penalty = context_matrix_penalty(candidate, rm, candidate_weight=1.3, context_weight=0.7, ridge=0.1)
    b = np.arange(6, dtype=float).reshape((3, 2), order="F")
    vec = b.reshape(-1, order="F")
    matrix_form = 1.3 * np.trace(b.T @ candidate.matrix @ b) + 0.7 * np.trace(b @ rm @ b.T) + 0.1 * np.sum(b * b)
    vector_form = vec @ penalty.matrix @ vec
    assert np.isclose(matrix_form, vector_form)

