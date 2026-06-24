import numpy as np

from bolr.config.foundation import CandidateGridConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.graph_residual import build_graph_residual_basis
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.tensor_basis import TensorProductBasis


def test_graph_residual_basis_is_centered_orthonormal_and_smooth_orthogonal() -> None:
    grid = load_candidate_grid("data/YM_grid.csv", CandidateGridConfig())
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(grid.entry_values, grid.stop_values)
    candidate_basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=6, degree=3),
            stop_basis=SplineAxisConfig(n_basis=8, degree=3),
        )
    ).fit_transform(coordinates).reduced_basis
    graph = build_canonical_grid_graph(grid)
    residual = build_graph_residual_basis(graph, candidate_basis, residual_dimension=8)
    assert residual.basis.shape == (1428, 8)
    assert np.allclose(residual.basis.T @ residual.basis, np.eye(8), atol=1e-8)
    assert np.max(np.abs(np.sum(residual.basis, axis=0))) < 1e-8
    assert np.max(np.abs(candidate_basis.T @ residual.basis)) < 1e-8
    eig_projected = residual.basis.T @ (graph.laplacian @ residual.basis)
    assert np.allclose(eig_projected, np.diag(residual.eigenvalues), atol=1e-6)
