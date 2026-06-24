import numpy as np

from bolr.data.candidate_grid import CandidateGrid
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.graph_residual import ProjectedLaplacianOperator, build_smooth_subspace_projector


def test_smooth_projector_is_idempotent_and_removes_constant_and_smooth_span() -> None:
    candidate_basis = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.2],
            [0.0, 1.0],
            [-0.3, 0.4],
            [-0.5, -0.2],
            [-0.7, -1.4],
        ]
    )
    projector = build_smooth_subspace_projector(candidate_basis)
    constant = np.ones(candidate_basis.shape[0])
    random_vector = np.array([0.3, -0.1, 0.5, 0.9, -0.2, 0.4])
    projected_once = projector.project(random_vector)
    projected_twice = projector.project(projected_once)
    assert np.allclose(projector.project(constant), 0.0, atol=1e-10)
    assert np.allclose(projector.project(candidate_basis[:, 0]), 0.0, atol=1e-10)
    assert np.allclose(projected_once, projected_twice, atol=1e-10)
    explicit = projector.explicit_projector()
    assert np.allclose(explicit, explicit.T, atol=1e-10)
    assert np.allclose(explicit @ explicit, explicit, atol=1e-10)
    assert np.allclose(projected_once, explicit @ random_vector, atol=1e-10)


def test_projected_laplacian_operator_matches_explicit_small_reference() -> None:
    grid = CandidateGrid(
        config_ids=np.arange(6),
        entry_values=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
        stop_values=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.1, 0.3): 2, (0.2, 0.1): 3, (0.2, 0.2): 4, (0.2, 0.3): 5},
        grid_shape=(2, 3),
    )
    graph = build_canonical_grid_graph(grid)
    candidate_basis = np.array(
        [
            [1.0],
            [0.5],
            [0.0],
            [-0.2],
            [-0.5],
            [-0.8],
        ]
    )
    projector = build_smooth_subspace_projector(candidate_basis)
    operator = ProjectedLaplacianOperator(graph, projector)
    vector = np.array([0.4, 0.1, -0.2, 0.5, -0.3, 0.8])
    explicit = projector.explicit_projector() @ (graph.laplacian.toarray() @ (projector.explicit_projector() @ vector))
    assert np.allclose(operator.apply(vector), explicit, atol=1e-10)
