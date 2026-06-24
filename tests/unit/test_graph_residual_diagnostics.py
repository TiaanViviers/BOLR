import numpy as np

from bolr.data.candidate_grid import CandidateGrid
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.graph_residual import smooth_plus_local_diagnostics


def test_smooth_plus_local_diagnostics_are_consistent() -> None:
    grid = CandidateGrid(
        config_ids=np.arange(6),
        entry_values=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
        stop_values=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.1, 0.3): 2, (0.2, 0.1): 3, (0.2, 0.2): 4, (0.2, 0.3): 5},
        grid_shape=(2, 3),
    )
    graph = build_canonical_grid_graph(grid)
    smooth = np.array([0.4, 0.2, 0.1, -0.1, -0.2, -0.4])
    local = np.array([0.1, -0.1, 0.0, 0.05, -0.02, -0.03])
    coeffs = np.array([0.5, -0.2])
    precision = np.array([[2.0, 0.0], [0.0, 3.0]])
    basis = np.array(
        [
            [0.8, 0.0],
            [0.1, 0.4],
            [-0.3, 0.2],
            [-0.4, -0.1],
            [-0.1, -0.3],
            [-0.1, -0.2],
        ]
    )
    covariance = np.array([[0.4, 0.1], [0.1, 0.3]])
    diagnostics = smooth_plus_local_diagnostics(
        smooth,
        local,
        graph,
        residual_coefficients=coeffs,
        residual_prior_precision=precision,
        residual_block_covariance=covariance,
        residual_basis=basis,
    )
    assert diagnostics["decomposition_inf_error"] < 1e-12
    assert diagnostics["local_score_norm"] > 0.0
    assert diagnostics["effective_residual_variance"] > 0.0
    assert diagnostics["prior_standardized_residual_norm"] > 0.0
