import numpy as np

from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import GraphResidualBlock


def test_graph_residual_block_matches_explicit_design() -> None:
    psi = np.array(
        [
            [0.7, 0.0],
            [0.2, 0.4],
            [-0.4, 0.3],
            [-0.5, -0.7],
        ]
    )
    block = GraphResidualBlock("local", psi)
    state = np.array([0.5, -0.2])
    scores = block.score_from_state({}, state)
    assert np.allclose(scores, psi @ state)
    assert np.allclose(block.transpose_multiply({}, np.array([1.0, -1.0, 0.2, 0.3])), psi.T @ np.array([1.0, -1.0, 0.2, 0.3]))
    model = CompositeScoreModel.from_blocks([], [block], {})
    assert np.allclose(model.explicit_design_matrix({}), psi)

