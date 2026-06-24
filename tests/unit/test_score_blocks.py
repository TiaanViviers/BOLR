import numpy as np

from bolr.model.score_blocks import ContextInteractionBlock, DynamicSurfaceBlock, LinearDesignBlock, StaticBaselineBlock


def test_dynamic_surface_block_forward_and_transpose() -> None:
    phi = np.array([[1.0, 0.5], [0.2, -0.3]])
    block = DynamicSurfaceBlock(name="surface", candidate_basis=phi)
    state = np.array([0.4, -0.1])
    score = block.score_from_state({}, state)
    assert np.allclose(score, phi @ state)
    assert np.allclose(block.transpose_multiply({}, np.array([1.0, 2.0])), phi.T @ np.array([1.0, 2.0]))


def test_context_interaction_block_matches_explicit_design() -> None:
    phi = np.array([[1.0, 0.0], [0.3, 0.7]])
    batch = {"context_vector": np.array([1.0, -0.2, 0.4])}
    block = ContextInteractionBlock(name="context", candidate_basis=phi)
    state = np.arange(6, dtype=float)
    assert np.allclose(block.score_from_state(batch, state), block.design_matrix(batch) @ state)


def test_linear_design_block_matches_matrix_ops() -> None:
    batch = {"history_design": np.array([[1.0, 0.2], [0.5, -0.3]])}
    block = LinearDesignBlock(name="history", design_key="history_design")
    state = np.array([0.4, 0.1])
    assert np.allclose(block.score_from_state(batch, state), batch["history_design"] @ state)
    assert np.allclose(block.transpose_multiply(batch, np.array([1.0, -1.0])), batch["history_design"].T @ np.array([1.0, -1.0]))

