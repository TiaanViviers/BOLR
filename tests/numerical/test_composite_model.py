import numpy as np

from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import ContextInteractionBlock, DynamicSurfaceBlock, LinearDesignBlock, StaticBaselineBlock


def test_composite_scores_transpose_and_hvp_match_explicit() -> None:
    phi = np.array([[1.0, 0.0], [0.2, 0.5], [-0.4, 0.8]])
    static = StaticBaselineBlock("baseline", phi, np.array([0.1, -0.2]), {})
    surface = DynamicSurfaceBlock("surface", phi)
    context = ContextInteractionBlock("context", phi)
    history = LinearDesignBlock("history", "history_design")
    batch = {"context_vector": np.array([1.0, -0.5]), "history_design": np.array([[0.3], [0.1], [-0.2]])}
    model = CompositeScoreModel.from_blocks([static], [surface, context, history], batch)
    state = np.arange(model.layout.total_dimension, dtype=float) * 0.1
    score_vector = np.array([0.7, -0.1, 0.2])
    explicit = model.explicit_design_matrix(batch)
    assert np.allclose(model.dynamic_scores(batch, state), explicit @ state)
    assert np.allclose(model.transpose_multiply(batch, score_vector), explicit.T @ score_vector)
    curvature = np.array([[1.2, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.9]])
    vector = np.array([0.2] * model.layout.total_dimension)
    assert np.allclose(model.parameter_hvp_from_score_curvature(batch, curvature, vector), explicit.T @ curvature @ explicit @ vector)


def test_identifiability_diagnostics_detect_duplicate_columns() -> None:
    design = np.array([[1.0, 1.0], [0.5, 0.5]])
    block_a = LinearDesignBlock("a", "Xa")
    block_b = LinearDesignBlock("b", "Xb")
    batch = {"Xa": design[:, :1], "Xb": design[:, 1:]}
    model = CompositeScoreModel.from_blocks([], [block_a, block_b], batch)
    diagnostics = model.identifiability_diagnostics(batch)
    assert any("duplicate" in warning for warning in diagnostics["warnings"])

