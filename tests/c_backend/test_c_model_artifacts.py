from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import ContextInteractionBlock, DynamicSurfaceBlock, GraphResidualBlock, StaticBaselineBlock


def test_c_model_artifacts_match_python_scores_and_transpose() -> None:
    backend = CBackend()
    phi = np.array([[1.0, 0.0], [0.4, 0.6], [-0.2, 0.7]])
    graph_basis = np.array([[1.0], [0.0], [-1.0]])
    static = StaticBaselineBlock("baseline", phi, np.array([0.1, -0.05]), {})
    surface = DynamicSurfaceBlock("surface", phi)
    context = ContextInteractionBlock("context", phi)
    graph = GraphResidualBlock("graph", graph_basis)
    batch = {"context_vector": np.array([1.0, 0.3])}
    model = CompositeScoreModel.from_blocks([static], [surface, context, graph], batch)
    state = np.array([0.25, -0.1, 0.2, -0.3, 0.05, 0.15, 0.4])
    score_vector = np.array([0.2, -0.1, 0.4])

    with backend.model_artifacts(model, batch) as artifacts:
        context_handle = backend.score_context(model, batch)
        assert np.allclose(artifacts.scores(state, context_handle), model.scores(batch, state))
        assert np.allclose(artifacts.scores(state, context_handle, dynamic_only=True), model.dynamic_scores(batch, state))
        assert np.allclose(artifacts.transpose(score_vector, context_handle), model.transpose_multiply(batch, score_vector))
