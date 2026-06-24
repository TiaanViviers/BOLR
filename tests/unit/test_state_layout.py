import numpy as np
import pytest

from bolr.model.state_layout import make_state_layout


def test_state_layout_extract_insert_and_reshape() -> None:
    layout = make_state_layout(
        [
            {"name": "surface", "shape": (3,)},
            {"name": "context", "shape": (2, 2), "vectorization_order": "F"},
        ]
    )
    state = np.arange(7, dtype=float)
    assert layout.slice_for("surface") == slice(0, 3)
    assert layout.shape_for("context") == (2, 2)
    context = layout.extract(state, "context")
    assert context.shape == (2, 2)
    replaced = layout.insert(state, "surface", np.array([10.0, 11.0, 12.0]))
    assert np.allclose(replaced[:3], [10.0, 11.0, 12.0])


def test_state_layout_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        make_state_layout([{"name": "x", "shape": (2,)}, {"name": "x", "shape": (1,)}])

