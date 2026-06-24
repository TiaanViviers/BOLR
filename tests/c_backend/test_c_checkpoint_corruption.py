from __future__ import annotations

import numpy as np
import pytest

from bolr.backend.c_api import CCheckpointError
from bolr.backend.c_backend import CCheckpointState, CGaussianState, CLibrary


def test_checkpoint_truncation_is_rejected() -> None:
    library = CLibrary()
    mean = np.array([0.25, -0.5], dtype=np.float64)
    covariance = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    with CGaussianState(mean, covariance, state_layout_hash=7, model_schema_hash=11, library=library) as state:
        with state.export_checkpoint() as checkpoint:
            payload = checkpoint.to_bytes()
    with pytest.raises(CCheckpointError):
        CCheckpointState.from_bytes(payload[:-3], library=library)
