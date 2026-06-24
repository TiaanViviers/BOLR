from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend, CInferenceWorkspace, CNewtonConfig, CStaticFitDataset
from bolr.initialization.static_surface import fit_static_surface
from bolr.targets.soft_target import SoftTargetBuilder


def test_c_static_fit_handles_zero_weight_days() -> None:
    backend = CBackend()
    design = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5]])
    observations = [SoftTargetBuilder().build(np.zeros(4)) for _ in range(3)]
    prior_mean = np.zeros(design.shape[1])
    prior_precision = np.eye(design.shape[1])
    python_fit = fit_static_surface(design, observations)

    with CStaticFitDataset(design, observations, library=backend.library) as dataset:
        with CInferenceWorkspace(design.shape[1], design.shape[0], library=backend.library) as workspace:
            coefficients, scores = backend.static_fit(dataset, prior_mean, prior_precision, workspace, CNewtonConfig())

    assert np.allclose(coefficients, python_fit.coefficients)
    assert np.allclose(scores, design @ coefficients)
