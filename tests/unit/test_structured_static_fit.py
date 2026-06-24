import numpy as np

from bolr.initialization.static_surface import fit_static_surface
from bolr.model.penalties import difference_penalty
from bolr.model.structured import prior_from_penalty
from bolr.targets.soft_target import SoftTargetBuilder


def test_structured_static_fit_runs_with_penalty_prior() -> None:
    phi = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0], [0.0, 0.7, 0.3], [-0.2, 0.1, 0.8]])
    observations = [
        SoftTargetBuilder().build(np.array([1.0, 0.3, -0.2, 0.0])),
        SoftTargetBuilder().build(np.array([0.8, 0.5, -0.4, -0.1])),
    ]
    penalty = difference_penalty(3, 2).properize(0.1)
    fit = fit_static_surface(phi, observations, prior_penalty=penalty)
    assert fit.converged
    assert fit.gradient_norm < 1e-5
    assert fit.diagnostics["penalty_name"] == penalty.name

