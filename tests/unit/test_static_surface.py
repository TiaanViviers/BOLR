import numpy as np

from bolr.config.foundation import SoftTargetConfig, StaticSurfaceConfig
from bolr.initialization.static_surface import fit_static_surface
from bolr.targets.soft_target import build_soft_target_observation


def test_static_surface_fit_converges_on_small_problem() -> None:
    basis = np.array([[1.0, 0.0], [0.3, 0.8], [-0.4, 0.6]])
    observations = [
        build_soft_target_observation(np.array([1.0, 0.1, -0.2]), SoftTargetConfig()),
        build_soft_target_observation(np.array([0.8, 0.0, -0.1]), SoftTargetConfig()),
    ]
    fit = fit_static_surface(basis, observations, StaticSurfaceConfig(regularization=0.1))
    assert fit.coefficients.shape == (2,)
    assert fit.converged
    assert fit.gradient_norm < 1e-6

