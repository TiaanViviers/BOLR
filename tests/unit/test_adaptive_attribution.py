import numpy as np

from bolr.adaptation.attribution import block_innovation_attribution
from bolr.model.state_layout import make_state_layout


def test_block_innovation_attribution_favors_changed_block() -> None:
    layout = make_state_layout([{"name": "surface", "shape": (2,)}, {"name": "residual", "shape": (2,)}])
    predictive_mean = np.zeros(4)
    posterior_mean = np.array([1.0, 0.5, 0.0, 0.0])
    predictive_covariance = np.eye(4)
    diagnostics = block_innovation_attribution(layout, predictive_mean, predictive_covariance, posterior_mean)
    assert diagnostics["surface"]["attribution_weight"] > diagnostics["residual"]["attribution_weight"]
