import numpy as np

from bolr.adaptation.standardizer import EWStandardizer
from bolr.config.foundation import SurpriseStandardizerConfig


def test_ew_standardizer_uses_prior_statistics_causally() -> None:
    standardizer = EWStandardizer(SurpriseStandardizerConfig(decay=0.5, variance_floor=1e-4, warmup_count=0, clip_z=None))
    state = standardizer.initial_state()
    state, first = standardizer.step(2.0, state)
    assert first["mean_before"] == 0.0
    state, second = standardizer.step(4.0, state)
    assert np.isclose(second["mean_before"], 1.0)
    assert second["z_score"] > 0.0


def test_ew_standardizer_missing_value_holds_state() -> None:
    standardizer = EWStandardizer()
    state = standardizer.initial_state()
    next_state, diagnostics = standardizer.step(None, state)
    assert next_state == state
    assert diagnostics["missing"] is True
