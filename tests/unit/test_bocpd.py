import numpy as np

from bolr.adaptation.bocpd import BOCPDDetector
from bolr.config.foundation import BOCPDConfig


def test_bocpd_run_length_posterior_stays_normalized_and_detects_shift() -> None:
    detector = BOCPDDetector(BOCPDConfig(hazard=0.1, max_run_length=20, prior_mean=0.0, prior_kappa=1.0, prior_alpha=2.0, prior_beta=1.0))
    state = detector.initial_state()
    diagnostics = None
    for value in [0.0, 0.1, -0.1, 0.05, 3.0]:
        state, diagnostics = detector.step(value, state)
    assert diagnostics is not None
    assert np.isclose(np.sum(diagnostics["run_length_posterior"]), 1.0)
    assert diagnostics["change_probability"] > 0.05


def test_bocpd_hold_missing_policy_keeps_state() -> None:
    detector = BOCPDDetector(BOCPDConfig(missing_policy="hold"))
    state = detector.initial_state()
    next_state, diagnostics = detector.step(None, state)
    assert next_state == state
    assert diagnostics["missing_policy"] == "hold"
