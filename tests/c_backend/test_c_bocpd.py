from __future__ import annotations

import numpy as np

from bolr.adaptation.bocpd import BOCPDDetector
from bolr.backend.c_backend import CBackend
from bolr.config.foundation import BOCPDConfig


def test_c_bocpd_sequence_matches_python() -> None:
    config = BOCPDConfig(hazard=0.2, max_run_length=8, prior_mean=0.0, prior_kappa=1.0, prior_alpha=2.0, prior_beta=1.0)
    backend = CBackend()
    python = BOCPDDetector(config)
    python_state = python.initial_state()
    c_state = backend.bocpd_state(config)
    try:
        for value in (0.1, 0.2, 0.3, None, 1.5):
            python_state, python_diag = python.step(value, python_state)
            c_diag = c_state.step(value)
            assert np.allclose(c_state.run_length_posterior(), python_diag["run_length_posterior"], atol=1e-10, rtol=1e-10)
            assert np.isclose(c_diag.change_probability, python_diag["change_probability"])
    finally:
        c_state.close()
