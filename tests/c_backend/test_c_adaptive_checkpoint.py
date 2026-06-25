from __future__ import annotations

import numpy as np

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy
from bolr.backend.c_backend import CBackend
from bolr.config.foundation import AdaptiveTransitionConfig, BlockAdaptationConfig, BOCPDConfig, SurpriseStandardizerConfig
from bolr.model.state_layout import make_state_layout


def test_c_adaptive_state_checkpoint_roundtrip() -> None:
    backend = CBackend()
    layout = make_state_layout([{"name": "surface", "shape": (1,)}])
    policy_py = AdaptiveAdditiveTransitionPolicy(
        np.array([[0.1]]),
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=2.0, decay=0.0),),
        ),
    )
    c_policy = backend.adaptive_policy(policy_py, layout)
    c_state = backend.adaptive_state(c_policy)
    try:
        payload = c_state.to_bytes()
        restored = c_state.from_bytes(c_policy, payload)
        try:
            assert np.allclose(restored.block_multipliers(), c_state.block_multipliers())
            assert np.allclose(restored.block_discounts(), c_state.block_discounts())
        finally:
            restored.close()
    finally:
        c_state.close()
        c_policy.close()
