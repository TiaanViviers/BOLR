import numpy as np

from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy, FixedAdditiveTransitionPolicy, HeterogeneousDiscountTransitionPolicy
from bolr.config.foundation import AdaptiveTransitionConfig, BlockAdaptationConfig, BOCPDConfig, SurpriseStandardizerConfig
from bolr.model.state_layout import make_state_layout
from bolr.posterior.state import GaussianPosterior


def test_heterogeneous_discount_scales_cross_block_covariance() -> None:
    layout = make_state_layout([{"name": "surface", "shape": (1,)}, {"name": "residual", "shape": (1,)}])
    posterior = GaussianPosterior(mean=np.zeros(2), covariance=np.array([[2.0, 0.5], [0.5, 1.0]]))
    policy = HeterogeneousDiscountTransitionPolicy({"surface": 0.5, "residual": 0.8})
    state = policy.initial_state(layout=layout)
    predicted, _, _ = policy.predict(posterior, state, layout=layout)
    expected = np.diag([0.5 ** -0.5, 0.8 ** -0.5]) @ posterior.covariance @ np.diag([0.5 ** -0.5, 0.8 ** -0.5])
    assert np.allclose(predicted.covariance, expected)


def test_adaptive_policy_updates_only_future_multipliers() -> None:
    layout = make_state_layout([{"name": "surface", "shape": (1,)}])
    policy = AdaptiveAdditiveTransitionPolicy(
        np.array([[0.2]]),
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=2.0, decay=0.0),),
        ),
    )
    state = policy.initial_state(layout=layout)
    posterior = GaussianPosterior(mean=np.zeros(1), covariance=np.array([[1.0]]))
    predictive, state, _ = policy.predict(posterior, state, layout=layout)
    assert np.allclose(predictive.covariance, np.array([[1.2]]))
    new_state, diagnostics = policy.observe_update(
        predictive_posterior=predictive,
        posterior=GaussianPosterior(mean=np.array([1.0]), covariance=np.array([[0.8]])),
        observation_diagnostics={},
        block_diagnostics={},
        policy_state=state,
        layout=layout,
        predictive_scores=np.array([0.0]),
        posterior_scores=np.array([1.0]),
        observation=type("Obs", (), {"update_weight": 1.0, "metadata": {}})(),
        observation_model=type("ObsModel", (), {"log_factor": lambda self, scores, obs: -2.0})(),
        date="2026-01-01",
    )
    assert diagnostics["activation_value"] >= 0.0
    assert new_state.block_multipliers["surface"] >= 1.0


def test_fixed_additive_policy_matches_q_addition() -> None:
    layout = make_state_layout([{"name": "surface", "shape": (2,)}])
    posterior = GaussianPosterior(mean=np.zeros(2), covariance=np.eye(2))
    q = np.diag([0.1, 0.2])
    policy = FixedAdditiveTransitionPolicy(q)
    state = policy.initial_state(layout=layout)
    predicted, _, _ = policy.predict(posterior, state, layout=layout)
    assert np.allclose(predicted.covariance, np.eye(2) + q)
