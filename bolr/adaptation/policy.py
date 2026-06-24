from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from bolr.adaptation.attribution import block_innovation_attribution
from bolr.adaptation.bocpd import BOCPDDetector, BOCPDState
from bolr.adaptation.reset import PendingReset, apply_partial_reset
from bolr.adaptation.standardizer import EWStandardizer, EWStandardizerState
from bolr.adaptation.surprise import GeneralizedPredictiveLossSurprise
from bolr.config.foundation import AdaptiveTransitionConfig, BlockAdaptationConfig
from bolr.model.state_layout import StateLayout
from bolr.posterior.state import GaussianPosterior


@dataclass(frozen=True)
class TransitionPolicyState:
    last_completed_date: str | None
    step_index: int
    block_multipliers: dict[str, float]
    block_discounts: dict[str, float]
    decay_state: dict[str, float]
    change_detector_state: BOCPDState | None
    online_standardizer_state: EWStandardizerState | None
    pending_resets: dict[str, PendingReset]
    last_surprise_values: dict[str, float | None]
    metadata: dict[str, Any]
    schema_version: str = "phase_j_v1"


def layout_signature(layout: StateLayout) -> str:
    payload = "|".join(f"{block.name}:{block.start}:{block.stop}:{block.shape}" for block in layout.blocks)
    return sha256(payload.encode("utf-8")).hexdigest()


class FixedAdditiveTransitionPolicy:
    def __init__(self, process_noise: np.ndarray) -> None:
        self.process_noise = np.asarray(process_noise, dtype=float)

    def initial_state(self, *, layout: StateLayout, base_dynamics: object | None = None) -> TransitionPolicyState:
        del base_dynamics
        return TransitionPolicyState(
            last_completed_date=None,
            step_index=0,
            block_multipliers={block.name: 1.0 for block in layout.blocks},
            block_discounts={block.name: 1.0 for block in layout.blocks},
            decay_state={block.name: 1.0 for block in layout.blocks},
            change_detector_state=None,
            online_standardizer_state=None,
            pending_resets={},
            last_surprise_values={},
            metadata={"layout_signature": layout_signature(layout), "policy_family": "fixed_additive"},
        )

    def predict(self, posterior: GaussianPosterior, policy_state: TransitionPolicyState, *, layout: StateLayout) -> tuple[GaussianPosterior, TransitionPolicyState, Mapping[str, object]]:
        _validate_policy_state(policy_state, layout)
        predicted = GaussianPosterior(
            mean=posterior.mean.copy(),
            covariance=posterior.covariance + self.process_noise,
            state_layout=posterior.state_layout,
            timestamp=posterior.timestamp,
            version=posterior.version,
            diagnostics=dict(posterior.diagnostics),
        ).with_diagnostics(predicted_by="fixed_additive")
        return predicted, policy_state, {"transition_policy_family": "fixed_additive"}

    def observe_update(self, **kwargs):
        state = kwargs["policy_state"]
        return replace(state, last_completed_date=None if kwargs.get("date") is None else str(kwargs["date"]), step_index=state.step_index + 1), {
            "change_probability": 0.0,
            "activation_value": 0.0,
        }

    def metadata(self) -> Mapping[str, object]:
        return {"family": "fixed_additive"}


class HeterogeneousDiscountTransitionPolicy:
    def __init__(self, block_discounts: Mapping[str, float]) -> None:
        self.block_discounts = {name: float(value) for name, value in block_discounts.items()}

    def initial_state(self, *, layout: StateLayout, base_dynamics: object | None = None) -> TransitionPolicyState:
        del base_dynamics
        return TransitionPolicyState(
            last_completed_date=None,
            step_index=0,
            block_multipliers={block.name: 1.0 for block in layout.blocks},
            block_discounts={block.name: self.block_discounts.get(block.name, 1.0) for block in layout.blocks},
            decay_state={block.name: 1.0 for block in layout.blocks},
            change_detector_state=None,
            online_standardizer_state=None,
            pending_resets={},
            last_surprise_values={},
            metadata={"layout_signature": layout_signature(layout), "policy_family": "heterogeneous_discount"},
        )

    def predict(self, posterior: GaussianPosterior, policy_state: TransitionPolicyState, *, layout: StateLayout) -> tuple[GaussianPosterior, TransitionPolicyState, Mapping[str, object]]:
        _validate_policy_state(policy_state, layout)
        scale = np.ones(layout.total_dimension, dtype=float)
        for block in layout.blocks:
            scale[block.start:block.stop] = policy_state.block_discounts[block.name] ** (-0.5)
        D = np.diag(scale)
        predicted = GaussianPosterior(
            mean=posterior.mean.copy(),
            covariance=D @ posterior.covariance @ D,
            state_layout=posterior.state_layout,
            timestamp=posterior.timestamp,
            version=posterior.version,
            diagnostics=dict(posterior.diagnostics),
        ).with_diagnostics(predicted_by="heterogeneous_discount")
        return predicted, policy_state, {"transition_policy_family": "heterogeneous_discount", "block_discounts": dict(policy_state.block_discounts)}

    def observe_update(self, **kwargs):
        state = kwargs["policy_state"]
        return replace(state, last_completed_date=None if kwargs.get("date") is None else str(kwargs["date"]), step_index=state.step_index + 1), {
            "change_probability": 0.0,
            "activation_value": 0.0,
        }

    def metadata(self) -> Mapping[str, object]:
        return {"family": "heterogeneous_discount", "block_discounts": self.block_discounts}


class AdaptiveAdditiveTransitionPolicy:
    def __init__(
        self,
        base_process_noise: np.ndarray,
        config: AdaptiveTransitionConfig,
        *,
        surprise_signal: object | None = None,
        initial_anchor_mean: Mapping[str, np.ndarray] | None = None,
        initial_anchor_covariance: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.base_process_noise = np.asarray(base_process_noise, dtype=float)
        self.config = config
        self.surprise_signal = surprise_signal or GeneralizedPredictiveLossSurprise()
        self.standardizer = EWStandardizer(config.standardizer)
        self.detector = BOCPDDetector(config.detector)
        self.block_configs = {block.block_name: block for block in config.blocks}
        self.initial_anchor_mean = {k: np.asarray(v, dtype=float) for k, v in (initial_anchor_mean or {}).items()}
        self.initial_anchor_covariance = {k: np.asarray(v, dtype=float) for k, v in (initial_anchor_covariance or {}).items()}

    def initial_state(self, *, layout: StateLayout, base_dynamics: object | None = None) -> TransitionPolicyState:
        del base_dynamics
        return TransitionPolicyState(
            last_completed_date=None,
            step_index=0,
            block_multipliers={block.name: 1.0 for block in layout.blocks},
            block_discounts={block.name: 1.0 for block in layout.blocks},
            decay_state={block.name: 0.0 for block in layout.blocks},
            change_detector_state=self.detector.initial_state(),
            online_standardizer_state=self.standardizer.initial_state(),
            pending_resets={},
            last_surprise_values={},
            metadata={"layout_signature": layout_signature(layout), "policy_family": "adaptive_additive", "days_since_reset": {block.name: 10**9 for block in layout.blocks}},
        )

    def predict(self, posterior: GaussianPosterior, policy_state: TransitionPolicyState, *, layout: StateLayout) -> tuple[GaussianPosterior, TransitionPolicyState, Mapping[str, object]]:
        _validate_policy_state(policy_state, layout)
        adjusted = posterior
        applied_resets = {}
        for block_name, pending in policy_state.pending_resets.items():
            adjusted = apply_partial_reset(adjusted, layout, pending)
            applied_resets[block_name] = pending.strength
        active_q = np.zeros_like(self.base_process_noise)
        for block in layout.blocks:
            sl = slice(block.start, block.stop)
            active_q[sl, sl] = policy_state.block_multipliers.get(block.name, 1.0) * self.base_process_noise[sl, sl]
        predicted = GaussianPosterior(
            mean=adjusted.mean.copy(),
            covariance=adjusted.covariance + active_q,
            state_layout=adjusted.state_layout,
            timestamp=adjusted.timestamp,
            version=adjusted.version,
            diagnostics=dict(adjusted.diagnostics),
        ).with_diagnostics(predicted_by="adaptive_additive")
        return predicted, replace(policy_state, pending_resets={}), {
            "transition_policy_family": "adaptive_additive",
            "active_process_noise_trace": float(np.trace(active_q)),
            "block_multipliers": dict(policy_state.block_multipliers),
            "applied_resets": applied_resets,
        }

    def observe_update(
        self,
        *,
        predictive_posterior: GaussianPosterior,
        posterior: GaussianPosterior,
        observation_diagnostics: Mapping[str, object],
        block_diagnostics: Mapping[str, object],
        policy_state: TransitionPolicyState,
        layout: StateLayout,
        predictive_scores: np.ndarray,
        posterior_scores: np.ndarray,
        observation: object,
        observation_model: object,
        date: object | None = None,
    ) -> tuple[TransitionPolicyState, Mapping[str, object]]:
        del block_diagnostics
        attribution = block_innovation_attribution(layout, predictive_posterior.mean, predictive_posterior.covariance, posterior.mean)
        surprise_value, surprise_diag = self.surprise_signal.compute(
            predictive_posterior=predictive_posterior,
            posterior=posterior,
            predictive_scores=predictive_scores,
            posterior_scores=posterior_scores,
            observation=observation,
            observation_model=observation_model,
            update_diagnostics=observation_diagnostics,
        )
        std_state, std_diag = self.standardizer.step(surprise_value, policy_state.online_standardizer_state or self.standardizer.initial_state())
        detector_state, detector_diag = self.detector.step(surprise_value, policy_state.change_detector_state or self.detector.initial_state())
        activation = _activation_value(float(detector_diag["change_probability"]), std_diag["z_score"] if std_diag["z_score"] is not None else 0.0, self.config.activation_parameters or {"beta": 1.0, "z0": 2.0})
        new_multipliers = dict(policy_state.block_multipliers)
        new_discounts = dict(policy_state.block_discounts)
        days_since_reset = dict(policy_state.metadata.get("days_since_reset", {}))
        pending_resets = dict(policy_state.pending_resets)
        per_block = {}
        for block in layout.blocks:
            cfg = self.block_configs.get(block.name, BlockAdaptationConfig(block_name=block.name, transition_family="fixed", adaptive_enabled=False))
            attrib = max(float(attribution[block.name]["attribution_weight"]), cfg.attribution_floor)
            if cfg.transition_family == "discount":
                target_discount = 1.0 - activation * attrib * (1.0 - (cfg.minimum_discount or 1.0))
                prev = policy_state.block_discounts.get(block.name, 1.0)
                new_discounts[block.name] = float(np.clip((1.0 - cfg.decay) * target_discount + cfg.decay * prev, cfg.minimum_discount or 1e-6, 1.0))
                new_multipliers[block.name] = 1.0
            elif cfg.transition_family in {"additive", "zero_noise"} and cfg.adaptive_enabled:
                target = 1.0 + cfg.amplitude * attrib * activation
                prev = policy_state.block_multipliers.get(block.name, 1.0)
                decayed = 1.0 + cfg.decay * (prev - 1.0) + (1.0 - cfg.decay) * (target - 1.0)
                new_multipliers[block.name] = float(np.clip(decayed, cfg.minimum_multiplier, cfg.maximum_multiplier))
            triggered = False
            if cfg.reset_enabled and cfg.reset_threshold is not None and cfg.reset_strength is not None:
                if activation >= cfg.reset_threshold and days_since_reset.get(block.name, 10**9) >= cfg.reset_cooldown:
                    pending_resets[block.name] = self._make_reset(block.name, cfg, layout, posterior)
                    days_since_reset[block.name] = 0
                    triggered = True
            days_since_reset[block.name] = days_since_reset.get(block.name, 10**9) + (0 if triggered else 1)
            per_block[block.name] = {
                **attribution[block.name],
                "process_noise_multiplier": float(new_multipliers[block.name]),
                "active_discount": float(new_discounts[block.name]),
                "reset_triggered": triggered,
                "reset_strength": float(cfg.reset_strength or 0.0),
                "days_since_reset": int(days_since_reset[block.name]),
            }
        new_state = replace(
            policy_state,
            last_completed_date=None if date is None else str(date),
            step_index=policy_state.step_index + 1,
            block_multipliers=new_multipliers,
            block_discounts=new_discounts,
            change_detector_state=detector_state,
            online_standardizer_state=std_state,
            pending_resets=pending_resets,
            last_surprise_values={"raw_surprise": surprise_diag.get("raw_surprise"), "standardised_surprise": std_diag.get("z_score")},
            metadata={**policy_state.metadata, "days_since_reset": days_since_reset},
        )
        return new_state, {
            **surprise_diag,
            "standardised_surprise": std_diag.get("z_score"),
            "standardiser_mean_before": std_diag.get("mean_before"),
            "standardiser_scale_before": std_diag.get("scale_before"),
            **detector_diag,
            "activation_value": activation,
            "block_diagnostics": per_block,
            "policy_state_schema_version": new_state.schema_version,
        }

    def _make_reset(self, block_name: str, cfg: BlockAdaptationConfig, layout: StateLayout, posterior: GaussianPosterior) -> PendingReset:
        block = layout._block(block_name)
        dim = block.dimension
        if cfg.reset_anchor == "initial_prior":
            mean = self.initial_anchor_mean.get(block_name, np.zeros(dim, dtype=float))
            cov = self.initial_anchor_covariance.get(block_name, np.eye(dim, dtype=float))
        elif cfg.reset_anchor == "configured_anchor":
            mean = self.initial_anchor_mean[block_name]
            cov = self.initial_anchor_covariance[block_name]
        else:
            mean = np.zeros(dim, dtype=float)
            cov = posterior.covariance[block.start:block.stop, block.start:block.stop].copy()
        return PendingReset(block_name=block_name, strength=float(cfg.reset_strength or 0.0), anchor_mean=mean, anchor_covariance=cov)

    def metadata(self) -> Mapping[str, object]:
        return {"family": "adaptive_additive", "config": self.config}


def _activation_value(change_probability: float, z_score: float, parameters: Mapping[str, float]) -> float:
    beta = float(parameters.get("beta", 1.0))
    z0 = float(parameters.get("z0", 2.0))
    sigmoid = 1.0 / (1.0 + np.exp(-beta * (z_score - z0)))
    return float(max(change_probability, sigmoid))


def _validate_policy_state(policy_state: TransitionPolicyState, layout: StateLayout) -> None:
    if policy_state.metadata.get("layout_signature") != layout_signature(layout):
        raise ValueError("Transition policy state is incompatible with the active layout.")
