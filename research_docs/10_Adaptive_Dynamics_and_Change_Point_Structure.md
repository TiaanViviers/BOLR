# Phase J: Adaptive Dynamics and Change-Point Structure

## Purpose

Phase J makes transition behaviour causal and adaptive. The update for day `t` may influence the transition into day `t+1`, but it may never alter the predictive prior used for day `t`.

The timing contract is:

1. predict day `t`;
2. persist prediction and selection;
3. reveal day `t` outcome;
4. update posterior;
5. compute surprise and innovation diagnostics;
6. configure transition state for day `t+1`.

## Generic replay

The reusable replay path now operates on a composed experiment:

- `CompositeScoreModel`
- `GaussianPosterior`
- transition policy
- target builder
- observation model
- decision policy
- batch builder

The runner no longer assumes the E0 single-block surface path internally. The old Candidate A entry point remains a compatibility wrapper over the generic composite replay loop.

## Surprise semantics

Phase J keeps observation-family-specific semantics inside the surprise signal.

The initial default signal is generalized predictive loss:

$$
x_t = -\log L_t(s_t^-).
$$

This is recorded as a generalized loss, not as a calibrated predictive probability unless the observation family is a proper likelihood.

Posterior update signals are also supported, including:

- prior-Mahalanobis displacement;
- Gaussian KL from predictive prior to posterior.

## Online standardisation

Raw surprise magnitudes can drift by observation family and by period. The initial standardiser uses exponentially weighted mean and variance with strictly causal standardisation:

$$
z_t = \frac{x_t - \mu_{t-1}}{\max(\sigma_{t-1}, \sigma_{\min})}.
$$

Only after computing `z_t` is the state updated with `x_t`.

## BOCPD

Phase J includes a dense Python reference BOCPD implementation for scalar surprise streams using a Gaussian observation model with Normal-Inverse-Gamma prior and Student-t posterior predictive density.

That detector should be treated as a reference backend, not as a hard architectural dependency. The adaptive transition layer is designed so a future detector backend, including `Fast_BOCPD`, can replace it without changing the rest of the transition contract.

Current detector diagnostics include:

- change probability;
- run-length posterior;
- MAP run length;
- expected run length;
- run-length entropy;
- predictive log density;
- truncation mass.

Missing-surprise handling supports:

- `hold`
- `hazard_only`

The default remains `hold`.

## Blockwise attribution

Global surprise does not identify which block should adapt. Phase J therefore attributes posterior movement using blockwise predictive-covariance-standardised displacement. This is a numerical attribution of posterior update energy, not a scientific proof of which latent block truly changed.

Confounded block designs can still split attribution. That limitation remains explicit.

## Adaptive transition policies

The implemented transition layer currently supports:

- fixed additive process noise;
- heterogeneous blockwise covariance discount;
- adaptive additive blockwise process-noise inflation;
- optional moment-matched partial reset.

Adaptive inflation uses bounded block multipliers driven by:

- standardised surprise;
- change probability;
- block attribution.

The response is then decayed deterministically back toward baseline.

## Partial reset

Partial resets are scheduled only after the day `t` update and applied at the next prediction step. The current reset is a moment-matched Gaussian operation that:

- shrinks the selected block mean toward an anchor;
- scales cross-block covariance involving that block;
- adds anchor uncertainty back into the reset block.

It is a transition policy, not an exact Bayesian mixture posterior.

## Checkpoints

Checkpoint payloads now persist:

- transition policy family;
- policy config hash;
- serialized policy state;
- serialized standardiser state;
- serialized detector state;
- block multipliers and discounts;
- pending resets;
- last surprise diagnostics.

Resume correctness is tested on adaptive replay slices.

## Synthetic and smoke scope

Phase J currently validates:

- causal timing;
- adaptive multiplier updates;
- BOCPD recurrence sanity;
- heterogeneous discount correctness;
- partial reset algebra;
- adaptive replay resume equivalence.

The current historical scope remains bounded smoke validation, not hyperparameter search or trading-performance optimisation.

## C-port contracts

The future C implementation must eventually support:

- `ew_standardizer_update`
- `student_t_predictive_log_density`
- `bocpd_step`
- `block_innovation_attribution`
- `adaptive_multiplier_update`
- `heterogeneous_discount_predict`
- `block_process_noise_scale`
- `partial_reset_apply`
- `transition_policy_step`

The dense Python implementation remains the numerical oracle until that port exists.

## Deferred work

Still deferred after Phase J:

- full-state particle or Rao-Blackwellised particle filtering;
- expert-switching regime models;
- final regime-model selection;
- PnL-driven tuning;
- posterior decision-layer probabilities;
- low-dimensional regime uncertainty over explicit latent regimes.
