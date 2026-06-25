# Posterior Decision and Region Inference

## Scope

Phase K adds the final Python reference decision layer on top of the existing predictive Gaussian posterior. It does not change the latent-state architecture, observation models, or adaptive transition family. It consumes only the predictive posterior for day `t` and produces decision quantities before the outcome is revealed.

## Predictive score posterior

For predictive state:

- `theta_t ~ N(m_t^-, P_t^-)`

and composite score model:

- `s_t = b_t + X_t theta_t`

the Python reference exposes:

- score mean `mu_s = b_t + X_t m_t^-`
- score variance diagonal `diag(X_t P_t^- X_t^T)`
- selected score covariance blocks `X_I P_t^- X_I^T`

The dense reference does not form the full `1428 x 1428` score covariance unless explicitly requested.

## Monte Carlo ranking summaries

The decision layer supports deterministic Gaussian state sampling with:

- fixed seeds
- optional antithetic pairs
- retained or transient score samples

From sampled score surfaces it estimates:

- probability best
- probability top-`K`
- expected rank
- rank standard deviation
- winner entropy and effective winner count

Exact floating-point ties use stable canonical ordering, which corresponds to lowest canonical candidate index after earlier tie-break stages.

## Pairwise probabilities

For requested candidate pairs `(i, j)`, the reference computes:

- analytic `P(s_i > s_j)` from the Gaussian difference distribution
- optional Monte Carlo validation against retained score samples

Zero difference variance is handled deterministically from the mean difference.

## Region inference

High-quality support is defined in rank space rather than absolute score space. Given a configured region top-`K` or top-fraction:

- candidate inclusion probability is estimated from sampled top-`K` membership
- a consensus set is built by threshold, top-count, or cumulative inclusion mass
- the induced subgraph on the canonical grid is decomposed into connected components

Each connected component records:

- inclusion mass
- probability-best mass
- mean-score and variance summaries
- entry and stop ranges
- graph diameter
- boundary edge count
- compactness
- deterministic weighted medoid

This lets BOLR describe broad plateaus and disconnected high-quality zones without pretending the posterior is confident about a unique point winner.

## Decision policies

The Python reference now includes:

- posterior-mean argmax
- maximum probability-best
- maximum probability top-`K`
- minimum expected rank
- Thompson decision
- highest-mass connected-region selection with representative choice by:
  - posterior mean
  - probability best
  - probability top-`K`
  - weighted graph medoid

Ranking-only policies always choose one candidate. Abstention is intentionally excluded unless an explicit outside-option provider is supplied. Candidate A and Candidate B scores remain ranking utilities, not expected PnL estimates.

## Calibration metrics

After the outcome is revealed, the replay layer records:

- probability-best Brier score
- top-`K` Brier score
- selected-region coverage of the realised best set

Best-set evaluation is tie-aware: when multiple realised winners are tied, target mass is distributed uniformly over the realised best set.

## Replay integration

The generic prequential runner now performs the day sequence:

1. predictive transition
2. predictive posterior score prediction
3. decision calculation
4. prediction persistence
5. outcome reveal
6. calibration and realised-metric evaluation
7. posterior update
8. adaptive transition diagnostics
9. checkpoint persistence

Decision configuration, Monte Carlo sampling configuration, and region-definition metadata are checkpointed so resume-equivalence remains testable.

## C-port contracts

The Python reference now stabilises the following kernel-level interfaces for the future C implementation:

- `score_variance_diagonal`
- `selected_score_covariance`
- `gaussian_state_sample`
- `composite_score_sample`
- `probability_best_accumulate`
- `probability_top_k_accumulate`
- `expected_rank_accumulate`
- `pairwise_win_probability`
- `consensus_set_build`
- `induced_connected_components`
- `region_summary_accumulate`
- `weighted_graph_medoid`
- `decision_policy_apply`
- `decision_calibration_update`

Arrays remain `float64`/`int64`, row-major where stored explicitly, and all tie conventions are deterministic.
