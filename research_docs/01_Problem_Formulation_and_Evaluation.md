# 01 - Problem Formulation and Evaluation

> **Project:** Bayesian Online Listwise Ranking (BOLR)  
> **Document purpose:** Formalise the prediction problem, information structure, decision timeline, target interface, operational constraints, and evaluation requirements before model design begins.  
> **Status:** Stage 1 specification  
> **Primary development asset:** YM futures  
> **Intended scope:** Asset-agnostic downstream deployment

---

# 1. Purpose of This Document

This document defines the exact problem BOLR is intended to solve.

It does **not** select the final:

- score function;
- likelihood;
- target construction;
- inference algorithm;
- nonlinear representation;
- adaptation mechanism;
- BOCPD design;
- decision rule beyond the current top-1 objective.

Those questions belong to later research documents.

The purpose here is to freeze the information structure and evaluation contract so that later theoretical and engineering work solves the correct problem.

The central requirement is:

> Given all information available immediately before the New York macro open, rank a fixed set of trading configurations and select the single configuration most likely to produce the strongest user-defined outcome over the remainder of the session.

---

# 2. Trading Hypothesis

The current strategy is based on the hypothesis that price often moves a statistically meaningful absolute distance around the New York macro open.

The direction of the move is not predicted directly.

Instead, the strategy places:

- one buy-stop order above the current market;
- one sell-stop order below the current market.

Once one directional order is filled:

- the opposite order is cancelled;
- the opened position is managed using a moving stop;
- the trade exits when the stop is hit or when the New York trading session ends.

The primary modelling problem is therefore not directional forecasting.

It is:

> Select the entry-distance and trailing-stop configuration best suited to the current market conditions.

---

# 3. Formal Online Learning Problem

For trading day $t$, let:

- $N$ be the number of candidate configurations;
- $c_i$ be the static or candidate-specific representation of configuration $i$;
- $m_t$ be the shared market context observed before the decision;
- $x_{it}$ be the complete feature representation for candidate $i$ on day $t$;
- $y_{it}$ be the realised user-defined outcome for configuration $i$;
- $s_{it}$ be the latent score assigned by BOLR;
- $\mathcal O_t$ be the ranking observation derived from the realised outcomes;
- $\theta_t$ denote the current model state.

The candidate set is currently fixed:

$$
\mathcal C
=
\{c_1,\ldots,c_N\},
$$

with approximately:

$$
N\approx1500.
$$

Before the New York macro open, BOLR observes:

$$
\mathcal I_t
=
\left(
m_t,
c_1,\ldots,c_N,
D_{1:t-1}
\right),
$$

where $D_{1:t-1}$ represents all information from completed previous days.

BOLR produces a ranking:

$$
\pi_t
=
\operatorname{Rank}
\left(
s_{1t},\ldots,s_{Nt}
\right),
$$

and selects the top-ranked candidate:

$$
i_t^\star
=
\arg\max_i s_{it}.
$$

After the trading session, the complete counterfactual outcome vector becomes available:

$$
y_t
=
\left(
y_{1t},\ldots,y_{Nt}
\right).
$$

The user-defined utility and target adapter transform these outcomes into a ranking observation:

$$
y_t
\longrightarrow
u_t
\longrightarrow
\mathcal O_t.
$$

The model state is then updated:

$$
p(\theta_t\mid D_{1:t-1})
\longrightarrow
p(\theta_t\mid D_{1:t}).
$$

---

# 4. Classification of the Learning Setting

BOLR is a:

> **full-information, once-daily, contextual online ranking problem with non-overlapping, same-day delayed feedback.**

This classification has several consequences.

## 4.1 Full-information feedback

Although only one configuration is deployed, the realised PnL of all approximately 1,500 configurations can be reconstructed after the session using the observed minute path and existing optimised C backtesting routines.

Therefore, BOLR observes:

$$
y_{1t},\ldots,y_{Nt},
$$

not only:

$$
y_{i_t^\star t}.
$$

This is not a contextual-bandit problem.

No exploration policy is required merely to observe the outcomes of unselected configurations.

## 4.2 Same-day delayed feedback

The outcome is not known at prediction time.

It becomes available when the trade path is complete, at the latest by the end of the New York session.

Therefore:

- prediction occurs before the open;
- deployment occurs during the session;
- posterior updating occurs after the session.

## 4.3 Non-overlapping outcomes

Trades do not remain active across decision periods.

Each day forms a complete prediction-decision-outcome cycle.

This avoids delayed overlapping labels and simplifies sequential updating.

---

# 5. Prediction Timeline

The production timeline for each trading day is divided into three stages.

## 5.1 Pre-open prediction stage

The decision is made immediately before:

$$
09{:}30
$$

New York wall-clock time.

The intended timezone is:

```text
America/New_York
```

The latest permitted market data is the final completed one-minute observation before 09:30.

The prediction stage is:

1. load the previous posterior checkpoint;
2. load the fixed candidate set;
3. compute the latest market-state features;
4. compute candidate-specific historical features;
5. score all candidates;
6. derive posterior ranking quantities;
7. select the top-ranked configuration;
8. place the buy-stop and sell-stop orders.

This stage must complete within the production latency budget.

## 5.2 Intraday execution stage

For the selected configuration:

1. place a buy-stop order;
2. place a sell-stop order;
3. cancel the opposite order after one side is filled;
4. maintain a dynamic trailing stop;
5. exit when the stop is hit or the session ends.

## 5.3 Post-session update stage

After the session:

1. reconstruct all candidate outcomes;
2. compute the user-defined utility;
3. build the ranking observation;
4. update the posterior;
5. perform any expensive resampling, optimisation, approximation, or rejuvenation;
6. update diagnostics;
7. save a new checkpoint.

The posterior update does not share the strict one-minute pre-open latency requirement.

It must only finish before the next daily prediction cycle.

---

# 6. Candidate Configuration Space

The current configuration space contains two primary parameters:

- `entry_percentage`;
- `sl_trail_percentage`.

The values are generated geometrically.

For entry distance:

$$
e_k
=
e_0(1+\delta_e)^k,
$$

and for trailing-stop distance:

$$
r_l
=
r_0(1+\delta_r)^l.
$$

The current granularity is approximately:

$$
\delta_e=\delta_r=0.10.
$$

Therefore, in log-space:

$$
\log e_k
=
\log e_0+k\log(1.1),
$$

and:

$$
\log r_l
=
\log r_0+l\log(1.1).
$$

The candidate set is therefore approximately a regular two-dimensional lattice in:

$$
\left(
\log(\text{entry percentage}),
\log(\text{trailing-stop percentage})
\right).
$$

This structure is important because candidates are not unrelated alternatives.

Neighbouring configurations are often similar and may produce:

- similar trade-entry behaviour;
- identical trade paths;
- identical PnL plateaus;
- locally smooth performance regions;
- abrupt discontinuities when an order or stop condition changes.

The model should be capable of exploiting configuration geometry without assuming that the entire PnL surface is globally smooth.

---

# 7. Fixed Versus Variable Candidate Sets

For a production deployment, the candidate set is expected to remain fixed for extended periods.

Configurations may be manually changed through code or system redesign, but they do not change randomly from day to day.

This permits:

- precomputation of static candidate features;
- precomputation of basis functions over the candidate grid;
- stable candidate identifiers;
- reusable covariance or neighbourhood structures;
- efficient checkpoint continuation.

The model design should nevertheless avoid making it impossible to:

- add candidates;
- remove candidates;
- rebuild the candidate grid;
- initialise a model on a new asset.

Candidate-set modification can require an explicit migration or reinitialisation step rather than being treated as an ordinary daily event.

---

# 8. Feature Information Structure

The feature space is divided into candidate-specific, shared-context, and interaction features.

## 8.1 Candidate-specific features

Current examples include:

### Configuration parameters

- `entry_percentage`
- `sl_trail_percentage`

### Configuration geometry

- `rr_ratio`
- `risk_size`

### Rolling counterfactual performance

- `pnl_lag_1d`
- `pnl_lag_5d`
- `win_rate_lag_3d`
- `win_rate_lag_5d`
- `pnl_mean_3d`
- `pnl_vol_3d`
- `pnl_sharpe_3d`
- `max_drawdown_3d`
- `pnl_mean_5d`
- `pnl_vol_5d`
- `pnl_sharpe_5d`
- `max_drawdown_5d`

These rolling features are calculated only from completed previous-day outcomes.

For example:

$$
\text{pnl\_mean\_5d}_{it}
=
\frac{1}{5}
\sum_{k=1}^{5}
y_{i,t-k}.
$$

## 8.2 Shared market-context features

Current shared features include:

- volatility;
- trend;
- momentum;
- calendar indicators;
- news-event indicators.

Possible future shared features include:

- volume;
- volume imbalance;
- relative volume;
- additional market-state representations.

These features are identical across all configurations on day $t$.

## 8.3 Existing explicit interactions

The current feature set already includes examples such as:

- `entry_x_stddev6900`
- `entry_x_stddev1400`
- `entry_x_stddev30`
- `sl_x_stddev6900`
- `sl_x_stddev1400`
- `sl_x_stddev30`

## 8.4 Central contextual-ranking requirement

A purely additive shared-market score cannot affect a listwise softmax ranking.

If:

$$
s_{it}
=
f(c_i)+g(m_t),
$$

then:

$$
\frac{
\exp(f(c_i)+g(m_t))
}{
\sum_j\exp(f(c_j)+g(m_t))
}
=
\frac{
\exp(f(c_i))
}{
\sum_j\exp(f(c_j))
}.
$$

Therefore, BOLR must model:

$$
s_{it}
=
f(c_i,m_t),
$$

where market context changes the relative suitability of configurations.

The model must preserve candidate identity while allowing shared market conditions to reshape the ranking surface over the fixed configuration grid.

---

# 9. Feature Availability and Leakage Contract

Only information available before the prediction cutoff may be used.

The prediction cutoff is the final completed minute before 09:30 New York time.

The following are permitted:

- historical minute OHLC data;
- historical volume data;
- calendar information known in advance;
- scheduled news-event indicators known before the decision;
- previous-day candidate outcomes;
- rolling features based only on completed previous days.

The following are prohibited:

- any 09:30 or later price information;
- same-day realised trade outcomes;
- revised future values unavailable at prediction time;
- rolling windows that accidentally include the current day;
- statistics fitted using future evaluation periods;
- target-derived normalisation using future data.

The existing preprocessing system is considered production-grade and already includes:

- verified leakage controls;
- no missing values;
- stable feature construction;
- reproducible historical processing.

BOLR experiments must preserve these guarantees.

---

# 10. Outcome Definition

The raw realised outcome is recorded as net points.

For example, if a long YM futures position enters at:

$$
3500
$$

and exits at:

$$
3550,
$$

then the gross movement is:

$$
50
$$

points.

The recorded outcome already incorporates:

- transaction costs;
- slippage assumptions;
- spread;
- relevant trade costs.

The current backtest evaluates every configuration using one-lot exposure.

However, equal lot size does not imply equal maximum risk because risk depends on `sl_trail_percentage`.

The BOLR core should remain agnostic to whether the user supplies:

- raw net points;
- return;
- risk-adjusted return;
- expected utility;
- drawdown-adjusted utility;
- another scalar objective.

This is an application-level decision.

---

# 11. Utility and Target Separation

BOLR should separate three distinct objects.

## 11.1 Raw outcome

The raw backtest output:

$$
y_{it}.
$$

For the current system, this is net points.

## 11.2 User-defined utility

The user may define:

$$
u_{it}
=
h(y_{it},c_i,m_t,\text{risk information}),
$$

where $h$ may be:

### Net points

$$
u_{it}=y_{it}.
$$

### Risk-adjusted utility

$$
u_{it}
=
\frac{y_{it}}
{\operatorname{Risk}(c_i)+\varepsilon}.
$$

### Penalised utility

$$
u_{it}
=
y_{it}
-
\lambda_{\text{risk}}\operatorname{Risk}(c_i).
$$

The core model must not prescribe one economic objective.

## 11.3 Ranking observation

The target adapter transforms utility into:

$$
\mathcal O_t.
$$

Possible observation forms include:

- a soft target distribution;
- ordinal relevance grades;
- a full ranking;
- a partial ranking;
- tied groups;
- pairwise preferences;
- top-$K$ membership;
- an explicit no-information observation.

BOLR is therefore:

> Economically agnostic to the definition of utility, but mathematically explicit about the observation model used for Bayesian updating.

---

# 12. Target Adapter Contract

The target adapter should expose a formal interface.

Conceptually:

$$
\mathcal O_t
=
\left(
T_t,
V_t,
W_t,
M_t,
I_t
\right),
$$

where:

- $T_t$ is the observation type;
- $V_t$ contains observation values;
- $W_t$ contains optional observation weights;
- $M_t$ contains candidate masks or eligibility information;
- $I_t$ indicates whether the observation contains ranking information.

Possible observation types include:

```text
SOFT_DISTRIBUTION
ORDINAL_GRADES
FULL_RANKING
PARTIAL_RANKING
PAIRWISE_PREFERENCES
TOP_K_SET
NO_UPDATE
```

The target adapter should be responsible for:

- converting utility into a supported ranking observation;
- handling ties;
- handling all-irrelevant days;
- validating that the observation is mathematically valid;
- recording any target hyperparameters;
- preserving raw outcomes for evaluation.

The model-specific likelihood or generalized Bayesian update must declare which observation types it supports.

---

# 13. Current Reference Target

The current LightGBM pipeline uses a four-grade positive-performance ladder.

Let $y_{it}$ denote net points.

The current interpretation is approximately:

## Grade 0

$$
y_{it}\leq0.
$$

## Grade 1

$$
y_{it}>0.
$$

## Grade 2

The configuration lies above a dynamic positive-PnL floor, currently based on approximately the 70th percentile of positive outcomes within the day, optionally combined with an absolute minimum threshold.

## Grade 3

The configuration lies in approximately the top 10% of the grade-2 candidate set under the selected top-label metric.

Ignoring ties and absolute thresholds, grade 3 therefore represents approximately:

$$
0.10\times0.30=0.03
$$

or the top 3% of positive configurations.

This target is retained as an important reference baseline.

It is not assumed to be the final BOLR observation model.

---

# 14. All-Non-Positive Days

Days on which no active configuration produces positive net utility occur approximately once every five to ten trading days.

These days require explicit handling.

If every active configuration is assigned the same irrelevant grade:

$$
b_{1t}=\cdots=b_{Nt}=0,
$$

then the target may contain no relative preference information under the user's utility definition.

Candidate policies include:

1. emit a `NO_UPDATE` observation;
2. rank negative outcomes relative to one another;
3. assign a uniform soft target;
4. include a no-trade outside candidate;
5. apply another user-defined rule.

The initial BOLR design should support an explicit:

```text
NO_UPDATE
```

observation.

Under this policy:

- the state transition still occurs;
- the model may become more uncertain through process noise;
- no ranking likelihood contribution is applied for that day.

Formally:

$$
p(\theta_t\mid D_{1:t})
\propto
p(\theta_t\mid D_{1:t-1})
$$

after the predictive state transition, when $\mathcal O_t$ contains no ranking information.

This policy is provisional and must remain configurable.

---

# 15. No-Trade and Allocation Scope

The current BOLR production objective is top-1 active-configuration selection.

The wider trading system is intended to use a stagewise architecture.

## Stage 1

BOLR selects the strongest active configuration.

## Stage 2

A separate capital-allocation or gating model may decide:

- whether to trade;
- how much capital to allocate;
- whether recent system performance supports deployment;
- whether current conditions resemble historically favourable states.

Possible future inputs to the allocation model include:

- BOLR posterior confidence;
- probability best;
- probability top-$K$;
- posterior score dispersion;
- predictive uncertainty;
- regime uncertainty;
- recent realised system performance.

A no-trade candidate may still be supported by BOLR, but the current project does not require BOLR to solve the complete capital-allocation problem.

---

# 16. Decision Objective

The primary production action is:

$$
i_t^\star
=
\arg\max_i
\operatorname{DecisionScore}_{it}.
$$

The exact posterior decision score remains a research question.

Candidates include:

- posterior mean latent score;
- expected ranking probability;
- probability of being best;
- probability of belonging to the top $K$;
- posterior expected utility;
- risk-adjusted posterior score.

The primary operational decision remains top-1.

Top-3 and top-5 quantities are useful for:

- diagnostics;
- robustness evaluation;
- posterior feature generation;
- possible future portfolio construction.

---

# 17. Dataset Scope

The current YM development dataset spans approximately:

```text
2008-01-04 to 2025-06-06
```

with approximately:

$$
4489
$$

complete trading days.

At approximately 1,500 candidates per day, the row count is roughly:

$$
4489\times1500
=
6,733,500.
$$

However, the effective number of distinct shared market contexts is approximately:

$$
4489,
$$

not 6.7 million.

Rows within a day are highly dependent because they share:

- the same market path;
- the same shared market features;
- related configuration geometry;
- correlated or identical realised outcomes.

This distinction must be respected when:

- selecting model capacity;
- estimating uncertainty;
- interpreting sample size;
- calibrating generalized Bayesian update strength;
- designing validation folds.

---

# 18. Asset Portability

YM is the primary development asset.

The intended downstream system should support additional assets by running separate BOLR instances on separate datasets and computing environments.

The model should therefore avoid hardcoding:

- YM-specific price levels;
- YM-specific volatility scales;
- YM-specific target thresholds;
- YM-specific grid dimensions;
- asset-specific session assumptions beyond configuration.

The following should be configurable per asset:

- instrument identifier;
- trading calendar;
- timezone;
- decision timestamp;
- session close;
- candidate grid;
- feature set;
- utility function;
- target adapter;
- prior and adaptation scales;
- checkpoint path.

The model is asset-agnostic at the architectural level, while fitted instances may remain asset-specific.

---

# 19. Canonical Evaluation Principles

The evaluation framework must compare BOLR against strong baselines using identical:

- feature information;
- prediction timestamps;
- candidate sets;
- target definitions;
- walk-forward periods;
- transaction-cost assumptions;
- decision rules.

BOLR must not receive information unavailable to the baseline or vice versa.

---

# 20. Evaluation Metric Families

No single metric is sufficient.

## 20.1 Primary business metrics

The most important metrics are based on the selected top-1 candidate:

- total net points;
- average daily net points;
- median daily net points;
- positive-day rate;
- downside quantiles;
- maximum drawdown;
- cumulative performance;
- regret relative to the ex-post best candidate.

Daily regret may be defined as:

$$
R_t
=
\max_i y_{it}
-
y_{i_t^\star t}.
$$

Aggregate regret is:

$$
R
=
\sum_tR_t.
$$

## 20.2 Ranking metrics

Candidate metrics include:

- NDCG;
- NDCG at $K$;
- precision at $K$;
- top-$K$ positive rate;
- selected candidate's ex-post rank;
- pairwise ranking accuracy;
- overlap between predicted and realised top sets.

Ranking metrics must be interpreted as diagnostic measures, not substitutes for realised trading utility.

## 20.3 Top-$K$ diagnostics

Even though production uses top-1, evaluate:

- positive rate among top 3;
- positive rate among top 5;
- average PnL among top 3;
- average PnL among top 5;
- whether the selected top 1 lies inside a stable high-quality cluster.

## 20.4 Uncertainty metrics

Once posterior uncertainty is available, evaluate:

- calibration of probability best;
- calibration of probability top-$K$;
- correlation between uncertainty and realised regret;
- usefulness of uncertainty for downstream gating;
- posterior predictive coverage;
- stability under uncertain days.

## 20.5 Adaptation metrics

Evaluate:

- recovery after distributional change;
- performance during stable periods;
- unnecessary adaptation frequency;
- sensitivity to abrupt versus gradual drift;
- posterior uncertainty around suspected regime transitions.

## 20.6 Computational metrics

Measure separately:

### Pre-open prediction

- feature completion time;
- candidate-scoring time;
- posterior-decision time;
- total latency.

### Post-session update

- target-construction time;
- posterior-update time;
- checkpoint time;
- peak memory.

Also record:

- scaling with candidate count;
- scaling with representation dimension;
- scaling with particle or mixture count;
- numerical stability;
- deterministic reproducibility.

---

# 21. Canonical Walk-Forward Evaluation

The exact final walk-forward protocol remains to be frozen after the data audit.

The current system has historically used:

- training windows of approximately 50 to 378 sequential days;
- evaluation windows of approximately 5 to 20 sequential days.

For BOLR, the primary protocol must reflect genuine online operation.

A candidate canonical protocol is:

1. initialise the model on an initial historical segment;
2. predict the next day;
3. record all ranking and business metrics;
4. reveal the complete daily outcome vector;
5. update the posterior;
6. repeat one day at a time.

This produces a true prequential evaluation:

$$
\text{predict}
\rightarrow
\text{observe}
\rightarrow
\text{update}.
$$

Static and retraining baselines should be evaluated on the same sequence.

The final protocol should include:

- one primary prequential evaluation;
- one or two robustness protocols;
- an untouched final test period;
- fixed hyperparameter-selection rules;
- no reuse of the final test period during design.

---

# 22. Baseline Families

BOLR should be compared against several levels of baseline.

## 22.1 Non-learning baselines

Examples:

- fixed best historical configuration;
- previous-day best configuration;
- recent-window best configuration;
- median configuration;
- configuration selected by rolling mean PnL.

## 22.2 Existing ranking baselines

- current LightGBM `rank_xendcg`;
- alternative LightGBM ranking objectives where relevant;
- rolling-window LightGBM;
- expanding-window LightGBM.

## 22.3 Simple contextual baselines

- linear ranking model;
- regularised linear contextual model;
- static low-rank interaction model;
- periodically retrained contextual ranker.

## 22.4 Online non-Bayesian baselines

- online gradient descent;
- forgetting-factor models;
- exponentially weighted updates;
- recursive regularised ranking methods.

## 22.5 Bayesian ablations

- static Bayesian model;
- dynamic model without shared context;
- contextual model without parameter drift;
- dynamic contextual model without nonlinear features;
- model without uncertainty-aware decisions.

The exact baseline set will be refined after the first theoretical research phase.

---

# 23. Success Criteria

BOLR should not be considered successful merely because it is Bayesian or mathematically novel.

It should demonstrate measurable value in at least several of the following areas:

1. higher top-1 net points;
2. lower regret;
3. higher positive-day rate;
4. improved top-$K$ concentration;
5. faster recovery after drift;
6. better stability during stationary periods;
7. useful posterior uncertainty;
8. improved downstream allocation features;
9. acceptable prediction latency;
10. acceptable update latency and memory;
11. robustness across assets or time periods;
12. superior performance after complexity penalties and ablation.

A complex component should remain in the final design only if it solves a clearly stated problem and produces measurable value.

---

# 24. Operational Constraints

## 24.1 Prediction latency

The pre-open prediction stage should complete within:

$$
60\text{ seconds}.
$$

A significantly smaller runtime is preferred.

The purpose of the limit is to avoid missing the intended entry because price moves before orders are placed.

## 24.2 Update latency

The posterior update may run after the session and may be substantially slower than prediction.

The hard requirement is that it completes before the next decision cycle.

## 24.3 Memory

Development hardware currently provides approximately:

$$
16\text{ GB}
$$

of memory.

The preferred fully active production footprint is below:

$$
32\text{ GB}.
$$

## 24.4 Implementation language

The performance-critical core may use modern C, with C99 or later acceptable.

Python may be used for:

- experiment orchestration;
- data loading;
- configuration;
- metrics;
- diagnostics;
- visualisation;
- comparison with existing models.

The project should avoid building a completely disposable Python implementation that must later be rewritten from scratch.

## 24.5 Checkpointing

Checkpointing is an eventual production requirement.

A checkpoint should eventually contain all information required for deterministic continuation, including where applicable:

- model parameters;
- posterior moments;
- particle states;
- particle weights;
- regime states;
- random-number-generator state;
- feature-schema version;
- candidate-grid version;
- target-adapter configuration;
- timestamp and asset metadata.

Checkpoint design must be considered during inference architecture research.

---

# 25. Stage 1 Decisions Considered Frozen

The following are considered fixed unless new evidence requires revision:

1. one ranking is produced per asset per trading day;
2. the decision occurs immediately before 09:30 New York time;
3. only information available through the final completed minute before 09:30 may be used;
4. the development asset is YM;
5. the architecture should remain asset-agnostic;
6. approximately 1,500 fixed configurations are ranked;
7. configurations are defined primarily by entry and trailing-stop percentages;
8. complete counterfactual outcomes are available after the session;
9. feedback is same-day and non-overlapping;
10. the current raw outcome is net points including costs;
11. BOLR is agnostic to the user's economic utility definition;
12. BOLR must explicitly support defined ranking-observation formats;
13. the current production objective is top-1 configuration selection;
14. downstream capital allocation is a separate modelling stage;
15. prediction must complete within one minute;
16. posterior updating may occur after the session;
17. modern C may be used for the core;
18. Python may be used for the research and orchestration layer;
19. checkpointing is a future production requirement;
20. the evaluation must be genuinely walk-forward and leakage-free.

---

# 26. Open Questions Remaining After Stage 1

The following remain unresolved and belong to later research documents.

## Target and observation model

- Should utility be converted into soft targets, grades, rankings, or partial orders?
- How should tied configurations be handled?
- Should all-non-positive days produce `NO_UPDATE`?
- Should target granularity be fixed or user-controlled?
- How should generalized Bayesian update strength be calibrated?

## Representation

- Should candidate geometry use splines, Fourier bases, RFF, low-rank factors, or local structures?
- How should shared market context modify the candidate-ranking surface?
- How much nonlinear capacity is justified by approximately 4,489 daily contexts?

## Inference

- Is full-state particle filtering viable?
- Should high-dimensional parameters use Gaussian or variational approximations?
- Can Rao-Blackwellised inference preserve important dependence?
- What covariance structure is computationally feasible?

## Adaptation

- Which parameter groups should drift?
- How quickly should different groups adapt?
- Should process-noise scales be fixed, learned, or regime-dependent?
- What role should BOCPD play?

## Decision policy

- Should top-1 use expected score or probability best?
- Which posterior summaries are most valuable for the allocation model?
- Should BOLR eventually support an outside no-trade candidate?

## Evaluation

- What exact initialisation period should be used?
- What exact untouched final period should be reserved?
- Which primary business metric should determine model selection?
- Which robustness protocols should be mandatory?

---

# 27. Required Next Steps

The next project steps are:

1. build a compact structural EDA and audit harness;
2. quantify candidate-surface properties;
3. quantify target sparsity and all-non-positive days;
4. freeze the canonical prequential evaluation protocol;
5. define the target-adapter software interface;
6. launch deep research into ranking observation models and generalized Bayesian updating.

The EDA should not be a one-off YM report.

It should become a reusable diagnostic tool that can be run for every asset before fitting BOLR.

---

# 28. Stage 1 Completion Gate

Stage 1 is complete when:

- the trading timeline is unambiguous;
- every input has a valid prediction-time timestamp;
- the feedback structure is formally classified;
- the candidate set is defined;
- the raw outcome and utility separation is explicit;
- target observations have a formal interface;
- the production decision is defined;
- operational constraints are recorded;
- baseline and evaluation families are documented;
- remaining questions are assigned to later research documents.

This document satisfies the conceptual part of that gate.

The remaining practical requirement is to validate key structural assumptions through the reusable EDA and audit process.
