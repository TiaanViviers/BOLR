# Bayesian Online Listwise Ranking for Adaptive Trading Systems

> **Working project name:** BOLR  
> **Project status:** Research proposal and architectural overview  
> **Primary application:** Daily ranking of approximately 1,500 trading-strategy configurations under market non-stationarity

---

# 1. Project Objective

The current trading system ranks approximately 1,500 strategy configurations each trading day and selects the most promising configuration or small set of configurations for deployment.

The existing ranking pipeline uses LightGBM with the `rank_xendcg` objective. This provides a strong practical baseline, but the trading problem has several properties that motivate a more specialised model:

1. **Non-stationarity**
   - Market regimes and market-state distributions change over time.
   - The relationship between configuration characteristics, market conditions, and future performance may drift.
   - A model trained on historical data may become poorly calibrated for the current environment.

2. **Small effective sample size**
   - Many years of observations may be available, but only a limited subset may be relevant to the current market state.
   - The effective amount of useful historical information is uncertain and time-varying.
   - Fixed rolling windows impose an arbitrary boundary between relevant and irrelevant history.

3. **High variance and weak signal**
   - Realised trading outcomes are noisy.
   - Flexible nonlinear models may overfit regime-specific relationships.
   - Time-series validation folds naturally contain different market conditions, making model selection difficult.

4. **Repeated retraining**
   - Most conventional ranking pipelines adapt through periodic retraining.
   - New information is incorporated by rebuilding a model rather than updating a persistent posterior belief.
   - Retraining frequency and training-window length become additional modelling decisions.

5. **Decision-focused prediction**
   - The system does not primarily need accurate absolute PnL forecasts for every configuration.
   - It needs to identify which configurations are most suitable relative to the alternatives available on the same day.
   - The top of the ranking matters substantially more than the middle or bottom.

The long-term objective is therefore:

> Develop a computationally efficient, fully online Bayesian contextual ranking system that continuously updates its beliefs, represents uncertainty over the ranking function, adapts to gradual and abrupt market change, and ranks approximately 1,500 configurations per inference call within a practical runtime budget.

The current runtime target is less than one minute per daily inference and update cycle, with substantially faster operation preferred.

---

# 2. Primary Research Question

> Can a dynamic probabilistic listwise ranking model, using market-conditioned configuration effects and sequential uncertainty updates, improve top-of-list net trading utility, adaptation after distributional change, and decision calibration relative to rolling-window and periodically retrained rankers, while remaining computationally feasible for approximately 1,500 daily candidates?

This question deliberately separates the desired behaviour of the model from the final choice of representation, likelihood, inference algorithm, and adaptation mechanism.

---

# 3. Scope and Design Philosophy

BOLR is not yet a fixed mathematical model. It is a research programme for designing a family of online Bayesian ranking models suited to a specific decision problem.

The proposal distinguishes between:

## 3.1 Core model design

The core model should define:

- how candidate configurations are scored;
- how shared market context changes the relative suitability of candidates;
- how ranking observations update beliefs;
- how parameter uncertainty evolves over time;
- how posterior predictive rankings are formed;
- how inference remains computationally practical.

## 3.2 Use-case modelling

The trading application must separately define:

- which strategy and market features are supplied;
- how realised outcomes are converted into ranking targets;
- whether net PnL, risk, costs, or other utilities are used;
- whether a no-trade candidate is included;
- whether the final decision is top-1, top-$K$, abstention, or allocation;
- how performance is evaluated.

This distinction is important. BOLR should not hardcode one specific label construction, one specific feature set, or one specific trading decision. However, the modules must still obey clear mathematical contracts so that the overall model remains coherent.

---

# 4. Core Bayesian Philosophy

A conventional retraining pipeline repeatedly solves a new estimation problem:

$$
\hat{\theta}_t
=
\arg\max_\theta p(D_{a_t:t}\mid\theta),
$$

where $D_{a_t:t}$ is a selected historical window.

BOLR instead aims to maintain a sequential posterior:

$$
p(\theta_t\mid D_{1:t}),
$$

where:

- $\theta_t$ represents the current ranking model and relevant adaptation variables;
- $D_{1:t}$ represents all observations available up to time $t$.

New information updates the current belief rather than discarding the old model and starting again:

$$
\text{prior}
\rightarrow
\text{posterior}
\rightarrow
\text{posterior}
\rightarrow
\cdots
$$

The intended advantages are:

- gradual information decay rather than hard window cut-offs;
- explicit uncertainty over the ranking function;
- continuous adaptation;
- principled posterior predictive quantities;
- the ability to represent several plausible market-model states simultaneously.

This does not imply that all historical information should be retained equally. Dynamic state evolution, process noise, changepoint beliefs, and likelihood tempering can all reduce the influence of outdated observations.

---

# 5. Data Structure and Notation

For trading day $t$:

- $N_t$ is the number of available configurations;
- $c_{it}$ contains features unique to configuration $i$;
- $m_t$ contains market features shared by all configurations on day $t$;
- $x_{it}$ denotes the complete model input for configuration $i$;
- $y_{it}$ denotes the realised future outcome or utility;
- $s_{it}$ is the latent score assigned to configuration $i$;
- $q_{it}$ is a predicted listwise probability or ranking mass;
- $\theta_t$ denotes the time-varying model parameters.

In many applications, configuration features may be static across days, in which case $c_{it}=c_i$. The more general notation allows configuration-derived features to change when necessary.

---

# 6. Central Modelling Requirement: Shared Context Must Change Relative Suitability

A central insight is that shared market variables cannot affect a softmax ranking through a purely additive term.

Suppose:

$$
s_{it}
=
f(c_{it})+g(m_t).
$$

Because $g(m_t)$ is identical for every candidate on day $t$:

$$
q_{it}
=
\frac{\exp(s_{it})}
{\sum_{j=1}^{N_t}\exp(s_{jt})}
=
\frac{\exp(f(c_{it}))}
{\sum_{j=1}^{N_t}\exp(f(c_{jt}))}.
$$

The shared market term cancels.

This does **not** mean that shared market information is unimportant. It means that market context must affect the ranking through a non-separable score:

$$
s_{it}
=
f(c_{it},m_t),
$$

where the effect of the market state depends on the characteristics of the configuration.

The central modelling task is therefore:

> Learn how shared market conditions reshape the relative suitability surface over configuration space.

The model should be able to express that a market state favours or suppresses broad families of configurations while still preserving the identity and geometry of individual configurations.

---

# 7. High-Level Architecture

The proposed BOLR system is organised into seven conceptual layers.

## 7.1 Layer 1: Candidate and Context Representation

The raw inputs are divided into two groups.

### Candidate-specific features

Examples include:

- entry and exit parameters;
- stop-loss and trailing-stop parameters;
- risk-reward geometry;
- position-sizing characteristics;
- holding-period characteristics;
- strategy-family indicators;
- candidate-specific historical summaries.

These are represented by:

$$
c_{it}.
$$

### Shared market-context features

Examples include:

- volatility;
- trend;
- momentum;
- liquidity;
- market breadth;
- macroeconomic or event indicators;
- recent distributional summaries;
- latent market-state embeddings.

These are represented by:

$$
m_t.
$$

The representation layer maps these inputs into a form suitable for contextual ranking:

$$
(c_{it},m_t)
\longrightarrow
\phi(c_{it},m_t).
$$

The final representation method remains an open research question.

### Candidate representation families

1. **Structured raw interactions**
2. **Varying-coefficient models**
3. **Low-rank bilinear interactions**
4. **Context-gated experts**
5. **Joint Random Fourier Features**
6. **Modular or product-kernel random features**
7. **Hybrid structured and nonlinear representations**

---

## 7.2 Layer 2: Contextual Score Function

Each configuration receives a latent score:

$$
s_{it}
=
f(c_{it},m_t;\theta_t).
$$

The exact score function is not yet fixed.

### Varying-coefficient form

$$
s_{it}
=
\phi_c(c_{it})^\top w(m_t),
$$

where market context determines the ranking coefficients.

A simple parameterisation is:

$$
w(m_t)
=
\alpha+B\phi_m(m_t),
$$

giving:

$$
s_{it}
=
\phi_c(c_{it})^\top\alpha
+
\phi_c(c_{it})^\top B\phi_m(m_t).
$$

The first term captures stable configuration quality. The second captures market-conditioned suitability.

### Low-rank bilinear form

$$
s_{it}
=
a(c_{it})
+
u(c_{it})^\top v(m_t).
$$

Here:

- $a(c_{it})$ is a baseline configuration effect;
- $u(c_{it})$ is a low-dimensional configuration embedding;
- $v(m_t)$ is a low-dimensional market embedding.

This can be interpreted as learned compatibility between configuration characteristics and market conditions.

If a full interaction matrix is factorised as:

$$
B=UV^\top,
$$

then parameter and computation costs can be reduced from approximately $d_cd_m$ to $r(d_c+d_m)$ for rank $r$.

### Context-gated expert form

$$
s_{it}
=
\sum_{k=1}^{K}
g_k(m_t)f_k(c_{it}),
$$

where:

$$
g_k(m_t)\geq0,
\qquad
\sum_{k=1}^{K}g_k(m_t)=1.
$$

The shared market context determines which ranking surfaces are most relevant.

### Joint nonlinear feature form

$$
s_{it}
=
w_t^\top\phi(c_{it},m_t).
$$

Random Fourier Features are a leading candidate for $\phi$ because they provide:

- nonlinear approximation;
- fixed-dimensional representations;
- bounded memory;
- online compatibility;
- efficient vectorised and C implementations;
- no support-vector growth.

For a joint input $x_{it}=[c_{it};m_t]$:

$$
\phi_j(x_{it})
=
\sqrt{\frac{2}{D}}
\cos(\omega_j^\top x_{it}+b_j).
$$

A modular alternative is to construct separate random feature maps for candidate and market variables, then combine them through structured interactions.

No representation family is assumed to be optimal at this stage.

---

## 7.3 Layer 3: Listwise Ranking Distribution

Given latent scores, the model constructs a listwise distribution:

$$
q_{it}
=
\frac{\exp(s_{it}/\tau)}
{\sum_{j=1}^{N_t}\exp(s_{jt}/\tau)}.
$$

The initial model should usually fix:

$$
\tau=1,
$$

because $w_t$ and $\tau$ are not separately identifiable without additional constraints.

Only the ratio $w_t/\tau$ is identified by the softmax probabilities.

The quantity $q_{it}$ should be interpreted carefully. Depending on the observation model, it may represent:

- listwise ranking mass;
- a random-utility choice probability;
- a component of a ranking likelihood.

It is not automatically equal to:

- probability of positive PnL;
- probability of belonging to the top $K$;
- calibrated probability of being truly optimal.

Posterior probabilities of being best or top-$K$ should be computed from the posterior distribution over latent scores.

---

## 7.4 Layer 4: Target or Observation Model

Realised outcomes must be converted into a ranking observation.

The mapping:

$$
y_t
\longrightarrow
\mathcal O_t
$$

is treated as a configurable target adapter rather than a fixed part of the core model.

Candidate observation types include:

1. **Robust continuous utility**
2. **Soft target distributions**
3. **Rank-based targets**
4. **Quantile or threshold bins**
5. **Hybrid absolute-relative relevance grades**
6. **Full or partial rankings**
7. **Top-$K$ membership**
8. **Pairwise comparisons**
9. **Ordinal relevance labels with ties**

### Soft target example

A robust within-day utility may be constructed as:

$$
\tilde y_{it}
=
\frac{
y_{it}-\operatorname{median}_j(y_{jt})
}{
\operatorname{MAD}_j(y_{jt})+\varepsilon
},
$$

then clipped:

$$
u_{it}
=
\operatorname{clip}(\tilde y_{it},-c,c),
$$

and converted to a target distribution:

$$
r_{it}
=
\frac{\exp(\kappa u_{it})}
{\sum_j\exp(\kappa u_{jt})}.
$$

### Binned relevance example

Let:

$$
b_{it}\in\{0,1,\ldots,K\}
$$

be a relevance grade based on realised net utility.

A hybrid target may:

- assign non-positive configurations to grade $0$;
- divide positive configurations into quantile or threshold bins;
- preserve stronger emphasis on the top of the list.

Binning may reduce variance and outlier sensitivity at the cost of discarding within-bin magnitude information.

### Model-design principle

The target adapter should be user-configurable, but its statistical consequences must be evaluated explicitly. The model must not assume that all target constructions are equivalent. This does also not imply that the model should cater directly towards the needs of every problem formulation.

---

## 7.5 Layer 5: Bayesian Updating

### Generalized Bayesian formulation

If the target is a soft distribution $r_t$, define listwise cross-entropy loss:

$$
\mathcal L_t(\theta_t)
=
-\sum_{i=1}^{N_t}
r_{it}\log q_{it}(\theta_t).
$$

A generalized Bayesian update is:

$$
p(\theta_t\mid D_{1:t})
\propto
p(\theta_t\mid D_{1:t-1})
\exp\left(
-\eta_t\mathcal L_t(\theta_t)
\right).
$$

The learning-rate parameter $\eta_t$ controls how strongly one day updates the posterior.

This formulation is useful when the ranking target is a constructed loss rather than a literal sample from a conventional generative likelihood.

The initial model does not need to maintain a posterior over $\eta_t$. It may be:

- fixed;
- selected through walk-forward validation;
- calibrated using predictive performance;
- linked to an effective list-size measure.

### Proper likelihood alternatives

Generalized Bayes is not the only candidate.

Alternative observation models include:

- Plackett-Luce likelihoods;
- grouped Plackett-Luce models with ties;
- ordinal ranking likelihoods;
- random-utility models;
- partial-ranking likelihoods.

A major research question is whether a proper generative ranking likelihood offers practical advantages over soft-target generalized Bayes.

---

## 7.6 Layer 6: Dynamic State Evolution and Inference

The ranking function is allowed to change over time.

A general state transition is:

$$
\theta_t
\sim
p(\theta_t\mid\theta_{t-1},z_t),
$$

where $z_t$ contains adaptation or regime variables.

A simple random-walk model is:

$$
w_t
=
w_{t-1}+\epsilon_t,
$$

with:

$$
\epsilon_t\sim N(0,Q_t).
$$

### Structured adaptation by feature family

A single process-noise scale is unlikely to be sufficient.

Partition the parameter vector:

$$
w_t
=
\begin{bmatrix}
w_t^{(\text{config})}\\
w_t^{(\text{context})}\\
w_t^{(\text{interaction})}\\
w_t^{(\text{other})}
\end{bmatrix},
$$

and allow separate drift scales:

$$
Q_t
=
\operatorname{blockdiag}
\left(
Q_t^{(\text{config})},
Q_t^{(\text{context})},
Q_t^{(\text{interaction})},
Q_t^{(\text{other})}
\right).
$$

Stable configuration effects may evolve slowly, while context interactions may adapt more quickly.

### Inference candidates

The final inference method is unresolved.

Candidate methods include:

1. **Full-state particle filtering**
2. **Rao-Blackwellised particle filtering**
3. **Laplace filtering**
4. **Assumed-density filtering**
5. **Online variational inference**
6. **Extended or iterated Gaussian filtering**
7. **Ensemble Kalman-style methods**
8. **Mixtures of approximate Gaussian filters**
9. **Hybrid SMC and optimisation methods**

A full particle filter over a high-dimensional weight vector is conceptually simple but may suffer from:

- particle degeneracy;
- poor exploration in high dimensions;
- expensive score evaluation;
- excessive particle requirements.

Rao-Blackwellised inference is a leading alternative:

$$
p(w_t,z_t\mid D_{1:t})
=
p(w_t\mid z_t,D_{1:t})
p(z_t\mid D_{1:t}).
$$

This factorisation is exact and does not assume independence. The approximation enters when the conditional posterior $p(w_t\mid z_t,D_{1:t})$ is represented by a tractable family such as a Gaussian.

A particle approximation may take the form:

$$
p(w_t,z_t\mid D_{1:t})
\approx
\sum_{p=1}^{P}
\alpha_t^{(p)}
q_t^{(p)}(w_t)
\delta_{z_t^{(p)}}(z_t).
$$

Each particle may carry a joint low-dimensional adaptation state and its own conditional approximation over the high-dimensional ranking parameters.

The validity and computational value of these approximations require dedicated research and controlled comparison.

---

## 7.7 Layer 7: Drift Detection and BOCPD Integration

BOLR should distinguish between:

### Covariate drift

$$
p_t(m)\neq p_{t-1}(m).
$$

The distribution of market conditions changes.

### Concept drift

$$
p_t(y\mid c,m)
\neq
p_{t-1}(y\mid c,m).
$$

The relationship between configurations, market state, and outcomes changes.

BOCPD may support two complementary roles.

### Exogenous market-state monitor

Run BOCPD over a suitable market-state representation.

This may provide a preventative signal before realised ranking performance deteriorates.

Possible reactions include:

- covariance inflation;
- increased process noise;
- wider posterior predictive uncertainty;
- stronger weighting of fast-adapting model components.

For example:

$$
Q_t
=
Q_{\text{base}}
+
P(\text{change at }t\mid m_{1:t})
Q_{\text{extra}}.
$$

### Endogenous model-mismatch monitor

Run BOCPD or a related surprise detector over:

- predictive listwise loss;
- posterior predictive residuals;
- realised selected-strategy performance;
- score calibration errors.

This signal is reactive but can detect concept drift that is not visible in the marginal market-state distribution.

A possible long-term architecture uses both:

- market-state change for preventative uncertainty inflation;
- predictive mismatch for stronger adaptation or partial reset.

The final BOCPD observation space, likelihood, and connection to the ranker remain open research questions.

---

# 8. Posterior Prediction and Decision Quantities

For a new day, BOLR should produce posterior predictive ranking information.

Let $\theta_t^{(p)}$ denote posterior samples or mixture components.

Possible outputs include:

## Expected latent score

$$
E[s_{it}\mid D_{1:t}].
$$

## Expected ranking mass

$$
E[q_{it}\mid D_{1:t}].
$$

## Probability best

$$
P\left(
s_{it}>s_{jt}
\text{ for all }j\neq i
\mid D_{1:t}
\right).
$$

## Probability top-$K$

$$
P\left(
i\in\operatorname{TopK}(s_t)
\mid D_{1:t}
\right).
$$

## Score uncertainty

$$
\operatorname{Var}(s_{it}\mid D_{1:t}).
$$

## Pairwise dominance probabilities

$$
P(s_{it}>s_{jt}\mid D_{1:t}).
$$

## Regime or adaptation uncertainty

Posterior uncertainty over:

- process-noise state;
- run length;
- changepoint probability;
- expert or regime allocation.

The decision layer may use these quantities for:

- top-1 selection;
- diversified top-$K$ selection;
- uncertainty-aware abstention;
- risk-adjusted allocation;
- confidence-controlled position sizing.

The decision policy must be evaluated separately from ranking quality.

---

# 9. Optional No-Trade or Outside Candidate

A listwise model always ranks the available alternatives. If every active trading configuration is unattractive, the least bad configuration will still appear at the top.

BOLR should therefore support an optional outside candidate representing no trade.

This candidate may be supplied naturally by the user:

$$
c_{0t}
=
\text{no-trade configuration},
$$

with realised utility:

$$
y_{0t}=0
$$

or another application-defined baseline.

The core model should not force an outside option, but it must permit one without special-case logic.

Without an outside candidate, BOLR is a relative ranking engine.

With an outside candidate, it can support abstention.

---

# 10. Computational Design

The model must remain practical for approximately 1,500 candidates per day.

Important computational quantities include:

- number of candidates $N$;
- representation dimension $D$;
- number of particles or mixture components $P$;
- covariance structure;
- number of adaptation states;
- frequency and cost of posterior updates.

A naive full-state particle filter may require approximately:

$$
O(PND)
$$

score operations per update.

For large $P$ and $D$, score evaluation rather than the generic particle-filter mechanics is likely to dominate runtime.

Promising computational strategies include:

- computing shared market embeddings once per day;
- precomputing static candidate representations;
- low-rank candidate-context interactions;
- matrix-matrix score evaluation;
- diagonal, block-diagonal, or low-rank covariance approximations;
- particles only over low-dimensional adaptation variables;
- OpenMP or SIMD parallelism;
- deterministic checkpointing;
- staged approximation levels;
- C implementation with Python bindings for research workflows.

Computational viability is a first-class research criterion, not an implementation detail to be considered after model design.

---

# 11. Candidate-Set Dependence

Softmax probabilities depend on the alternatives included in the list.

This creates possible sensitivity to:

- duplicated configurations;
- near-duplicate configurations;
- uneven grid density;
- adding or removing irrelevant alternatives;
- dense regions of configuration space splitting probability mass.

This does not invalidate a listwise softmax model, but it must become an explicit evaluation criterion.

Possible remedies to investigate include:

- local-density weighting;
- duplicate robustness;
- hierarchical candidate grouping;
- continuous configuration-space modelling;
- alternative ranking likelihoods;
- score-based decisions that are less sensitive to list composition.

---

# 12. Existing Technical Infrastructure

Two existing C-based libraries may provide useful infrastructure.

## Fast_BOCPD

Repository:

https://github.com/TiaanViviers/Fast_BOCPD

Relevant capabilities include:

- online run-length posterior updates;
- changepoint probabilities;
- streaming and batch interfaces;
- conjugate observation models;
- a pure C backend with Python access.

The most relevant contribution is the ability to maintain uncertainty over run length rather than returning only a point changepoint decision.

## fastpf

Repository:

https://github.com/TiaanViviers/fastpf

Relevant capabilities include:

- model-agnostic sequential importance resampling;
- callback-based transition and likelihood evaluation;
- log-space weights;
- ESS diagnostics;
- adaptive resampling;
- optional rejuvenation;
- OpenMP parallelism;
- checkpointing and deterministic continuation.

These libraries are implementation assets, but they should not determine the final statistical model. The inference and adaptation mechanisms must be selected on theoretical and empirical grounds.

---

# 13. Research and Implementation Strategy

BOLR should be developed through controlled increases in complexity.

## Phase 0: Formal problem definition

Before implementation:

- define the candidate/context split;
- define available information at prediction time;
- define outcome delay;
- define utility and target adapters;
- define top-of-list and trading metrics;
- define runtime and memory budgets;
- define walk-forward evaluation procedures.

## Phase 1: Minimal dynamic contextual ranker

Start with:

- linear or low-rank contextual score;
- fixed target adapter;
- fixed process-noise structure;
- no BOCPD;
- no generic nonlinear random features;
- one or more tractable inference baselines.

Goal:

> Determine whether dynamic Bayesian contextual adaptation provides value beyond static and periodically retrained rankers.

## Phase 2: Observation-model comparison

Compare:

- robust soft targets;
- binned relevance labels;
- rank-based targets;
- partial-order or ordinal likelihoods;
- proper ranking likelihoods;
- generalized Bayesian updates.

Goal:

> Determine which observation model best aligns learning with stable top-of-list net utility.

## Phase 3: Inference comparison

Compare:

- full-state particle filtering;
- Rao-Blackwellised SMC;
- Gaussian filtering approximations;
- online variational methods;
- mixture approximations.

Goal:

> Determine the best accuracy, calibration, robustness, and computational trade-off.

## Phase 4: Nonlinear representation

Compare:

- structured manual interactions;
- low-rank bilinear interactions;
- context-gated experts;
- joint RFF;
- modular or product-kernel RFF;
- hybrid models.

Goal:

> Add nonlinear capacity without losing statistical efficiency or online feasibility.

## Phase 5: Structured adaptation

Introduce:

- feature-family-specific process noise;
- latent adaptation states;
- adaptive covariance inflation;
- stable and fast parameter blocks.

Goal:

> Learn which parts of the ranking function should adapt and at what speed.

## Phase 6: BOCPD integration

Investigate:

- market-state BOCPD;
- predictive-surprise BOCPD;
- dual exogenous/endogenous monitoring;
- run-length-conditioned rankers;
- changepoint-triggered covariance inflation or partial reset.

Goal:

> Improve preventative and reactive adaptation to abrupt distributional change.

## Phase 7: Decision policy

Investigate:

- posterior mean ranking;
- probability-best selection;
- probability-top-$K$ selection;
- outside-option selection;
- uncertainty thresholds;
- allocation and position sizing.

Goal:

> Convert posterior ranking information into robust trading decisions.

## Phase 8: Production implementation

After the mathematical design is validated:

- implement performance-critical components in C;
- expose Python bindings;
- support checkpointing and deterministic continuation;
- add benchmark and diagnostic tooling;
- preserve modular research interfaces.

---

# 14. Evaluation Philosophy

BOLR must not be judged by one ranking metric alone.

Evaluation should include:

## Ranking quality

- NDCG or top-weighted ranking metrics;
- precision at $K$;
- regret relative to the best available candidate;
- pairwise ranking accuracy;
- selected-candidate rank.

## Trading utility

- realised net PnL;
- risk-adjusted return;
- drawdown;
- turnover and transaction costs;
- stability of selected configurations;
- performance of top-$K$ portfolios.

## Adaptation

- recovery time after drift;
- performance before and after changepoints;
- sensitivity to gradual drift;
- stability during stationary periods;
- false adaptation frequency.

## Uncertainty quality

- calibration of probability-best and probability-top-$K$;
- posterior predictive coverage;
- usefulness of uncertainty for abstention;
- correlation between uncertainty and realised error.

## Robustness

- sensitivity to target construction;
- sensitivity to candidate duplication;
- sensitivity to configuration-grid density;
- sensitivity to hyperparameters;
- performance under reduced sample size;
- performance under synthetic and historical regime changes.

## Computation

- update latency;
- prediction latency;
- memory usage;
- scaling in $N$, $D$, and $P$;
- reproducibility;
- numerical stability.

All complex additions should be evaluated through ablation. A component should remain in the final system only if it solves a clearly stated problem and provides measurable value.

---

# 15. Main Research Questions

1. Does dynamic Bayesian adaptation outperform static estimation, rolling windows, and periodic retraining?

2. Does explicit modelling of configuration-market interactions improve regime-dependent ranking?

3. Which contextual representation provides the best trade-off between flexibility, interpretability, sample efficiency, and computation?

4. Which target construction best aligns model learning with realised top-of-list net trading utility?

5. Does generalized Bayesian cross-entropy provide a better practical update rule than proper ranking likelihoods?

6. Can binning or ordinal relevance grades reduce variance without discarding economically important information?

7. Is full-state particle filtering viable at the required dimensionality?

8. Do Rao-Blackwellised, Gaussian, variational, or ensemble approximations provide better posterior quality per unit of computation?

9. How should process noise and adaptation be structured across feature families?

10. Can market-state BOCPD provide useful preventative adaptation before ranking performance deteriorates?

11. Does predictive-surprise monitoring detect concept drift that is invisible in market-state covariates?

12. Does combining exogenous and endogenous drift monitoring outperform either signal alone?

13. Do Random Fourier Features provide useful nonlinear capacity under limited effective sample size?

14. Are modular or product-kernel random features superior to a generic joint RBF approximation?

15. How sensitive is the model to duplicate candidates, near-duplicate candidates, and uneven configuration-space density?

16. Does posterior uncertainty improve top-1 selection, top-$K$ selection, abstention, or allocation?

17. Can a user-supplied no-trade candidate be calibrated reliably as an outside option?

18. Which posterior decision quantity is most useful: expected score, expected ranking mass, probability best, probability top-$K$, or a risk-adjusted posterior utility?

19. Can the complete system meet the required runtime and memory budget in a C implementation?

20. Which components remain valuable after controlled ablation against strong ranking and online-learning baselines?

---

# 16. Planned Sub-Research Documents

The main proposal provides the project overview. Each major unresolved area will receive a dedicated research document.

## `01 - Problem Formulation and Evaluation.md`

Topics:

- prediction-time information;
- outcome timing;
- leakage prevention;
- trading utility;
- walk-forward validation;
- baselines;
- ranking and trading metrics;
- falsifiable success criteria.

## `02 - Contextual Representation and Score Functions.md`

Topics:

- candidate versus shared context;
- additive cancellation;
- varying-coefficient models;
- low-rank bilinear interactions;
- context-gated experts;
- representation identifiability;
- computational structure.

## `03 - Nonlinear Features and Random Fourier Features.md`

Topics:

- joint RFF;
- modular random features;
- product kernels;
- feature scaling;
- kernel bandwidth;
- approximation dimension;
- static versus adaptive random bases;
- comparison with structured interactions.

## `04 - Target Construction and Observation Models.md`

Topics:

- raw and robust utility;
- binning;
- ordinal grades;
- hybrid absolute-relative labels;
- soft targets;
- Plackett-Luce;
- partial rankings;
- generalized Bayes;
- likelihood and target temperature.

## `05 - Sequential Inference.md`

Topics:

- full-state particle filtering;
- degeneracy in high dimensions;
- Rao-Blackwellised SMC;
- Laplace filtering;
- assumed-density filtering;
- variational filtering;
- ensemble methods;
- posterior diagnostics;
- calibration and approximation error.

## `06 - Dynamic Adaptation and Process Noise.md`

Topics:

- random-walk dynamics;
- structured $Q$;
- feature-family drift;
- latent adaptation states;
- shrinkage;
- covariance inflation;
- stability-adaptivity trade-offs.

## `07 - BOCPD and Drift Detection.md`

Topics:

- covariate drift;
- concept drift;
- market-state representations;
- predictive surprise;
- exogenous and endogenous monitors;
- run-length-conditioned inference;
- preventative versus reactive adaptation.

## `08 - Posterior Prediction and Decision Policy.md`

Topics:

- expected score;
- expected ranking mass;
- probability best;
- probability top-$K$;
- pairwise dominance;
- uncertainty-aware abstention;
- outside options;
- allocation and position sizing.

## `09 - Candidate-Set Dependence and Configuration Geometry.md`

Topics:

- duplicate candidates;
- grid density;
- irrelevant alternatives;
- continuous configuration space;
- local-density weighting;
- hierarchical candidate structure;
- robustness tests.

## `10 - Computational Architecture and C Implementation.md`

Topics:

- complexity analysis;
- matrix operations;
- memory layout;
- covariance representation;
- parallelism;
- fastpf integration;
- Fast_BOCPD integration;
- checkpointing;
- Python bindings;
- benchmarks.

## `11 - Experimental Programme and Ablation Plan.md`

Topics:

- staged experiments;
- synthetic drift experiments;
- historical walk-forward experiments;
- baseline selection;
- ablation matrix;
- statistical comparison;
- stopping and rejection criteria.

---

# 17. Current Working Hypotheses

The following are working hypotheses rather than settled design decisions.

1. A contextual score that separates stable configuration effects from market-conditioned effects will be more data-efficient than an unstructured nonlinear model.

2. A dynamic Bayesian model will adapt more smoothly than periodic retraining and avoid arbitrary rolling-window boundaries.

3. Structured process noise will outperform one global adaptation rate.

4. Robust or binned ranking targets will be more stable than raw PnL softmax targets.

5. Full-state particle filtering will become inefficient as representation dimension grows.

6. Rao-Blackwellised or approximate Gaussian methods may provide a better computational trade-off, but their assumptions must be tested.

7. Random Fourier Features may provide useful modular nonlinear capacity, especially when combined with structured candidate-context decomposition.

8. Market-state BOCPD may provide preventative adaptation, while predictive-surprise monitoring may be necessary for detecting concept drift.

9. Posterior uncertainty will be most valuable when connected to abstention, top-$K$ selection, or allocation rather than used only as a descriptive output.

10. Candidate-set dependence will be practically important because the configurations form a dense and highly related grid rather than a collection of independent alternatives.

---

# 18. Non-Goals at the Current Stage

The current proposal does not assume:

- that particle filtering is the final inference method;
- that Gaussian conditional approximations are sufficient;
- that RFF is the final nonlinear representation;
- that BOCPD is necessarily beneficial;
- that softmax probabilities are calibrated probabilities of being best;
- that raw PnL is an appropriate target;
- that every day should produce an active trade;
- that all parameters should adapt at the same rate;
- that one ranking metric is sufficient for validation.

These questions must be resolved through research and experiments.

---

# 19. Long-Term Vision

The long-term goal is a probabilistic ranking engine capable of:

- ranking approximately 1,500 configurations per daily inference call;
- adapting continuously to changing relationships;
- maintaining uncertainty over ranking functions and adaptation states;
- incorporating shared market context without losing candidate-specific structure;
- supporting several coherent observation models;
- representing gradual and abrupt change;
- providing probability-best, probability-top-$K$, and uncertainty measures;
- supporting a user-supplied no-trade alternative;
- operating efficiently in C;
- exposing Python bindings for research and integration;
- remaining modular enough to support new features, targets, and decision policies.

The intended contribution is not a single isolated algorithmic novelty.

It is the design and validation of a coherent system combining:

> **dynamic Bayesian learning, contextual listwise ranking, structured market-configuration interactions, uncertainty-aware decisions, and computationally bounded online inference for non-stationary trading environments.**
