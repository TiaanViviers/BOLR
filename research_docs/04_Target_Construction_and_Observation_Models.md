# 04 - Target Construction and Observation Models

> **Project:** Bayesian Online Listwise Ranking (BOLR)  
> **Document purpose:** Define how user-supplied utility becomes a statistical observation for sequential Bayesian updating.  
> **Status:** Research specification with provisional decisions  
> **Empirical foundation:** `01.5_EDA.md`  
> **Representation foundation:** `02 - Contextual Representation and Score Functions.md`  
> **Research basis:** `Target Construction and Bayesian Observation Models research.md`

---

# 1. Purpose of This Document

BOLR must update a dynamic posterior ranking surface from the full realised utility vector of approximately 1,428 configurations after each trading day.

The observation problem is not equivalent to assigning a strict rank to every configuration.

The EDA shows:

- many exact and near ties;
- large plateaus;
- strong local dependence;
- broad connected top regions;
- unstable exact winners;
- variable information content across days.

The statistical observation should therefore represent daily utility vectors as structured, partially ordered, and information-variable objects.

This document defines:

1. the separation between economic utility, target construction, observation model, and posterior decision;
2. the observation families retained for prototype comparison;
3. the treatment of ties, near-ties, plateaus, and weak-information days;
4. the provisional target-adapter interface;
5. the decisions that are now accepted;
6. the questions that remain unresolved.

This document does **not** yet select one final observation likelihood.

---

# 2. Core Conceptual Pipeline

The full BOLR observation pipeline is:

$$
y_{it}
\rightarrow
u_{it}
\rightarrow
\mathcal O_t
\rightarrow
p(\theta_t\mid D_{1:t})
\rightarrow
d_{it}.
$$

Where:

- $y_{it}$ is the raw realised outcome;
- $u_{it}$ is the user-defined economic utility;
- $\mathcal O_t$ is the statistical observation produced by the target adapter;
- $p(\theta_t\mid D_{1:t})$ is the updated posterior;
- $d_{it}$ is the production decision quantity.

These objects must remain separate.

A specific economic utility does not uniquely determine:

- a ranking target;
- a likelihood;
- a posterior probability interpretation;
- a deployment decision rule.

BOLR should remain agnostic to the user's economic definition of utility while requiring the mathematical type of the resulting observation to be explicit.

---

# 3. Raw Outcome and Economic Utility

Let:

$$
y_{it}
$$

be the realised raw outcome of configuration $i$ on day $t$.

Examples include:

- net points;
- net currency return;
- gross PnL;
- execution-adjusted PnL.

The user supplies a utility transformation:

$$
u_{it}
=
U(y_{it},a_{it},\xi_t),
$$

where:

- $a_{it}$ may contain candidate-specific risk information;
- $\xi_t$ may contain day-level information;
- $U$ is externally defined by the user.

Possible utilities include:

- raw net points;
- risk-adjusted PnL;
- downside-penalised utility;
- drawdown-aware utility;
- another scalar objective.

The core model should not hardcode one economic objective.

However, the observation adapter must know whether the supplied utility is:

- cardinal;
- ordinal;
- thresholded;
- relative;
- bounded;
- heavy-tailed;
- comparable across days.

---

# 4. Why a Strict Total Ranking Is Rejected

A strict daily total order would impose:

$$
i_{1,t}
\succ
i_{2,t}
\succ
\cdots
\succ
i_{N,t}.
$$

For BOLR, this is often artificial.

The EDA found:

- mean identical-neighbour fraction of approximately 35%;
- mean largest plateau fraction of approximately 16.87%;
- broad connected top regions;
- exact best configuration changing on approximately 98.62% of transitions;
- very low consecutive top-set overlap.

A strict ranking would create preferences between candidates that:

- earned exactly the same utility;
- differ only by an economically meaningless amount;
- followed the same realised trade path;
- belong to the same broad high-quality region.

Therefore:

> Full strict ranking is not the default semantic observation for BOLR.

A complete Plackett–Luce likelihood may remain useful as a methodological reference, but it is not the leading observation model.

---

# 5. Observation Types Retained for Research

The observation layer should support a small set of explicit types.

A provisional daily observation object is:

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
- $V_t$ contains the target values or groups;
- $W_t$ contains weights, temperatures, or update strength;
- $M_t$ contains masks, group membership, or ignored candidates;
- $I_t$ contains information diagnostics.

Candidate observation types include:

```text
SOFT_TARGET
ORDERED_GROUPS
TOP_SET
DIRECT_UTILITY
NO_UPDATE
```

The first prototype programme should focus on three families:

1. tolerance-aware soft-target generalized Bayes;
2. tolerance-aware nested top-set or ordered groups;
3. robust direct utility.

---

# 6. Ties and Near-Ties

Ties are a first-class part of the observation.

## 6.1 Exact ties

If:

$$
u_{it}=u_{jt},
$$

then configurations $i$ and $j$ should normally be treated as statistically equivalent within the target adapter.

No arbitrary internal ordering should be introduced.

## 6.2 Near-ties

Configurations may be treated as equivalent when:

$$
|u_{it}-u_{jt}|
\leq
\varepsilon_t.
$$

The tolerance may combine several concepts:

$$
\varepsilon_t
=
\max
\left(
\varepsilon_{\text{abs}},
\lambda\widehat{\sigma}^{\text{robust}}_t,
\varepsilon_{\text{exec}}
\right).
$$

Where:

- $\varepsilon_{\text{abs}}$ is a fixed economic tolerance;
- $\widehat{\sigma}^{\text{robust}}_t$ is a robust daily spread estimate;
- $\lambda$ controls the relative tolerance;
- $\varepsilon_{\text{exec}}$ represents execution or reconstruction uncertainty.

This formula is provisional.

The adapter should allow:

- absolute tolerance only;
- relative tolerance only;
- hybrid tolerance;
- no tolerance beyond exact equality.

## 6.3 Connected and disconnected ties

Connectedness in configuration space should not create an artificial economic order.

Two disconnected candidates with equal utility remain tied.

Geometry may still be used for:

- plateau diagnostics;
- region construction;
- effective-information diagnostics;
- evaluation.

The target adapter should avoid using geometry to invent preferences not present in the utility.

---

# 7. Plateaus and High-Quality Regions

A plateau is a set of configurations whose utilities are equal or tolerance-equivalent.

A high-quality region may be defined as:

$$
\mathcal T_t
=
\left\{
i:
u_{it}
\geq
u_t^{\max}
-
\varepsilon_t
\right\}.
$$

Alternatively, an absolute-relative rule may be used:

$$
\mathcal T_t
=
\left\{
i:
u_{it}
\geq
\max
\left(
\tau_{\text{abs}},
u_t^{\max}-\varepsilon_t
\right)
\right\}.
$$

Possible region definitions include:

- tolerance around the best;
- top percentage;
- top $K$;
- above an absolute floor;
- above a relative quantile;
- hybrid absolute-relative rule;
- connected component of a broader qualifying set.

The initial model should avoid assuming that the exact winner is the only informative target.

Posterior region membership is a central decision quantity:

$$
P(i\in\mathcal T_t\mid D_{1:t-1}).
$$

---

# 8. Candidate A: Tolerance-Aware Soft-Target Generalized Bayes

This is the simplest credible observation model.

## 8.1 Model probabilities

Let:

$$
q_{it}(\theta_t)
=
\frac{\exp(s_{it}(\theta_t))}
{\sum_j\exp(s_{jt}(\theta_t))}.
$$

## 8.2 Robust utility transformation

Define a transformed utility:

$$
v_{it}
=
g_t(u_{it}).
$$

One possible transform is:

$$
v_{it}
=
\operatorname{clip}
\left(
\frac{
u_{it}-\operatorname{median}(u_t)}{a_t},
-c,c
\right),
$$

where:

$$
a_t
=
\max
\left(
\operatorname{MAD}(u_t),
c_{\text{IQR}}\operatorname{IQR}(u_t),
a_{\min}
\right).
$$

Alternative transforms include:

- rank transformation;
- ordinal grade scores;
- thresholded positive utility;
- clipped raw utility;
- tolerance-collapsed utility groups.

## 8.3 Soft target

Define:

$$
r_{it}
=
\frac{\exp(\kappa v_{it})}
{\sum_j\exp(\kappa v_{jt})}.
$$

Where $\kappa$ controls target concentration.

## 8.4 Loss

Use listwise cross-entropy:

$$
\mathcal L_t(\theta_t)
=
-
\sum_i
r_{it}
\log q_{it}(\theta_t).
$$

## 8.5 Generalized Bayesian update

$$
p(\theta_t\mid D_{1:t})
\propto
p(\theta_t\mid D_{1:t-1})
\exp
\left(
-\eta_t\mathcal L_t(\theta_t)
\right).
$$

This should be interpreted as a Gibbs or generalized posterior unless an explicit generative interpretation is adopted.

## 8.6 Initial policy

For the first prototype:

$$
\eta_t=\eta_0
$$

on ordinary update days.

Do not initially use a complex adaptive update-strength formula.

Possible exceptions:

$$
\eta_t=0
$$

for `NO_UPDATE`, or:

$$
0<\eta_t<\eta_0
$$

for an explicitly reduced-strength policy.

## 8.7 Advantages

- simple;
- smooth;
- low computational cost;
- easy gradients and Hessians;
- easy C implementation;
- compatible with Gaussian posterior approximations;
- useful baseline for representation experiments.

## 8.8 Limitations

- not automatically a literal likelihood;
- sensitive to utility transformation;
- $\kappa$ and $\eta$ may be confounded;
- probability outputs need careful interpretation;
- duplicate candidates can split target and model mass;
- ties are only handled correctly if the adapter is tolerance-aware.

---

# 9. Candidate B: Tolerance-Aware Nested Top-Set or Ordered Groups

This is the leading semantically structured observation family.

## 9.1 Ordered groups

Construct:

$$
G_{H,t}
\succ
G_{M,t}
\succ
G_{L,t},
$$

where:

- $G_{H,t}$ is the high-quality group;
- $G_{M,t}$ is a secondary acceptable or positive group;
- $G_{L,t}$ is the remainder.

No internal order is assumed within each group.

A binary version is:

$$
G_{H,t}
\succ
G_{L,t}.
$$

## 9.2 Possible group construction

One possible rule is:

$$
G_{H,t}
=
\left\{
i:
u_{it}
\geq
\max
\left(
\tau_{\text{abs}},
u_t^{\max}-\varepsilon_t
\right)
\right\}.
$$

A middle group may be:

$$
G_{M,t}
=
\left\{
i:
u_{it}>\tau_{\text{rel}}
\right\}
\setminus G_{H,t}.
$$

The remaining candidates form:

$$
G_{L,t}
=
\{1,\ldots,N\}
\setminus
(G_{H,t}\cup G_{M,t}).
$$

Other group rules may use:

- quantiles;
- current relevance grades;
- positive versus non-positive utility;
- tolerance-collapsed levels;
- user-controlled ordinal bins.

## 9.3 Desired likelihood semantics

The observation should state:

- every member of $G_H$ is preferred to every member of $G_M$;
- every member of $G_M$ is preferred to every member of $G_L$;
- no order is imposed within a group.

This is an ordered partition or partitioned preference.

## 9.4 Candidate likelihood families

The exact likelihood remains unresolved.

Research candidates include:

- grouped Plackett–Luce;
- tied Plackett–Luce;
- partitioned-preference likelihoods;
- setwise choice models;
- cumulative ordinal constructions;
- generalized Bayesian ordered-group losses.

## 9.5 Advantages

- direct treatment of ties;
- direct treatment of broad top regions;
- strong top-of-list emphasis;
- no artificial within-group ordering;
- interpretable posterior region probabilities;
- less sensitive to tiny utility differences.

## 9.6 Limitations

- exact grouped likelihood may be computationally difficult;
- group construction introduces thresholds;
- cardinal utility information is discarded within groups;
- candidate-set dependence may remain;
- sequential approximation depends strongly on the final likelihood.

## 9.7 Current status

Candidate B is the leading observation family conceptually.

The unresolved issue is:

> Which exact ordered-group likelihood or loss is computationally feasible and statistically coherent for BOLR?

This must be answered jointly with the sequential-inference research.

---

# 10. Candidate C: Robust Direct Utility Model

This is the information-rich alternative.

## 10.1 Observation model

Model the utility vector directly:

$$
u_t
\mid
\theta_t
\sim
t_\nu
\left(
\mu_t(\theta_t),
\Sigma_t
\right).
$$

Where:

$$
\mu_{it}(\theta_t)
=
s_{it}(\theta_t).
$$

A Gaussian model may be used as a simpler baseline, but heavy tails are likely more appropriate.

## 10.2 Initial covariance choices

Possible starting points:

### Diagonal

$$
\Sigma_t=D.
$$

### Low-rank plus diagonal

$$
\Sigma_t=UU^\top+D.
$$

### Graph-structured plus diagonal

$$
\Sigma_t=\Sigma_{\text{graph}}+D.
$$

### Low-rank plus graph plus diagonal

$$
\Sigma_t=UU^\top+\Sigma_{\text{graph}}+D.
$$

A full dense covariance is unlikely to be practical.

## 10.3 Posterior decision quantities

From posterior predictive draws:

$$
u_t^{(b)}
\sim
p(u_t\mid D_{1:t-1}),
$$

estimate:

$$
P(i=\text{best})
\approx
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf 1
\left\{
i=\arg\max_j u_{jt}^{(b)}
\right\},
$$

and:

$$
P(i\in\operatorname{TopK})
\approx
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf 1
\left\{
i\in\operatorname{TopK}(u_t^{(b)})
\right\}.
$$

Posterior expected utility is:

$$
E[u_{it}\mid D_{1:t-1}].
$$

## 10.4 Advantages

- retains cardinal information;
- supports expected utility;
- supports predictive ranking probabilities;
- does not require arbitrary grades;
- naturally includes negative days;
- clean generative interpretation when the likelihood is credible.

## 10.5 Limitations

- highly sensitive to utility scale;
- cardinal fit may not emphasise the top sufficiently;
- within-day dependence must be modelled for calibrated uncertainty;
- probability-best may be overconfident under diagonal noise;
- heavier computational burden;
- covariance design becomes a major research problem.

## 10.6 Current role

Candidate C should be treated as an information-rich reference model.

Its posterior mean predictions may be useful even when its ranking-probability calibration is not yet fully credible.

---

# 11. Ordinal Grades

Ordinal grades remain a useful adapter family.

Let:

$$
b_{it}
\in
\{0,1,\ldots,K\}.
$$

Possible meanings include:

- non-positive;
- positive;
- strong positive;
- top region.

An ordinal likelihood does not require equal spacing between grades.

The current four-grade target is useful as:

- a reference baseline;
- an interpretable adapter;
- a comparison against continuous and set-based targets.

However, a plain itemwise ordinal likelihood may overcount same-day information because candidates are strongly dependent.

Ordinal grades are therefore more promising when used as:

- ordered groups;
- a coarse target for generalized Bayes;
- a structured adapter before a listwise model.

The exact current heuristic should not be treated as the final BOLR target.

---

# 12. Pairwise Models

Pairwise models include:

- Bradley–Terry;
- Thurstone–Mosteller;
- pairwise logistic likelihood;
- pairwise probit likelihood.

A pairwise observation is:

$$
i\succ j.
$$

Full pair construction costs:

$$
O(N^2).
$$

For:

$$
N=1428,
$$

this is excessive.

Possible reductions include:

- cross-group pairs only;
- top-versus-bottom pairs;
- sampled pairs;
- neighbour pairs;
- hard-negative pairs.

Pairwise methods are retained as:

- sparse baselines;
- fallbacks;
- diagnostic models.

They are not the leading observation family because they discard available list structure.

---

# 13. Plackett–Luce and Random-Utility Models

For a strict ranking:

$$
\pi_t
=
(\pi_{1,t},\ldots,\pi_{N,t}),
$$

the Plackett–Luce likelihood is:

$$
P(\pi_t\mid s_t)
=
\prod_{k=1}^{N}
\frac{
\exp(s_{\pi_{k,t},t})
}{
\sum_{\ell=k}^{N}
\exp(s_{\pi_{\ell,t},t})
}.
$$

Advantages include:

- proper ranking likelihood;
- random-utility interpretation;
- efficient strict-ranking factorisation;
- natural top-$K$ truncation.

However:

- strict order is not credible for plateau-heavy outcomes;
- tied-group handling is more complex;
- independence of irrelevant alternatives is problematic for dense grids;
- duplicate or near-duplicate candidates may split probability mass.

Full strict Plackett–Luce is therefore rejected as the default.

Grouped, tied, or partial Plackett–Luce remains a candidate implementation for Candidate B.

---

# 14. Candidate-Set Dependence

The candidate grid is dense and highly structured.

Observation models based on discrete softmax worths may change when:

- duplicate candidates are added;
- near-duplicates are added;
- grid density changes;
- one region is sampled more densely;
- candidates are removed.

This affects:

- probability-best;
- softmax target mass;
- Plackett–Luce worth;
- pairwise comparison counts.

The current fixed candidate set reduces immediate operational risk.

Nevertheless, robustness tests must include:

- grid thinning;
- grid densification;
- duplicate injection;
- near-duplicate injection.

Region-level quantities are expected to be more stable than exact candidate probabilities.

---

# 15. Effective Information Content

The daily list contains 1,428 candidates, but far fewer independent outcomes.

Define:

$$
N_{\text{eff},t}
<
N.
$$

Possible diagnostics include:

## Unique or tolerance-collapsed utility levels

$$
N_{\text{eff},t}^{(u)}
=
\#\{
\text{tolerance groups}
\}.
$$

## Target perplexity

$$
N_{\text{eff},t}^{(H)}
=
\exp(H(r_t)).
$$

## Plateau count

$$
N_{\text{eff},t}^{(\pi)}
=
\#\{
\text{plateaus or ordered groups}
\}.
$$

## Distinct trade paths

$$
N_{\text{eff},t}^{(\text{path})}
=
\#\{
\text{unique realised trade paths}
\}.
$$

These quantities should initially be:

- recorded;
- analysed;
- used for calibration stratification;
- used for robustness diagnostics.

They should not initially control update strength automatically.

An adaptive rule such as:

$$
\eta_t
=
\eta_0
\left(
\frac{N_{\text{eff},t}}{N_{\text{ref}}}
\right)^\alpha
\left(
1-
\frac{H_t}{\log N}
\right)^\beta
$$

is a research hypothesis, not a frozen design.

---

# 16. All-Irrelevant and Weak-Information Days

A day may contain:

- no positive utility;
- no candidate above an absolute threshold;
- nearly identical utilities;
- only weakly distinguishable candidates.

Possible policies are:

```text
NO_UPDATE
REDUCED_STRENGTH_RELATIVE_UPDATE
FULL_RELATIVE_UPDATE
OUTSIDE_OPTION
```

## 16.1 No update

The state evolves according to its transition model, but no observation update is applied.

## 16.2 Reduced-strength relative update

The model learns relative differences among poor candidates, but the day receives a smaller update weight.

## 16.3 Full relative update

The least-negative candidates are treated as fully informative relative winners.

## 16.4 Outside option

A no-trade candidate with baseline utility is included in the observation.

## 16.5 Current provisional policy

No single policy is frozen.

For initial raw-PnL experiments compare:

1. `NO_UPDATE`;
2. `REDUCED_STRENGTH_RELATIVE_UPDATE`.

Because all-non-positive days are only approximately 2.27% of the dataset, this choice should not dominate the first prototype.

A uniform soft target is not equivalent to no information and should not be used as the default no-update representation.

---

# 17. Target Concentration and Update Strength

Several scale parameters must remain distinct.

## Target concentration

$$
\kappa
$$

controls how sharply utility differences become target preferences.

## Generalized Bayes learning rate

$$
\eta
$$

controls how strongly the posterior trusts the daily loss.

## Model prediction temperature

$$
\tau
$$

controls score-to-probability scaling:

$$
q_{it}
=
\frac{
\exp(s_{it}/\tau)
}{
\sum_j\exp(s_{jt}/\tau)
}.
$$

These parameters can be confounded.

The initial design should therefore:

- fix prediction temperature, likely $\tau=1$;
- tune a small set of target concentrations $\kappa$;
- use fixed update strength $\eta_0$;
- avoid simultaneous unconstrained estimation of all scales.

---

# 18. Posterior Decision Quantities

The observation model does not determine the production decision automatically.

Possible posterior quantities include:

## Posterior mean score

$$
E[s_{it}\mid D_{1:t-1}].
$$

## Posterior expected utility

$$
E[u_{it}\mid D_{1:t-1}].
$$

## Probability best

$$
P(i=\arg\max_j u_{jt}\mid D_{1:t-1}).
$$

## Probability top-$K$

$$
P(i\in\operatorname{TopK}(u_t)\mid D_{1:t-1}).
$$

## Probability high-quality region

$$
P(i\in\mathcal T_t\mid D_{1:t-1}).
$$

## Pairwise dominance

$$
P(u_{it}>u_{jt}\mid D_{1:t-1}).
$$

## Posterior entropy

Used as a measure of uncertainty or ambiguity.

The downstream allocation model may consume several of these quantities.

The ranking model should not hardcode the final capital-allocation policy.

---

# 19. Provisional Target-Adapter Interface

A target adapter should produce a structured object similar to:

```text
Observation {
    type
    utility_values
    transformed_values
    tolerance
    groups
    target_probabilities
    relevance_baseline
    all_irrelevant_flag
    update_weight
    diagnostics
}
```

Suggested diagnostics:

```text
positive_count
non_positive_count
unique_utility_count
tolerance_group_count
target_entropy
top_group_size
middle_group_size
plateau_count
largest_plateau_fraction
all_irrelevant_flag
utility_scale
utility_spread
```

The adapter should be deterministic for fixed inputs and configuration.

The adapter should not mutate the original dataset.

---

# 20. Prototype Comparison Programme

Use one common score representation:

$$
s_{it}
=
\phi_c(c_i)^\top
\left(
\alpha+B\psi(m_t)
\right)
+
\gamma^\top z_{it}.
$$

Hold the representation and inference approximation constant where possible.

Only the observation layer should change.

## 20.1 Prototype A

**Tolerance-aware soft-target generalized Bayes**

Initial choices:

- robust or rank-based utility transform;
- small grid over $\kappa$;
- fixed $\eta$;
- fixed prediction temperature;
- optional `NO_UPDATE`.

## 20.2 Prototype B

**Nested top-set or ordered groups**

Initial choices:

- two or three groups;
- no within-group order;
- tolerance-aware high group;
- exact likelihood selected after feasibility research;
- same score representation as Prototype A.

## 20.3 Prototype C

**Student-$t$ direct utility**

Initial choices:

- same mean surface;
- diagonal or simple low-rank-plus-diagonal covariance;
- fixed degrees of freedom or small candidate set;
- posterior mean and predictive ranking diagnostics.

---

# 21. Evaluation Contract

All observation models should use true prequential evaluation:

1. predict day $t$;
2. rank candidates;
3. select the deployment candidate;
4. reveal the full utility vector;
5. update once;
6. proceed to day $t+1$.

## 21.1 Business metrics

- top-1 net points;
- average daily selected utility;
- positive-day rate;
- regret;
- downside performance;
- cumulative utility.

## 21.2 Ranking metrics

- selected-candidate rank;
- NDCG;
- top-$K$ positive rate;
- top-set overlap;
- probability of selecting from the high-quality region.

## 21.3 Calibration metrics

- probability-best calibration;
- probability-top-$K$ calibration;
- high-quality-region calibration;
- predictive log score;
- calibration by plateau size;
- calibration by target entropy;
- calibration by effective-information diagnostics.

## 21.4 Robustness

- utility rescaling;
- outlier injection;
- exact ties;
- near-ties;
- grid thinning;
- grid densification;
- duplicate injection;
- all-non-positive days;
- broad versus narrow top regions.

## 21.5 Computation

- update runtime;
- prediction runtime;
- memory;
- numerical stability;
- C implementation complexity.

---

# 22. Accepted Design Principles

The following decisions are now provisionally accepted.

1. Economic utility and statistical observation are separate layers.

2. Full strict rankings are not the default observation.

3. Exact ties should not be broken arbitrarily.

4. Near-tie tolerance should be configurable.

5. High-quality regions are more meaningful than exact winners on many days.

6. Soft-target generalized Bayes is the primary engineering baseline.

7. Nested top-set or ordered-group observations are the leading structured family.

8. Direct robust utility is the information-rich alternative.

9. Adaptive daily update strength should not be included in the first prototype.

10. Effective-information diagnostics should be recorded before being used.

11. Full Plackett–Luce is not the leading model.

12. Pairwise methods are secondary baselines.

13. The all-irrelevant-day policy remains configurable.

14. Posterior decision quantities remain separate from training observations.

---

# 23. Unresolved Questions

The following remain open.

1. What exact tolerance rule should be used?

2. Should tolerance operate on raw utility, transformed utility, or both?

3. What exact high-quality-region definition is appropriate?

4. Should the target use two groups, three groups, or variable group count?

5. Which exact grouped or setwise likelihood should implement Candidate B?

6. Can that likelihood be updated efficiently under sequential approximate inference?

7. Should the soft-target baseline use robust cardinal scaling, ranks, or ordinal scores?

8. How should $\kappa$ be tuned?

9. How should generalized Bayes learning rate $\eta$ be calibrated?

10. Should all-negative days be ignored or weakly ranked?

11. Is an outside no-trade candidate part of the ranker or a downstream allocation model?

12. What covariance structure is required for calibrated direct-utility uncertainty?

13. Which posterior quantity is most stable under grid changes?

14. How should candidate-set density affect region probabilities?

15. Can effective-information measures improve calibration without double-counting entropy?

---

# 24. Sequential-Inference Implications

The observation choice constrains the inference architecture.

## Soft-target generalized Bayes

Likely compatible with:

- Laplace filtering;
- assumed-density filtering;
- online variational inference;
- ensemble Gaussian approximations;
- particle methods for small nonlinear state blocks.

## Ordered groups

Compatibility depends on:

- concavity;
- gradient and Hessian structure;
- tied-group likelihood cost;
- augmentation schemes;
- approximation quality.

## Direct utility

Likely compatible with:

- Gaussian or Student-$t$ filtering;
- latent scale-mixture augmentation;
- robust variational filtering;
- Rao–Blackwellised approaches if covariance structure permits.

The next deep-research stage should therefore focus on:

> Sequential Bayesian inference for a dynamic contextual ranking surface under soft-target generalized Bayes, ordered-group observations, and robust direct-utility likelihoods.

---

# 25. Provisional Direction

The leading design direction is:

$$
\boxed{
\text{tolerance-aware nested top-region observation}
}
$$

with:

$$
\boxed{
\text{soft-target generalized Bayes as the baseline}
}
$$

and:

$$
\boxed{
\text{robust direct utility as the information-rich reference}
}
$$

The first implementation should not begin until the sequential-inference research determines:

- which state representation can be updated reliably;
- which ordered-group likelihood is computationally viable;
- which posterior approximation preserves useful uncertainty;
- which components should be dynamic.

---

# 26. Stage Completion Criteria

This stage is complete when:

- the utility-to-observation separation is explicit;
- strict total rankings are rejected as the default;
- tie and near-tie policies are documented;
- the three-model shortlist is accepted;
- the target-adapter interface is defined;
- all-irrelevant-day policies are explicit;
- effective-information diagnostics are specified;
- unresolved likelihood and calibration questions are recorded;
- the sequential-inference research task is ready.

The next project stage is sequential Bayesian inference.
