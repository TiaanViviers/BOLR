# 05 - Sequential Bayesian Inference

> **Project:** Bayesian Online Listwise Ranking (BOLR)  
> **Document purpose:** Define the first credible sequential Bayesian inference architecture for the dynamic contextual ranking surface.  
> **Status:** Research specification with provisional architectural decisions  
> **Representation foundation:** `02 - Contextual Representation and Score Functions.md`  
> **Observation foundation:** `04 - Target Construction and Observation Models.md`  
> **Research basis:** Sequential-inference deep research report

---

# 1. Purpose of This Document

BOLR must produce one ranking before the market open, receive the full realised utility vector after the session, and update its posterior before the next prediction.

The central inference question is:

> How can BOLR adapt a structured contextual suitability surface through time while retaining useful posterior uncertainty and remaining computationally feasible?

The research has narrowed the first serious architecture to:

$$
\boxed{\text{partially dynamic contextual tensor surface}}
$$

updated by:

$$
\boxed{\text{Laplace-style Gaussian filtering}}
$$

with:

$$
\boxed{\text{soft-target generalized Bayes as the baseline observation}}
$$

and:

$$
\boxed{\text{partitioned-preference Plackett--Luce as the leading proper grouped likelihood}}.
$$

A cross-group logistic generalized-Bayes loss is retained as the simpler Candidate B fallback.

This document records what can now be frozen and what must still be resolved through the mathematical proof-of-concept.

---

# 2. Operational Inference Cycle

For each trading day $t$:

1. propagate the posterior state from day $t-1$ to day $t$;
2. construct the current design matrix from configuration geometry and market context;
3. compute posterior predictive scores for all candidates;
4. derive deployment quantities;
5. select the deployment candidate;
6. observe the full realised utility vector after the session;
7. construct the statistical observation;
8. apply one Bayesian update;
9. checkpoint the posterior state.

The information sequence is:

$$
p(\theta_{t-1}\mid D_{1:t-1})
\rightarrow
p(\theta_t\mid D_{1:t-1})
\rightarrow
p(\theta_t\mid D_{1:t}).
$$

Prediction must complete comfortably within the pre-open latency budget. Posterior updating may be slower, provided it finishes before the next trading day.

---

# 3. First-Prototype Score Model

The leading score family is:

$$
s_{it}
=
\phi_c(c_i)^\top\alpha
+
\phi_c(c_i)^\top B_t\psi(m_t)
+
\gamma^\top z_{it}.
$$

Where:

- $c_i$ contains the configuration coordinates;
- $\phi_c(c_i)$ is a centred tensor-product configuration basis;
- $m_t$ is shared market context;
- $\psi(m_t)$ is a compact context basis;
- $z_{it}$ contains optional candidate-specific rolling features;
- $\alpha$ is the long-run baseline surface;
- $B_t$ is the dynamic contextual-deformation block;
- $\gamma$ controls candidate-specific rolling features.

For the first mathematical POC, use the smaller model:

$$
s_{it}
=
\phi_c(c_i)^\top B_t\psi(m_t).
$$

The baseline and rolling-performance terms can be added after the central filtering mechanism is validated.

---

# 4. Configuration Basis

Let:

$$
u_i=\log(\text{entry percentage}_i),
$$

$$
v_i=\log(\text{sl-trail percentage}_i).
$$

Define one-dimensional bases:

$$
\phi_e(u_i)\in\mathbb R^A,
\qquad
\phi_r(v_i)\in\mathbb R^B.
$$

The tensor-product basis is:

$$
\phi_c(c_i)
=
\phi_e(u_i)\otimes\phi_r(v_i).
$$

The initial basis should be:

- low-dimensional;
- centred over the candidate grid;
- deterministic;
- precomputed;
- numerically well conditioned.

B-splines are the preferred first choice because of their local support and interpretability.

---

# 5. Context Basis

The shared market feature vector is too large to interact naively with every configuration basis term.

Define:

$$
\psi(m_t)\in\mathbb R^H.
$$

The first context basis should be compact, with approximately:

$$
H\in[5,10].
$$

Possible constructions include:

- selected raw features;
- feature-family summaries;
- PCA fitted only on historical training data;
- a small hand-designed market-state vector.

The basis must be centred, scaled, leakage-safe, and checkpointed with the model.

---

# 6. Dynamic State

The first dynamic state is:

$$
\theta_t=\operatorname{vec}(B_t).
$$

If:

$$
B_t\in\mathbb R^{p_c\times H},
$$

then:

$$
P=p_cH
$$

is the dynamic-state dimension.

A practical first target is:

$$
P\in[100,300].
$$

This is intentionally smaller than the possible final production state.

---

# 7. Static and Dynamic Blocks

The following roles are provisionally accepted.

## 7.1 Baseline surface

$\alpha$ should initially be static or estimated offline. It may later be allowed to evolve very slowly.

## 7.2 Contextual deformation

$B_t$ is the primary dynamic block. It controls how current market context changes relative suitability across the grid.

## 7.3 Rolling-performance coefficients

$\gamma$ should initially be static. A selected subset may later become dynamic if feature ablation supports it.

## 7.4 Graph or node residual

A dynamic node-level residual $r_t$ is excluded from Prototype 1. It would add approximately 1,428 state dimensions before the core filter is validated.

---

# 8. State Evolution

Use a Gaussian random walk:

$$
\theta_t
=
\theta_{t-1}+\epsilon_t,
\qquad
\epsilon_t\sim N(0,Q).
$$

The predictive moments are:

$$
m_{t\mid t-1}=m_{t-1\mid t-1},
$$

$$
P_{t\mid t-1}=P_{t-1\mid t-1}+Q.
$$

Initial process-noise choices are:

## Isotropic process noise

$$
Q=qI.
$$

## Discount-factor evolution

$$
P_{t\mid t-1}
=
\frac{1}{\delta}P_{t-1\mid t-1},
\qquad 0<\delta\le1.
$$

The first POC should compare a scalar $q$ against a scalar discount factor $\delta$. Do not begin with a separate process variance for every coefficient.

Mean reversion, heavy-tailed state noise, and regime-switching evolution are deferred.

---

# 9. Gaussian Posterior Approximation

Maintain:

$$
p(\theta_t\mid D_{1:t})
\approx
N(m_{t\mid t},P_{t\mid t}).
$$

The daily update is:

$$
p(\theta_t\mid D_{1:t})
\propto
p(\theta_t\mid D_{1:t-1})L_t(\theta_t),
$$

where $L_t$ may be a proper likelihood or a generalized-Bayesian exponential loss.

The resulting posterior is generally non-Gaussian. Laplace filtering approximates it locally around the posterior mode.

---

# 10. Laplace Filtering

Let the predictive prior be:

$$
\theta_t\sim N(m^-,P^-).
$$

Define the log posterior up to a constant:

$$
\log\widetilde p(\theta_t)
=
-\frac12
(\theta_t-m^-)^\top(P^-)^{-1}(\theta_t-m^-)
+
\log L_t(\theta_t).
$$

The posterior mode is:

$$
\hat\theta_t
=
\arg\max_{\theta_t}\log\widetilde p(\theta_t).
$$

The Laplace covariance is:

$$
P^+
=
\left[
(P^-)^{-1}
-
\nabla_\theta^2\log L_t(\hat\theta_t)
\right]^{-1}.
$$

The posterior mean approximation is:

$$
m^+=\hat\theta_t.
$$

---

# 11. Newton Update and Safeguards

At iteration $k$:

$$
g_k
=
\nabla_\theta\log\widetilde p(\theta^{(k)}),
$$

$$
H_k
=
-\nabla_\theta^2\log\widetilde p(\theta^{(k)}).
$$

Solve:

$$
H_k\Delta_k=g_k,
$$

then update:

$$
\theta^{(k+1)}
=
\theta^{(k)}+\rho_k\Delta_k.
$$

Use the predictive mean $m^-$ as the warm start.

Required safeguards:

- damped Newton steps;
- backtracking line search;
- Cholesky failure detection;
- diagonal jitter;
- maximum iteration count;
- gradient-norm convergence test;
- fallback to the predictive posterior if the update fails.

---

# 12. Candidate A: Soft-Target Generalized Bayes

Let:

$$
s_t=X_t\theta_t,
$$

where:

$$
X_t\in\mathbb R^{N\times P}.
$$

The predicted list distribution is:

$$
q_{it}
=
\frac{\exp(s_{it})}{\sum_j\exp(s_{jt})}.
$$

The target adapter supplies:

$$
r_t\in\Delta^{N-1}.
$$

The daily loss is:

$$
\mathcal L_t
=
-\sum_i r_{it}\log q_{it}.
$$

The generalized update factor is:

$$
L_t(\theta_t)
=
\exp(-\eta\mathcal L_t).
$$

---

# 13. Candidate A Derivatives

The score-space gradient is:

$$
\nabla_s\mathcal L_t=q_t-r_t.
$$

The score-space Hessian is:

$$
\nabla_s^2\mathcal L_t
=
\operatorname{Diag}(q_t)-q_tq_t^\top.
$$

Because $s_t=X_t\theta_t$:

$$
\nabla_\theta\mathcal L_t
=
X_t^\top(q_t-r_t),
$$

$$
\nabla_\theta^2\mathcal L_t
=
X_t^\top
\left[
\operatorname{Diag}(q_t)-q_tq_t^\top
\right]
X_t.
$$

Therefore:

$$
\nabla_\theta\log L_t
=
-\eta X_t^\top(q_t-r_t),
$$

and:

$$
-\nabla_\theta^2\log L_t
=
\eta X_t^\top
\left[
\operatorname{Diag}(q_t)-q_tq_t^\top
\right]
X_t.
$$

The Gaussian prior supplies curvature in directions not identified by the listwise observation.

---

# 14. Efficient Candidate A Curvature

Do not construct a dense $N\times N$ softmax covariance.

Let:

$$
W=\operatorname{Diag}(q)-qq^\top.
$$

Then:

$$
X^\top W X
=
X^\top\operatorname{Diag}(q)X
-
(X^\top q)(X^\top q)^\top.
$$

A Hessian-vector product is:

$$
Hv
=
X^\top
\left[
q\odot(Xv)-q(q^\top Xv)
\right].
$$

For the small POC, explicit $P\times P$ curvature is acceptable. Matrix-free methods become useful only when the state grows.

---

# 15. Score Identifiability

Softmax and Plackett--Luce scores are invariant to a common shift:

$$
s_{it}'=s_{it}+a_t.
$$

The initial design should remove this null direction using a centred configuration basis:

$$
\sum_{i=1}^{N}\phi_c(c_i)=0.
$$

Additional safeguards are:

- no candidate-independent intercept in the listwise score;
- a proper Gaussian prior;
- posterior precision regularisation.

A reference-candidate constraint is less attractive because it gives one arbitrary configuration a special role.

---

# 16. Candidate B Proper Likelihood

The leading proper observation model is partitioned-preference Plackett--Luce.

The adapter constructs:

$$
G_{1,t}\succ G_{2,t}\succ\cdots\succ G_{M,t},
$$

with no assumed order inside a group.

The preferred first form is:

$$
G_{H,t}\succ G_{M,t}\succ G_{L,t}.
$$

The high group contains the tolerance-aware high-quality region, the middle group contains the remaining positive or acceptable configurations, and the lower group contains the rest.

A naive likelihood sums Plackett--Luce probability over every internal permutation consistent with the ordered groups and can therefore become factorial.

The research identified partitioned-preference methods with approximate scaling:

$$
O(N+S^3),
$$

where $S$ is the largest upper partition.

This makes the method credible, but the exact equations, assumptions, derivatives, and numerical algorithm must still be implemented from the primary reference and validated.

---

# 17. Candidate B Proper-Likelihood Requirements

Before integration into the full filter, the implementation must provide:

$$
\ell_t(s_t)
=
\log p(G_{1,t}\succ\cdots\succ G_{M,t}\mid s_t),
$$

and either analytic or reliably differentiated:

$$
\nabla_s\ell_t,
$$

$$
\nabla_s^2\ell_t,
$$

or stable Hessian-vector products.

Required tests:

- brute-force agreement on tiny lists;
- finite-difference gradient checks;
- Hessian symmetry;
- translation invariance;
- extreme-score stability;
- tied-score stability;
- singleton groups;
- empty middle groups;
- large upper groups;
- duplicate candidates;
- runtime scaling.

---

# 18. Candidate B Fallback Loss

The fallback is a cross-group logistic composite loss.

For:

$$
G_1\succ G_2\succ\cdots\succ G_M,
$$

define:

$$
\mathcal P_{ab}
=
\{(i,j):i\in G_a,\ j\in G_b\},
\qquad a<b.
$$

The loss is:

$$
\mathcal L_t
=
\sum_{a<b}w_{ab}
\sum_{(i,j)\in\mathcal P_{ab}}
\log(1+\exp(s_j-s_i)).
$$

The generalized update is:

$$
L_t(\theta_t)=\exp(-\eta\mathcal L_t).
$$

This is not a proper full grouped-ranking likelihood. It is a transparent generalized posterior based on cross-group pairwise loss.

---

# 19. Cross-Group Pair Reduction

The complete cross-group pair set may still be large.

Possible strategies include:

- uniform sampling within group pairs;
- a fixed pair budget per day;
- balanced sampling across group comparisons;
- hard-negative sampling;
- neighbour-aware sampling;
- aggregation of repeated score differences.

The first POC should use deterministic random seeds, a fixed pair budget, balanced group-pair sampling, and importance weights when sampling probabilities differ.

---

# 20. Candidate B Fallback Derivatives

For one ordered pair $i\succ j$, define:

$$
d_{ij}=s_j-s_i.
$$

The loss is:

$$
\ell_{ij}=\log(1+\exp(d_{ij})).
$$

Let:

$$
p_{ij}=\sigma(d_{ij}).
$$

Then:

$$
\frac{\partial\ell_{ij}}{\partial s_i}=-p_{ij},
$$

$$
\frac{\partial\ell_{ij}}{\partial s_j}=p_{ij}.
$$

The curvature contribution is:

$$
p_{ij}(1-p_{ij})
\begin{bmatrix}
1 & -1\\
-1 & 1
\end{bmatrix}.
$$

For linear scores $s_i=x_i^\top\theta$:

$$
\nabla_\theta\ell_{ij}
=
p_{ij}(x_j-x_i),
$$

$$
\nabla_\theta^2\ell_{ij}
=
p_{ij}(1-p_{ij})
(x_j-x_i)(x_j-x_i)^\top.
$$

---

# 21. Candidate C: Robust Direct Utility

The information-rich reference model is:

$$
u_t\mid\theta_t
\sim
t_\nu(X_t\theta_t,\Sigma_t).
$$

The first version may use:

$$
\Sigma_t=\sigma^2I.
$$

This is acceptable for mean-prediction comparison, but it may overstate confidence in ranking probabilities because same-day candidate residuals are dependent.

More credible later structures include:

$$
\Sigma_t=UU^\top+D,
$$

or:

$$
\Sigma_t=\Sigma_{\text{graph}}+D.
$$

Candidate C should not block the initial Candidate A and B experiments.

---

# 22. Student-$t$ Scale Mixture

A Student-$t$ observation can be represented as:

$$
u_t\mid\lambda_t,\theta_t
\sim
N\left(X_t\theta_t,\frac{\Sigma_t}{\lambda_t}\right),
$$

with:

$$
\lambda_t
\sim
\operatorname{Gamma}\left(\frac{\nu}{2},\frac{\nu}{2}\right).
$$

This may later support conditionally Gaussian updates, robust variational filtering, or Rao--Blackwellisation if $\lambda_t$ is the only sampled variable.

---

# 23. Covariance Representation

For state dimension $P$, a full covariance costs:

$$
O(P^2)
$$

memory.

For the first POC with:

$$
P\le300,
$$

a dense covariance is feasible and gives the cleanest reference implementation.

At $P=300$, the covariance contains 90,000 entries, or about 720 KB in double precision before workspace. The main cost is the repeated $O(P^3)$ factorisation, which is acceptable for one daily update in a mathematical POC.

Later covariance families may include:

- block diagonal;
- low rank plus diagonal;
- Kronecker-factored covariance;
- sparse precision.

Do not compress covariance before the dense small-state filter is validated.

---

# 24. Matrix-Normal and Kronecker Structure

For:

$$
B_t\in\mathbb R^{p_c\times H},
$$

a matrix-normal approximation gives:

$$
\operatorname{Cov}(\operatorname{vec}(B_t))
\approx
P_m\otimes P_c.
$$

This may reduce storage and computation substantially.

However, non-Gaussian daily updates may destroy exact Kronecker structure. Projection back into a Kronecker family would introduce another approximation.

This should be considered only after the dense POC establishes the required covariance fidelity.

---

# 25. Alternative Online Approximations

## Assumed-density filtering

ADF retains a Gaussian posterior but projects the one-step updated distribution back into that family. It is the primary fallback if Laplace filtering shows unstable modes, poor calibration, or covariance collapse.

## Online variational inference

Structured Gaussian VI may use full, block, or low-rank covariance. Its main risk is posterior underdispersion, particularly with mean-field or overly restrictive covariance families.

## Expectation propagation

EP may be useful when moment matching is tractable, but site management, damping, and possible non-convergence make it unsuitable for Prototype 1.

---

# 26. Particle Methods

Full particle filtering over the coefficient state is rejected for Prototype 1 because high-dimensional importance weights are likely to collapse.

The existing `fastpf` library should not determine the model architecture.

Rao--Blackwellised particle filtering remains a later option for a genuinely low-dimensional state such as:

- latent regime;
- process-noise multiplier;
- BOCPD reset state;
- likelihood temperature;
- expert identity;
- heavy-tailed scale.

A future factorisation could be:

$$
p(\theta_t,k_t\mid D_{1:t})
=
p(\theta_t\mid k_t,D_{1:t})p(k_t\mid D_{1:t}).
$$

It should only be introduced if the small sampled state provides measurable benefits.

---

# 27. BOCPD Integration

BOCPD is deferred until the base filter is stable.

A later system may use it to monitor:

- market-context drift;
- predictive ranking mismatch;
- residual utility errors;
- posterior surprise.

Its output may control process-noise inflation, partial resets, regime probabilities, or mixture weights.

---

# 28. Offline Reference Inference

Use HMC or NUTS on reduced problems:

- smaller candidate subsets;
- smaller tensor bases;
- short windows;
- static-state versions;
- one observation family at a time.

The purpose is to compare:

- posterior means;
- posterior covariance;
- credible intervals;
- score uncertainty;
- probability-top-region estimates.

The offline reference does not need to scale to the production sequence.

---

# 29. Calibration Requirements

## State calibration

On synthetic data with known state:

- posterior coverage;
- state mean squared error;
- covariance calibration;
- adaptation delay.

## Score calibration

- credible interval coverage;
- surface recovery;
- posterior rank uncertainty.

## Decision calibration

- probability-best calibration;
- probability-top-$K$ calibration;
- probability-high-region calibration.

## Approximation calibration

Compare online posterior moments with HMC or NUTS on reduced problems.

---

# 30. Approximation-Failure Diagnostics

Track:

- Newton iteration count;
- gradient norm;
- line-search reductions;
- Cholesky failures;
- covariance condition number;
- covariance eigenvalue range;
- posterior variance shrinkage;
- posterior-predictive entropy;
- failed update count;
- fallback update count.

Warning signals include repeated maximum-iteration termination, covariance near-singularity, monotonic variance collapse, extreme sensitivity to process noise, unstable mode jumps, and posterior confidence inconsistent with realised errors.

---

# 31. Synthetic Validation Programme

Required scenarios:

1. stationary smooth surface;
2. slowly drifting high-quality region;
3. abrupt regime jump;
4. broad plateau;
5. narrow ridge;
6. hard local discontinuity;
7. multiple disconnected high-quality regions;
8. weak-information day;
9. all-irrelevant day;
10. heavy-tailed noise.

Evaluate:

- latent-state recovery;
- surface recovery;
- top-region recovery;
- adaptation speed;
- false adaptation;
- posterior coverage;
- numerical failures;
- update runtime.

---

# 32. Candidate B Tolerance Audit

Before implementing partitioned-preference PL in the full filter, apply several target adapters to the historical utility vectors.

For each rule, record:

- $|G_H|$;
- $|G_M|$;
- $|G_L|$;
- largest upper-group size;
- distribution of $S^3$;
- empty-group frequency;
- top-group fraction;
- tolerance-group count;
- all-irrelevant frequency.

Candidate rules should include:

## Relative to best

$$
G_H
=
\{i:u_i\ge u_{\max}-\varepsilon_t\}.
$$

## Absolute-relative hybrid

$$
G_H
=
\left\{
i:u_i\ge\max(\tau_{\text{abs}},u_{\max}-\varepsilon_t)
\right\}.
$$

## Positive middle group

$$
G_M
=
\{i:u_i>0\}\setminus G_H.
$$

This audit determines whether exact partitioned preference is computationally realistic under the observed group sizes.

---

# 33. Likelihood Correctness Harness

Before implementing the dynamic model, build a standalone likelihood test harness supporting:

- Candidate A soft-target loss;
- Candidate B partitioned-preference PL;
- Candidate B cross-group logistic loss.

Required tests:

- brute-force agreement on tiny lists;
- finite-difference gradient checks;
- finite-difference Hessian checks;
- Hessian symmetry;
- translation invariance;
- extreme-score stability;
- tied-score stability;
- deterministic outputs;
- runtime scaling;
- memory scaling.

This harness is a prerequisite for the historical model.

---

# 34. First POC Boundary

The first POC should include:

- centred tensor basis;
- compact context basis;
- dynamic contextual interaction state;
- Gaussian random-walk or discount evolution;
- dense Gaussian covariance;
- Laplace filtering;
- Candidate A soft-target observation;
- synthetic data first;
- historical data only after derivative validation.

Exclude:

- graph residual;
- BOCPD;
- particle filtering;
- dynamic rolling-performance coefficients;
- adaptive generalized-Bayes learning rate;
- covariance compression;
- full direct-utility covariance;
- gated experts;
- RFF.

---

# 35. Second POC Extension

After Candidate A is stable:

1. add the cross-group logistic Candidate B fallback;
2. run the Candidate B tolerance audit;
3. implement partitioned-preference PL;
4. validate derivatives and runtime;
5. compare Candidate A and Candidate B under the same filter.

Only after these steps should the project consider rolling-performance corrections, low-rank states, graph residuals, BOCPD, RBPF regimes, or direct robust utility.

---

# 36. Historical Prequential Protocol

For each day $t$:

1. load the posterior from day $t-1$;
2. apply the state transition;
3. build $X_t$;
4. compute predictive score moments;
5. derive decision quantities;
6. choose the deployment candidate;
7. reveal the full utility vector;
8. construct the observation;
9. update the posterior once;
10. write diagnostics;
11. checkpoint.

No information from day $t$ outcomes may enter the prediction step.

---

# 37. Computational Budget

For the first POC:

- $N=1428$ candidates;
- $P\in[100,300]$ dynamic parameters;
- dense covariance;
- one update per day;
- several Newton iterations;
- full-history walk-forward replay.

Expected dominant costs are:

$$
O(NP)
$$

for scores and gradients, and:

$$
O(P^3)
$$

for dense factorisation.

These costs should be practical for a mathematical POC.

---

# 38. C Implementation Implications

The eventual C core should require:

- dense matrix multiplication;
- Cholesky factorisation;
- triangular solves;
- weighted cross-products;
- rank-one updates;
- log-sum-exp;
- stable logistic functions;
- deterministic random-number generation;
- checkpoint serialization.

BLAS and LAPACK should cover most first-stage linear algebra.

The first POC may be implemented in Python with NumPy and SciPy, provided the mathematics and data flow are designed for later translation to C.

---

# 39. Accepted Decisions

1. The first state is partially dynamic.
2. The contextual deformation block is the primary dynamic state.
3. The baseline surface is static or very slow initially.
4. Rolling-performance coefficients are static initially.
5. No dynamic node residual is included in Prototype 1.
6. The initial transition is a Gaussian random walk or discount evolution.
7. The primary online approximation is Laplace-style Gaussian filtering.
8. ADF or structured Gaussian VI is the fallback.
9. Reduced HMC or NUTS is the offline reference.
10. Full-state particle filtering is rejected for Prototype 1.
11. RBPF is reserved for a small later regime or scale state.
12. Candidate A is the first observation implementation.
13. Partitioned-preference PL is the leading proper Candidate B likelihood.
14. Cross-group logistic loss is the Candidate B fallback.
15. The first covariance may be dense because the POC state is deliberately small.
16. Candidate B must pass tolerance and derivative audits before integration.
17. BOCPD is deferred until the base filter is stable.

---

# 40. Unresolved Questions

1. Which spline basis and dimensions should be used?
2. How should the compact market basis be constructed?
3. Should $\alpha$ be pre-estimated offline or learned jointly?
4. Should $\gamma$ be included in the first historical POC?
5. Should the transition use $Q=qI$ or covariance discounting?
6. How should $q$ or $\delta$ be selected?
7. What prior covariance should initialize the state?
8. How should generalized-Bayes learning rate $\eta$ be selected?
9. What tolerance rule should construct Candidate B groups?
10. What are the actual historical upper-group sizes?
11. Can partitioned-preference PL derivatives be implemented stably?
12. Is exact Hessian calculation necessary, or are Hessian-vector products enough?
13. How accurately does Laplace filtering reproduce reduced HMC posteriors?
14. Does posterior covariance remain calibrated through long sequences?
15. When should covariance structure be compressed?
16. When should BOCPD or a regime-state particle filter be introduced?

---

# 41. Immediate Next Stage

The next project stage is:

```text
05.5 - Mathematical Prototype and Likelihood Validation
```

Its purpose is to build:

1. a Candidate B tolerance audit;
2. a standalone likelihood correctness harness;
3. a small synthetic dynamic-surface generator;
4. a dense-covariance Laplace filter;
5. Candidate A as the first end-to-end observation;
6. Candidate B fallback and proper likelihood as later extensions.

The project should now shift from broad architecture research to compact, falsifiable mathematical experiments.

---

# 42. Stage Completion Criteria

This inference stage is complete when:

- the first dynamic state is defined;
- the state transition is explicit;
- the Gaussian posterior approximation is specified;
- Laplace update equations are documented;
- Candidate A derivatives are explicit;
- Candidate B finalists are chosen;
- particle filtering is correctly scoped;
- covariance strategy is specified;
- synthetic validation requirements are defined;
- the first POC boundary is explicit.

The next discussion should define the exact scope, modules, tests, and experiment order for the first proof-of-concept implementation.
