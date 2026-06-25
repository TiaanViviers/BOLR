# 02 - Contextual Representation and Score Functions

> **Project:** Bayesian Online Listwise Ranking (BOLR)  
> **Document purpose:** Define how BOLR should represent the configuration space, shared market context, and candidate-specific state before selecting a final mathematical representation.  
> **Status:** Research specification  
> **Empirical foundation:** `01.5_EDA.md`

---

# 1. Purpose of This Document

The BOLR problem is not simply to assign unrelated scores to approximately 1,428 candidate configurations.

The candidate set forms a complete and highly structured two-dimensional grid over:

- `entry_percentage`;
- `sl_trail_percentage`.

The empirical analysis shows that neighbouring configurations often produce similar or identical outcomes, while the location of the best-performing region moves rapidly through time.

This suggests a revised modelling objective:

> Learn a market-conditioned posterior suitability surface over configuration space, rather than learning an independent score for every configuration ID.

This document formalises that objective and records the main candidate representation families.

It does **not** yet choose the final representation.

The final choice requires deeper theoretical research and controlled empirical comparison.

---

# 2. Empirical Motivation

The YM structural EDA produced the following key findings in [[01.5_EDA]]:

- 4,494 trading days;
- 1,428 configurations per day;
- complete $34\times42$ configuration grid;
- stable candidate set;
- mean positive-configuration fraction of 15.32%;
- mean normalised neighbour PnL difference of 0.0230;
- mean identical-neighbour fraction of 35.03%;
- mean largest plateau fraction of 16.87%;
- mean largest connected fraction within the top 5% of 77.20%;
- exact best configuration changes on 98.62% of day-to-day transitions;
- mean Manhattan movement of the best configuration of 18.80 grid cells;
- tie-aware previous-best to current-best-plateau distance of 17.91;
- previous best remains in the current top 5% only 14.20% of the time;
- mean best-plateau Jaccard overlap of 0.0363.

These findings establish two apparently opposing facts.

## 2.1 Strong local structure

Neighbouring configurations are often similar.

Large plateaus and connected top regions are common.

This implies that the model should share statistical information across nearby configurations.

## 2.2 Rapid temporal movement

The exact best configuration and the best region move aggressively over time.

This implies that the model cannot rely on a static global optimum or excessive temporal inertia.

The representation must therefore support:

$$
\text{local smoothness}
+
\text{time-varying contextual deformation}
+
\text{possible local discontinuities}.
$$

---

# 3. Configuration Space

Let:

$$
e_i = \text{entry percentage of configuration }i,
$$

and:

$$
r_i = \text{trailing-stop percentage of configuration }i.
$$

Because the grid is approximately geometrically spaced, define log coordinates:

$$
u_i=\log e_i,
$$

$$
v_i=\log r_i.
$$

Each configuration is represented by:

$$
c_i=
\begin{bmatrix}
u_i\\v_i\end{bmatrix}.
$$

The observed configuration domain is:

$$
\mathcal C=\{(u_i,v_i)\}_{i=1}^{N},
$$

with:

$$
N=34\times42=1428.
$$

The grid is nearly regular in log-space. This makes log coordinates a natural default representation because equal grid steps correspond approximately to equal proportional changes in the raw configuration parameters.

---

# 4. Revised Modelling Objective

The original row-wise view is:

$$
s_{it}=f(x_{it}),
$$

where each candidate is treated as a row with ordinary predictor values.

The revised structural view is:

$$
s_t(u,v)=f(u,v,m_t,z_t),
$$

where:

- $(u,v)$ describes a location in configuration space;
- $m_t$ is the market context shared across the day;
- $z_t$ represents candidate-specific dynamic information;
- $s_t(u,v)$ is a time-dependent suitability surface.

For an observed configuration $i$:

$$
s_{it}=s_t(u_i,v_i).
$$

The production decision remains:

$$
i_t^\star=\arg\max_i \operatorname{DecisionScore}_{it}.
$$

However, the model should ideally learn the shape and uncertainty of the full surface rather than only a collection of unrelated candidate scores.

---

# 5. Three Information Components

The score should distinguish three sources of information.

## 5.1 Structural configuration geometry

Examples:

- $\log(\text{entry percentage})$;
- $\log(\text{sl-trail percentage})$;
- `rr_ratio`;
- `risk_size`;
- basis functions over the configuration grid;
- neighbourhood relationships.

Denote this information by $c_i$.

## 5.2 Shared market context

Examples:

- volatility;
- trend;
- momentum;
- calendar indicators;
- macro-event indicators;
- possible future volume features.

Denote this information by $m_t$.

## 5.3 Candidate-specific dynamic state

Examples:

- lagged PnL;
- rolling mean PnL;
- rolling PnL volatility;
- rolling win rate;
- rolling drawdown;
- rolling Sharpe-like summaries.

Denote this information by $z_{it}$.

A general decomposition is:

$$
s_{it}=f_{\text{surface}}(c_i,m_t)+f_{\text{dynamic}}(z_{it},m_t)+\delta_{it},
$$

where $\delta_{it}$ may represent a candidate-specific deviation or residual component.

---

# 6. Why Shared Context Must Enter Through Interactions

Suppose the score is purely additive:

$$
s_{it}=f(c_i,z_{it})+g(m_t).
$$

For a softmax ranking:

$$
q_{it}=\frac{\exp(s_{it})}{\sum_j\exp(s_{jt})}.
$$

Because $g(m_t)$ is identical for all candidates:

$$
q_{it}=\frac{\exp(f(c_i,z_{it}))}{\sum_j\exp(f(c_j,z_{jt}))}.
$$

The shared market term cancels.

Therefore, BOLR must use a non-separable score:

$$
s_{it}=f(c_i,z_{it},m_t),
$$

where the effect of market state depends on candidate characteristics.

The central representation problem is:

> How should market context reshape the suitability surface over configuration space?

---

# 7. Baseline Surface Decomposition

A useful conceptual decomposition is:

$$
s_{it}=a(c_i)+h(c_i,m_t)+g(z_{it},m_t).
$$

Here:

- $a(c_i)$ is the long-run baseline suitability of a configuration region;
- $h(c_i,m_t)$ is the market-conditioned deformation of that baseline surface;
- $g(z_{it},m_t)$ captures candidate-specific dynamic information.

This allows the model to represent:

1. persistent structural preferences;
2. market-dependent movement of the favourable region;
3. short-run candidate-specific evidence.

---

# 8. Candidate Representation Family A: Tensor-Product Bases

Define separate bases for the two configuration axes:

$$
\phi_e(u)=
\begin{bmatrix}
\phi_{e1}(u)\\
\vdots\\
\phi_{eA}(u)
\end{bmatrix},
$$

$$
\phi_r(v)=
\begin{bmatrix}
\phi_{r1}(v)\\
\vdots\\
\phi_{rB}(v)
\end{bmatrix}.
$$

The joint basis is:

$$
\phi_c(u,v)=\phi_e(u)\otimes\phi_r(v).
$$

A static surface is:

$$
a(u,v)=\phi_c(u,v)^\top\beta.
$$

Equivalently:

$$
a(u,v)=\sum_{a=1}^{A}\sum_{b=1}^{B}W_{ab}\phi_{ea}(u)\phi_{rb}(v).
$$

Possible bases include:

- B-splines;
- natural cubic splines;
- Fourier bases;
- radial basis functions;
- compact local basis functions;
- multi-resolution bases.

## Advantages

- explicitly uses the two-dimensional geometry;
- supports interpolation;
- may require fewer parameters than one coefficient per configuration;
- candidate basis values can be precomputed;
- may support separable computation.

## Risks

- excessive smoothness may erase genuine discontinuities;
- basis resolution becomes a hyperparameter;
- high-resolution tensor products can still become large;
- boundary behaviour requires care.

---

# 9. Candidate Representation Family B: Low-Rank Surfaces

Factorise the coefficient matrix:

$$
W_t=U_tV_t^\top,
$$

with:

$$
R\ll\min(A,B).
$$

Then:

$$
s_t(u,v)=\phi_e(u)^\top U_tV_t^\top\phi_r(v).
$$

Equivalently:

$$
s_t(u,v)=\sum_{k=1}^{R}a_{k,t}(u)b_{k,t}(v).
$$

## Advantages

- reduced parameter count;
- lower memory use;
- potential computational savings;
- useful when the surface has a few dominant patterns.

## Risks

- the true surface may not be low rank;
- abrupt local behaviour may require larger rank;
- factorisation introduces non-identifiability;
- online Bayesian inference over factors may be difficult.

---

# 10. Candidate Representation Family C: Kronecker Structure

Let $K_e$ describe similarity across entry values and $K_r$ similarity across stop values.

A separable covariance over the full grid is:

$$
K_c=K_e\otimes K_r.
$$

For two configurations:

$$
K_c(c_i,c_j)=K_e(u_i,u_j)K_r(v_i,v_j).
$$

This represents the idea that two configurations are similar when both their entry and stop coordinates are similar.

## Potential computational benefit

Instead of storing a generic dense:

$$
1428\times1428
$$

covariance matrix, calculations may exploit:

$$
K_e\in\mathbb R^{34\times34},
$$

$$
K_r\in\mathbb R^{42\times42}.
$$

Kronecker algebra may reduce storage and linear-algebra costs.

## Risks

A purely separable structure may be too restrictive.

Possible extensions include:

$$
K_c=\sum_{k=1}^{R}K_{e,k}\otimes K_{r,k},
$$

or a separable baseline plus a local residual component.

---

# 11. Candidate Representation Family D: Graph-Based Structure

Represent the grid as a graph:

$$
G=(V,E),
$$

where configurations are nodes and direct grid neighbours are connected.

A graph-smoothing penalty is:

$$
\sum_{(i,j)\in E}w_{ij}(s_{it}-s_{jt})^2.
$$

Using graph Laplacian $L$:

$$
s_t^\top Ls_t.
$$

A Bayesian analogue is:

$$
p(s_t)\propto\exp\left(-\frac{\lambda}{2}s_t^\top Ls_t\right).
$$

## Advantages

- directly matches the observed grid;
- supports local information sharing;
- graph matrices are sparse;
- can handle incomplete grids;
- does not require a global smooth function.

## Risks

- ordinary quadratic smoothing may over-smooth discontinuities;
- a pure graph model does not naturally interpolate to unseen coordinates;
- the prior may require constraints;
- smoothing strength must be chosen or inferred.

## Robust graph alternatives

Research should consider:

- graph total variation;
- fused penalties;
- edge-specific smoothing weights;
- heavy-tailed difference priors;
- adaptive graph Laplacians;
- smooth-region and boundary mixtures.

---

# 12. Candidate Representation Family E: Random Fourier Features

Let:

$$
c_i=\begin{bmatrix}u_i\\v_i\end{bmatrix}.
$$

Random Fourier Features approximate a stationary kernel using:

$$
\phi_j(c_i)=\sqrt{\frac{2}{D}}\cos(\omega_j^\top c_i+b_j).
$$

A candidate-only surface is:

$$
a(c_i)=w^\top\phi(c_i).
$$

A joint candidate-market map is:

$$
\phi(c_i,m_t)=\phi\left(\begin{bmatrix}c_i\\m_t\end{bmatrix}\right).
$$

## Advantages

- fixed-dimensional nonlinear representation;
- interpolation in continuous coordinates;
- modular support for new numeric features;
- efficient matrix operations;
- suitable for C implementation;
- avoids support-vector growth.

## Risks

- a generic isotropic RBF kernel may poorly match heterogeneous features;
- joint RFF may require many dimensions;
- bandwidth selection matters;
- high-dimensional Bayesian inference may become difficult;
- local grid structure may be used inefficiently.

---

# 13. Candidate Representation Family F: Modular and Product Kernels

Let $K_c(c,c')$ be a configuration kernel and $K_m(m,m')$ a market kernel.

A combined kernel may be:

$$
K((c,m),(c',m'))=K_c(c,c')+\lambda K_c(c,c')K_m(m,m').
$$

Interpretation:

- $K_c$ captures stable configuration similarity;
- $K_cK_m$ captures market-conditioned similarity.

A random-feature approximation may use:

$$
z_c=\phi_c(c),
$$

$$
z_m=\phi_m(m),
$$

and:

$$
s_{it}=\alpha^\top z_{c,i}+z_{c,i}^\top Bz_{m,t}.
$$

This preserves modularity while making the market-conditioning structure explicit.

---

# 14. Contextual Surface Models

## 14.1 Varying-coefficient surface

Let $\phi_c(c_i)$ be the configuration basis.

Define:

$$
s_{it}=\phi_c(c_i)^\top w(m_t).
$$

A linear context map is:

$$
w(m_t)=\alpha+B\phi_m(m_t).
$$

Therefore:

$$
s_{it}=\phi_c(c_i)^\top\alpha+\phi_c(c_i)^\top B\phi_m(m_t).
$$

The market state changes the coefficients of the configuration surface.

## 14.2 Low-rank contextual compatibility

Define:

$$
u(c_i)\in\mathbb R^R,
$$

$$
v(m_t)\in\mathbb R^R.
$$

Then:

$$
s_{it}=a(c_i)+u(c_i)^\top v(m_t).
$$

This is a learned compatibility between configuration and market embeddings.

## 14.3 Context-gated surfaces

Let $f_k(c_i)$ be candidate-ranking surfaces and $g_k(m_t)$ non-negative weights summing to one.

Then:

$$
s_{it}=\sum_{k=1}^{K}g_k(m_t)f_k(c_i).
$$

The market context mixes several latent ranking experts.

---

# 15. Dynamic Candidate-Specific Information

The structural surface alone does not use rolling-performance features.

Let $z_{it}$ contain candidate-specific dynamic information.

A general score is:

$$
s_{it}=f_{\text{surface}}(c_i,m_t)+g(z_{it},m_t).
$$

Possible forms include:

## Additive dynamic correction

$$
g(z_{it},m_t)=\gamma_t^\top z_{it}.
$$

## Contextual dynamic correction

$$
g(z_{it},m_t)=z_{it}^\top C\phi_m(m_t).
$$

## Nonlinear residual model

$$
g(z_{it},m_t)=w_z^\top\phi_z(z_{it},m_t).
$$

## Hierarchical residual

$$
s_{it}=f_{\text{surface}}(c_i,m_t)+\delta_{it},
$$

with $\delta_{it}$ following a structured dynamic prior.

The dynamic component should not destroy the local sharing induced by configuration geometry.

---

# 16. Surface Versus Candidate-ID Modelling

## Independent candidate scores

$$
s_{it}=\alpha_{i,t}.
$$

Flexible, but statistically inefficient and unable to interpolate.

## Pure continuous surface

$$
s_{it}=f_t(u_i,v_i).
$$

Strong sharing, but may miss candidate-specific effects.

## Structured surface plus residual

$$
s_{it}=f_t(u_i,v_i)+\delta_{i,t}.
$$

This combines surface structure with candidate-specific deviations.

The residual must be regularised strongly enough that the model does not collapse back into 1,428 unrelated parameters.

---

# 17. Interpolation and Extrapolation

A continuous coordinate representation can score unseen configurations.

For:

$$
c^\star=(u^\star,v^\star),
$$

the model can compute:

$$
s_t(c^\star).
$$

## Interpolation

Inside the observed domain, interpolation may support future grid refinement.

## Extrapolation

Outside the observed range, predictions should be treated cautiously and posterior uncertainty should ideally increase.

The initial BOLR production action may remain restricted to the fixed observed grid.

---

# 18. Smoothness Versus Discontinuity

The EDA supports local smoothness, but trading mechanics can create real discontinuities.

Examples:

- one entry threshold is crossed while a neighbour is not;
- one stop is hit while another survives;
- a small parameter change alters the entire trade path;
- session-end effects.

Therefore, BOLR should support:

- local sharing;
- plateaus;
- sharp boundaries;
- isolated high-value structures when supported by evidence.

Candidate mechanisms include:

- adaptive smoothing;
- multi-resolution bases;
- graph total variation;
- heavy-tailed local differences;
- smooth baseline plus sparse residual;
- mixtures of smooth experts.

---

# 19. Posterior Suitability Surface

BOLR should maintain uncertainty over:

$$
p(s_t(u,v)\mid D_{1:t-1},m_t).
$$

Useful quantities include:

## Posterior mean surface

$$
E[s_t(u,v)\mid D_{1:t-1},m_t].
$$

## Posterior variance surface

$$
\operatorname{Var}(s_t(u,v)\mid D_{1:t-1},m_t).
$$

## Probability best

$$
P(s_{it}>s_{jt}\ \forall j\neq i\mid D_{1:t-1}).
$$

## Probability top-$K$

$$
P(i\in\operatorname{TopK}(s_t)\mid D_{1:t-1}).
$$

## Region-level uncertainty

The posterior may reveal:

- one broad plausible region;
- several disconnected plausible regions;
- a narrow high-confidence optimum;
- a diffuse uncertain surface.

---

# 20. Exact Winner Versus High-Quality Region

The EDA shows broad connected top regions within days but extremely low persistence of the exact winner.

The model should distinguish:

## Point objective

$$
\arg\max_i y_{it}.
$$

## Region objective

$$
\mathcal T_t=\{i:y_{it}\text{ belongs to a high-quality region}\}.
$$

## Utility-weighted surface objective

A soft distribution over candidate quality.

The final observation model may emphasise regions or tied groups rather than a strict total ordering.

---

# 21. Computational Opportunities

The fixed grid allows precomputation of:

- log coordinates;
- tensor or spline bases;
- RFF candidate maps;
- graph edges and Laplacian;
- kernel factors;
- candidate embeddings.

Per day, shared computations may include:

- market representation;
- context embedding;
- context-dependent deformation;
- market-state change signal.

Ideally, scoring reduces to:

- matrix-vector multiplication;
- low-rank dot products;
- sparse graph operations;
- a small candidate-specific correction.

Kronecker identities such as:

$$
(A\otimes B)\operatorname{vec}(X)=\operatorname{vec}(BXA^\top)
$$

may avoid constructing large dense matrices.

---

# 22. Identifiability and Scaling Concerns

## Low-rank factorisation

Many pairs $(U,V)$ can produce the same $UV^\top$.

## Embeddings

Rotations of candidate and market embeddings can preserve the same scores.

## Softmax translation invariance

For any daily constant $a_t$:

$$
s_{it}'=s_{it}+a_t
$$

produces the same listwise probabilities.

## Softmax scale and temperature

Score scale and temperature may be confounded.

The final design should constrain or normalise:

- score location;
- factor scales;
- embedding scales;
- temperature;
- basis coefficients.

---

# 23. Candidate Representation Requirements

Any serious representation should satisfy the following.

## Statistical requirements

- use local configuration structure;
- allow market-conditioned deformation;
- preserve candidate-specific dynamic information;
- represent uncertainty;
- avoid excessive effective dimension;
- permit shrinkage;
- avoid assuming perfect global smoothness.

## Computational requirements

- score 1,428 candidates within the pre-open latency budget;
- fit within memory limits;
- support overnight posterior updates;
- allow precomputation;
- be implementable efficiently in modern C;
- support checkpointing.

## Generalisation requirements

- support different assets and grid sizes;
- tolerate incomplete grids where possible;
- avoid hardcoded YM-specific parameter values;
- support interpolation where justified.

---

# 24. Candidate Model Families for Deep Research

The theoretical research should compare at least:

1. tensor-product spline surfaces;
2. low-rank tensor surfaces;
3. separable Gaussian-process or kernel structures;
4. Kronecker-structured covariance models;
5. graph-Laplacian Gaussian priors;
6. graph total-variation or fused priors;
7. smooth baseline plus local residual models;
8. low-rank candidate-market compatibility models;
9. context-gated surface experts;
10. joint RFF;
11. modular or product-kernel RFF;
12. hybrid tensor/graph/RFF approaches.

---

# 25. Proposed Initial Baseline Hierarchy

## Baseline A: independent linear candidate rows

Use raw candidate features and explicit market interactions.

## Baseline B: smooth static configuration surface

Use a low-dimensional tensor basis without market conditioning.

## Baseline C: contextual tensor surface

Allow market state to deform the tensor surface.

## Baseline D: contextual surface plus rolling-performance correction

Add candidate-specific dynamic features.

## Baseline E: graph-smoothed contextual surface

Add neighbourhood regularisation.

## Baseline F: nonlinear modular representation

Add RFF or another nonlinear candidate-market map.

---

# 26. Working Hypotheses

1. A structured configuration surface will outperform treating candidate IDs independently.
2. A smooth baseline plus adaptive local residuals will outperform both rigid smoothing and complete independence.
3. Market-conditioned deformation is more important than a static long-run optimum.
4. Low-rank or separable structures may provide major computational benefits.
5. A representation focused on high-quality regions will be more stable than one trained only on the exact winner.
6. Candidate geometry and rolling performance contain complementary information.
7. Modular candidate and market representations will be more data-efficient than one generic nonlinear map.

---

# 27. Main Open Research Questions

1. What basis best represents the two-dimensional configuration surface?
2. Should coordinates be raw, logged, normalised, or domain-transformed?
3. How much smoothness is appropriate?
4. Should smoothness be fixed, learned, or time-varying?
5. Can graph priors preserve discontinuities better than splines or kernels?
6. Does Kronecker structure remain useful after introducing a listwise likelihood?
7. Is a low-rank contextual surface expressive enough?
8. How should rolling-performance features modify the structural surface?
9. Should market context deform basis coefficients or gate among surfaces?
10. Can RFF add useful capacity without making inference infeasible?
11. Should the model interpolate to unseen configurations?
12. How should uncertainty behave near grid boundaries?
13. Should the target describe exact winners, tied groups, or high-quality regions?
14. How should identifiability constraints be imposed?
15. Which structures survive sequential Bayesian updating?
16. Which structures can be implemented efficiently in C?

---

# 28. Deep-Research Assignment

The dedicated research task should be:

> Investigate structured representations for a dynamic Bayesian contextual ranking surface over a fixed two-dimensional configuration grid, with strong local dependence, plateaus, occasional discontinuities, and rapidly moving high-quality regions.

The report should compare:

- tensor-product splines;
- low-rank tensor decompositions;
- Kronecker kernels and covariances;
- graph-Laplacian and graph total-variation priors;
- smooth-plus-residual models;
- contextual varying-coefficient models;
- low-rank candidate-market embeddings;
- gated experts;
- modular RFF and product kernels.

For each method, report:

1. mathematical formulation;
2. assumptions;
3. ability to model local smoothness;
4. ability to preserve discontinuities;
5. interpolation behaviour;
6. posterior representation;
7. compatibility with sequential inference;
8. time and memory complexity;
9. identifiability issues;
10. implementation implications;
11. expected suitability for BOLR;
12. primary references;
13. recommended experiments.

---

# 29. Current Provisional Direction

The strongest provisional architecture is:

$$
s_{it}=f_{\text{surface}}(c_i,m_t)+f_{\text{dynamic}}(z_{it},m_t),
$$

where:

- $f_{\text{surface}}$ is a structured market-conditioned function over the two-dimensional candidate grid;
- $f_{\text{dynamic}}$ captures rolling candidate-specific information.

A likely first research prototype should use:

- log configuration coordinates;
- a low-dimensional tensor or low-rank basis;
- a simple contextual deformation;
- strong regularisation;
- no graph discontinuity model initially;
- no generic high-dimensional RFF initially.

This is not a final decision. It is a practical starting point for controlled representation experiments.

---

# 30. Relationship to Other BOLR Research Areas

## Observation model

Broad tied regions may favour:

- grouped relevance grades;
- partial rankings;
- soft utility surfaces;
- tolerance-aware top sets.

## Inference

High-dimensional bases may rule out full-state particle filtering.

Kronecker or low-rank structure may enable more efficient approximate inference.

## Adaptation

The baseline surface and contextual deformation may require different process-noise scales.

## BOCPD

Market-state changes may alter the contextual deformation before realised ranking performance deteriorates.

## Decision policy

A posterior suitability surface supports:

- probability best;
- probability top-$K$;
- region-level confidence;
- uncertainty-aware downstream allocation.

---

# 31. Stage Completion Criteria

This representation stage is conceptually complete when:

- the suitability-surface formulation is accepted;
- configuration geometry, market context, and dynamic candidate state are separated;
- candidate representation families are documented;
- computational structure is identified;
- interpolation and smoothing assumptions are explicit;
- deep-research questions are defined;
- no final method is selected prematurely.

The next step is to conduct the dedicated representation research and connect its findings to the observation-model and inference research.
