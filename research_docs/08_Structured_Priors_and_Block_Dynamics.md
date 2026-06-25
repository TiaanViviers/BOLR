# Phase H: Structured Priors and Block-Specific Dynamics

## Purpose

Phase H adds statistically meaningful regularisation and transition structure on top of the dense Gaussian posterior and composite score architecture from Phase G. The Python implementation remains the mathematical reference. No covariance approximation is introduced.

## Quadratic penalties

The core abstraction is `QuadraticPenalty`, representing:

$$
\mathcal P(\beta)=\frac12 \beta^\top R \beta
$$

with symmetric positive-semidefinite matrix `R`.

The implementation supports:

- penalty value, gradient, and Hessian access;
- rank, nullity, eigenvalue, and PSD checks;
- ridge properisation;
- scalar weighting;
- additive combination.

This becomes the common language for surface smoothness priors, context-matrix shrinkage, and penalty-shaped process noise.

## Difference and tensor penalties

The repo now includes deterministic first- and second-difference matrices and their corresponding penalties:

- first order penalises local slope;
- second order penalises local curvature.

For raw tensor-product candidate coefficients, the smoothness penalty is:

$$
R_{\text{tensor}}
=
\lambda_e (R_e \otimes I_B)
+
\lambda_r (I_A \otimes R_r).
$$

The stop-axis index remains the fast-varying axis so the tensor penalty ordering matches the current basis vectorisation convention.

## Reduced-basis projection

The centred raw candidate basis has width `48`, while the effective centred basis has width `47`. The basis layer now exposes the explicit lift matrix `L` so reduced coefficients satisfy:

$$
\Phi_{\text{reduced}} \beta = \Phi_{\text{centred,raw}} L \beta.
$$

Structured penalties are projected into reduced coordinates using:

$$
R_{\text{reduced}} = L^\top R_{\text{tensor}} L.
$$

Tests cover score equivalence and quadratic-form equivalence.

## Proper Gaussian priors

Singular roughness penalties are converted into proper Gaussian priors through:

$$
\Lambda = \lambda_{\text{smooth}} R + \lambda_0 I,
\qquad
P_0 = \Lambda^{-1}.
$$

The implementation records:

- penalty rank;
- precision and covariance eigenvalue ranges;
- condition number;
- average marginal prior variance;
- properisation ridge.

Surface and context blocks can therefore use smoothness structure without giving up a well-defined dense posterior covariance.

## Context matrix penalties

For context interaction matrices `B` stored with column-major vectorisation, the structured precision is:

$$
\Lambda_{\text{context}}
=
\lambda_c (I_{p_m} \otimes R_c)
+
\lambda_m (R_m \otimes I_{p_c})
+
\lambda_0 I.
$$

Current support includes:

- candidate-surface smoothness on the configuration axis;
- context ridge or diagonal shrinkage;
- optional ridge properisation.

The matrix-form and vectorised-form penalties are regression-tested for exact agreement.

## Block-specific dynamics

Process-noise support now includes:

- isotropic random walk;
- diagonal random walk;
- penalty-shaped random walk.

Penalty-shaped drift uses:

$$
Q = q (R + \tau I)^{-1},
$$

which increases drift variance in smoother modes and decreases it in rougher modes. Diagnostics expose:

- penalty eigenvalues;
- process-noise eigenvalues;
- minimum and maximum drift variance;
- smoothest and roughest mode variances;
- process-noise condition number.

Blockwise dense process noise is still assembled into a single dense Gaussian prediction step, preserving cross-block posterior covariance from the previous update.

## Static fitting

Static baseline fitting now accepts a general quadratic prior precision rather than only scalar ridge regularisation. This preserves the original ridge path while allowing structured smoothness penalties for the baseline surface.

## Innovation and roughness diagnostics

The diagnostics layer now records:

- Euclidean and Mahalanobis state updates;
- prior and posterior log factors;
- posterior objective improvement;
- Gaussian KL divergence between predictive prior and posterior;
- blockwise update summaries;
- roughness energy and prior-standardised state norm.

These diagnostics are recorded only. Adaptive process-noise policies remain deferred.

## Identifiability limits

Structured priors regularise confounded blocks numerically, but they do not create scientific identifiability where the likelihood cannot distinguish blocks. The correct interpretation is:

- the posterior becomes proper and stable;
- the individual block decomposition can still remain interpretation-sensitive.

That distinction should remain explicit in every later multi-block analysis.

## Synthetic and historical scope

Phase H adds bounded structured-prior tests:

- smooth and rough surface behaviour checks;
- structured-prior composite smoke tests;
- continued E0 ridge/isotropic compatibility.

This phase does not add full historical structured experiments, feature selection, or covariance approximations.

## C-port contract additions

The future C implementation must support these kernels exactly:

- `difference_matrix_build`
- `tensor_penalty_build`
- `penalty_project`
- `quadratic_penalty_value`
- `quadratic_penalty_gradient`
- `block_precision_assemble`
- `block_process_noise_assemble`
- `penalty_shaped_process_noise`
- `gaussian_kl`
- `block_diagnostics`

For each kernel, the Python oracle now fixes:

- dense `float64` arrays;
- explicit input and output shapes;
- immutable-style semantics for mathematical objects;
- column-major vectorisation for context matrices;
- deterministic golden-fixture references for later C equivalence.

## Deferred work

Still deferred after Phase H:

- graph residual blocks;
- total variation;
- adaptive dynamics and BOCPD;
- particle filtering;
- covariance approximation;
- Python runtime optimisation;
- real context-feature modelling experiments.
