# Python Reference Model Freeze

## Status

After Phase K, the BOLR Python stack is treated as the mathematical reference architecture. The next implementation milestone is C-port equivalence, not another Python-side latent-model redesign.

## Frozen components

### Mathematical reference complete

- representation:
  - coordinate transforms
  - spline and tensor bases
  - centered rank-reduced candidate basis
  - explicit and operator design construction
- observations:
  - Candidate A soft-target generalized Bayes
  - Candidate B cross-group logistic
  - Candidate B partitioned Plackett-Luce
- priors:
  - isotropic, diagonal, and structured Gaussian block priors
- graph residual:
  - orthogonal low-dimensional residual block over the canonical grid graph
- dynamics:
  - dense Gaussian random-walk and discount-style reference dynamics
- adaptive transitions:
  - surprise standardisation
  - BOCPD-backed activation
  - blockwise additive/discount adaptation
  - partial reset hooks
- decision layer:
  - predictive posterior score summaries
  - Monte Carlo ranking probabilities
  - connected-region inference
  - deterministic decision policies
  - tie-aware calibration metrics
- checkpointing:
  - posterior state
  - adaptive transition state
  - decision-policy metadata
  - Monte Carlo sampling metadata
- golden fixtures:
  - composite update
  - structured priors
  - graph residual
  - adaptive transitions
  - posterior decision layer
- backend contracts:
  - Python interface contracts for future C kernels

### Operational Python reference

- generic historical replay runner
- Candidate A historical replay path
- bounded Candidate B replay/integration path
- synthetic scenario coverage
- deterministic resume-equivalence tests

### Reduced-scale reference

- full Candidate B historical execution remains intentionally reduced in Python
- dense posterior covariance remains the reference covariance form
- Monte Carlo decision summaries prioritise correctness over throughput

### Deferred optional extensions

- Rao-Blackwellised particle regime layer
- additional regime-transition machinery
- structured posterior covariance approximations
- direct robust cardinal utility observations
- feature-selection and large empirical model-selection sweeps
- production C optimisation work

## Production interpretation

The frozen Python system is the source of truth for:

- mathematics
- tie conventions
- deterministic ranking probabilities
- checkpoint schemas
- replay ordering guarantees
- golden-fixture regression expectations

It is not the intended production runtime. Production implementation now shifts to a C backend that must match these numerical contracts on golden fixtures and bounded replay tests before broader feature or hyperparameter work resumes.
