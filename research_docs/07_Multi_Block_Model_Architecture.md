# 07 Multi-Block Model Architecture

## Status

Phase G establishes the permanent Python-side block architecture that future C kernels will implement.

Recorded project status:

- Candidate B mathematical implementation: complete
- Candidate B reduced exact validation: complete
- Candidate B generic Laplace integration: complete
- Candidate B realistic pure-Python historical execution: deferred
- Candidate B C acceleration: future production phase

## Score Decomposition

The composite score is represented as:

`s_t = s_t^static + sum_b X_{b,t} theta_{b,t}`

with separate registered blocks rather than one hardcoded giant design matrix.

## State Layout

Implemented in:

- `bolr/model/state_layout.py`

Properties:

- contiguous non-overlapping slices
- explicit vectorization order
- block extraction, insertion, flattening, and reshaping
- immutable metadata suitable for future C offsets

## Score Blocks

Implemented blocks:

- static baseline block
- dynamic surface block
- context-interaction block
- generic linear history block
- generic supplied-design block

All blocks expose:

- forward score evaluation
- transpose multiplication
- optional explicit design matrix
- metadata

## Composite Model

Implemented in:

- `bolr/model/composite.py`

It provides:

- static score aggregation
- dynamic score aggregation
- explicit composite design
- operator-based transpose multiplication
- operator-based curvature pullback
- block score decomposition
- identifiability diagnostics

## Priors and Dynamics

Implemented in:

- `bolr/model/priors.py`

Supports:

- isotropic block priors
- diagonal block priors
- isotropic block process noise
- diagonal block process noise
- frozen blocks
- fixed blocks represented in layout but excluded from dynamic updates through the composite model

Dense posterior covariance remains fully dense after updating.

## Composite Laplace Path

Implemented in:

- `bolr/inference/laplace.py`

The composite update consumes:

- composite score forward map
- composite transpose multiplication
- score-space curvature from the observation model

The surface-only composite model reproduces the existing E0 path exactly on the tested reduced cases.

## Identifiability

Diagnostics currently report:

- numerical rank
- singular values
- condition number
- block cross-Gram matrices
- duplicate column warnings
- zero-column warnings
- near-zero context warnings

This is the architectural point where block confounding becomes explicit instead of hidden inside one design matrix.

## C-Port Conventions

Implemented conventions:

- one contiguous `float64` state vector
- one layout object with integer offsets
- explicit row/column vectorization order
- narrow backend contract in `bolr/backend/base.py`
- golden deterministic fixtures in `tests/fixtures/golden`

The Python reference is now the mathematical oracle and interface freezer for the future C kernels.

## Golden Fixtures

Committed fixture:

- `tests/fixtures/golden/composite_reference.npz`
- `tests/fixtures/golden/composite_reference.json`

It records a small composite baseline + surface Laplace update and target tolerances for future backend equivalence.

## Commands Executed

```bash
source ~/environments/pyenv/bin/activate && python -m pytest -q tests/unit/test_state_layout.py tests/unit/test_score_blocks.py tests/numerical/test_composite_model.py tests/unit/test_block_priors.py tests/synthetic/test_multiblock_recovery.py tests/integration/test_composite_e0_equivalence.py tests/integration/test_golden_fixtures.py
source ~/environments/pyenv/bin/activate && python -m pytest -q
```

## Deferred Work

Explicitly deferred by design:

- Python Candidate B runtime optimisation
- full Python Candidate B historical replay as a milestone
- real feature-selection work
- structured covariance
- C kernels

## Unresolved Architectural Issues

1. The current composite Laplace path forms dense parameter-space information by repeated HVP pullback, which is correct but not yet the final scalable route.
2. Fixed-parameter blocks are represented architecturally but the historical replay engine has not yet been expanded into a full experiment composer with heterogeneous real blocks.
3. The backend contract is now concrete enough for C-port work, but the exact kernel partitioning for Candidate B proper partitioned-PL remains a future implementation decision.
