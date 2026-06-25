# Phase L4A: C11 Deterministic Posterior Decisions and Regions

Phase L4A ports the deterministic subset of the Phase K posterior-decision layer into the pure-C11 backend and exposes it through the existing Python orchestration path.

Implemented native components:

- ABI bump to `1.4.0`
- posterior prediction handle for deterministic score summaries
- posterior score mean extraction
- posterior marginal score variance extraction
- selected score covariance for requested candidate subsets
- analytic pairwise score-difference moments and win probabilities
- entropy/effective-count utilities for supplied probability vectors
- canonical grid-graph handle
- consensus candidate-set construction from supplied inclusion probabilities
- connected-component decomposition on the induced consensus subgraph
- region posterior summaries
- weighted graph medoid selection
- deterministic candidate policies:
  - posterior-mean argmax
  - maximum supplied probability-best
  - maximum supplied top-k probability
  - minimum supplied expected rank
- deterministic region policy with representative selection
- tie-aware realised calibration helpers
- Python ctypes wrappers for posterior prediction, grid graph, regions, and decision policies
- focused Python/C equivalence tests for posterior summaries, regions, deterministic decisions, and calibration

The native L4A path intentionally remains deterministic. It does not yet include:

- RNG
- Gaussian posterior sampling
- score sampling
- Monte Carlo probability-best or probability-top-k
- expected-rank accumulation from samples
- Thompson policies
- replay checkpoint composition for stochastic state

## Corrected integration defects

Two end-to-end aliasing defects were found during L4A integration.

### 1. Posterior-prediction score-buffer alias

`bolr_posterior_prediction_create()` initially passed a workspace-owned score buffer into `bolr_model_forward()` while that same call also acquired the same workspace score buffer as internal scratch.

That alias caused block contributions to be added into the same storage twice, which inflated posterior score means in composite models.

The fix was to route prediction outputs into owned `prediction->score_mean` storage and keep workspace score buffers scratch-only.

Regression coverage:

- `csrc/tests/test_score_uncertainty.c`
- `tests/c_backend/test_c_posterior_decision.py`

### 2. Composite-forward scratch-buffer alias

`bolr_model_forward()` reused the score scratch buffer across dynamic blocks without clearing it before each block forward pass.

That allowed one block's score contribution to leak into subsequent block accumulations, which is mathematically incorrect whenever multiple dynamic blocks are present.

The fix was to zero the per-block score scratch before every block forward call.

Regression coverage:

- `csrc/tests/test_score_uncertainty.c`
- `csrc/tests/test_decision_policy.c`
- `tests/c_backend/test_c_posterior_decision.py`

## Validation

L4A was validated with:

- `make -C csrc BUILD_DIR=build/l4a-debug-gcc test CC=gcc`
- `make -C csrc BUILD_DIR=build/l4a-debug-clang test CC=clang`
- `make -C csrc BUILD_DIR=build/l4a-sanitize-gcc sanitize CC=gcc`
- `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q tests/c_backend`
- `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q`

Observed release-gate result for this phase:

- GCC native tests passed
- Clang native tests passed
- ASAN/UBSAN native tests passed
- `tests/c_backend` passed
- full Python test suite passed

## Freeze point

The intended freeze point for this phase is the deterministic native posterior-decision layer at ABI `1.4.0`.

Because the current worktree contains uncommitted L4A changes, a Git tag should be created only after those validated changes are committed. Tagging the current `HEAD` without a commit would not identify the actual L4A result.
