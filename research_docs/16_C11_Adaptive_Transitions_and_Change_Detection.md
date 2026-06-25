# Phase L3B: C11 Adaptive Transitions and Change Detection

This phase ports the frozen Phase J adaptive-transition reference mathematics into the pure-C11 backend without introducing RNG, Fast-BOCPD coupling, or a separate native replay runner.

Implemented native components:

- ABI bump to `1.3.0`
- causal EW surprise standardizer
- scalar Gaussian-NIG BOCPD reference detector
- generalized-loss and posterior-update surprise scalar bridge
- blockwise innovation attribution
- partial-reset transform with cross-covariance scaling
- adaptive policy/state handles
- adaptive-state byte checkpoint round-trip
- Python ctypes wrappers for BOCPD and adaptive policy/state
- bounded Candidate A and Candidate B sequential equivalence tests

Current native adaptive semantics intentionally mirror the existing Python reference:

- adaptation is configured after the day-`t` posterior update and affects only day `t+1`
- pending resets are applied on the next prediction step
- additive block multipliers are used in predictive covariance construction
- adaptive discount state is tracked but not yet used in the adaptive predictor because the current Python `AdaptiveAdditiveTransitionPolicy` does not apply those discounts during `predict()`

Validation completed in this slice:

- `make -C csrc BUILD_DIR=build/l3b-debug-gcc test CC=gcc`
- `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q tests/c_backend`
- focused Python adaptive tests covering standardizer, BOCPD, surprise signals, attribution, reset, and adaptive replay timing

Deferred items remain unchanged:

- native full historical replay runner
- native adaptive-discount transition predictor beyond Python parity
- Fast-BOCPD adapter integration
- RNG and posterior sampling
- final checkpoint-file schema
