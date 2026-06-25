# BOLR
Bayesian Online Listwise Ranking for applications to quantative trading

## Phase Status

- Phase A/B: complete and frozen
- Phase C: complete
- Phase D first slice: complete
- Phase E0: complete
- Phase F: Candidate B mathematics, reduced exact validation, generic Laplace integration, and real-data resume-equivalence are complete; realistic pure-Python Candidate B historical execution is deferred
- Phase G: multi-block score/state architecture, composite Laplace compatibility, backend contracts, and golden numerical fixtures are implemented
- Phase H: structured priors, penalty-shaped block dynamics, structured static fitting, diagnostics, and golden structured-prior fixtures are implemented
- Phase I: orthogonal graph residual architecture, constrained spectral basis construction, graph priors/dynamics, and graph-residual smoke coverage are implemented
- Phase J: generic composite replay, causal adaptive transition policies, online surprise standardisation, BOCPD-backed adaptation, and adaptive golden fixtures are implemented
- Phase K: predictive posterior decision objects, Monte Carlo ranking probabilities, connected-region inference, calibration metrics, decision-policy replay integration, and Phase K golden fixtures are implemented
- Phase L1: pure-C11 backend foundation, Make build, versioned ABI, allocator/ownership model, checkpoint-ready handles, Candidate A reference kernels, ctypes binding, and initial Python/C equivalence harness are implemented
- Phase L2: Python-accessible C Gaussian inference integration is complete for Candidate A, including C-owned state/layout/model handles, composite Laplace updates, bounded historical replay equivalence, sequential checkpoint continuity, backend ownership/error handling, isolated build profiles, and a concurrency-safe Python loader; Candidate B, adaptation, decisions, and RNG remain unported
- Phase L3A: native Candidate A target construction, ordered-partition construction, exact and deterministic-sampled cross-group Candidate B observations, generic Laplace integration, Python/C derivative and Laplace equivalence, bounded sequential Candidate B validation, and compiler/sanitizer coverage are implemented; native RNG, adaptive transitions, and full historical Candidate B replay remain deferred
- Phase L3B: ABI `1.3.0` native adaptive-transition support is implemented for the reference path, including C11 BOCPD state evolution, causal EW surprise standardisation, blockwise innovation attribution, adaptive additive process-noise policies, pending reset scheduling, adaptive-state checkpoint round-trips, Python ctypes wrappers, and bounded Candidate A/Candidate B sequential equivalence tests; full replay orchestration, native discount-family prediction semantics, RNG, and Fast-BOCPD integration remain deferred
- Phase L4A: ABI `1.4.0` native deterministic posterior-decision support is implemented, including posterior score summaries, selected score covariance, analytic pairwise win probabilities, canonical-grid consensus regions, connected-component summaries, weighted graph medoids, deterministic candidate and region decision policies, ctypes wrappers, and Python/C equivalence coverage; RNG, posterior sampling, Monte Carlo ranking probabilities, Thompson policies, and replay checkpoint composition remain deferred

## C Backend ABI

- Current native ABI: `1.4.0`
- Release gate validated for L4A:
  - `make -C csrc BUILD_DIR=build/l4a-debug-gcc test CC=gcc`
  - `make -C csrc BUILD_DIR=build/l4a-debug-clang test CC=clang`
  - `make -C csrc BUILD_DIR=build/l4a-sanitize-gcc sanitize CC=gcc`
  - `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q tests/c_backend`
  - `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q`

## L4A Integration Notes

Two aliasing defects were found and corrected while integrating deterministic posterior decisions through the end-to-end native path:

- Posterior-prediction score-buffer alias:
  - `bolr_posterior_prediction_create()` originally reused the workspace score buffer as both internal scratch and output storage, which doubled dynamic score contributions in composite models.
  - Regression coverage: `csrc/tests/test_score_uncertainty.c` and `tests/c_backend/test_c_posterior_decision.py`.
- Composite-forward scratch-buffer alias:
  - `bolr_model_forward()` originally reused a score scratch buffer across blocks without clearing it before each block forward pass, which leaked prior block contributions into later block accumulations.
  - Regression coverage: `csrc/tests/test_score_uncertainty.c`, `csrc/tests/test_decision_policy.c`, and `tests/c_backend/test_c_posterior_decision.py`.

## Phase A/B Baseline

The current repository contains the structural foundation for the first BOLR reference implementation:

- canonical candidate-grid loading from `data/YM_grid.csv`
- log-coordinate transformation over `entry_percentage` and `sl_trail_percentage`
- tensor-product B-spline candidate basis with centering and rank reduction
- selected-column context basis
- explicit and structured dynamic design construction
- Candidate A tolerance-aware soft-target generalized-Bayes kernels
- unit, numerical, derivative, and integration tests

Run the validation summary with:

```bash
source ~/environments/pyenv/bin/activate
python scripts/validate_foundation.py
```

Run the full test suite with:

```bash
source ~/environments/pyenv/bin/activate
python -m pytest -q
```

### Recorded Diagnostics

Using the real YM configuration grid and the current default tensor basis (`6 x 8`, cubic splines on each axis), the centered candidate basis has:

- raw width `48`
- effective rank `p_c = 47`
- largest retained singular value `5.668189618668`
- smallest retained singular value `0.295215090099`
- retained-basis condition number `19.200202864835`
- maximum absolute centered-column mean `2.552e-16`

This means the first dynamic state dimension is:

- `P = p_c * p_m`
- for the intended initial context width `p_m = 6`, `P = 282`
- the current validation script uses a 4-dimensional demo context basis, so its explicit design has width `47 * 4 = 188`

Candidate A derivative baselines from the finite-difference checks are:

- score-space gradient max absolute error `7.902353771350e-11`
- score-space curvature max absolute error `1.010611278054e-05`
- score-space HVP max absolute error `1.010611278053e-05`
- parameter-space gradient max absolute error `4.749506343771e-11`
- parameter-space HVP max absolute error `5.394207672420e-06`

The current soft-target defaults (`kappa = 1`, robust median/MAD-IQR scaling, no tolerance collapse) produce the following representative real-day summaries:

| Case | Day | Entropy | Perplexity | Max target mass | Clipping fraction | Utility scale | All irrelevant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lowest entropy | `2016.12.23` | `5.356071758224` | `211.890950569241` | `0.011999507142` | `0.264705882353` | `0.000001000000` | `False` |
| Median entropy | `2013.09.24` | `6.022378120705` | `412.558544062667` | `0.012234416258` | `0.044117647059` | `8.369276999999` | `False` |
| Highest entropy | `2019.07.04` | `7.264030142900` | `1427.999999999998` | `0.000700280112` | `0.000000000000` | `0.000001000000` | `True` |

The highest-entropy case is effectively uniform across all `1,428` candidates and is currently flagged as `all_irrelevant`, which is the expected baseline behavior for a fully degenerate utility day.
