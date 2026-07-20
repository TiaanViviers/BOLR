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
- Phase L4B1: ABI `1.5.0` native stochastic posterior-sampling support is implemented, including PCG32 stream-selectable RNG state, immutable 128-layer Ziggurat normals, exact RNG checkpoint export/import, Gaussian posterior state sampling, antithetic sampling, composite score sampling, ctypes wrappers, deterministic integer/checkpoint regression tests, and bounded sampling-moment validation; posterior-rank Monte Carlo summaries, Thompson policies, and replay checkpoint composition remain deferred
- Phase L4B2: ABI `1.6.0` native Monte Carlo ranking support is implemented, including exact sampled-score rank accumulation, probability-best and probability-top-k summaries, expected rank, rank standard deviation, optional retained score samples, Thompson sample-zero selection, ctypes wrappers, GCC/Clang/sanitizer coverage, and Python/C retained-sample equivalence tests; streaming accumulation, replay state machines, and durable checkpoint files remain deferred
- Phase L4B2.2: ABI `1.7.0` native bounded-memory replay support is implemented, including streaming Monte Carlo rank accumulation with optional sample-zero retention, exact reusable rank accumulators, a native causal replay state machine, in-memory ready/pending replay checkpoints, fixed and adaptive replay handles, Candidate A and Candidate B replay integration, transactional failure semantics, ctypes bindings, and native/Python checkpoint-resume equivalence coverage
- Phase L4B2.3: ABI `1.8.0` versioned portable checkpoint codec and atomic file persistence are implemented, including little-endian sectioned format `v1.0` (`BOLRCP01`), CRC32 integrity checks, ready/pending encode-decode, restore-context validation, atomic POSIX write/read, injectable I/O failure hooks, Python byte/file wrappers, golden fixtures, and file-based replay restart coverage
- Phase L5.1: full native Candidate A historical replay harness is implemented, including Python orchestration around the C replay engine, durable checkpoint scheduling, forced ready/pending restart, daily/manifest/summary outputs, timing diagnostics, fixed-transition historical runs, and bounded adaptive fixture coverage; a documented full-YM command is provided, and smoke/restart historical executions validate the harness (full-period production evidence remains an operator run, not a pytest gate)
- Phase L5.2: Candidate A policy/static-baseline matrix is implemented, including always-41, warm-up static, trailing mean, oracle replay static, and Candidate A fixed/adaptive × {posterior_mean, probability_best, Thompson} comparison with selection/turnover/bad-switch and candidate-41 delta diagnostics. L5.1 finding: Candidate A fixed/probability-best behaved almost statically (candidate 41). L5.2 compares Candidate A policies against static and trailing baselines.

## C Backend ABI

- Current native ABI: `1.8.0`
- Checkpoint format: `1.0` (`BOLRCP01`)
- Release gate validated for L5.2:
  - `make -C csrc BUILD_DIR=build/l5-debug-gcc clean test CC=gcc`
  - `make -C csrc BUILD_DIR=build/l5-debug-clang clean test CC=clang`
  - `make -C csrc BUILD_DIR=build/l5-sanitize-gcc clean sanitize CC=gcc`
  - `make -C csrc BUILD_DIR=build/l5-release-gcc clean release CC=gcc`
  - `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q tests/c_backend`
  - `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q`
  - `PYTHONPATH=. ~/environments/pyenv/bin/pytest -q tests/c_backend/test_l5_candidate_a_policy_matrix.py`
  - documented matrix command in `research_docs/23_L5_2_Candidate_A_Policy_Matrix_and_Static_Baselines.md` and `scripts/run_l5_candidate_a_policy_matrix.py`

## L4B2 Ranking Notes

- Native RNG: PCG32 XSH-RR with explicit `(seed, stream)` selection, where `inc = (stream << 1) | 1`.
- Native normal sampler: immutable 128-layer Marsaglia-Tsang Ziggurat with committed lookup tables and no mutable global state.
- Native checkpoint scope: exact continuation of the PCG/Ziggurat stream on supported Linux GCC/Clang builds through `bolr_rng_checkpoint`.
- Gaussian sampling: one Cholesky factorization per call, optional antithetic ordering matched to the Python reference convention, and caller-owned row-major output buffers.
- Remaining reproducibility caveat: the integer stream is exact by construction; normal draws rely on `libm` for `exp` and `log`, so cross-platform bitwise identity is only claimed for the validated Linux toolchains above.
- Rank ordering semantics are frozen to the Python reference: stable descending score sort with original candidate index tie-breaks, ranks numbered `1..N`, and winner equal to rank-1 under that stable order.
- Retained score samples are optional and diagnostic. The current `bolr_posterior_prediction_monte_carlo_rank()` path may retain all score samples for debugging and equivalence, but production replay will move to streaming accumulation by default.
- Thompson selection is frozen to Monte Carlo sample `0`. Future chunked replay must preserve sample `0` exactly but does not need to retain the full sampled-score matrix.
- L4B2.2 extends this with `bolr_posterior_prediction_monte_carlo_rank_streaming()`, which accumulates exact rank statistics in chunks and can retain only sample `0` for Thompson semantics.
- L4B2.2 also adds a native daily replay state machine with exact in-memory checkpoint export/import for both ready and pending states.
- L4B2.3 adds portable sectioned binary checkpoints (`BOLRCP01` v1.0) with CRC32 integrity, atomic POSIX persistence, and file-based ready/pending restart.
- L5.1 adds the native Candidate A historical replay harness with durable checkpoint restart.
- L5.2 adds Candidate A policy/static-baseline comparison; Candidate B historical replay remains later L5 work.

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
