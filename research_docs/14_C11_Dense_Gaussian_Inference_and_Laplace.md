# C11 Dense Gaussian Inference and Laplace Integration

## Status

Phase L2.2 closes the first usable C inference path for Candidate A.

Implemented:

- owned C Gaussian posterior states with checkpoint encode/decode
- additive prediction from Python through the C ABI
- C-owned state-layout and composite-model artifact handles
- Python wrappers for model artifacts, score context, Candidate A observations, inference workspaces, and Newton configuration
- high-level `CBackend.laplace_update`
- bounded `CBackend.static_fit` exposure
- composite score equality checks for surface, context, and graph blocks
- sequential C/Python posterior equivalence
- bounded historical surface replay equivalence on `YM_full.parquet`

Still intentionally excluded:

- Candidate B in C
- adaptive transitions in C
- posterior sampling and RNG
- decision-layer C port
- final production checkpoint schema

## Ownership Model

Persistent numerical artifacts are copied into C-owned memory:

- state layouts
- static score vectors
- dense block designs
- context block candidate bases
- graph residual bases
- Candidate A targets
- Gaussian means and covariances

Daily context is lightweight and currently passed as a short-lived Python wrapper over a contiguous `float64` vector. The wrapper keeps strong references to any borrowed Python owners for the duration of the inference call.

The practical contract is:

1. `CModelArtifacts` owns immutable score artifacts.
2. `CScoreContext` supplies daily context input.
3. `CGaussianState` owns predictive or posterior state memory.
4. `CInferenceWorkspace` is reusable across same-dimension updates.
5. `CCandidateAObservation` owns the per-day target copy.

All wrappers support explicit `.close()` and `weakref.finalize`.

## High-Level Python API

The integrated path is:

```python
backend = CBackend()
artifacts = backend.model_artifacts(model, batch)
context = backend.score_context(model, batch)
predictive = artifacts.state_from_posterior(prior)
observation = backend.candidate_a_observation(target_observation)
workspace = CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count)
posterior_handle, diagnostics = backend.laplace_update(
    predictive,
    artifacts,
    context,
    observation,
    workspace,
    CNewtonConfig(),
)
posterior = posterior_handle.to_posterior(state_layout=model.layout.metadata())
```

`CBackend` also exposes debug helpers for:

- posterior objective
- posterior gradient
- dense posterior Hessian

These are used for Python/C equivalence tests.

## Numerical Equivalence Coverage

The current acceptance coverage includes:

- direct Gaussian state round-trip
- checkpoint truncation rejection
- composite score and transpose equality
- multiblock Laplace posterior equality
- sequential synthetic prediction/update equality
- bounded 10-day historical Candidate A replay equality
- native C test binary after Laplace workspace fixes

The two key native solver defects fixed during L2.2 were:

- aliased current-state and trial-state buffers in the Newton loop
- aliased score-HVP input/output and aliased gradient scratch during Hessian assembly

Without those fixes, the mode and covariance could drift away from the Python reference even though the objective and gradient at the predictive mean were correct.

## Validation Commands

Primary commands used for L2.2:

```bash
make -C csrc test
PYTHONPATH=. ~/environments/pyenv/bin/pytest -q tests/c_backend
PYTHONPATH=. ~/environments/pyenv/bin/pytest -q
```

## Remaining Differences

The C diagnostics surface is usable but not yet identical to the Python diagnostic vocabulary.

The exposed `static_fit` path is available through Python, but broader warm-up equivalence beyond bounded acceptance fixtures should still be expanded before Phase L3.
