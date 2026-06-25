# Phase L4B1: C11 RNG and Gaussian Posterior Sampling

Phase L4B1 adds the stochastic foundation required for posterior sampling without changing the deterministic Laplace mathematics frozen in L4A.

## Scope

- ABI bump to `1.5.0`
- Native `bolr_rng` handle with explicit `(seed, stream)` inputs
- PCG32 XSH-RR integer generation
- Open-interval uniform mapping `(u + 1) / (2^32 + 1)`
- Immutable committed 128-layer Ziggurat lookup tables
- Native `N(0, 1)` draws with no cached secondary normal
- Exact typed RNG checkpoint export/import and byte encoding
- Native Gaussian state sampling with optional antithetic ordering
- Native composite score sampling and direct posterior-to-score sampling
- Python `ctypes` wrappers: `CRNG`, `CRNGCheckpoint`, and sampling helpers

## Design choices

- Stream semantics use `inc = (stream << 1) | 1`, so identical seed/stream pairs replay exactly and different streams move onto different PCG trajectories.
- The runtime library performs no mutable global initialization. Ziggurat tables are compiled in as `static const`.
- The RNG handle owns all mutable state: PCG state, increment, seed metadata, stream metadata, and draw counters.
- No C RNG sampling policy is introduced yet for Candidate B pair materialization. Python still materializes deterministic sampled pairs and passes them into the native observation handle.

## Checkpoint surface

`bolr_rng_checkpoint` stores:

- algorithm family and version
- schema version
- PCG state and increment
- table hash
- seed and stream metadata
- integer, uniform, and normal draw counters

Decode rejects:

- even increments
- wrong checkpoint magic or version
- incompatible schema or algorithm metadata
- mismatched committed table hash
- wrong payload size

## Sampling path

Gaussian state sampling uses:

`theta = m + L z`

with one Cholesky factorization per call. Antithetic mode mirrors the Python ordering:

- draw `half = ceil(B / 2)` latent normals
- emit transformed positive samples first
- append reflected samples `2m - theta`
- truncate naturally for odd `B`

Composite score sampling applies the existing model forward operator row-by-row to caller-supplied state samples. The direct posterior-to-score convenience path uses the same Gaussian sampling kernel and then applies the same composite transform.

## Validation

Native C coverage now includes:

- fixed PCG32 integer vectors
- stream separation
- RNG clone equivalence
- RNG checkpoint continuation
- invalid checkpoint rejection
- Gaussian antithetic sampling
- composite score transformation

Python C-backend coverage now includes:

- deterministic RNG vectors and metadata
- RNG byte checkpoint round-trips
- Gaussian antithetic sampling
- score sampling from supplied state samples
- posterior-to-score sampling equivalence
- bounded empirical sampling-moment checks

## Reproducibility contract

- The PCG32 integer stream is exact across supported builds.
- Exact checkpoint continuation is guaranteed on the validated Linux GCC/Clang toolchains.
- Standard-normal draws use `exp` and `log`, so the project does not yet claim universal bitwise identity across arbitrary `libm` implementations.

## Deferred work

L4B1 intentionally excludes:

- probability-best and probability-top-k Monte Carlo accumulation
- Thompson policies
- replay checkpoint composition
- native general-purpose RNG policy utilities
- native Candidate B pair sampling
