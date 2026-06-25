# Phase L4B2: C11 Monte Carlo Ranking and Thompson Semantics

Phase L4B2 extends the pure-C11 backend from stochastic score generation to stochastic posterior ranking without changing the Gaussian sampling or deterministic decision mathematics frozen in earlier phases.

## Scope

- ABI bump to `1.6.0`
- Native posterior-rank accumulation from sampled score matrices
- Probability-best summaries
- Probability-top-k summaries
- Expected-rank summaries
- Rank-standard-deviation summaries
- Optional retained score samples on `bolr_posterior_prediction`
- Thompson decision support through the generic C decision engine
- Python `ctypes` wrappers and retained-sample equivalence coverage

## Frozen semantics

- Rank ordering matches the Python Phase K reference exactly: stable descending score sort with original candidate index tie-breaks.
- Winner identity is the first element of that stable order.
- Candidate ranks are indexed `1..N`.
- `probability_top_k[k]` is the Monte Carlo frequency of `rank <= k`.
- `expected_rank` and `rank_stddev` are computed directly from those integer ranks.
- Thompson selection uses retained score sample `0`.

## Design choices

- The rank accumulator stores exact integer counts and rank sums before converting to probabilities or moments.
- Retained score samples are optional and diagnostic rather than part of the production replay contract.
- The validated L4B2 reference path can retain all sampled scores to separate three error sources cleanly:
  - Gaussian sampling
  - score transformation
  - ranking and accumulation
- Python/C equivalence compares ranking summaries against the exact same retained sampled-score matrix, so ranking semantics are tested independently of RNG-stream identity.

## Current limitation

The current L4B2 implementation performs Monte Carlo accumulation over an in-memory sampled-score matrix attached to `bolr_posterior_prediction`.

This is intentionally a reference path. It is not yet the bounded-memory production replay design.

## Validation

Native and Python validation currently covers:

- GCC debug native tests
- Clang debug native tests
- GCC sanitizer native tests
- GCC release build
- Python `tests/c_backend`
- full Python test suite
- retained-score ranking equivalence
- Thompson sample-zero equivalence

## Deferred work

L4B2 intentionally excludes:

- streaming or chunked Monte Carlo accumulation
- sample-zero-only retention mode
- replay-engine state transitions
- ready/pending replay checkpoints
- durable binary checkpoint files
- CRC or corruption-handling file codecs
