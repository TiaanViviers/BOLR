# Phase L4B2.2: C11 Streaming Monte Carlo Ranking and In-Memory Replay

This phase extends the L4B2 stochastic-ranking backend into a bounded-memory, causal, restartable daily engine without changing the frozen ranking, sampling, Laplace, or adaptive mathematics.

## Implemented scope

- ABI bump to `1.7.0`
- public replay API through `bolr/replay.h`
- explicit replay phases:
  - `READY`
  - `AWAITING_OUTCOME`
- streaming posterior ranking through `bolr_posterior_prediction_monte_carlo_rank_streaming()`
- exact integer rank accumulation reused across retained and streaming paths
- chunked posterior sampling with chunk-invariant rank summaries
- optional production retention modes:
  - `BOLR_SCORE_RETENTION_NONE`
  - `BOLR_SCORE_RETENTION_SAMPLE_ZERO`
- exact Thompson sample-zero preservation without retaining the full score matrix
- fixed-transition replay engine
- adaptive-transition replay engine
- Candidate A replay updates
- Candidate B exact replay updates
- Candidate B deterministic-sampled replay updates through Python-materialized pairs
- in-memory replay checkpoint export/import for:
  - ready states
  - pending states
- transactional begin/finish failure semantics
- Python ctypes wrappers for streaming ranking and replay handles
- native and Python checkpoint-resume equivalence coverage

## Frozen semantics

- Ranking order remains the L4B2 order:
  - stable descending score sort
  - original candidate index breaks exact ties
  - ranks numbered `1..N`
- Streaming accumulation must match the retained reference path exactly for:
  - probability-best
  - probability-top-`K`
  - expected rank
  - rank standard deviation
  - tie counts
- Thompson selection remains defined by retained Monte Carlo sample `0`.
- `begin_day()` remains outcome-free. Realized outcomes enter only through `finish_day()`.

## Replay contract

The native replay lifecycle is now:

```text
READY
  -> begin_day()
AWAITING_OUTCOME
  -> finish_day()
READY
```

`begin_day()` performs:

1. predictive state construction;
2. streaming posterior ranking;
3. optional region construction;
4. deterministic or stochastic decision selection;
5. pending-state capture for later outcome-conditioned update.

`finish_day()` performs:

1. observation-conditioned Laplace update from the stored predictive state;
2. optional adaptive-state observation update;
3. pending-state clearance;
4. promotion back to `READY`.

The replay engine is transactional:

- if `begin_day()` fails, the engine remains in `READY`;
- if `finish_day()` fails, the engine remains in `AWAITING_OUTCOME`;
- checkpoint export failure does not mutate replay state;
- adaptive replay imports require a matching adaptive-policy configuration hash.

## Validation added in this phase

- native rank-accumulator merge/reset test coverage
- native replay failure-path coverage:
  - invalid region request without a graph
  - finish failure under model-schema mismatch
  - checkpoint export allocation failure
- native pending-checkpoint resume coverage
- Python streaming-versus-retained ranking equivalence
- Python fixed replay checkpoint-resume equivalence
- Python adaptive replay checkpoint-resume equivalence across multiple days
- Python Candidate B exact and deterministic-sampled replay equivalence

## Deliberately deferred

This phase still excludes:

- sectioned binary checkpoint files
- CRC or corruption envelopes around replay checkpoints
- POSIX atomic checkpoint writes
- mid-batch Monte Carlo checkpoints
- full native historical replay
- multithreading or SIMD-specific optimization
- native pair sampling RNG inside Candidate B replay

The retained-score path from L4B2 remains available as the debugging and equivalence reference path. Production replay is expected to use the streaming accumulator by default.
