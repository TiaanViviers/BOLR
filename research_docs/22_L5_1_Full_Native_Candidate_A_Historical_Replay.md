# Phase L5.1 — Full Native Candidate A Historical Replay Harness

## Goal

Answer the engineering question:

> Can the complete native backend run a full historical Candidate A replay safely, restartably, and observably?

This phase does **not** answer profitability. PnL and regret are reported as observed metrics only.

## Scope

Candidate A only, on top of ABI `1.8.0` and checkpoint format `v1.0` (`BOLRCP01`).

Implemented:

- Python orchestration around the C replay engine
- durable ready/pending checkpoint scheduling
- forced ready and pending restart during runs
- daily results, manifest, summary, checkpoint metadata
- timing and optional Linux RSS diagnostics
- fixed-transition historical runs
- adaptive additive transition on bounded fixtures / optional historical runs

Not changed:

- Candidate A targets/likelihood
- score model / priors / adaptation mathematics
- replay state machine semantics
- checkpoint format
- RNG algorithm
- decision-policy definitions
- historical feature construction

Deferred: Candidate B full replay, keyed RNG, Fast-BOCPD, tuning, parallel replay, compression.

## Data source

- `data/YM_full.parquet` via `HistoricalDataset`
- `data/YM_grid.csv` via `load_candidate_grid`
- Projected columns only: `date`, `config_id`, `entry_percentage`, `sl_trail_percentage`, `pnl`
- Validated calendar: `2021-01-29` → `2024-10-08` (954 days, 1428 candidates/day)

## Replay protocol

Default split (configurable):

- warm-up: first 504 days
- replay: remaining days (450 unless `maximum_days` / date windows truncate)

Native loop:

1. `get_predictors(date)`
2. `begin_day(...)` (C prediction, MC ranking, decision)
3. optional pending checkpoint / forced pending restore
4. `reveal_outcomes(date)`
5. native Candidate A target build + observation
6. `finish_day(...)`
7. record row; optional ready checkpoint / forced ready restore

## Warm-up protocol

Preferred path (implemented):

- Python `SoftTargetBuilder` + `fit_static_surface` over warm-up days
- `make_initial_dynamic_prior`
- static baseline + dynamic surface composite model
- hand Gaussian prior into `CBackend.replay_engine_fixed` / `_adaptive`

Manifest records warm-up backend, dates, state dimension, posterior hash, covariance trace.

## Decision policy

CLI aliases:

- `posterior_mean` → `posterior_mean_argmax` (default recommendation)
- `probability_best` → `maximum_probability_best`
- `thompson` / `thompson_sample_zero` → Thompson with sample-zero retention
- `region_medoid` → highest-mass region + weighted medoid (requires graph wiring for production use)

## Checkpoint schedule

Defaults:

- `checkpoint_every_n_days = 25`
- `checkpoint_at_end = true`
- `checkpoint_at_pending_day = false`

Forced interruption:

- `--force-restart-after-day K`
- `--force-pending-restart-on-day K`

## Restart validation

Bounded synthetic fixtures prove:

- ready file restart matches uninterrupted decisions
- pending file restart preserves selected candidate, then matches finish posterior path

Historical runs can exercise the same forced-restart flags on real YM days.

## Output schema

```text
outputs/<run_name>/
  manifest.json
  summary.json
  daily_results.csv
  checkpoints.jsonl
  config.json
  checkpoints/*.bolrcp
  logs/replay.log
```

## Commands

Full historical fixed-transition run:

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_native_candidate_a_replay.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_a_native_fixed \
  --run-name candidate_a_native_fixed_v1 \
  --transition fixed \
  --decision-policy probability_best \
  --mc-samples 512 \
  --mc-chunk-size 64 \
  --top-k 10 \
  --rng-seed 20260720 \
  --rng-stream 1 \
  --checkpoint-every-n-days 25 \
  --overwrite-outputs
```

Restart-test historical command:

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_native_candidate_a_replay.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_a_native_restart_test \
  --run-name candidate_a_native_restart_test \
  --transition fixed \
  --decision-policy probability_best \
  --mc-samples 512 \
  --mc-chunk-size 64 \
  --top-k 10 \
  --rng-seed 20260720 \
  --rng-stream 1 \
  --checkpoint-every-n-days 10 \
  --force-restart-after-day 40 \
  --force-pending-restart-on-day 55 \
  --overwrite-outputs
```

Harness module: `bolr/evaluation/native_candidate_a_replay.py`.

## Timing results (executed smoke)

Validated historical smoke on 2026-07-20:

```text
outputs/l5_candidate_a_native_smoke/candidate_a_native_smoke_v1
replay days: 5 (after 504-day warm-up)
decision: probability_best
mc samples: 128
forced ready restart: passed (after day index 1)
forced pending restart: passed (on day index 3)
total elapsed: ~86.6 s (dominated by warm-up)
```

Full 450-day MC-512 execution is supported by the same CLI; it is an operator evidence run rather than a pytest gate.

## Summary metrics

Smoke metrics are written to `summary.json` for each run (selected PnL totals/means, regret, hit rates, Laplace stats, checkpoint counts, timing, optional Linux RSS). These are observational only.

## Limitations

- Engine-owned RNG draw counts are not exposed on the daily CSV without exporting checkpoints each day; seed/stream provenance is recorded instead.
- Adaptive per-day surprise/BOCPD fields are optional and may be blank when not surfaced by finish diagnostics.
- `selected_probability_top_k` is blank unless a future ABI exposes it on the decision object.
- Full 450-day MC-512 historical runtime is substantial; use `maximum_days` for smoke runs.
- Region-medoid historical use still needs an explicit grid-graph attachment in the harness.

## Next phase

L5.x evidence expansion: longer production runs, Candidate B native historical harness, optional comparison packs against frozen Python Candidate A outputs, and measured optimisation only if correctness-preserving bottlenecks appear.


Full Candidate A replay succeeded technically, but fixed/probability-best selection collapsed almost entirely to candidate 41 and underperformed the always-41 static baseline due to one bad switch.