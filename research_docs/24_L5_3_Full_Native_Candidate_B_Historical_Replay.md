# Phase L5.3 — Full Native Candidate B Historical Replay

## Motivation

L5.2 showed Candidate A did **not** produce a non-degenerate dynamic win over always-candidate-41:

| Strategy | total PnL | effective selected |
|---|---:|---:|
| baseline_always_41 | 1055.24 | 1.0 |
| candidate_a_fixed_probability_best | 1089.63 | ≈1.02 |
| candidate_a_adaptive_probability_best | 1055.24 | 1.0 |
| oracle_static_best_replay | 2865.32 | leakage |

L5.3 therefore evaluates **Candidate B**, the ordered-partition / pairwise ranking observation model, under the same historical protocol.

Central question:

> Does Candidate B produce useful dynamic ranking behaviour beyond static candidate 41 and beyond Candidate A?

## Protocol (unchanged)

| Window | Dates | Days |
|---|---|---:|
| Warm-up | 2021-01-29 → 2023-01-11 | 504 |
| Replay | 2023-01-12 → 2024-10-08 | 450 |
| Candidates | YM grid | 1428 |

Data: `data/YM_full.parquet`, grid: `data/YM_grid.csv`.

MC defaults: 512 samples, chunk 64, top-k 10, RNG seed `20260720`, stream `1`.

## Candidate B sampled vs exact

- **Exact**: all cross-group preferred pairs. Too dense for full 1428-candidate replay (hundreds of thousands of pairs/day).
- **Sampled** (`candidate_b_sampled`): deterministic date-seeded sampling with `sampled_pair_budget=4096`, `sampling_seed=0`, `normalize_pair_losses=True`.
- Exact is retained only for bounded correctness validation.

Partition defaults: `OrderedPartitionConfig` with `relative_tolerance=0.1`, `positive_threshold=0.0`, `all_irrelevant_policy=no_update`.

No Candidate B hyperparameter tuning in this phase.

## Warm-up note

L5.3 warm-up uses the **same SoftTarget static surface** as L5.1/L5.2. Candidate B observations apply only at `finish_day`.

Rationale:

1. Fair A/B comparison on a shared prior surface.
2. Candidate B static-surface fitting over 504 days × dense pairwise losses is prohibitively expensive in the current Python materialisation path (measured bottleneck before any optimisation).

This does **not** change Candidate B mathematics, partition semantics, or pair weighting.

## Replay loop invariants

```text
for each replay day:
    begin_day(context)          # decision issued; no current-day outcomes
    optional pending checkpoint
    reveal outcomes
    build ordered partition
    materialize deterministic sampled pairs
    finish_day(observation)     # does not recompute decision
    record + optional ready checkpoint
```

## Strategy set

| Observation | Variant | Transition | Decision |
|---|---|---|---|
| Candidate B | sampled | fixed | probability_best |
| Candidate B | sampled | fixed | posterior_mean |
| Candidate B | sampled | fixed | Thompson |
| Candidate B | sampled | adaptive_additive | probability_best |
| Candidate B | sampled | adaptive_additive | posterior_mean |
| Candidate B | sampled | adaptive_additive | Thompson |

Minimum required first full run: `candidate_b_sampled_fixed_probability_best`.

L5.2 comparison rows are imported when available (not re-run):

- `baseline_always_41`
- `oracle_static_best_replay`
- `candidate_a_fixed_probability_best`
- `candidate_a_adaptive_probability_best`
- `candidate_a_fixed_thompson`
- `candidate_a_adaptive_thompson`

## Outputs

Root: `outputs/l5_candidate_b_native_replay/`

Per run: `manifest.json`, `summary.json`, `daily_results.csv`, `checkpoints.jsonl`, `logs/replay.log`.

Comparison: `strategy_summary.csv/json`, `strategy_daily_panel.csv`, `selection_diagnostics.csv`, `candidate_b_pair_diagnostics.csv`, `probability_best_calibration.csv`, `README.md`.

## Restart validation

Bounded tests cover:

- ready checkpoint restart matches uninterrupted run
- pending checkpoint restart preserves selected candidate
- sampled pair counts are deterministic across restart

Smoke historical command also forces ready after day 8 and pending on day 12.

## Commands

### Smoke (20 days)

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_native_candidate_b_replay.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_b_native_replay_smoke \
  --run-name candidate_b_sampled_fixed_probability_best_smoke \
  --variant sampled \
  --transition fixed \
  --decision-policy probability_best \
  --mc-samples 512 \
  --mc-chunk-size 64 \
  --top-k 10 \
  --rng-seed 20260720 \
  --rng-stream 1 \
  --checkpoint-every-n-days 5 \
  --maximum-days 20 \
  --force-restart-after-day 8 \
  --force-pending-restart-on-day 12
```

### Full first strategy

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_native_candidate_b_replay.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_b_native_replay \
  --run-name candidate_b_sampled_fixed_probability_best \
  --variant sampled \
  --transition fixed \
  --decision-policy probability_best \
  --mc-samples 512 \
  --mc-chunk-size 64 \
  --top-k 10 \
  --rng-seed 20260720 \
  --rng-stream 1 \
  --checkpoint-every-n-days 25
```

### Matrix

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_l5_candidate_b_matrix.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_b_native_replay \
  --variant sampled \
  --mc-samples 512 \
  --mc-chunk-size 64 \
  --top-k 10 \
  --rng-seed 20260720 \
  --rng-stream 1 \
  --checkpoint-every-n-days 25 \
  --include-fixed \
  --include-adaptive \
  --l5-2-comparison-dir outputs/l5_candidate_a_policy_matrix/comparison
```

## Smoke results (completed)

| Field | Value |
|---|---|
| days | 20 |
| total_pnl | -194.79 |
| mean sampled pairs | 3401.85 |
| max sampled pairs | 4096 |
| forced ready restart | passed |
| forced pending restart | passed |
| total elapsed | ~84.4 s |
| warm-up | ~54.4 s |
| mean day | ~995 ms |

## Full-run results

All six sampled Candidate B strategies completed over 450 replay days.

| Strategy | total PnL | Δ always-41 | Δ A fixed/pbest | unique | N_eff | c41 share | net_switch vs 41 | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **candidate_b_sampled_fixed_thompson** | **1414.72** | **+359.48** | **+325.09** | 80 | 24.1 | 0.36 | **+359.48** | ~429 s |
| candidate_a_fixed_probability_best (L5.2) | 1089.63 | +34.39 | 0 | ~2 | ≈1.02 | ~1.0 | near-static | — |
| always-41 / B fixed&adaptive probability_best | 1055.24 | 0 | -34.39 | 1 | 1.0 | 1.0 | 0 | ~490/431 s |
| B fixed&adaptive posterior_mean | 808.28 | -246.96 | -281.35 | 4 | 1.09 | 0.987 | -246.96 | ~733/427 s |
| B adaptive thompson | -1706.43 | -2761.67 | -2796.06 | 75 | 22.3 | 0.36 | -2761.67 | ~421 s |
| oracle_static_best_replay (leakage) | 2865.32 | +1810 | — | 1 | 1 | — | — | — |

### `candidate_b_sampled_fixed_probability_best`

- Selected candidate **41 on 450/450 days** (exact always-41 collapse).
- mean sampled pairs/day 3269.7 (max 4096); possible pairs mean/max ≈ 265k / 537k.
- informative days 444/450; fallback 6; checkpoints 18; Laplace failures 0.
- Probability-best calibration: selected pbest mostly ≥0.30 while top-1 hit rate ≈3% (overconfident).

### Pair / timing (representative fixed/pbest)

| Timing | seconds |
|---|---:|
| warm-up (SoftTarget shared) | ~55 |
| begin_day total | ~317 |
| finish_day total | ~59 |
| pair sampling total | ~47 |
| total run | ~490 |

## Direct answers

1. **Did Candidate B beat always-candidate-41 in a non-degenerate way?**  
   - **probability_best / posterior_mean: No** (collapse or near-collapse to 41).  
   - **fixed Thompson: Yes, observationally** — +359.48 PnL, 80 unique selections, net_switch_value +359.48.  
   - **adaptive Thompson: No** — dynamic but large loss.

2. **Did Candidate B beat Candidate A fixed/probability-best?**  
   - Only **fixed Thompson** did (+325.09). Other B policies did not.

## Recommended next phase

Mixed signal:

- Probability-best / posterior-mean paths reinforce the **static favourite (41)** story → L5.4 static structure investigation remains relevant.
- Fixed Thompson shows a **non-degenerate dynamic** observational edge → also justifies **L5.4 baseline expansion / robustness** (costs, date splits, alternate seeds/streams) before trusting the Thompson edge.
- Adaptive Thompson’s large loss argues against treating adaptation as free performance.

PnL is observational evidence only. Do not overstate profitability.
