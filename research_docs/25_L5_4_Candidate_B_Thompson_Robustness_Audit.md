# Phase L5.4 — Candidate B Fixed Thompson Robustness Audit

## Why L5.4

L5.3 found one non-degenerate Candidate B result:

| Strategy | total PnL | vs always-41 | unique / N_eff |
|---|---:|---:|---|
| Candidate B fixed Thompson | 1414.72 | +359.48 | 80 / 24.1 |
| Candidate A fixed probability-best | 1089.63 | +34.39 | ~2 / 1.02 |
| always-41 / B probability-best | 1055.24 | 0 | 1 / 1 |
| B adaptive Thompson | -1706.43 | -2761.67 | 75 / 22.3 |

That +359 edge is interesting but untrusted until stress-tested for seed luck, period concentration, costs, and pair-sampling fragility.

Central question:

> Is the Candidate B fixed Thompson edge versus always-candidate-41 robust enough to justify further development?

## Strict boundary

No model mathematics, partition rules, pair weighting, architecture, priors, dynamics, checkpoints, or date protocol were changed. L5.4 only varies Thompson RNG streams (and optionally pair sampling configs) and analyses daily deltas.

## Audited strategy

```text
candidate_b_sampled_fixed_thompson
variant: sampled
transition: fixed
decision: Thompson (sample-zero)
pair_budget: 4096
pair_sampling_seed: 0 (Stage 2 fixed)
mc_samples: 512
rng_seed: 20260720
```

Warm-up remains the SoftTarget static surface shared with L5.1–L5.3.

## Design

### Stage 1 — Bounded smoke

`maximum_days=20`, streams `1:3`, mode `all`.

### Stage 2 — Seed/stream robustness

Full 450-day replay, streams `1:30`, pair sampling fixed.

### Stage 3 — Split / cost / bootstrap analysis

Computed from Stage 2 daily delta panel (no reruns).

### Stage 4 — Pair sampling robustness

Deferred until Stage 2 is reviewed. Contained design:

```text
pair_budgets = 2048,4096,8192
pair_sampling_seeds = 0,1,2
rng_streams = 1:3
```

## Baselines

| Baseline | Source | Role |
|---|---|---|
| always candidate 41 | L5.2 / recomputed | primary static |
| Candidate A fixed probability-best | L5.2 daily panel | best A (near-static) |
| oracle static best replay | L5.2 | leakage ceiling |
| Candidate B fixed probability-best | L5.3 | collapsed to 41 |

## Outputs

Root: `outputs/l5_candidate_b_thompson_robustness/`

Key artefacts:

- `run_registry.csv`
- `seed_robustness/comparison/seed_daily_delta_panel.csv`
- `seed_robustness/comparison/seed_robustness_summary.csv`
- `split_analysis/split_summary.csv`
- `cost_analysis/cost_sensitivity.csv`
- `statistical_tests/block_bootstrap_summary.json`
- `selection_robustness.csv` / `bad_switch_robustness.csv`

## Smoke result (completed)

```text
streams 1:3, maximum_days=20
completed=3 failed=0
share_beating_41 = 0.667
median_delta_vs_41 = 128.28
mean_delta_vs_41 = 124.07
best_stream=1 worst_stream=2
```

Aggregation, registry, and delta panel validated.

## Full seed robustness (Stage 2) — completed

Command:

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_l5_candidate_b_thompson_robustness.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_b_thompson_robustness \
  --rng-seed 20260720 \
  --rng-streams 1:30 \
  --pair-budget 4096 \
  --pair-sampling-seed 0 \
  --mode seed_robustness \
  --resume-existing
```

| Metric | Value |
|---|---:|
| streams completed | **30 / 30** |
| failed | 0 |
| wall clock | ~3.69 h |
| mean total PnL | −158.81 |
| median total PnL | −262.93 |
| mean Δ vs always-41 | **−1214.05** |
| median Δ vs always-41 | **−1318.17** |
| p05 / p95 Δ vs 41 | −3270.8 / +1052.8 |
| **share_beating_41** | **0.267 (8/30)** |
| share_beating Candidate A fixed/pbest | 0.267 |
| best stream | 11 (Δ = +2229.37, PnL = 3284.61) |
| worst stream | 24 (Δ = −6611.65, PnL = −5556.41) |
| L5.3 stream | **stream 3** (Δ = +359.48) — one of the lucky minority |

### Block bootstrap (mean daily-delta path across streams)

| block | observed total Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| 5 | −1214.05 | [−4296, +1681] | 0.197 |
| 10 | −1214.05 | [−4487, +1232] | 0.144 |
| 20 | −1214.05 | [−4463, +696] | 0.082 |

Even the L5.3 stream (stream 3) has bootstrap P(Δ>0) only ≈0.50–0.58 — not decisive.

### Cost sensitivity

At zero costs, only 26.7% of streams beat always-41. Any positive cost per non-41 day makes this worse (share → 0 at cost 10). Tiny costs do **not** create an edge; the raw edge is already negative in the median.

### Splits

| Split | mean Δ vs 41 | share > 0 |
|---|---:|---:|
| full | −1214 | 0.27 |
| 2023 | −447 | 0.40 |
| 2024 | −767 | 0.30 |
| H1 / H2 | −535 / −679 | 0.30 / 0.33 |
| Q1 2023 | +366 | 0.73 |
| other quarters | mostly negative | ≤0.43 |

Performance is not a single-day fluke, but it is **not** robust across streams; Q1 2023 is the only clearly positive quarter on average.

### Selection / switches

- Mean unique selections ≈ 72; mean N_eff ≈ 20.7; mean candidate-41 share ≈ 0.37
- Mean net_switch_value vs 41 = **−1214** (same as total Δ)
- Share of streams with net_switch > 0 = **0.267**

Thompson is dynamically diverse — and that diversity is usually harmful versus always-41.

## Pair sampling robustness

**Deferred.** Stage 2 already falsifies robustness of the L5.3 Thompson edge under fixed pair sampling. Running 27–100 additional full 450-day pair-config × stream combinations would not change the pause decision and would cost many more hours. The harness supports `--mode pair_sampling` when needed later.

## Direct answers

1. **Is Candidate B fixed Thompson robust across Thompson RNG streams?**  
   **No.** Only 8/30 streams beat always-41; median Δ = −1318.

2. **Does it still look better than always-candidate-41?**  
   **No** in the aggregate. The L5.3 +359 result was a lucky stream (stream 3).

3. **Does it survive basic cost sensitivity?**  
   **No.** The median stream already loses before costs; costs only worsen the picture.

4. **Is the result concentrated in one period?**  
   Partly — Q1 2023 is the only consistently positive quarter on average, but the dominant failure mode is **stream luck**, not a single week.

5. **Are switches away from candidate 41 net positive?**  
   **No** for most streams (net_switch > 0 in only 26.7%).

6. **Should we tune Candidate B, proceed to stronger baselines, or pause?**  
   **Pause model development** (and do not tune). The L5.3 +359 edge disappears across streams.

## Recommendation

Per the L5.4 decision rule: **pause**. Do not chase Thompson seed luck or Candidate B hyperparameters. If work continues, it should be static-structure / always-41 investigation (L5.4 static track from L5.3 decision tree) or a broader audit of whether any dynamic policy can beat always-41 out-of-sample — not Thompson retuning.

PnL remains observational evidence only.
