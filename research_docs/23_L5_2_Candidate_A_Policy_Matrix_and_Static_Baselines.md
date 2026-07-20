# Phase L5.2 — Candidate A Policy Matrix and Static Baselines

## Motivation

L5.1 showed Candidate A with fixed transition and `probability_best` selected candidate **41** on 449/450 days. The single switch (`2024-08-06` → candidate 1427) lost value versus staying on 41.

L5.2 asks:

> Is Candidate A doing anything useful beyond selecting a static favourite?

## Candidate 41 finding (L5.1)

- Canonical index `41` = entry `0.0004`, stop `0.004979`, `config_id=41`
- Fixed/`probability_best`: almost static
- Fixed/`posterior_mean`: exact match to frozen Python Candidate A historical selections (28 unique configs, 64% on 41)

## Strategy list

### Deployable / no-leakage baselines

- `baseline_always_41`
- `baseline_best_warmup_static` (warm-up mean PnL argmax)
- `baseline_best_warmup_sharpe` (optional)
- `baseline_trailing_mean_5` / `_20`
- `baseline_trailing_positive_5` / `_20` (optional)

### Oracle diagnostic (leakage)

- `oracle_static_best_replay` — best single candidate over replay period itself

### Candidate A native matrix

| Transition | Decision |
|---|---|
| fixed | posterior_mean |
| fixed | probability_best |
| fixed | Thompson (sample-zero) |
| adaptive_additive | posterior_mean |
| adaptive_additive | probability_best |
| adaptive_additive | Thompson |

## No-leakage rules

- Warm-up static uses warm-up days only
- Trailing baselines use history before the decision day only
- Current-day PnL never enters the decision
- Oracle is labelled and separated in summaries

## Replay protocol

Default L5.1 dates:

- warm-up: `2021-01-29` → `2023-01-11`
- replay: `2023-01-12` → `2024-10-08`
- MC: 512 samples, chunk 64, top-k 10
- RNG seed `20260720`, base stream `1`, strategy stream offsets `+0..+5`

## Commands

```bash
PYTHONPATH=. ~/environments/pyenv/bin/python \
  scripts/run_l5_candidate_a_policy_matrix.py \
  --data data/YM_full.parquet \
  --output-dir outputs/l5_candidate_a_policy_matrix \
  --warmup-start 2021-01-29 \
  --warmup-end 2023-01-11 \
  --replay-start 2023-01-12 \
  --replay-end 2024-10-08 \
  --mc-samples 512 \
  --mc-chunk-size 64 \
  --top-k 10 \
  --rng-seed 20260720 \
  --rng-stream 1 \
  --checkpoint-every-n-days 25 \
  --overwrite-outputs
```

Bounded smoke:

```bash
... --maximum-days 20 --strategies required
```

## Outputs

```text
outputs/l5_candidate_a_policy_matrix/
  <strategy>/
  comparison/
    strategy_summary.csv
    strategy_summary.json
    strategy_daily_panel.csv
    selection_diagnostics.csv
    probability_best_bins.csv
    README.md
```

Key diagnostics: `candidate_41_delta_*`, selection entropy / effective candidates, turnover, bad-switch costs.

## Interpretation protocol

Answer directly:

> Did any Candidate A strategy beat always-candidate-41 without simply being another static selector?

Require both:

1. positive `candidate_41_delta_total_pnl`
2. non-degenerate selection (`effective_selected_candidates` meaningfully > 1, or good switches that are not a different static favourite)

If not, Candidate B historical replay (L5.3) is the natural next research step.

## Full matrix results (executed)

450 replay days (`2023-01-12` → `2024-10-08`). Wall time ~49 minutes.

| strategy | total_pnl | vs_41 | unique | N_eff |
|---|---:|---:|---:|---:|
| oracle_static_best_replay | 2865.32 | +1810 | 1 | 1.0 |
| candidate_a_fixed_probability_best | 1089.63 | **+34.39** | 2 | 1.02 |
| candidate_a_adaptive_probability_best | 1055.24 | 0.00 | 1 | 1.0 |
| baseline_always_41 | 1055.24 | 0.00 | 1 | 1.0 |
| baseline_best_warmup_sharpe | 516.85 | -538 | 1 | 1.0 |
| candidate_a_fixed_thompson | -1140.34 | -2196 | 125 | 48.9 |
| candidate_a_adaptive_thompson | -1233.81 | -2289 | 124 | 50.7 |
| baseline_best_warmup_static | -1694.43 | -2750 | 1 | 1.0 |
| candidate_a_adaptive_posterior_mean | -2275.30 | -3331 | 29 | 6.2 |
| candidate_a_fixed_posterior_mean | -2941.47 | -3997 | 28 | 5.6 |
| trailing baselines | worse | … | high | high |

### Direct answer

> Did any Candidate A strategy beat always-candidate-41 without simply being another static selector?

**No clear dynamic win.** Fixed/`probability_best` edged always-41 by ~+34 via one good switch, but effective selected candidates ≈ 1.02 — still essentially the static favourite. Adaptive/`probability_best` collapsed exactly to always-41. Posterior-mean and Thompson explored more and lost large amounts versus always-41. Trailing baselines also underperformed always-41.

Natural next step: **L5.3 Candidate B native historical replay**.

## Limitations

- Adaptive surprise fields may be blank under current finish diagnostics
- Region-medoid excluded until grid-graph wiring is production-clean
- Full matrix runtime is roughly order(number of Candidate A strategies) × L5.1 runtime after a shared warm-up
- PnL is observational evidence, not a correctness gate

## Next phase

- If Candidate A fails to beat static/trailing baselines in a non-degenerate way → **L5.3 Candidate B native historical replay**
- If it does → **L5.2B sensitivity analysis** (only then consider target strength / process noise / adaptation knobs)
