# BOLR

**Bayesian Online Listwise Ranking** for quantitative trading research.

BOLR scores a large candidate configuration grid each trading day, updates a Gaussian posterior online from realised outcomes, and selects a configuration under posterior decision policies (posterior mean, probability-best, Thompson).

This repository contains:

- a frozen **Python mathematical reference** (Phases A–K)
- a pure **C11 inference/replay backend** (ABI `1.8.0`) with ctypes bindings
- durable **checkpoint** persistence (`BOLRCP01` v1.0)
- native **historical replay** harnesses and evaluation matrices (Phase L5)

License: [MIT](LICENSE).

---

## What problem it solves

Each day the system faces ~1,400 candidate *(entry, stop)* configurations. The goal is not to forecast a single price path, but to maintain a posterior over a smooth score surface and pick a configuration before outcomes are known.

Two observation models were studied:

| Model | Idea |
| --- | --- |
| **Candidate A** | Soft-target / generalized-Bayes observation from daily PnL utilities |
| **Candidate B** | Ordered-partition pairwise ranking (sampled cross-group logistic pairs) |

Transitions may be fixed (additive process noise) or adaptive (BOCPD-backed). Decisions may be posterior-mean, Monte Carlo probability-best, or Thompson (sample-zero).

---

## Repository layout

```text
bolr/           Python reference + evaluation harnesses
csrc/           Pure C11 backend (Make build, ABI, tests)
scripts/        Replay / matrix / robustness CLIs
tests/          Python + ctypes equivalence tests
research_docs/  Design notes and phase write-ups
data/           Candidate grid (+ local historical parquet; not always in git)
outputs/        Local run artefacts (gitignored under outputs/l5_*/)
```

---

## Findings (L5)

Frozen historical protocol:

| Window | Dates | Days |
| --- | --- | ---: |
| Warm-up | 2021-01-29 → 2023-01-11 | 504 |
| Replay | 2023-01-12 → 2024-10-08 | 450 |
| Candidates | YM grid | 1428 |

**Headline results (observational PnL only):**

| Strategy | Total PnL | vs always-41 | Notes |
| --- | ---: | ---: | --- |
| Oracle static best (leakage) | 2865.32 | +1810 | Upper bound only |
| Candidate A fixed probability-best | 1089.63 | +34 | Nearly static (~candidate 41) |
| Always candidate 41 | 1055.24 | 0 | Strong static baseline |
| Candidate B fixed probability-best | 1055.24 | 0 | Collapsed to always-41 |
| Candidate B fixed Thompson (L5.3 stream) | 1414.72 | +359 | Looked promising once |
| Candidate B fixed Thompson (30 streams) | median Δ **−1318** | — | **Not robust** (`share_beating_41 = 26.7%`) |

Interpretation:

1. Candidate A’s “best” policy was essentially static.
2. Candidate B only became dynamic under Thompson, and that edge failed a 30-stream robustness audit.
3. Further model tuning was paused rather than chasing seed luck.

Details: [`research_docs/23_…`](research_docs/23_L5_2_Candidate_A_Policy_Matrix_and_Static_Baselines.md), [`24_…`](research_docs/24_L5_3_Full_Native_Candidate_B_Historical_Replay.md), [`25_…`](research_docs/25_L5_4_Candidate_B_Thompson_Robustness_Audit.md).

---

## Architecture (short)

```text
features / basis  →  composite score model  →  Gaussian posterior
        ↑                                            │
   daily context                              begin_day → decision
                                                     │
                              outcomes → observation → finish_day (Laplace)
                                                     │
                                              durable checkpoint
```

- **Python** owns feature construction, warm-up static surface fit, and evaluation orchestration.
- **C11** owns dense Gaussian inference, observations, ranking/MC decisions, adaptive transitions, replay state machine, and checkpoints.
- **No leakage rule:** `begin_day()` issues the decision before current-day outcomes are revealed; Candidate A/B observations are built only for `finish_day()`.

Current native ABI: **`1.8.0`**. Checkpoint format: **`1.0` (`BOLRCP01`)**.

---

## Quick start

### Requirements

- Python ≥ 3.10 (`numpy`, `pytest`; pandas used by evaluation scripts)
- GCC or Clang with C11
- Linux recommended for claimed RNG/checkpoint reproducibility

### Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# evaluation scripts also expect pandas/pyarrow for parquet workflows
pip install pandas pyarrow
```

### Build & test the C backend

```bash
make -C csrc BUILD_DIR=build/debug-gcc clean test CC=gcc
make -C csrc BUILD_DIR=build/debug-clang clean test CC=clang
make -C csrc BUILD_DIR=build/sanitize-gcc clean sanitize CC=gcc
make -C csrc BUILD_DIR=build/release-gcc clean release CC=gcc
```

### Python tests

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m pytest -q tests/c_backend
```

The ctypes layer auto-builds an isolated debug shared library under `csrc/build/python-debug-gcc/` when needed.

### Foundation smoke

```bash
PYTHONPATH=. python scripts/validate_foundation.py
```

---

## Historical replay CLIs

These are operator runs (full 450-day matrices are slow; not pytest gates).

```bash
# Candidate A native replay
PYTHONPATH=. python scripts/run_native_candidate_a_replay.py --help

# Candidate A policy / baseline matrix
PYTHONPATH=. python scripts/run_l5_candidate_a_policy_matrix.py --help

# Candidate B native replay
PYTHONPATH=. python scripts/run_native_candidate_b_replay.py --help

# Candidate B matrix (+ optional L5.2 comparison import)
PYTHONPATH=. python scripts/run_l5_candidate_b_matrix.py --help

# Candidate B Thompson robustness audit
PYTHONPATH=. python scripts/run_l5_candidate_b_thompson_robustness.py --help
```

Typical data inputs (local):

- `data/YM_grid.csv` — canonical candidate grid
- `data/YM_full.parquet` — historical day×candidate PnL / features

---

## Research documentation

Design and phase write-ups live in [`research_docs/`](research_docs/).

| Doc | Topic |
| --- | --- |
| [`BOLR_Main.md`](research_docs/BOLR_Main.md) | High-level research narrative |
| [`12_Python_Reference_Model_Freeze.md`](research_docs/12_Python_Reference_Model_Freeze.md) | Frozen Python reference |
| [`13_…`–`21_…`](research_docs/) | C11 ABI, inference, decisions, RNG, replay, checkpoints |
| [`22_L5_1_…`](research_docs/22_L5_1_Full_Native_Candidate_A_Historical_Replay.md) | Native Candidate A historical replay |
| [`23_L5_2_…`](research_docs/23_L5_2_Candidate_A_Policy_Matrix_and_Static_Baselines.md) | Candidate A policy matrix |
| [`24_L5_3_…`](research_docs/24_L5_3_Full_Native_Candidate_B_Historical_Replay.md) | Candidate B historical replay |
| [`25_L5_4_…`](research_docs/25_L5_4_Candidate_B_Thompson_Robustness_Audit.md) | Thompson robustness audit / pause |

---

## Implementation milestones (compact)

| Stage | Status |
| --- | --- |
| A–K Python reference (basis, obs models, Laplace, adaptation, decisions) | Frozen |
| L1–L2 C foundation + Candidate A Laplace path | Done (ABI through early 1.x) |
| L3A–L3B Candidate B + adaptive transitions | Done |
| L4A–L4B2.3 Decisions, RNG, MC ranking, replay, checkpoints | Done — ABI **1.8.0** |
| L5.1–L5.3 Native historical A/B evaluation | Done |
| L5.4 Thompson robustness | Done — **pause** |

Older per-phase changelog detail is retained in the research docs rather than this README.

---

## Technical notes worth keeping

- **RNG:** PCG32 with explicit `(seed, stream)`; normals via immutable 128-layer Ziggurat. Integer streams are exact; normal draws depend on `libm`, so cross-platform bitwise identity is only claimed for validated Linux GCC/Clang builds.
- **Thompson:** frozen to Monte Carlo sample `0` (streaming rank accumulators may discard other samples).
- **Rank ties:** stable descending score sort with original index tie-break; ranks `1..N`.
- **Checkpoints:** little-endian sectioned `BOLRCP01` with CRC32 and atomic POSIX write/replace.

---

## Disclaimer

PnL figures in this repository are **observational research evidence**, not trading advice. Historical results can reflect leakage (explicitly labelled oracles), seed luck, or protocol choices. Always-41 and related static baselines are part of the evaluation story for a reason.
