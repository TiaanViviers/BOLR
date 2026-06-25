# 06 Candidate B Ordered-Region Observations

## Status

Implemented in the reference Python stack:

- generic `TargetBuilder` and `ObservationModel` interfaces
- Candidate A integrated through the generic interfaces without changing its numerical behaviour
- deterministic ordered-partition target construction
- tolerance calculation with absolute / relative / hybrid components
- exact cross-group logistic observation with dense gradient and curvature
- deterministic sampled cross-group logistic mode with fixed sampled pairs per update
- strict Plackett-Luce reference on tiny lists
- brute-force partitioned-preference Plackett-Luce reference
- exact reduced-size partitioned-preference Plackett-Luce dynamic-programming implementation with exact gradient and dense curvature on reduced problems
- generic static-surface fitting
- generic historical replay engine
- exact bounded real-data resume-equivalence test
- full historical tolerance audit

## Group Semantics

The current ordered partition builder produces:

- `G_H = { i : u_i >= u_max - eps_t }`
- `G_M = { i : u_i > tau_positive } \\ G_H`
- `G_L = remaining candidates`

with deterministic removal of empty redundant groups.

If only one informative group remains, the observation is treated as uninformative and the configured all-irrelevant policy is applied.

## Tolerance Semantics

Tolerance supports:

- absolute component
- relative robust component
- execution component

The current implementation records all three components in the observation metadata together with the selected robust scale.

## Cross-Group Logistic

The default cross-group logistic observation uses mean loss inside each ordered group pair and equal weights across active group pairs.

The sampled mode uses a fixed deterministic sample for the whole update so Newton iterations optimise a fixed approximation rather than changing the objective between iterations.

## Proper Partitioned PL

The proper partitioned-PL path is exact for reduced upper-partition sizes through a subset dynamic program. It is validated against brute-force enumeration on tiny lists.

This implementation is mathematically correct for reduced problems but is not yet a practical full-history Python replay engine for large real-world group sizes.

## Historical Tolerance Audit

Generated output:

- `outputs/candidate_b_tolerance_audit/config.json`
- `outputs/candidate_b_tolerance_audit/daily_group_diagnostics.parquet`
- `outputs/candidate_b_tolerance_audit/summary.json`
- `outputs/candidate_b_tolerance_audit/summary.md`

Using the frozen provisional configuration:

- absolute tolerance: `0.0`
- relative tolerance: `0.1`

key results on the 954 historical days:

- median `|G_H| = 2`
- mean `|G_H| = 10.46`
- median `|G_M| = 219`
- median `|G_L| = 1203`
- median possible cross-group pair count `= 270214`
- 95th percentile possible cross-group pair count `= 475253.65`
- no-update frequency `= 0.00839`

These results make exact all-pair historical Candidate B updates expensive in pure Python even before considering the proper partition likelihood.

## Resume Gate

The real-data bounded resume-equivalence test passes exactly on a 10-day slice after the 504-day warm-up:

- final posterior mean infinity-norm difference `< 1e-12`
- final posterior covariance infinity-norm difference `< 1e-12`
- prediction outputs agree exactly on the tested fields

## Commands Executed

```bash
source ~/environments/pyenv/bin/activate && python -m pytest -q
source ~/environments/pyenv/bin/activate && python scripts/audit_candidate_b_groups.py --historical-parquet data/YM_full.parquet --grid-csv data/YM_grid.csv --output-dir outputs/candidate_b_tolerance_audit --absolute-tolerance 0.0 --relative-tolerance 0.1
```

## Unresolved Issues

1. The current pure-Python Candidate B historical replay path is mathematically integrated but operationally heavy because Candidate B static warm-up fitting and daily observation evaluation are much more expensive than Candidate A.
2. The reduced exact partitioned-PL implementation is suitable for correctness tests and reduced problems, but not yet for large historical group sizes.
3. The historical cross-group replay budget needs an explicit practical runtime envelope before it should be promoted to a default release command.
