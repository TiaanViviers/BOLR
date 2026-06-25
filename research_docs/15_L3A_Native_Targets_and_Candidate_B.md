# 15 L3A Native Targets and Candidate B

## Scope

Phase L3A extends the native C11 backend without changing the dense Gaussian inference engine introduced in Phase L2.

The milestone is:

$$
\text{Candidate A and cross-group Candidate B use the same generic C Laplace engine.}
$$

## Frozen boundary

The sampled Candidate B path is native only at the observation-evaluation layer.

It does **not** introduce a general C random number generator.

Instead:

1. Python materializes a deterministic sampled pair set using the frozen reference semantics.
2. The C sampled-observation handle owns a copy of:
   - winner indices;
   - loser indices;
   - per-pair coefficients.
3. The same owned pair set is reused for every Newton objective, gradient, and curvature-HVP evaluation.

This preserves exact Newton consistency while keeping RNG work out of L3A.

## Candidate A target semantics

The native Candidate A target builder mirrors the frozen Python implementation in `bolr/targets/soft_target.py`.

It preserves:

- robust tolerance collapse;
- robust scaling by `max(MAD, 0.7413 * IQR, min_scale)`;
- clipping;
- softmax target construction;
- degenerate no-update behaviour through `update_weight`.

## Ordered partition semantics

The native ordered-partition builder mirrors `bolr/targets/ordered_partition.py`.

It preserves:

- tolerance calculation from absolute, relative, and execution components;
- robust scale selection mode;
- the `high`, `middle`, `low` grouping rule;
- all-irrelevant policy handling;
- canonical within-group candidate ordering;
- immutable owned partition storage.

## Candidate B cross-group logistic semantics

The exact native Candidate B observation uses all cross-group pairs implied by the ordered partition.

The sampled native Candidate B observation uses the deterministic pre-materialized pair set supplied by Python.

For each active group pair `(a, b)` with `a < b`, the loss is the frozen Python cross-group logistic loss:

$$
\ell_{ij}(s)=\operatorname{softplus}(s_j - s_i), \qquad i \in G_a,\; j \in G_b.
$$

The native log-factor is:

$$
\log f(s \mid O) = -w \sum_{(i,j)\in\mathcal P} c_{ij}\,\operatorname{softplus}(s_j-s_i),
$$

where:

- `w` is the observation update weight;
- `\mathcal P` is either the full exact pair set or the deterministic sampled pair set;
- `c_{ij}` is the frozen per-pair coefficient implied by the Python normalization policy.

The score gradient is therefore:

$$
\nabla_{s_i}\log f = w \sum_{(i,j)\in\mathcal P} c_{ij}\,\sigma(s_j-s_i),
\qquad
\nabla_{s_j}\log f = -w \sum_{(i,j)\in\mathcal P} c_{ij}\,\sigma(s_j-s_i).
$$

The curvature operator remains positive semidefinite and is evaluated in native code through the score-space HVP callback expected by the generic Laplace engine.

## Validation summary

L3A validation now includes:

- native GCC and Clang test passes;
- sanitizer pass;
- Candidate A target equivalence;
- ordered-partition equivalence;
- exact Candidate B value/gradient/HVP equivalence;
- deterministic sampled Candidate B value/gradient/HVP equivalence;
- exact Candidate B Laplace equivalence;
- sampled Candidate B Laplace equivalence;
- bounded sequential Candidate B equivalence.
