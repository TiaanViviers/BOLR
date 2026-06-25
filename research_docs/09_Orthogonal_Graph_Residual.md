# Phase I: Orthogonal Graph Residual Architecture

## Purpose

Phase I adds a low-dimensional local correction block on top of the existing smooth tensor surface. The intent is not to replace the spline surface with a free node model. The intent is to capture boundaries, plateaus, bumps, and disconnected favourable regions that the globally smooth basis cannot represent cleanly.

The score decomposition is:

$$
s_t
=
s_t^{\text{smooth}}
+
\Psi_r \rho_t,
$$

where the residual basis is explicitly constrained to lie outside the constant direction and outside the smooth tensor-score subspace.

## Canonical graph

The residual basis is built on the canonical `34 x 42` configuration grid with deterministic entry-major, stop-minor ordering. The graph uses four-neighbour Manhattan adjacency:

- previous and next entry positions at fixed stop;
- previous and next stop positions at fixed entry.

For the full grid this yields:

- `1428` nodes;
- `2780` undirected edges.

The default Laplacian is combinatorial:

$$
L = D - A.
$$

An optional symmetric normalised form is exposed only as an architectural extension point.

## Orthogonal residual subspace

Let `Phi_c` be the centred reduced tensor basis and let:

$$
q_0 = \frac{1}{\sqrt N}\mathbf 1.
$$

The smooth-space basis is:

$$
S = [q_0 \;\; Q_c],
$$

with `Q_c = orth(Phi_c)`.

Residual directions are projected with:

$$
P_\perp v = v - S(S^\top v).
$$

This enforces:

- zero residual mean across candidates;
- zero overlap with the static spline surface;
- zero overlap with dynamic surface and context-surface blocks, whose scores also live in `col(Phi_c)`.

## Constrained spectral basis

The graph residual basis is built from the smallest positive eigenmodes of:

$$
L_\perp = P_\perp L P_\perp.
$$

The retained basis `Psi_r` therefore:

- is orthonormal;
- is centred;
- is orthogonal to the smooth spline span;
- diagonalises graph roughness inside the constrained residual space.

The current implementation uses:

- projector actions instead of a dense `N x N` projector in normal operator application;
- sparse eigensolver support through a SciPy `LinearOperator`;
- deterministic eigenvector sign handling;
- deterministic canonicalisation for clustered eigenvalues through projector-based QR.

Golden validation prioritises subspace projectors and downstream numerical outputs rather than raw eigenvector equality alone.

## Residual priors and dynamics

In residual coordinates the graph penalty is diagonal:

$$
\Psi_r^\top L \Psi_r = \operatorname{Diag}(\mu_1,\dots,\mu_k).
$$

This yields a Gaussian prior precision:

$$
\Lambda_r
=
\lambda_L \operatorname{Diag}(\mu)
+
\lambda_0 I.
$$

Supported residual dynamics are:

- fixed or frozen residual;
- zero-noise observable residual;
- isotropic random walk;
- graph-penalty-shaped random walk:

$$
Q_r = q_r [\operatorname{Diag}(\mu) + \tau I]^{-1}.
$$

The graph-shaped transition allows smoother residual modes to move more than rougher modes.

## Diagnostics

Phase I adds:

- total graph energy;
- entry-axis and stop-axis edge energy;
- maximum edge jump;
- smooth/local score norms and their ratio;
- residual coefficient norm;
- prior-standardised residual norm;
- residual effective variance in score space;
- spectral diagnostics such as retained eigenvalues and inverse participation ratio.

These diagnostics are descriptive only. No adaptive residual policy is introduced here.

## Synthetic findings

The synthetic regression coverage now checks:

- smooth truth does not produce a dominant residual;
- sharp boundary truth benefits from the graph residual;
- reduced Candidate B compatibility still holds with a residual block present.

The residual block is therefore behaving as intended: local correction, not a second unconstrained surface.

## Historical smoke scope

Phase I adds a bounded historical smoke path only:

- E0 static baseline;
- dynamic smooth surface;
- small residual basis;
- Candidate A update;
- dense covariance;
- additive blockwise `Q`.

This validates composition, convergence, covariance stability, checkpoint metadata, and basis compatibility. It is not a historical performance study.

## Checkpoint and compatibility rules

Graph-aware checkpoints now persist:

- graph schema version;
- graph definition hash;
- smooth-basis fingerprint;
- residual-basis hash;
- residual dimension;
- Laplacian type;
- edge weights;
- retained residual eigenvalues.

Checkpoints are validated against the active graph and residual basis before reuse.

## Total variation status

Total variation remains deferred. It is a distinct future residual prior:

$$
\lambda_{\text{TV}} \sum_{(i,j)\in E} |r_i-r_j|,
$$

and is not representable exactly inside the current Gaussian reference filter.

## C-port contracts

The future C implementation must support these frozen kernels:

- `grid_graph_build`
- `graph_laplacian_apply`
- `subspace_project`
- `projected_laplacian_apply`
- `residual_basis_forward`
- `residual_basis_transpose`
- `graph_energy`
- `graph_prior_assemble`
- `graph_process_noise_assemble`

The preferred production path is still to compute the spectral residual basis offline in Python and load it into C with hashes and metadata validation.

## Limits

Phase I still does not introduce:

- total variation inference;
- adaptive process-noise inflation;
- residual feature selection;
- covariance approximation;
- C execution kernels.
