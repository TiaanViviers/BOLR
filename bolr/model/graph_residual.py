from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from bolr.model.graph import GridGraph, graph_energy
from bolr.model.penalties import QuadraticPenalty
from bolr.model.structured import ProcessNoiseModel, prior_from_penalty
from bolr.posterior.diagnostics import covariance_condition_number


def _fingerprint_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(array, dtype=float)).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SmoothSubspaceProjector:
    q0: np.ndarray
    smooth_basis: np.ndarray
    subspace_basis: np.ndarray
    metadata: Mapping[str, object]

    def project(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        return vector - self.subspace_basis @ (self.subspace_basis.T @ vector)

    def explicit_projector(self) -> np.ndarray:
        n = self.subspace_basis.shape[0]
        return np.eye(n, dtype=float) - self.subspace_basis @ self.subspace_basis.T


@dataclass(frozen=True)
class ProjectedLaplacianOperator:
    graph: GridGraph
    projector: SmoothSubspaceProjector

    def apply(self, vector: np.ndarray) -> np.ndarray:
        projected = self.projector.project(vector)
        applied = self.graph.laplacian @ projected
        return self.projector.project(np.asarray(applied, dtype=float))

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            shape=(self.graph.node_count, self.graph.node_count),
            matvec=self.apply,
            dtype=float,
        )


@dataclass(frozen=True)
class GraphResidualBasis:
    basis: np.ndarray
    eigenvalues: np.ndarray
    graph: GridGraph
    projector: SmoothSubspaceProjector
    metadata: Mapping[str, object]

    def forward(self, coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        return self.basis @ coefficients

    def transpose(self, scores: np.ndarray) -> np.ndarray:
        return self.basis.T @ np.asarray(scores, dtype=float).reshape(-1)

    def design_matrix(self) -> np.ndarray:
        return self.basis

    def subspace_projector(self) -> np.ndarray:
        return self.basis @ self.basis.T

    def penalty(self) -> QuadraticPenalty:
        return QuadraticPenalty(
            matrix=np.diag(self.eigenvalues),
            dimension=self.eigenvalues.size,
            name="graph_residual_mode_penalty",
            metadata={"basis_hash": self.metadata["residual_basis_hash"]},
        )


def build_smooth_subspace_projector(candidate_basis: np.ndarray, *, tolerance: float = 1e-10) -> SmoothSubspaceProjector:
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    node_count = candidate_basis.shape[0]
    q0 = np.ones(node_count, dtype=float) / np.sqrt(node_count)
    q_candidate, _ = np.linalg.qr(candidate_basis, mode="reduced")
    q_candidate = q_candidate - q0[:, None] @ (q0[None, :] @ q_candidate)
    q_candidate, _ = np.linalg.qr(q_candidate, mode="reduced")
    combined = np.column_stack([q0, q_candidate])
    subspace_basis, _ = np.linalg.qr(combined, mode="reduced")
    gram = subspace_basis.T @ subspace_basis
    if np.max(np.abs(gram - np.eye(gram.shape[0], dtype=float))) > 1e-8:
        raise ValueError("Smooth subspace basis is not orthonormal within tolerance.")
    rank = int(np.linalg.matrix_rank(subspace_basis, tol=tolerance))
    return SmoothSubspaceProjector(
        q0=subspace_basis[:, 0],
        smooth_basis=subspace_basis[:, 1:],
        subspace_basis=subspace_basis,
        metadata={
            "projection_rank": rank,
            "smooth_basis_fingerprint": _fingerprint_array(candidate_basis),
            "orthogonality_error": float(np.max(np.abs(gram - np.eye(gram.shape[0], dtype=float)))),
        },
    )


def build_graph_residual_basis(
    graph: GridGraph,
    candidate_basis: np.ndarray,
    residual_dimension: int,
    *,
    near_zero_tolerance: float = 1e-8,
    leakage_tolerance: float = 1e-8,
    cluster_tolerance: float = 1e-6,
) -> GraphResidualBasis:
    if residual_dimension <= 0:
        raise ValueError("residual_dimension must be positive.")
    projector = build_smooth_subspace_projector(candidate_basis)
    operator = ProjectedLaplacianOperator(graph=graph, projector=projector)
    if graph.node_count <= 128:
        explicit = np.column_stack([operator.apply(np.eye(graph.node_count, dtype=float)[:, idx]) for idx in range(graph.node_count)])
        explicit = 0.5 * (explicit + explicit.T)
        eigenvalues, eigenvectors = np.linalg.eigh(explicit)
        search_dimension = int(graph.node_count)
    else:
        search_dimension = min(graph.node_count - 1, residual_dimension + projector.subspace_basis.shape[1] + 16)
        eigenvalues, eigenvectors = eigsh(operator.as_linear_operator(), k=search_dimension, which="SM")
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

    retained_values: list[float] = []
    retained_vectors: list[np.ndarray] = []
    rejected_near_zero = 0
    for idx in range(eigenvalues.size):
        vector = projector.project(eigenvectors[:, idx])
        norm = float(np.linalg.norm(vector))
        if norm <= near_zero_tolerance:
            rejected_near_zero += 1
            continue
        vector = vector / norm
        leakage = float(np.linalg.norm(projector.subspace_basis.T @ vector))
        if eigenvalues[idx] <= near_zero_tolerance or leakage > leakage_tolerance:
            rejected_near_zero += 1
            continue
        retained_values.append(float(eigenvalues[idx]))
        retained_vectors.append(vector)
        if len(retained_values) >= residual_dimension:
            break
    if len(retained_values) < residual_dimension:
        raise ValueError("Unable to construct the requested residual basis dimension.")

    raw_basis = np.column_stack(retained_vectors)
    canonical_basis = _canonicalize_residual_basis(raw_basis, np.asarray(retained_values), cluster_tolerance=cluster_tolerance)
    canonical_basis, _ = np.linalg.qr(canonical_basis, mode="reduced")
    canonical_basis = _canonicalize_signs(canonical_basis)
    leakage = np.linalg.norm(projector.subspace_basis.T @ canonical_basis, axis=0)
    laplacian_diag = np.diag(canonical_basis.T @ (graph.laplacian @ canonical_basis))
    metadata = {
        "graph_type": graph.metadata["graph_type"],
        "node_ordering": "entry_major_stop_minor",
        "edge_definition": "manhattan_4_neighbour",
        "edge_weights": {"entry": graph.metadata["entry_weight"], "stop": graph.metadata["stop_weight"]},
        "laplacian_type": graph.metadata["laplacian_type"],
        "smooth_basis_fingerprint": projector.metadata["smooth_basis_fingerprint"],
        "projection_rank": projector.metadata["projection_rank"],
        "requested_residual_dimension": residual_dimension,
        "retained_residual_dimension": int(canonical_basis.shape[1]),
        "eigenvalues": np.asarray(retained_values, dtype=float),
        "eigenvalue_cluster_tolerance": cluster_tolerance,
        "orthogonality_tolerance": leakage_tolerance,
        "sign_convention": "max_abs_loading_positive",
        "basis_schema_version": "phase_i_v1",
        "graph_definition_hash": graph.metadata["graph_definition_hash"],
        "residual_basis_hash": _fingerprint_array(canonical_basis),
        "computed_eigenpairs": int(search_dimension),
        "rejected_near_zero_count": int(rejected_near_zero),
        "smallest_retained_eigenvalue": float(np.min(retained_values)),
        "largest_retained_eigenvalue": float(np.max(retained_values)),
        "orthogonality_error": float(np.max(np.abs(canonical_basis.T @ canonical_basis - np.eye(canonical_basis.shape[1])))),
        "smooth_space_leakage": float(np.max(leakage)),
        "residual_norms": np.linalg.norm(canonical_basis, axis=0),
        "graph_spectral_diagnostics": graph_spectral_diagnostics(graph, canonical_basis, np.asarray(retained_values, dtype=float)),
    }
    return GraphResidualBasis(
        basis=canonical_basis,
        eigenvalues=np.asarray(retained_values, dtype=float),
        graph=graph,
        projector=projector,
        metadata=metadata,
    )


def graph_residual_prior(
    residual_basis: GraphResidualBasis,
    *,
    graph_energy_weight: float = 1.0,
    ridge_weight: float = 1e-6,
) -> object:
    penalty = residual_basis.penalty()
    return prior_from_penalty(penalty, smooth_weight=graph_energy_weight, ridge=ridge_weight)


def graph_penalty_shaped_process_noise(
    residual_basis: GraphResidualBasis,
    *,
    scale: float,
    properization: float,
) -> ProcessNoiseModel:
    if scale < 0.0 or properization <= 0.0:
        raise ValueError("scale must be non-negative and properization positive.")
    diagonal = scale / (residual_basis.eigenvalues + properization)
    covariance = np.diag(diagonal)
    return ProcessNoiseModel(
        covariance=covariance,
        family="graph_penalty_shaped_random_walk",
        metadata={
            "scale": scale,
            "properization": properization,
            "eigenvalues": residual_basis.eigenvalues.copy(),
            "min_drift_variance": float(np.min(diagonal)),
            "max_drift_variance": float(np.max(diagonal)),
            "smoothest_mode_drift_variance": float(np.max(diagonal)),
            "roughest_mode_drift_variance": float(np.min(diagonal)),
            "condition_number": float(np.max(diagonal) / np.min(diagonal)),
            "residual_basis_hash": residual_basis.metadata["residual_basis_hash"],
        },
    )


def smooth_plus_local_diagnostics(
    smooth_scores: np.ndarray,
    local_scores: np.ndarray,
    graph: GridGraph,
    *,
    residual_coefficients: np.ndarray | None = None,
    residual_prior_precision: np.ndarray | None = None,
    residual_block_covariance: np.ndarray | None = None,
    residual_basis: np.ndarray | None = None,
) -> dict[str, float]:
    smooth_scores = np.asarray(smooth_scores, dtype=float).reshape(-1)
    local_scores = np.asarray(local_scores, dtype=float).reshape(-1)
    total_scores = smooth_scores + local_scores
    smooth_energy = graph_energy(smooth_scores, graph)
    local_energy = graph_energy(local_scores, graph)
    total_energy = graph_energy(total_scores, graph)
    diagnostics = {
        "smooth_score_norm": float(np.linalg.norm(smooth_scores)),
        "local_score_norm": float(np.linalg.norm(local_scores)),
        "total_score_norm": float(np.linalg.norm(total_scores)),
        "local_to_total_norm_ratio": float(np.linalg.norm(local_scores) / max(np.linalg.norm(total_scores), 1e-12)),
        "smooth_graph_energy": smooth_energy["total_graph_energy"],
        "local_graph_energy": local_energy["total_graph_energy"],
        "total_graph_energy": total_energy["total_graph_energy"],
        "maximum_local_score": float(np.max(local_scores)),
        "minimum_local_score": float(np.min(local_scores)),
        "maximum_local_edge_jump": local_energy["maximum_edge_difference"],
        "surface_residual_inner_product": float(smooth_scores @ local_scores),
        "decomposition_inf_error": float(np.max(np.abs(total_scores - smooth_scores - local_scores))),
    }
    if residual_coefficients is not None:
        coeffs = np.asarray(residual_coefficients, dtype=float).reshape(-1)
        diagnostics["residual_coefficient_norm"] = float(np.linalg.norm(coeffs))
        if residual_prior_precision is not None:
            precision = np.asarray(residual_prior_precision, dtype=float)
            diagnostics["prior_standardized_residual_norm"] = float(coeffs @ precision @ coeffs)
        diagnostics["score_energy_ratio"] = float((local_scores @ local_scores) / max(smooth_scores @ smooth_scores, 1e-12))
    if residual_block_covariance is not None and residual_basis is not None:
        cov = np.asarray(residual_block_covariance, dtype=float)
        basis = np.asarray(residual_basis, dtype=float)
        diagnostics["effective_residual_variance"] = float(np.trace(basis @ cov @ basis.T))
        diagnostics["residual_covariance_condition_number"] = covariance_condition_number(cov)
    return diagnostics


def graph_spectral_diagnostics(graph: GridGraph, basis: np.ndarray, eigenvalues: np.ndarray) -> dict[str, np.ndarray | float]:
    basis = np.asarray(basis, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    edge_diffs = basis[graph.edge_index[:, 0], :] - basis[graph.edge_index[:, 1], :]
    entry_mask = graph.edge_axis == "entry"
    stop_mask = graph.edge_axis == "stop"
    inverse_mass = np.cumsum(1.0 / eigenvalues) / np.sum(1.0 / eigenvalues)
    return {
        "retained_eigenvalues": eigenvalues.copy(),
        "cumulative_inverse_eigenvalue_mass": inverse_mass,
        "basis_graph_energy": np.sum(graph.edge_weights[:, None] * edge_diffs * edge_diffs, axis=0),
        "entry_axis_energy": np.sum(graph.edge_weights[entry_mask, None] * edge_diffs[entry_mask] ** 2, axis=0),
        "stop_axis_energy": np.sum(graph.edge_weights[stop_mask, None] * edge_diffs[stop_mask] ** 2, axis=0),
        "maximum_basis_loading": np.max(np.abs(basis), axis=0),
        "inverse_participation_ratio": np.sum(basis**4, axis=0),
    }


def validate_residual_basis_compatibility(residual_basis: GraphResidualBasis, graph: GridGraph, candidate_basis: np.ndarray) -> None:
    expected_smooth = _fingerprint_array(np.asarray(candidate_basis, dtype=float))
    if residual_basis.metadata["graph_definition_hash"] != graph.metadata["graph_definition_hash"]:
        raise ValueError("Residual basis graph definition is incompatible with the active graph.")
    if residual_basis.metadata["smooth_basis_fingerprint"] != expected_smooth:
        raise ValueError("Residual basis smooth-space fingerprint is incompatible with the active candidate basis.")


def validate_checkpoint_graph_metadata(checkpoint_graph_metadata: Mapping[str, object] | None, residual_basis: GraphResidualBasis, graph: GridGraph) -> None:
    if checkpoint_graph_metadata is None:
        raise ValueError("Checkpoint does not contain graph metadata.")
    if checkpoint_graph_metadata.get("graph_definition_hash") != graph.metadata["graph_definition_hash"]:
        raise ValueError("Checkpoint graph definition is incompatible with the active graph.")
    if checkpoint_graph_metadata.get("smooth_basis_fingerprint") != residual_basis.metadata["smooth_basis_fingerprint"]:
        raise ValueError("Checkpoint smooth-basis fingerprint is incompatible with the active residual basis.")
    if checkpoint_graph_metadata.get("residual_basis_hash") != residual_basis.metadata["residual_basis_hash"]:
        raise ValueError("Checkpoint residual-basis hash is incompatible with the active residual basis.")


def _canonicalize_residual_basis(vectors: np.ndarray, eigenvalues: np.ndarray, *, cluster_tolerance: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    groups: list[tuple[int, int]] = []
    start = 0
    while start < eigenvalues.size:
        stop = start + 1
        while stop < eigenvalues.size:
            scale = max(abs(eigenvalues[start]), abs(eigenvalues[stop]), 1.0)
            if abs(eigenvalues[stop] - eigenvalues[stop - 1]) > cluster_tolerance * scale:
                break
            stop += 1
        groups.append((start, stop))
        start = stop
    pieces = []
    for start, stop in groups:
        block = vectors[:, start:stop]
        if block.shape[1] == 1:
            pieces.append(_canonicalize_signs(block))
            continue
        projector = block @ block.T
        q, _, pivots = linalg.qr(projector, pivoting=True, mode="economic")
        canonical = q[:, : block.shape[1]]
        order = np.argsort(pivots[: block.shape[1]])
        pieces.append(_canonicalize_signs(canonical[:, order]))
    return np.column_stack(pieces)


def _canonicalize_signs(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float).copy()
    for idx in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, idx])))
        if vectors[pivot, idx] < 0.0:
            vectors[:, idx] *= -1.0
    return vectors
