from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np
from scipy import sparse

from bolr.data.candidate_grid import CandidateGrid


@dataclass(frozen=True)
class GridGraph:
    node_count: int
    edge_count: int
    edge_index: np.ndarray
    edge_weights: np.ndarray
    edge_axis: np.ndarray
    adjacency: sparse.csr_matrix
    degree: np.ndarray
    laplacian: sparse.csr_matrix
    entry_laplacian: sparse.csr_matrix
    stop_laplacian: sparse.csr_matrix
    entry_indices: np.ndarray
    stop_indices: np.ndarray
    canonical_config_ids: np.ndarray
    metadata: Mapping[str, object]


def build_canonical_grid_graph(
    candidate_grid: CandidateGrid,
    *,
    entry_weight: float = 1.0,
    stop_weight: float = 1.0,
    laplacian_type: str = "combinatorial",
) -> GridGraph:
    if entry_weight <= 0.0 or stop_weight <= 0.0:
        raise ValueError("Graph edge weights must be positive.")
    if laplacian_type not in {"combinatorial", "symmetric_normalized"}:
        raise ValueError("Unsupported laplacian_type.")
    entry_count, stop_count = candidate_grid.grid_shape
    node_count = candidate_grid.n_candidates
    expected_nodes = entry_count * stop_count
    if node_count != expected_nodes:
        raise ValueError("Candidate grid does not match its declared rectangular shape.")

    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    edge_axis: list[str] = []
    for entry_idx in range(entry_count):
        for stop_idx in range(stop_count):
            node = entry_idx * stop_count + stop_idx
            if entry_idx + 1 < entry_count:
                edge_pairs.append((node, (entry_idx + 1) * stop_count + stop_idx))
                edge_weights.append(entry_weight)
                edge_axis.append("entry")
            if stop_idx + 1 < stop_count:
                edge_pairs.append((node, entry_idx * stop_count + (stop_idx + 1)))
                edge_weights.append(stop_weight)
                edge_axis.append("stop")

    edge_index = np.asarray(edge_pairs, dtype=int)
    weights = np.asarray(edge_weights, dtype=float)
    axis = np.asarray(edge_axis, dtype=object)

    adjacency = _build_adjacency(node_count, edge_index, weights)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if laplacian_type == "combinatorial":
        laplacian = sparse.diags(degree) - adjacency
    else:
        inv_sqrt = np.zeros_like(degree)
        positive = degree > 0.0
        inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        d_half = sparse.diags(inv_sqrt)
        laplacian = sparse.eye(node_count, format="csr") - d_half @ adjacency @ d_half

    entry_mask = axis == "entry"
    stop_mask = axis == "stop"
    entry_adjacency = _build_adjacency(node_count, edge_index[entry_mask], weights[entry_mask])
    stop_adjacency = _build_adjacency(node_count, edge_index[stop_mask], weights[stop_mask])
    entry_degree = np.asarray(entry_adjacency.sum(axis=1)).reshape(-1)
    stop_degree = np.asarray(stop_adjacency.sum(axis=1)).reshape(-1)

    return GridGraph(
        node_count=node_count,
        edge_count=int(edge_index.shape[0]),
        edge_index=edge_index,
        edge_weights=weights,
        edge_axis=axis,
        adjacency=adjacency.tocsr(),
        degree=degree,
        laplacian=laplacian.tocsr(),
        entry_laplacian=(sparse.diags(entry_degree) - entry_adjacency).tocsr(),
        stop_laplacian=(sparse.diags(stop_degree) - stop_adjacency).tocsr(),
        entry_indices=np.repeat(np.arange(entry_count, dtype=int), stop_count),
        stop_indices=np.tile(np.arange(stop_count, dtype=int), entry_count),
        canonical_config_ids=np.asarray(candidate_grid.config_ids, dtype=int),
        metadata={
            "graph_type": "rectangular_manhattan_4_neighbour",
            "grid_shape": tuple(int(x) for x in candidate_grid.grid_shape),
            "entry_weight": float(entry_weight),
            "stop_weight": float(stop_weight),
            "laplacian_type": laplacian_type,
            "graph_definition_hash": graph_definition_hash(candidate_grid, entry_weight=entry_weight, stop_weight=stop_weight, laplacian_type=laplacian_type),
        },
    )


def graph_definition_hash(
    candidate_grid: CandidateGrid,
    *,
    entry_weight: float,
    stop_weight: float,
    laplacian_type: str,
) -> str:
    payload = np.column_stack(
        [
            candidate_grid.config_ids.astype(float),
            candidate_grid.entry_values,
            candidate_grid.stop_values,
        ]
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(payload).tobytes())
    digest.update(f"{candidate_grid.grid_shape}|{entry_weight}|{stop_weight}|{laplacian_type}".encode("utf-8"))
    return digest.hexdigest()


def graph_energy(scores: np.ndarray, graph: GridGraph) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    diffs = scores[graph.edge_index[:, 0]] - scores[graph.edge_index[:, 1]]
    weighted = graph.edge_weights * diffs * diffs
    entry_mask = graph.edge_axis == "entry"
    stop_mask = graph.edge_axis == "stop"
    return {
        "total_graph_energy": float(np.sum(weighted)),
        "mean_edge_squared_difference": float(np.mean(diffs * diffs)),
        "maximum_edge_difference": float(np.max(np.abs(diffs))),
        "entry_axis_edge_energy": float(np.sum(weighted[entry_mask])),
        "stop_axis_edge_energy": float(np.sum(weighted[stop_mask])),
    }


def _build_adjacency(node_count: int, edge_index: np.ndarray, weights: np.ndarray) -> sparse.csr_matrix:
    if edge_index.size == 0:
        return sparse.csr_matrix((node_count, node_count), dtype=float)
    rows = np.concatenate([edge_index[:, 0], edge_index[:, 1]])
    cols = np.concatenate([edge_index[:, 1], edge_index[:, 0]])
    data = np.concatenate([weights, weights])
    return sparse.coo_matrix((data, (rows, cols)), shape=(node_count, node_count), dtype=float).tocsr()
