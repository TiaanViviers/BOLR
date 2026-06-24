from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from bolr.config.foundation import RegionDefinitionConfig
from bolr.model.graph import GridGraph


@dataclass(frozen=True)
class RegionComponentSummary:
    region_id: int
    candidate_indices: np.ndarray
    candidate_count: int
    inclusion_mass: float
    probability_best_mass: float
    maximum_score_mean: float
    average_score_mean: float
    inclusion_weighted_score_mean: float
    average_score_variance: float
    maximum_score_variance: float
    inclusion_weighted_variance: float
    entry_index_range: tuple[int, int]
    stop_index_range: tuple[int, int]
    entry_value_range: tuple[float, float]
    stop_value_range: tuple[float, float]
    graph_diameter: float
    boundary_edge_count: int
    compactness: float
    representative_medoid_index: int | None = None


@dataclass(frozen=True)
class RegionSummary:
    top_k: int
    inclusion_probability: np.ndarray
    consensus_indices: np.ndarray
    consensus_family: str
    threshold: float
    total_inclusion_mass: float
    empty_consensus: bool
    components: tuple[RegionComponentSummary, ...]
    edge_comembership: dict[tuple[int, int], float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


def build_consensus_set(inclusion_probability: np.ndarray, config: RegionDefinitionConfig) -> np.ndarray:
    inclusion_probability = np.asarray(inclusion_probability, dtype=float)
    order = np.argsort(-inclusion_probability, kind="mergesort")
    if config.consensus_family == "threshold":
        indices = np.flatnonzero(inclusion_probability >= config.inclusion_threshold)
    elif config.consensus_family == "top_count":
        count = max(1, int(round(config.inclusion_threshold)))
        indices = order[:count]
    else:
        cumulative = np.cumsum(inclusion_probability[order])
        target_mass = float(config.inclusion_threshold)
        cutoff = int(np.searchsorted(cumulative, target_mass, side="left")) + 1
        indices = order[: min(cutoff, inclusion_probability.size)]
    return np.sort(indices.astype(int))


def weighted_graph_medoid(indices: np.ndarray, weights: np.ndarray, graph: GridGraph) -> tuple[int, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if indices.size == 0:
        raise ValueError("Cannot compute a medoid on an empty set.")
    if indices.size == 1:
        return int(indices[0]), np.zeros((1, 1), dtype=float)
    subgraph = graph.adjacency[indices][:, indices]
    distances = shortest_path(subgraph, directed=False, unweighted=True)
    weighted_distance = distances @ weights
    return int(indices[np.argmin(weighted_distance)]), distances


def summarize_regions(
    *,
    score_mean: np.ndarray,
    score_variance: np.ndarray,
    probability_best: np.ndarray,
    probability_top_k: dict[int, np.ndarray],
    candidate_grid,
    graph: GridGraph,
    region_config: RegionDefinitionConfig,
    score_samples: np.ndarray | None = None,
) -> RegionSummary:
    if candidate_grid is None:
        raise ValueError("Region summaries require a candidate_grid.")
    top_k = int(region_config.top_k or max(1, int(np.ceil(float(region_config.top_fraction) * score_mean.size))))
    if top_k not in probability_top_k:
        raise ValueError(f"Top-k inclusion probabilities for K={top_k} were not provided.")
    inclusion_probability = np.asarray(probability_top_k[top_k], dtype=float)
    consensus_indices = build_consensus_set(inclusion_probability, region_config)
    if consensus_indices.size == 0:
        consensus_indices = np.array([int(np.argmax(inclusion_probability))], dtype=int)
        empty_consensus = True
    else:
        empty_consensus = False

    indicator = np.zeros(graph.node_count, dtype=bool)
    indicator[consensus_indices] = True
    rows = []
    cols = []
    data = []
    for edge_idx, (left, right) in enumerate(graph.edge_index):
        if indicator[left] and indicator[right]:
            rows.extend([left, right])
            cols.extend([right, left])
            data.extend([graph.edge_weights[edge_idx], graph.edge_weights[edge_idx]])
    induced = csr_matrix((data, (rows, cols)), shape=(graph.node_count, graph.node_count))
    _, labels = connected_components(induced[consensus_indices][:, consensus_indices], directed=False, return_labels=True)

    components: list[RegionComponentSummary] = []
    for component_id in np.unique(labels):
        members = consensus_indices[labels == component_id]
        entry_idx = graph.entry_indices[members]
        stop_idx = graph.stop_indices[members]
        weights = inclusion_probability[members]
        internal_edges = 0
        boundary_edges = 0
        for left, right in graph.edge_index:
            left_in = left in set(members.tolist())
            right_in = right in set(members.tolist())
            if left_in and right_in:
                internal_edges += 1
            elif left_in or right_in:
                boundary_edges += 1
        medoid, distances = weighted_graph_medoid(members, weights, graph)
        finite = distances[np.isfinite(distances)]
        graph_diameter = float(finite.max()) if finite.size else 0.0
        weight_sum = float(weights.sum()) if float(weights.sum()) > 0.0 else 1.0
        components.append(
            RegionComponentSummary(
                region_id=int(component_id),
                candidate_indices=members.copy(),
                candidate_count=int(members.size),
                inclusion_mass=float(weights.sum()),
                probability_best_mass=float(np.asarray(probability_best, dtype=float)[members].sum()),
                maximum_score_mean=float(np.max(score_mean[members])),
                average_score_mean=float(np.mean(score_mean[members])),
                inclusion_weighted_score_mean=float(np.dot(weights, score_mean[members]) / weight_sum),
                average_score_variance=float(np.mean(score_variance[members])),
                maximum_score_variance=float(np.max(score_variance[members])),
                inclusion_weighted_variance=float(np.dot(weights, score_variance[members]) / weight_sum),
                entry_index_range=(int(entry_idx.min()), int(entry_idx.max())),
                stop_index_range=(int(stop_idx.min()), int(stop_idx.max())),
                entry_value_range=(float(candidate_grid.entry_values[members].min()), float(candidate_grid.entry_values[members].max())),
                stop_value_range=(float(candidate_grid.stop_values[members].min()), float(candidate_grid.stop_values[members].max())),
                graph_diameter=graph_diameter,
                boundary_edge_count=int(boundary_edges),
                compactness=float(internal_edges / max(1, members.size - 1)),
                representative_medoid_index=medoid,
            )
        )

    edge_comembership: dict[tuple[int, int], float] = {}
    if region_config.edge_comembership_enabled and score_samples is not None and score_samples.shape[0] > 0:
        order = np.argsort(-score_samples, axis=1, kind="mergesort")
        membership = np.zeros_like(score_samples, dtype=bool)
        sample_ids = np.arange(score_samples.shape[0])[:, None]
        membership[sample_ids, order[:, :top_k]] = True
        for left, right in graph.edge_index:
            if indicator[left] and indicator[right]:
                edge_comembership[(int(left), int(right))] = float(np.mean(membership[:, left] & membership[:, right]))

    return RegionSummary(
        top_k=top_k,
        inclusion_probability=inclusion_probability.copy(),
        consensus_indices=consensus_indices.copy(),
        consensus_family=region_config.consensus_family,
        threshold=float(region_config.inclusion_threshold),
        total_inclusion_mass=float(inclusion_probability[consensus_indices].sum()),
        empty_consensus=empty_consensus,
        components=tuple(sorted(components, key=lambda component: component.region_id)),
        edge_comembership=edge_comembership,
        metadata={"component_count": len(components)},
    )
