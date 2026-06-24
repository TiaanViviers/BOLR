import numpy as np

from bolr.data.candidate_grid import CandidateGrid, load_candidate_grid
from bolr.model.graph import build_canonical_grid_graph, graph_energy


def test_canonical_grid_graph_has_expected_counts_and_degrees() -> None:
    grid = load_candidate_grid("data/YM_grid.csv")
    graph = build_canonical_grid_graph(grid)
    assert graph.node_count == 1428
    assert graph.edge_count == (34 - 1) * 42 + 34 * (42 - 1)
    assert np.allclose(graph.adjacency.toarray(), graph.adjacency.toarray().T)
    assert graph.degree[0] == 2.0
    assert graph.degree[41] == 2.0
    assert graph.degree[42] == 3.0
    assert graph.degree[43] == 4.0
    assert np.array_equal(graph.canonical_config_ids, np.arange(1428))


def test_graph_energy_matches_laplacian_quadratic_form() -> None:
    grid = CandidateGrid(
        config_ids=np.arange(6),
        entry_values=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
        stop_values=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.1, 0.3): 2, (0.2, 0.1): 3, (0.2, 0.2): 4, (0.2, 0.3): 5},
        grid_shape=(2, 3),
    )
    graph = build_canonical_grid_graph(grid, entry_weight=1.5, stop_weight=0.7)
    scores = np.array([1.0, 0.5, -0.2, 0.4, -0.1, 0.3])
    diagnostics = graph_energy(scores, graph)
    quadratic = float(scores @ (graph.laplacian @ scores))
    assert np.isclose(diagnostics["total_graph_energy"], quadratic)
    assert np.isclose(diagnostics["entry_axis_edge_energy"] + diagnostics["stop_axis_edge_energy"], quadratic)
