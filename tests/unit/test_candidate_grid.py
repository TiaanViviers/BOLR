import numpy as np

from bolr.data.candidate_grid import load_candidate_grid, load_day_utility_vector


def test_candidate_grid_is_complete_and_canonical() -> None:
    grid = load_candidate_grid("data/YM_grid.csv")
    assert grid.grid_shape == (34, 42)
    assert grid.n_candidates == 1428
    assert np.array_equal(grid.config_ids, np.arange(1428))
    assert grid.pair_to_id[(0.0004, 0.0001)] == 0


def test_day_utility_vector_covers_full_grid() -> None:
    grid = load_candidate_grid("data/YM_grid.csv")
    utilities = load_day_utility_vector("data/YM_grid.csv", "2008.01.02", grid)
    assert utilities.shape == (1428,)
    assert np.isfinite(utilities).all()
