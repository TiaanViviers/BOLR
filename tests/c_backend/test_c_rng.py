from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CRNG, CLibrary


def test_c_rng_deterministic_u32_and_clone() -> None:
    library = CLibrary()
    expected = [3837872008, 932996374, 1548399547, 1612522464, 473443212]
    with CRNG(seed=0, stream=0, library=library) as rng:
        assert [rng.u32() for _ in range(len(expected))] == expected

    with CRNG(seed=9, stream=5, library=library) as left:
        with left.clone() as right:
            left_draws = np.array([left.normal() for _ in range(8)])
            right_draws = np.array([right.normal() for _ in range(8)])
            assert np.array_equal(left_draws, right_draws)
            metadata = left.metadata()
            assert metadata.stream == 5
            assert metadata.algorithm_family == 1
            assert metadata.ziggurat_layers == 128


def test_c_rng_uniform_range_and_bulk_fill() -> None:
    with CRNG(seed=17, stream=3) as rng:
        values = rng.fill_uniform(256)
        assert values.shape == (256,)
        assert np.all(values > 0.0)
        assert np.all(values < 1.0)
        normals = rng.fill_standard_normal(128)
        assert normals.shape == (128,)
        assert np.isfinite(normals).all()
