import numpy as np

from bolr.config.foundation import SplineAxisConfig, TensorBasisConfig
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.spline_basis import BSplineBasis1D
from bolr.representation.tensor_basis import TensorProductBasis


def test_bspline_basis_has_exact_width_and_partition_of_unity() -> None:
    values = np.linspace(-2.0, 2.0, 11)
    basis = BSplineBasis1D(SplineAxisConfig(n_basis=6, degree=3)).fit_transform(values)
    assert basis.shape == (11, 6)
    assert np.allclose(basis.sum(axis=1), 1.0)


def test_tensor_basis_is_centered_and_rank_reduced() -> None:
    entry, stop = np.meshgrid(np.linspace(0.1, 1.0, 6), np.linspace(0.2, 1.2, 7), indexing="ij")
    coordinates = LogCoordinateTransform().fit(entry.ravel(), stop.ravel()).transform(entry.ravel(), stop.ravel())
    basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=4, degree=2),
            stop_basis=SplineAxisConfig(n_basis=5, degree=2),
        )
    ).fit_transform(coordinates)
    assert np.allclose(basis.centered_basis.sum(axis=0), 0.0, atol=1e-10)
    assert basis.reduced_basis.shape[1] < basis.original_basis.shape[1]
    assert np.linalg.matrix_rank(basis.reduced_basis) == basis.reduced_basis.shape[1]
