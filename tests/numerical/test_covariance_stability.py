import numpy as np

from bolr.posterior.diagnostics import jittered_cholesky


def test_jittered_cholesky_stabilizes_near_singular_matrix() -> None:
    matrix = np.array([[1.0, 0.9999999999], [0.9999999999, 0.9999999998]])
    result = jittered_cholesky(matrix, initial_jitter=1e-12, max_attempts=10)
    reconstructed = result.factor @ result.factor.T
    assert result.jitter > 0.0
    assert np.all(np.linalg.eigvalsh(reconstructed) > 0.0)

