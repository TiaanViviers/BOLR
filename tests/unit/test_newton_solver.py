import numpy as np

from bolr.inference.newton import NewtonOptions, damped_newton_solve


def test_damped_newton_solver_recovers_quadratic_mode() -> None:
    target = np.array([0.7, -1.2, 0.4])
    precision = np.array(
        [
            [2.0, 0.2, 0.0],
            [0.2, 1.5, 0.1],
            [0.0, 0.1, 1.0],
        ]
    )

    def objective(x):
        delta = x - target
        return -0.5 * delta @ precision @ delta

    def gradient(x):
        return -precision @ (x - target)

    def information(_x):
        return precision

    result = damped_newton_solve(
        start=np.zeros(3),
        objective_fn=objective,
        gradient_fn=gradient,
        information_fn=information,
        options=NewtonOptions(max_iterations=10),
    )
    assert result.converged
    assert np.allclose(result.point, target, atol=1e-10)

