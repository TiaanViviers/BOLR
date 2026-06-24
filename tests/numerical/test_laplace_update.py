import numpy as np

from bolr.inference.laplace import laplace_update
from bolr.inference.newton import NewtonOptions
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign
from bolr.targets.soft_target import build_soft_target_observation


def test_structured_and_explicit_laplace_updates_match() -> None:
    candidate_basis = np.array(
        [
            [1.0, 0.0],
            [0.4, 0.2],
            [-0.3, 0.8],
            [0.1, -0.5],
        ]
    )
    context = np.array([1.0, -0.4, 0.3])
    design = DailyDesign(candidate_basis=candidate_basis, context_vector=context)
    observation = build_soft_target_observation(np.array([1.0, 0.1, -0.3, 0.4]))
    prior = GaussianPosterior(mean=np.zeros(design.state_dim), covariance=np.eye(design.state_dim) * 1.5)
    options = NewtonOptions(max_iterations=20)

    structured = laplace_update(prior, design, observation, options=options, use_structured_curvature=True)
    explicit = laplace_update(prior, design, observation, options=options, use_structured_curvature=False)

    assert structured.newton_result.converged
    assert explicit.newton_result.converged
    assert np.allclose(structured.posterior.mean, explicit.posterior.mean, atol=1e-9)
    assert np.allclose(structured.posterior.covariance, explicit.posterior.covariance, atol=1e-9)

