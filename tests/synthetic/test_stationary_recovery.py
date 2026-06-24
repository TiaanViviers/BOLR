import numpy as np

from bolr.dynamics.random_walk import IsotropicRandomWalkDynamics
from bolr.inference.laplace import laplace_update
from bolr.inference.newton import NewtonOptions
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign
from bolr.synthetic.generator import SyntheticContextualSurfaceGenerator, SyntheticGeneratorConfig
from bolr.targets.soft_target import build_soft_target_observation


def test_stationary_scenario_reduces_final_score_error() -> None:
    candidate_basis = np.array(
        [
            [1.0, 0.0],
            [0.6, 0.2],
            [0.0, 1.0],
            [-0.5, 0.6],
            [0.2, -0.7],
        ]
    )
    scenario = SyntheticContextualSurfaceGenerator(
        SyntheticGeneratorConfig(
            n_days=10,
            candidate_basis=candidate_basis,
            context_dim=3,
            observation_noise=0.02,
            seed=4,
        )
    ).stationary_scenario()
    posterior = GaussianPosterior(mean=np.zeros(6), covariance=np.eye(6) * 2.0)
    dynamics = IsotropicRandomWalkDynamics(process_variance=0.01)
    options = NewtonOptions(max_iterations=25)

    initial_design = DailyDesign(candidate_basis, scenario.days[0].context_vector)
    initial_error = np.sqrt(np.mean((initial_design.scores(posterior.mean) - scenario.days[-1].scores) ** 2))

    for day in scenario.days:
        prior = dynamics.predict(posterior)
        design = DailyDesign(candidate_basis, day.context_vector)
        observation = build_soft_target_observation(day.utilities)
        posterior = laplace_update(prior, design, observation, options=options).posterior

    final_design = DailyDesign(candidate_basis, scenario.days[-1].context_vector)
    final_error = np.sqrt(np.mean((final_design.scores(posterior.mean) - scenario.days[-1].scores) ** 2))
    assert final_error < initial_error
    assert final_error < 0.6

