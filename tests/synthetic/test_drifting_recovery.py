import numpy as np

from bolr.dynamics.random_walk import IsotropicRandomWalkDynamics
from bolr.inference.laplace import laplace_update
from bolr.inference.newton import NewtonOptions
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign
from bolr.synthetic.generator import SyntheticContextualSurfaceGenerator, SyntheticGeneratorConfig
from bolr.targets.soft_target import build_soft_target_observation


def test_drifting_scenario_tracks_last_day_reasonably() -> None:
    candidate_basis = np.array(
        [
            [1.0, 0.0],
            [0.7, 0.1],
            [0.2, 0.9],
            [-0.4, 0.8],
            [0.1, -0.6],
        ]
    )
    scenario = SyntheticContextualSurfaceGenerator(
        SyntheticGeneratorConfig(
            n_days=12,
            candidate_basis=candidate_basis,
            context_dim=3,
            observation_noise=0.03,
            seed=9,
        )
    ).drifting_scenario()
    posterior = GaussianPosterior(mean=np.zeros(6), covariance=np.eye(6) * 2.0)
    dynamics = IsotropicRandomWalkDynamics(process_variance=0.03)
    options = NewtonOptions(max_iterations=25)

    final_design = DailyDesign(candidate_basis, scenario.days[-1].context_vector)
    baseline_error = np.sqrt(np.mean((final_design.scores(posterior.mean) - scenario.days[-1].scores) ** 2))
    errors = []
    for day in scenario.days:
        prior = dynamics.predict(posterior)
        design = DailyDesign(candidate_basis, day.context_vector)
        observation = build_soft_target_observation(day.utilities)
        posterior = laplace_update(prior, design, observation, options=options).posterior
        errors.append(np.sqrt(np.mean((design.scores(posterior.mean) - day.scores) ** 2)))

    assert errors[-1] < 0.8
    assert errors[-1] < baseline_error
