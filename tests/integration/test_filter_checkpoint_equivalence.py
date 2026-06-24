import numpy as np

from bolr.dynamics.random_walk import IsotropicRandomWalkDynamics
from bolr.inference.laplace import laplace_update
from bolr.inference.newton import NewtonOptions
from bolr.posterior.state import GaussianPosterior
from bolr.representation.score_design import DailyDesign
from bolr.synthetic.generator import SyntheticContextualSurfaceGenerator, SyntheticGeneratorConfig
from bolr.targets.soft_target import build_soft_target_observation


def _run_filter(days, posterior):
    dynamics = IsotropicRandomWalkDynamics(process_variance=0.02)
    options = NewtonOptions(max_iterations=25)
    for day in days:
        prior = dynamics.predict(posterior)
        design = DailyDesign(day_scores_basis, day.context_vector)
        observation = build_soft_target_observation(day.utilities)
        posterior = laplace_update(prior, design, observation, options=options).posterior
    return posterior


day_scores_basis = np.array(
    [
        [1.0, 0.0],
        [0.5, 0.3],
        [0.0, 1.0],
        [-0.4, 0.7],
    ]
)


def test_filter_checkpoint_equivalence() -> None:
    scenario = SyntheticContextualSurfaceGenerator(
        SyntheticGeneratorConfig(
            n_days=8,
            candidate_basis=day_scores_basis,
            context_dim=3,
            observation_noise=0.02,
            seed=21,
        )
    ).stationary_scenario()
    initial = GaussianPosterior(mean=np.zeros(6), covariance=np.eye(6) * 1.5)
    full = _run_filter(scenario.days, initial)
    first_half = _run_filter(scenario.days[:4], initial)
    resumed = _run_filter(scenario.days[4:], first_half)
    assert np.allclose(full.mean, resumed.mean, atol=1e-10)
    assert np.allclose(full.covariance, resumed.covariance, atol=1e-10)
