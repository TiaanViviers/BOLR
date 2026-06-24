import numpy as np

from bolr.config.foundation import SoftTargetConfig
from bolr.inference.laplace import laplace_update, laplace_update_composite
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.representation.score_design import DailyDesign
from bolr.targets.soft_target import SoftTargetBuilder


def test_composite_e0_path_matches_existing_design_path() -> None:
    phi = np.array([[1.0, 0.0], [0.3, 0.7], [-0.2, 0.4], [0.1, -0.5]])
    static_coeffs = np.array([0.2, -0.1])
    static_scores = phi @ static_coeffs
    prior = make_initial_dynamic_prior(2, sigma0=1.0)
    observation = SoftTargetBuilder(SoftTargetConfig()).build(np.array([1.0, 0.3, -0.4, 0.2]))
    design = DailyDesign(candidate_basis=phi, context_vector=np.array([1.0]), static_scores=static_scores)
    existing = laplace_update(prior, design, observation, observation_model=SoftTargetObservationModel())
    model = CompositeScoreModel.from_blocks([StaticBaselineBlock("baseline", phi, static_coeffs, {})], [DynamicSurfaceBlock("surface", phi)], {})
    composite = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    assert np.allclose(existing.posterior.mean, composite.posterior.mean, atol=1e-12)
    assert np.allclose(existing.posterior.covariance, composite.posterior.covariance, atol=1e-12)
