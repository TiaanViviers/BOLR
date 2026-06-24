import numpy as np

from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.penalties import difference_penalty
from bolr.model.priors import BlockPriorSpec, assemble_block_prior
from bolr.model.score_blocks import DynamicSurfaceBlock
from bolr.model.structured import prior_from_penalty
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder


def test_smooth_truth_structured_prior_completes() -> None:
    phi = np.eye(5)
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi)], {})
    penalty = difference_penalty(5, 2)
    structured = prior_from_penalty(penalty, smooth_weight=5.0, ridge=0.2)
    prior = assemble_block_prior(model.layout, [BlockPriorSpec("surface", family="structured_gaussian", mean=structured.mean, covariance=structured.covariance)])
    observation = SoftTargetBuilder().build(np.array([0.8, 0.5, 0.2, -0.1, -0.4]))
    result = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    assert result.newton_result.converged


def test_structured_prior_improves_smooth_truth_but_biases_rough_truth() -> None:
    phi = np.eye(5)
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi)], {})
    penalty = difference_penalty(5, 2)
    observation_model = SoftTargetObservationModel()
    structured = prior_from_penalty(penalty, smooth_weight=5.0, ridge=0.2)
    structured_prior = assemble_block_prior(
        model.layout,
        [BlockPriorSpec("surface", family="structured_gaussian", mean=structured.mean, covariance=structured.covariance)],
    )
    isotropic_prior = assemble_block_prior(model.layout, [BlockPriorSpec("surface", isotropic_scale=1.0)])

    smooth_truth = np.array([0.8, 0.5, 0.2, -0.1, -0.4])
    rough_truth = np.array([1.0, -1.0, 1.0, -1.0, 1.0])

    smooth_observation = SoftTargetBuilder().build(smooth_truth)
    rough_observation = SoftTargetBuilder().build(rough_truth)

    smooth_structured = laplace_update_composite(structured_prior, model, {}, smooth_observation, observation_model=observation_model)
    smooth_isotropic = laplace_update_composite(isotropic_prior, model, {}, smooth_observation, observation_model=observation_model)
    rough_structured = laplace_update_composite(structured_prior, model, {}, rough_observation, observation_model=observation_model)
    rough_isotropic = laplace_update_composite(isotropic_prior, model, {}, rough_observation, observation_model=observation_model)

    smooth_structured_mse = float(np.mean((smooth_structured.posterior.mean - smooth_truth) ** 2))
    smooth_isotropic_mse = float(np.mean((smooth_isotropic.posterior.mean - smooth_truth) ** 2))
    rough_structured_mse = float(np.mean((rough_structured.posterior.mean - rough_truth) ** 2))
    rough_isotropic_mse = float(np.mean((rough_isotropic.posterior.mean - rough_truth) ** 2))

    assert smooth_structured_mse < smooth_isotropic_mse
    assert rough_structured_mse > rough_isotropic_mse
