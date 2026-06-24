import numpy as np

from bolr.inference.laplace import laplace_update, laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.priors import BlockPriorSpec, assemble_block_prior
from bolr.model.score_blocks import ContextInteractionBlock, DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.representation.score_design import DailyDesign
from bolr.targets.soft_target import SoftTargetBuilder


def test_surface_only_composite_matches_single_block_path() -> None:
    phi = np.array([[1.0, 0.0], [0.4, 0.6], [-0.2, 0.7]])
    static_coeffs = np.array([0.1, -0.05])
    dynamic_state = np.array([0.3, -0.2])
    utilities = np.array([1.0, 0.1, -0.3])
    observation = SoftTargetBuilder().build(utilities)
    prior = assemble_block_prior(
        CompositeScoreModel.from_blocks(
            [StaticBaselineBlock("baseline", phi, static_coeffs, {})],
            [DynamicSurfaceBlock("surface", phi)],
            {},
        ).layout,
        [BlockPriorSpec("surface", isotropic_scale=1.0)],
    )
    design = DailyDesign(candidate_basis=phi, context_vector=np.array([1.0]), static_scores=phi @ static_coeffs)
    single = laplace_update(prior, design, observation, observation_model=SoftTargetObservationModel())
    model = CompositeScoreModel.from_blocks([StaticBaselineBlock("baseline", phi, static_coeffs, {})], [DynamicSurfaceBlock("surface", phi)], {})
    composite = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    assert np.allclose(single.posterior.mean, composite.posterior.mean)
    assert np.allclose(single.posterior.covariance, composite.posterior.covariance)


def test_surface_plus_context_composite_runs() -> None:
    phi = np.array([[1.0, 0.0], [0.4, 0.6], [-0.2, 0.7]])
    static = StaticBaselineBlock("baseline", phi, np.array([0.1, -0.05]), {})
    surface = DynamicSurfaceBlock("surface", phi)
    context = ContextInteractionBlock("context", phi)
    batch = {"context_vector": np.array([1.0, 0.3])}
    model = CompositeScoreModel.from_blocks([static], [surface, context], batch)
    prior = assemble_block_prior(
        model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("context", isotropic_scale=1.0),
        ],
    )
    observation = SoftTargetBuilder().build(np.array([1.0, -0.2, 0.4]))
    result = laplace_update_composite(prior, model, batch, observation, observation_model=SoftTargetObservationModel())
    assert result.newton_result.converged

