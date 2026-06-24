import numpy as np

from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.penalties import difference_penalty
from bolr.model.priors import BlockPriorSpec, assemble_block_prior, assemble_block_process_noise, BlockDynamicsSpec
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.model.structured import penalty_shaped_process_noise
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder


def test_structured_prior_smoke_for_surface_block() -> None:
    phi = np.array([[1.0, 0.0, 0.0], [0.4, 0.4, 0.0], [0.0, 0.6, 0.2], [-0.1, 0.2, 0.8]])
    static = StaticBaselineBlock("baseline", phi, np.array([0.1, -0.05, 0.0]), {})
    surface = DynamicSurfaceBlock("surface", phi)
    model = CompositeScoreModel.from_blocks([static], [surface], {})
    penalty = difference_penalty(3, 2)
    prior = assemble_block_prior(model.layout, [BlockPriorSpec("surface", isotropic_scale=1.0)])
    q = penalty_shaped_process_noise(penalty, scale=0.05, properization=0.2)
    assembled_q = assemble_block_process_noise(model.layout, [BlockDynamicsSpec("surface", process_noise=q)])
    assert assembled_q.shape == (3, 3)
    observation = SoftTargetBuilder().build(np.array([1.0, 0.4, -0.1, 0.2]))
    result = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    assert result.newton_result.converged
