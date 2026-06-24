import numpy as np

from bolr.config.foundation import BlockDynamicsConfig, BlockPriorConfig, PenaltyConfig
from bolr.model.composer import compose_model_from_config
from bolr.model.penalties import difference_penalty
from bolr.model.score_blocks import DynamicSurfaceBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder


def test_compose_model_from_structured_configs_builds_prior_and_process_noise() -> None:
    phi = np.array([[1.0, 0.0, 0.0], [0.4, 0.5, 0.1], [-0.2, 0.3, 0.8]])
    surface = DynamicSurfaceBlock("surface", phi)
    penalty_registry = {"surface": difference_penalty(3, 2)}
    composed = compose_model_from_config(
        static_blocks=[],
        dynamic_blocks=[surface],
        sample_batch={},
        prior_configs=[
            BlockPriorConfig(
                block_name="surface",
                family="structured_gaussian",
                penalty=PenaltyConfig(family="named", weight=2.0, ridge=0.3),
            )
        ],
        dynamics_configs=[
            BlockDynamicsConfig(
                block_name="surface",
                family="penalty_shaped_random_walk",
                scale=0.05,
                properization=0.2,
            )
        ],
        penalty_registry=penalty_registry,
        observation_model=SoftTargetObservationModel(),
        target_builder=SoftTargetBuilder(),
    )
    assert composed.prior.covariance.shape == (3, 3)
    assert composed.process_noise.shape == (3, 3)
    assert composed.observation_model is not None
    assert composed.target_builder is not None
    assert composed.metadata["prior_families"]["surface"] == "structured_gaussian"
    assert composed.metadata["dynamics_families"]["surface"] == "penalty_shaped_random_walk"


def test_compose_model_from_config_rejects_missing_penalty() -> None:
    phi = np.eye(2)
    surface = DynamicSurfaceBlock("surface", phi)
    try:
        compose_model_from_config(
            static_blocks=[],
            dynamic_blocks=[surface],
            sample_batch={},
            prior_configs=[
                BlockPriorConfig(
                    block_name="surface",
                    family="structured_gaussian",
                    penalty=PenaltyConfig(family="named", weight=1.0, ridge=0.1),
                )
            ],
            dynamics_configs=[BlockDynamicsConfig(block_name="surface", family="frozen")],
            penalty_registry={},
        )
    except KeyError as exc:
        assert "surface" in str(exc)
    else:
        raise AssertionError("Expected missing structured penalty to raise KeyError.")
