from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from bolr.config.foundation import BlockDynamicsConfig, BlockPriorConfig
from bolr.model.composite import CompositeScoreModel
from bolr.model.priors import BlockDynamicsSpec, BlockPriorSpec, assemble_block_prior, assemble_block_process_noise
from bolr.model.structured import penalty_shaped_process_noise, prior_from_penalty
from bolr.model.penalties import QuadraticPenalty
from bolr.model.score_blocks import ScoreBlock
from bolr.observations.base import ObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.targets.base import TargetBuilder


@dataclass(frozen=True)
class ComposedModel:
    score_model: CompositeScoreModel
    prior: GaussianPosterior
    process_noise: np.ndarray
    observation_model: ObservationModel | None
    target_builder: TargetBuilder | None
    metadata: dict[str, Any]


def compose_model(
    *,
    static_blocks: list[ScoreBlock],
    dynamic_blocks: list[ScoreBlock],
    sample_batch: object,
    prior_specs: list[BlockPriorSpec],
    dynamics_specs: list[BlockDynamicsSpec],
    fixed_blocks: set[str] | None = None,
) -> ComposedModel:
    score_model = CompositeScoreModel.from_blocks(static_blocks, dynamic_blocks, sample_batch, fixed_blocks=fixed_blocks)
    prior = assemble_block_prior(score_model.layout, prior_specs)
    process_noise = assemble_block_process_noise(score_model.layout, dynamics_specs)
    return ComposedModel(
        score_model=score_model,
        prior=prior,
        process_noise=process_noise,
        observation_model=None,
        target_builder=None,
        metadata={
            "layout": score_model.layout.metadata(),
            "fixed_blocks": sorted(fixed_blocks or set()),
        },
    )


def _mean_array(values: tuple[float, ...] | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def build_prior_specs_from_config(
    layout: object,
    configs: list[BlockPriorConfig],
    penalty_registry: Mapping[str, QuadraticPenalty] | None = None,
) -> list[BlockPriorSpec]:
    penalty_registry = penalty_registry or {}
    specs: list[BlockPriorSpec] = []
    for config in configs:
        if config.family == "isotropic_gaussian":
            specs.append(
                BlockPriorSpec(
                    block_name=config.block_name,
                    family=config.family,
                    mean=_mean_array(config.mean),
                    isotropic_scale=config.isotropic_scale,
                )
            )
            continue
        if config.family == "diagonal_gaussian":
            specs.append(
                BlockPriorSpec(
                    block_name=config.block_name,
                    family=config.family,
                    mean=_mean_array(config.mean),
                    diagonal=np.asarray(config.diagonal, dtype=float),
                )
            )
            continue
        penalty_name = config.penalty.source_name if config.penalty and config.penalty.source_name else config.block_name
        if penalty_name not in penalty_registry:
            raise KeyError(f"Missing penalty for block '{config.block_name}'.")
        structured = prior_from_penalty(
            penalty_registry[penalty_name],
            mean=_mean_array(config.mean),
            smooth_weight=config.penalty.weight,
            ridge=config.penalty.ridge,
        )
        specs.append(
            BlockPriorSpec(
                block_name=config.block_name,
                family=config.family,
                mean=structured.mean,
                covariance=structured.covariance,
                metadata=dict(structured.metadata),
            )
        )
    del layout
    return specs


def build_dynamics_specs_from_config(
    configs: list[BlockDynamicsConfig],
    penalty_registry: Mapping[str, QuadraticPenalty] | None = None,
) -> list[BlockDynamicsSpec]:
    penalty_registry = penalty_registry or {}
    specs: list[BlockDynamicsSpec] = []
    for config in configs:
        if config.family == "frozen":
            specs.append(BlockDynamicsSpec(block_name=config.block_name, family=config.family, frozen=True))
            continue
        if config.family == "isotropic_random_walk":
            specs.append(
                BlockDynamicsSpec(
                    block_name=config.block_name,
                    family=config.family,
                    isotropic_process_variance=config.scale,
                )
            )
            continue
        if config.family == "diagonal_random_walk":
            specs.append(
                BlockDynamicsSpec(
                    block_name=config.block_name,
                    family=config.family,
                    diagonal_process_variance=np.asarray(config.diagonal, dtype=float),
                )
            )
            continue
        if config.family == "covariance_discount":
            specs.append(BlockDynamicsSpec(block_name=config.block_name, family=config.family))
            continue
        penalty_name = config.block_name
        if penalty_name not in penalty_registry:
            raise KeyError(f"Missing penalty for block '{config.block_name}'.")
        process_noise = penalty_shaped_process_noise(
            penalty_registry[penalty_name],
            scale=config.scale,
            properization=config.properization or 1e-6,
        )
        specs.append(
            BlockDynamicsSpec(
                block_name=config.block_name,
                family=config.family,
                process_noise=process_noise,
                metadata=dict(process_noise.metadata),
            )
        )
    return specs


def compose_model_from_config(
    *,
    static_blocks: list[ScoreBlock],
    dynamic_blocks: list[ScoreBlock],
    sample_batch: object,
    prior_configs: list[BlockPriorConfig],
    dynamics_configs: list[BlockDynamicsConfig],
    penalty_registry: Mapping[str, QuadraticPenalty] | None = None,
    observation_model: ObservationModel | None = None,
    target_builder: TargetBuilder | None = None,
    fixed_blocks: set[str] | None = None,
) -> ComposedModel:
    score_model = CompositeScoreModel.from_blocks(static_blocks, dynamic_blocks, sample_batch, fixed_blocks=fixed_blocks)
    prior_specs = build_prior_specs_from_config(score_model.layout, prior_configs, penalty_registry)
    dynamics_specs = build_dynamics_specs_from_config(dynamics_configs, penalty_registry)
    prior = assemble_block_prior(score_model.layout, prior_specs)
    process_noise = assemble_block_process_noise(score_model.layout, dynamics_specs)
    return ComposedModel(
        score_model=score_model,
        prior=prior,
        process_noise=process_noise,
        observation_model=observation_model,
        target_builder=target_builder,
        metadata={
            "layout": score_model.layout.metadata(),
            "fixed_blocks": sorted(fixed_blocks or set()),
            "prior_families": {config.block_name: config.family for config in prior_configs},
            "dynamics_families": {config.block_name: config.family for config in dynamics_configs},
        },
    )
