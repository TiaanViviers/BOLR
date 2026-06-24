from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bolr.model.structured import ProcessNoiseModel
from bolr.model.state_layout import StateLayout
from bolr.posterior.state import GaussianPosterior


@dataclass(frozen=True)
class BlockPriorSpec:
    block_name: str
    family: str = "isotropic_gaussian"
    mean: np.ndarray | None = None
    isotropic_scale: float | None = None
    diagonal: np.ndarray | None = None
    covariance: np.ndarray | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class BlockDynamicsSpec:
    block_name: str
    family: str = "isotropic_random_walk"
    isotropic_process_variance: float | None = None
    diagonal_process_variance: np.ndarray | None = None
    process_noise: ProcessNoiseModel | None = None
    frozen: bool = False
    fixed_parameter: bool = False
    metadata: dict[str, object] | None = None


def _validated_spec_map(layout: StateLayout, specs: list[BlockPriorSpec | BlockDynamicsSpec]) -> dict[str, BlockPriorSpec | BlockDynamicsSpec]:
    names = [spec.block_name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate block configurations are not allowed.")
    valid = {block.name for block in layout.blocks}
    unknown = sorted(set(names) - valid)
    if unknown:
        raise ValueError(f"Unknown block configurations: {unknown}")
    missing = sorted(valid - set(names))
    if missing:
        raise ValueError(f"Missing block configurations: {missing}")
    return {spec.block_name: spec for spec in specs}


def assemble_block_prior(layout: StateLayout, specs: list[BlockPriorSpec]) -> GaussianPosterior:
    mean = np.zeros(layout.total_dimension, dtype=float)
    covariance = np.zeros((layout.total_dimension, layout.total_dimension), dtype=float)
    spec_map = _validated_spec_map(layout, specs)
    for block in layout.blocks:
        spec = spec_map[block.name]
        block_mean = np.zeros(block.dimension, dtype=float) if spec.mean is None else np.asarray(spec.mean, dtype=float).reshape(-1)
        if block_mean.size != block.dimension:
            raise ValueError("Block prior mean dimension mismatch.")
        mean[block.start:block.stop] = block_mean
        if spec.covariance is not None:
            block_covariance = np.asarray(spec.covariance, dtype=float)
            if block_covariance.shape != (block.dimension, block.dimension):
                raise ValueError("Block prior covariance mismatch.")
            covariance[block.start:block.stop, block.start:block.stop] = block_covariance
        elif spec.diagonal is not None:
            diagonal = np.asarray(spec.diagonal, dtype=float).reshape(-1)
            if diagonal.size != block.dimension:
                raise ValueError("Block prior diagonal mismatch.")
            if np.any(diagonal <= 0.0):
                raise ValueError("Block prior diagonal entries must be positive.")
            covariance[block.start:block.stop, block.start:block.stop] = np.diag(diagonal)
        elif spec.isotropic_scale is not None:
            if spec.isotropic_scale <= 0.0:
                raise ValueError("Block isotropic prior scale must be positive.")
            covariance[block.start:block.stop, block.start:block.stop] = (spec.isotropic_scale**2) * np.eye(block.dimension)
        else:
            raise ValueError("Each block prior must specify either isotropic_scale or diagonal.")
    return GaussianPosterior(mean=mean, covariance=covariance)


def assemble_block_process_noise(layout: StateLayout, specs: list[BlockDynamicsSpec]) -> np.ndarray:
    covariance = np.zeros((layout.total_dimension, layout.total_dimension), dtype=float)
    spec_map = _validated_spec_map(layout, specs)
    for block in layout.blocks:
        spec = spec_map[block.name]
        if spec.frozen or spec.fixed_parameter:
            continue
        if spec.process_noise is not None:
            block_cov = np.asarray(spec.process_noise.covariance, dtype=float)
            if block_cov.shape != (block.dimension, block.dimension):
                raise ValueError("Structured process noise dimension mismatch.")
            covariance[block.start:block.stop, block.start:block.stop] = block_cov
        elif spec.diagonal_process_variance is not None:
            diagonal = np.asarray(spec.diagonal_process_variance, dtype=float).reshape(-1)
            if diagonal.size != block.dimension:
                raise ValueError("Block process diagonal mismatch.")
            if np.any(diagonal < 0.0):
                raise ValueError("Block process diagonal entries must be non-negative.")
            covariance[block.start:block.stop, block.start:block.stop] = np.diag(diagonal)
        elif spec.isotropic_process_variance is not None:
            if spec.isotropic_process_variance < 0.0:
                raise ValueError("Block isotropic process variance must be non-negative.")
            covariance[block.start:block.stop, block.start:block.stop] = spec.isotropic_process_variance * np.eye(block.dimension)
        elif spec.family == "covariance_discount":
            continue
        else:
            raise ValueError("Each block process spec must specify isotropic or diagonal variance unless frozen.")
    return covariance
