from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class StateBlockLayout:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    vectorization_order: str
    parameter_family: str
    dynamic: bool
    metadata: Mapping[str, object]

    @property
    def dimension(self) -> int:
        return int(self.stop - self.start)


@dataclass(frozen=True)
class StateLayout:
    blocks: tuple[StateBlockLayout, ...]
    total_dimension: int

    def __post_init__(self) -> None:
        names = [block.name for block in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate block names are not allowed.")
        expected_start = 0
        for block in self.blocks:
            if block.start != expected_start:
                raise ValueError("State block slices must be contiguous and non-overlapping.")
            if block.stop <= block.start:
                raise ValueError("Invalid block slice.")
            if int(np.prod(block.shape, dtype=int)) != block.dimension:
                raise ValueError("Block shape does not match slice dimension.")
            if block.vectorization_order not in {"C", "F"}:
                raise ValueError("vectorization_order must be 'C' or 'F'.")
            expected_start = block.stop
        if expected_start != self.total_dimension:
            raise ValueError("total_dimension does not match block coverage.")

    def slice_for(self, block_name: str) -> slice:
        block = self._block(block_name)
        return slice(block.start, block.stop)

    def shape_for(self, block_name: str) -> tuple[int, ...]:
        return self._block(block_name).shape

    def extract(self, full_state: np.ndarray, block_name: str) -> np.ndarray:
        full_state = np.asarray(full_state, dtype=float)
        block = self._block(block_name)
        values = full_state[block.start:block.stop]
        return self.reshape_block(block_name, values)

    def insert(self, full_state: np.ndarray, block_name: str, block_values: np.ndarray) -> np.ndarray:
        full_state = np.asarray(full_state, dtype=float).copy()
        block = self._block(block_name)
        flat = self.flatten_block(block_name, block_values)
        full_state[block.start:block.stop] = flat
        return full_state

    def flatten_block(self, block_name: str, values: np.ndarray) -> np.ndarray:
        block = self._block(block_name)
        values = np.asarray(values, dtype=float)
        if values.shape != block.shape:
            raise ValueError("Block value shape mismatch.")
        return values.reshape(-1, order=block.vectorization_order)

    def reshape_block(self, block_name: str, values: np.ndarray) -> np.ndarray:
        block = self._block(block_name)
        values = np.asarray(values, dtype=float)
        if values.size != block.dimension:
            raise ValueError("Block vector length mismatch.")
        return values.reshape(block.shape, order=block.vectorization_order)

    def metadata(self) -> dict[str, object]:
        return {
            "total_dimension": self.total_dimension,
            "blocks": [
                {
                    "name": block.name,
                    "start": block.start,
                    "stop": block.stop,
                    "shape": block.shape,
                    "vectorization_order": block.vectorization_order,
                    "parameter_family": block.parameter_family,
                    "dynamic": block.dynamic,
                    "metadata": dict(block.metadata),
                }
                for block in self.blocks
            ],
        }

    def _block(self, block_name: str) -> StateBlockLayout:
        for block in self.blocks:
            if block.name == block_name:
                return block
        raise KeyError(block_name)


def make_state_layout(block_specs: list[dict[str, object]]) -> StateLayout:
    blocks = []
    start = 0
    for spec in block_specs:
        shape = tuple(spec["shape"])
        dimension = int(np.prod(shape, dtype=int))
        block = StateBlockLayout(
            name=str(spec["name"]),
            start=start,
            stop=start + dimension,
            shape=shape,
            vectorization_order=str(spec.get("vectorization_order", "C")),
            parameter_family=str(spec.get("parameter_family", "generic")),
            dynamic=bool(spec.get("dynamic", True)),
            metadata=dict(spec.get("metadata", {})),
        )
        blocks.append(block)
        start += dimension
    return StateLayout(blocks=tuple(blocks), total_dimension=start)

