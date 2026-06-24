from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class ScoreBlock(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def dynamic(self) -> bool: ...
    @property
    def state_shape(self) -> tuple[int, ...]: ...
    @property
    def state_dimension(self) -> int: ...
    def static_scores(self, batch: object) -> np.ndarray: ...
    def score_from_state(self, batch: object, block_state: np.ndarray) -> np.ndarray: ...
    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray: ...
    def design_matrix(self, batch: object) -> np.ndarray: ...
    def metadata(self) -> Mapping[str, object]: ...


def _batch_get(batch: object, key: str) -> np.ndarray:
    if isinstance(batch, dict):
        return np.asarray(batch[key], dtype=float)
    return np.asarray(getattr(batch, key), dtype=float)


@dataclass(frozen=True)
class StaticBaselineBlock:
    name: str
    candidate_basis: np.ndarray
    coefficients: np.ndarray
    fit_metadata: Mapping[str, object]

    @property
    def dynamic(self) -> bool:
        return False

    @property
    def state_shape(self) -> tuple[int, ...]:
        return tuple()

    @property
    def state_dimension(self) -> int:
        return 0

    def static_scores(self, batch: object) -> np.ndarray:
        del batch
        return np.asarray(self.candidate_basis, dtype=float) @ np.asarray(self.coefficients, dtype=float)

    def score_from_state(self, batch: object, block_state: np.ndarray) -> np.ndarray:
        del batch, block_state
        return np.zeros(self.candidate_basis.shape[0], dtype=float)

    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray:
        del batch, score_vector
        return np.zeros(0, dtype=float)

    def design_matrix(self, batch: object) -> np.ndarray:
        del batch
        return np.zeros((self.candidate_basis.shape[0], 0), dtype=float)

    def metadata(self) -> Mapping[str, object]:
        return {"type": "static_baseline", **dict(self.fit_metadata)}


@dataclass(frozen=True)
class DynamicSurfaceBlock:
    name: str
    candidate_basis: np.ndarray
    parameter_family: str = "surface"

    @property
    def dynamic(self) -> bool:
        return True

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.candidate_basis.shape[1],)

    @property
    def state_dimension(self) -> int:
        return int(self.candidate_basis.shape[1])

    def static_scores(self, batch: object) -> np.ndarray:
        del batch
        return np.zeros(self.candidate_basis.shape[0], dtype=float)

    def score_from_state(self, batch: object, block_state: np.ndarray) -> np.ndarray:
        del batch
        return np.asarray(self.candidate_basis, dtype=float) @ np.asarray(block_state, dtype=float).reshape(-1)

    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray:
        del batch
        return np.asarray(self.candidate_basis, dtype=float).T @ np.asarray(score_vector, dtype=float)

    def design_matrix(self, batch: object) -> np.ndarray:
        del batch
        return np.asarray(self.candidate_basis, dtype=float)

    def metadata(self) -> Mapping[str, object]:
        return {"type": "dynamic_surface", "parameter_family": self.parameter_family}


@dataclass(frozen=True)
class ContextInteractionBlock:
    name: str
    candidate_basis: np.ndarray
    context_key: str = "context_vector"
    parameter_family: str = "context"

    @property
    def dynamic(self) -> bool:
        return True

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.candidate_basis.shape[1], 0)

    @property
    def state_dimension(self) -> int:
        raise RuntimeError("ContextInteractionBlock dimension depends on the configured context dimension.")

    def state_shape_for_batch(self, batch: object) -> tuple[int, ...]:
        context = _batch_get(batch, self.context_key)
        return (self.candidate_basis.shape[1], context.size)

    def state_dimension_for_batch(self, batch: object) -> int:
        shape = self.state_shape_for_batch(batch)
        return int(shape[0] * shape[1])

    def static_scores(self, batch: object) -> np.ndarray:
        del batch
        return np.zeros(self.candidate_basis.shape[0], dtype=float)

    def score_from_state(self, batch: object, block_state: np.ndarray) -> np.ndarray:
        context = _batch_get(batch, self.context_key)
        matrix = np.asarray(block_state, dtype=float).reshape((self.candidate_basis.shape[1], context.size), order="F")
        return np.asarray(self.candidate_basis, dtype=float) @ matrix @ context

    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray:
        context = _batch_get(batch, self.context_key)
        candidate_component = np.asarray(self.candidate_basis, dtype=float).T @ np.asarray(score_vector, dtype=float)
        return np.kron(context, candidate_component)

    def design_matrix(self, batch: object) -> np.ndarray:
        context = _batch_get(batch, self.context_key)
        return np.kron(context.reshape(1, -1), np.asarray(self.candidate_basis, dtype=float))

    def metadata(self) -> Mapping[str, object]:
        return {"type": "context_interaction", "parameter_family": self.parameter_family, "context_key": self.context_key}


@dataclass(frozen=True)
class LinearDesignBlock:
    name: str
    design_key: str
    parameter_family: str = "history"
    dynamic: bool = True

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (0,)

    @property
    def state_dimension(self) -> int:
        raise RuntimeError("LinearDesignBlock dimension depends on the supplied design matrix.")

    def state_shape_for_batch(self, batch: object) -> tuple[int, ...]:
        design = _batch_get(batch, self.design_key)
        return (design.shape[1],)

    def state_dimension_for_batch(self, batch: object) -> int:
        design = _batch_get(batch, self.design_key)
        return int(design.shape[1])

    def static_scores(self, batch: object) -> np.ndarray:
        design = _batch_get(batch, self.design_key)
        return np.zeros(design.shape[0], dtype=float)

    def score_from_state(self, batch: object, block_state: np.ndarray) -> np.ndarray:
        design = _batch_get(batch, self.design_key)
        return design @ np.asarray(block_state, dtype=float).reshape(-1)

    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray:
        design = _batch_get(batch, self.design_key)
        return design.T @ np.asarray(score_vector, dtype=float)

    def design_matrix(self, batch: object) -> np.ndarray:
        return _batch_get(batch, self.design_key)

    def metadata(self) -> Mapping[str, object]:
        return {"type": "linear_design", "parameter_family": self.parameter_family, "design_key": self.design_key}


@dataclass(frozen=True)
class SuppliedDesignBlock(LinearDesignBlock):
    parameter_family: str = "supplied"


@dataclass(frozen=True)
class GraphResidualBlock:
    name: str
    residual_basis: np.ndarray
    parameter_family: str = "graph_residual"

    @property
    def dynamic(self) -> bool:
        return True

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.residual_basis.shape[1],)

    @property
    def state_dimension(self) -> int:
        return int(self.residual_basis.shape[1])

    def static_scores(self, batch: object) -> np.ndarray:
        del batch
        return np.zeros(self.residual_basis.shape[0], dtype=float)

    def score_from_state(self, batch: object, block_state: np.ndarray) -> np.ndarray:
        del batch
        return np.asarray(self.residual_basis, dtype=float) @ np.asarray(block_state, dtype=float).reshape(-1)

    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray:
        del batch
        return np.asarray(self.residual_basis, dtype=float).T @ np.asarray(score_vector, dtype=float)

    def design_matrix(self, batch: object) -> np.ndarray:
        del batch
        return np.asarray(self.residual_basis, dtype=float)

    def metadata(self) -> Mapping[str, object]:
        return {
            "type": "graph_residual",
            "parameter_family": self.parameter_family,
            "residual_dimension": int(self.residual_basis.shape[1]),
        }
