from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from bolr.model.score_blocks import ContextInteractionBlock, LinearDesignBlock, ScoreBlock
from bolr.model.state_layout import StateLayout, make_state_layout


@dataclass(frozen=True)
class CompositeScoreModel:
    static_blocks: tuple[ScoreBlock, ...]
    dynamic_blocks: tuple[ScoreBlock, ...]
    layout: StateLayout
    fixed_blocks: frozenset[str] = frozenset()

    @classmethod
    def from_blocks(
        cls,
        static_blocks: list[ScoreBlock],
        dynamic_blocks: list[ScoreBlock],
        sample_batch: object,
        fixed_blocks: set[str] | None = None,
    ) -> "CompositeScoreModel":
        block_specs = []
        for block in dynamic_blocks:
            if isinstance(block, ContextInteractionBlock):
                shape = block.state_shape_for_batch(sample_batch)
                order = "F"
            elif isinstance(block, LinearDesignBlock):
                shape = block.state_shape_for_batch(sample_batch)
                order = "C"
            else:
                shape = block.state_shape
                order = "C"
            block_specs.append(
                {
                    "name": block.name,
                    "shape": shape,
                    "vectorization_order": order,
                    "parameter_family": block.metadata().get("parameter_family", "generic"),
                    "dynamic": True,
                    "metadata": dict(block.metadata()),
                }
            )
        return cls(
            static_blocks=tuple(static_blocks),
            dynamic_blocks=tuple(dynamic_blocks),
            layout=make_state_layout(block_specs),
            fixed_blocks=frozenset(fixed_blocks or set()),
        )

    def static_scores(self, batch: object) -> np.ndarray:
        if not self.static_blocks:
            n = self.dynamic_blocks[0].design_matrix(batch).shape[0]
            return np.zeros(n, dtype=float)
        return sum((block.static_scores(batch) for block in self.static_blocks), start=np.zeros(self.static_blocks[0].static_scores(batch).shape[0], dtype=float))

    def dynamic_scores(self, batch: object, full_state: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.static_scores(batch).shape[0], dtype=float)
        for block in self.dynamic_blocks:
            block_state = self.layout.extract(full_state, block.name)
            scores += block.score_from_state(batch, block_state)
        return scores

    def scores(self, batch: object, full_state: np.ndarray) -> np.ndarray:
        return self.static_scores(batch) + self.dynamic_scores(batch, full_state)

    def transpose_multiply(self, batch: object, score_vector: np.ndarray) -> np.ndarray:
        parts = []
        for block in self.dynamic_blocks:
            if block.name in self.fixed_blocks:
                parts.append(np.zeros(self.layout.slice_for(block.name).stop - self.layout.slice_for(block.name).start, dtype=float))
            else:
                parts.append(block.transpose_multiply(batch, score_vector).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def explicit_design_matrix(self, batch: object) -> np.ndarray:
        matrices = []
        for block in self.dynamic_blocks:
            matrix = block.design_matrix(batch)
            if block.name in self.fixed_blocks:
                matrix = np.zeros_like(matrix)
            matrices.append(matrix)
        return np.hstack(matrices) if matrices else np.zeros((self.static_scores(batch).size, 0), dtype=float)

    def block_scores(self, batch: object, full_state: np.ndarray) -> dict[str, np.ndarray]:
        outputs = {block.name: block.static_scores(batch) for block in self.static_blocks}
        for block in self.dynamic_blocks:
            outputs[block.name] = block.score_from_state(batch, self.layout.extract(full_state, block.name))
        return outputs

    def state_layout(self) -> StateLayout:
        return self.layout

    def metadata(self) -> dict[str, object]:
        return {
            "static_blocks": [block.metadata() for block in self.static_blocks],
            "dynamic_blocks": [block.metadata() for block in self.dynamic_blocks],
            "fixed_blocks": sorted(self.fixed_blocks),
            "layout": self.layout.metadata(),
        }

    def parameter_hvp_from_score_curvature(self, batch: object, score_curvature: np.ndarray, vector: np.ndarray) -> np.ndarray:
        projected = self.explicit_design_matrix(batch) @ np.asarray(vector, dtype=float)
        curved = np.asarray(score_curvature, dtype=float) @ projected
        return self.transpose_multiply(batch, curved)

    def identifiability_diagnostics(self, batch: object, warn: bool = False) -> dict[str, object]:
        design = self.explicit_design_matrix(batch)
        singular_values = np.linalg.svd(design, compute_uv=False)
        rank = int(np.linalg.matrix_rank(design))
        positive = singular_values[singular_values > 0.0]
        condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
        cross_grams = {}
        warnings_list = []
        for a in self.dynamic_blocks:
            Xa = a.design_matrix(batch)
            if np.allclose(Xa, 0.0):
                warnings_list.append(f"{a.name}: zero design columns")
            if np.any(np.linalg.norm(Xa, axis=0) == 0.0):
                warnings_list.append(f"{a.name}: contains zero columns")
            for b in self.dynamic_blocks:
                Xb = b.design_matrix(batch)
                cross_grams[f"{a.name}|{b.name}"] = Xa.T @ Xb
            if isinstance(a, ContextInteractionBlock):
                context = np.asarray(batch["context_vector"] if isinstance(batch, dict) else getattr(batch, "context_vector"), dtype=float)
                if np.linalg.norm(context) < 1e-12:
                    warnings_list.append(f"{a.name}: near-zero context vector")
        if design.shape[1] != np.unique(np.round(design, 12), axis=1).shape[1]:
            warnings_list.append("duplicate design columns detected")
        if warn and warnings_list:
            for message in warnings_list:
                warnings.warn(message, RuntimeWarning)
        return {
            "rank": rank,
            "singular_values": singular_values,
            "condition_number": condition,
            "cross_grams": cross_grams,
            "warnings": warnings_list,
        }

