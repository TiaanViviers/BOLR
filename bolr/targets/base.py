from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np


class TargetBuilder(Protocol):
    def build(
        self,
        utilities: np.ndarray,
        *,
        date: object | None = None,
        candidate_metadata: object | None = None,
    ) -> object:
        ...

    def metadata(self) -> Mapping[str, object]:
        ...
