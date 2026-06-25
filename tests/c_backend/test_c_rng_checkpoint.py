from __future__ import annotations

import pytest

from bolr.backend.c_api import CCheckpointError
from bolr.backend.c_backend import CRNG, CRNGCheckpoint


def test_c_rng_checkpoint_round_trip_and_continuation() -> None:
    with CRNG(seed=123, stream=7) as rng:
        prefix = [rng.u32() for _ in range(6)]
        assert len(prefix) == 6
        with rng.export_checkpoint() as checkpoint:
            payload = checkpoint.to_bytes()
            with CRNGCheckpoint.from_bytes(payload) as decoded:
                with CRNG.import_checkpoint(decoded) as restored:
                    left = [rng.normal() for _ in range(8)]
                    right = [restored.normal() for _ in range(8)]
                    assert left == right


def test_c_rng_checkpoint_rejects_corruption() -> None:
    with CRNG(seed=123, stream=7) as rng:
        with rng.export_checkpoint() as checkpoint:
            payload = bytearray(checkpoint.to_bytes())
    payload[16] &= 0xFE
    with pytest.raises(CCheckpointError):
        CRNGCheckpoint.from_bytes(bytes(payload))
