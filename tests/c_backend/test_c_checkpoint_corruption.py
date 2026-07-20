from __future__ import annotations

import numpy as np
import pytest

from bolr.backend.c_api import CCheckpointCorruptionError, CCheckpointError
from bolr.backend.c_backend import CCheckpointState, CReplayEngine, CGaussianState, CLibrary, make_restore_context
from bolr.config.foundation import DecisionPolicyConfig
from bolr.posterior.state import GaussianPosterior
from tests.c_backend.test_c_replay import _candidate_a_model, _fixed_transition


def test_checkpoint_truncation_is_rejected() -> None:
    library = CLibrary()
    mean = np.array([0.25, -0.5], dtype=np.float64)
    covariance = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    with CGaussianState(mean, covariance, state_layout_hash=7, model_schema_hash=11, library=library) as state:
        with state.export_checkpoint() as checkpoint:
            payload = checkpoint.to_bytes()
    with pytest.raises(CCheckpointError):
        CCheckpointState.from_bytes(payload[:-3], library=library)


def _replay_pending_payload():
    from bolr.backend.c_backend import CBackend

    backend = CBackend()
    model = _candidate_a_model()
    prior = GaussianPosterior(mean=np.zeros(model.layout.total_dimension), covariance=np.eye(model.layout.total_dimension))
    transition = _fixed_transition(model.layout.total_dimension)
    artifacts = backend.model_artifacts(model, {})
    c_posterior = artifacts.state_from_posterior(prior)
    rng = backend.rng(seed=17, stream=1)
    engine = backend.replay_engine_fixed(c_posterior, transition, rng)
    decision_policy = backend.decision_policy(DecisionPolicyConfig(family="thompson"))
    engine.begin_day(
        artifacts,
        decision_policy,
        ranking_sample_count=16,
        chunk_size=4,
        top_k_values=(1,),
        antithetic=True,
        retention="sample_zero",
    )
    payload = engine.encode_checkpoint()
    context = make_restore_context(artifacts)
    engine.close()
    decision_policy.close()
    rng.close()
    c_posterior.close()
    artifacts.close()
    return payload, context


def test_c_replay_checkpoint_bad_magic_is_rejected() -> None:
    payload, context = _replay_pending_payload()
    corrupted = bytearray(payload)
    corrupted[0] ^= 0xFF
    with pytest.raises(CCheckpointCorruptionError):
        CReplayEngine.decode_checkpoint(bytes(corrupted), context)


def test_c_replay_checkpoint_checksum_mismatch_is_rejected() -> None:
    payload, context = _replay_pending_payload()
    corrupted = bytearray(payload)
    corrupted[64] ^= 0x01
    with pytest.raises(CCheckpointCorruptionError):
        CReplayEngine.decode_checkpoint(bytes(corrupted), context)


def test_c_replay_checkpoint_truncation_is_rejected() -> None:
    payload, context = _replay_pending_payload()
    with pytest.raises(CCheckpointCorruptionError):
        CReplayEngine.decode_checkpoint(payload[:-8], context)

