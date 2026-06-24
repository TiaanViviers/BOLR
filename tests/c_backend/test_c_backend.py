from __future__ import annotations

import gc
import json
import multiprocessing
import subprocess
from pathlib import Path

import numpy as np
from bolr.backend.c_api import (
    BolrCError,
    PYTHON_BUILD_STAMP,
    PYTHON_LIB_PATH,
    ensure_debug_build,
    load_library,
)
from bolr.backend.c_backend import CBackend, CWorkspace
from bolr.decision.metrics import probability_best_brier, top_k_brier
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel, score_hvp
from bolr.targets.soft_target import SoftTargetBuilder


def _concurrent_build_worker(_: object) -> tuple[str, int]:
    lib_path = ensure_debug_build()
    lib = load_library()
    return (str(lib_path), int(lib.bolr_abi_version_major()))


def test_abi_and_library_load() -> None:
    ensure_debug_build()
    lib = load_library()
    assert lib.bolr_abi_version_major() == 1


def test_workspace_close_and_double_protection() -> None:
    workspace = CWorkspace(4, 4, 4)
    workspace.close()
    try:
        workspace._require_open()
    except BolrCError:
        pass
    else:
        raise AssertionError("closed handle should raise")


def test_automatic_finalization() -> None:
    workspace = CWorkspace(2, 2, 2)
    finalizer = workspace._finalizer
    del workspace
    gc.collect()
    assert not finalizer.alive


def test_backend_matches_composite_and_candidate_a_fixtures() -> None:
    backend = CBackend()
    arrays = np.load(Path("tests/fixtures/golden/composite_reference.npz"))
    score = backend.composite_forward(arrays["design"], arrays["posterior_mean"], arrays["static_scores"])
    assert np.allclose(score, arrays["design"] @ arrays["posterior_mean"] + arrays["static_scores"])

    pd_arrays = np.load(Path("tests/fixtures/golden/posterior_decision_reference.npz"))
    observation = SoftTargetBuilder().build(pd_arrays["utilities"])
    model = SoftTargetObservationModel()
    value, gradient, hvp = backend.observation_value_gradient_hvp(pd_arrays["score_mean"], observation, pd_arrays["score_mean"])
    assert np.isclose(value, model.log_factor(pd_arrays["score_mean"], observation))
    assert np.allclose(gradient, model.score_gradient(pd_arrays["score_mean"], observation))
    assert np.allclose(hvp, -score_hvp(pd_arrays["score_mean"], pd_arrays["score_mean"], observation))
    assert probability_best_brier(pd_arrays["probability_best"], pd_arrays["utilities"]) >= 0.0
    assert top_k_brier(pd_arrays["probability_top_2"], pd_arrays["utilities"], k=2) >= 0.0


def test_make_targets_exist() -> None:
    result = subprocess.run(["make", "-C", "csrc", "print-config"], check=True, capture_output=True, text=True)
    assert "BUILD=" in result.stdout


def test_ensure_debug_build_isolated_under_concurrency() -> None:
    ensure_debug_build()
    baseline_stamp = json.loads(PYTHON_BUILD_STAMP.read_text(encoding="utf-8"))
    baseline_count = int(baseline_stamp["build_count"])

    PYTHON_LIB_PATH.unlink()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_concurrent_build_worker, [()] * 4)

    rebuilt_stamp = json.loads(PYTHON_BUILD_STAMP.read_text(encoding="utf-8"))
    rebuilt_count = int(rebuilt_stamp["build_count"])
    assert rebuilt_count == baseline_count + 1
    assert all(Path(path) == PYTHON_LIB_PATH for path, _ in results)
    assert all(abi_version == 1 for _, abi_version in results)

    with ctx.Pool(processes=4) as pool:
        current_results = pool.map(_concurrent_build_worker, [()] * 4)

    current_stamp = json.loads(PYTHON_BUILD_STAMP.read_text(encoding="utf-8"))
    current_count = int(current_stamp["build_count"])
    assert current_count == rebuilt_count
    assert all(Path(path) == PYTHON_LIB_PATH for path, _ in current_results)
    assert all(abi_version == 1 for _, abi_version in current_results)
