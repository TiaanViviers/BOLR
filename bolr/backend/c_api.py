from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import shlex
import subprocess
import weakref
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSRC_DIR = PROJECT_ROOT / "csrc"
PYTHON_BUILD_PROFILE = "python-debug-gcc"
PYTHON_BUILD_CC = "gcc"
PYTHON_BUILD_DIR = Path("build") / PYTHON_BUILD_PROFILE
PYTHON_BUILD_OUT_DIR = CSRC_DIR / PYTHON_BUILD_DIR / "debug"
PYTHON_LIB_PATH = PYTHON_BUILD_OUT_DIR / "libbolr.so"
PYTHON_BUILD_LOCK = CSRC_DIR / "build" / ".locks" / f"{PYTHON_BUILD_PROFILE}.lock"
PYTHON_BUILD_STAMP = PYTHON_BUILD_OUT_DIR / "build_stamp.json"


class CBackendError(RuntimeError):
    pass


class CInvalidArgumentError(CBackendError):
    pass


class CShapeError(CBackendError):
    pass


class CNonFiniteInputError(CBackendError):
    pass


class CAllocationError(CBackendError):
    pass


class CPositiveDefiniteError(CBackendError):
    pass


class CNumericalFailureError(CBackendError):
    pass


class CSchemaMismatchError(CBackendError):
    pass


class CVersionMismatchError(CBackendError):
    pass


class CCheckpointError(CBackendError):
    pass


class CCheckpointVersionError(CCheckpointError):
    pass


class CCheckpointCorruptionError(CCheckpointError):
    pass


class CCheckpointCompatibilityError(CCheckpointError):
    pass


class CCheckpointLimitError(CCheckpointError):
    pass


class CCheckpointIOError(CCheckpointError):
    pass


class CUnsupportedOperationError(CBackendError):
    pass


class CClosedHandleError(CBackendError):
    pass


_STATUS_TO_EXCEPTION = {
    1: CInvalidArgumentError,
    2: CShapeError,
    3: CNonFiniteInputError,
    4: CAllocationError,
    6: CPositiveDefiniteError,
    8: CNumericalFailureError,
    9: CSchemaMismatchError,
    10: CVersionMismatchError,
    11: CCheckpointCompatibilityError,
    12: CUnsupportedOperationError,
    13: CClosedHandleError,
    14: CCheckpointCorruptionError,
    15: CCheckpointVersionError,
    16: CCheckpointCorruptionError,
    17: CCheckpointCorruptionError,
    18: CCheckpointCorruptionError,
    19: CCheckpointCorruptionError,
    20: CCheckpointCorruptionError,
    21: CCheckpointLimitError,
    22: CCheckpointIOError,
}

BolrCError = CBackendError


class CHandle:
    def __init__(self, handle, destroy):
        self._handle = handle
        self._destroy = destroy
        self._closed = False
        self._finalizer = weakref.finalize(self, self._finalize_handle, handle, destroy)

    @staticmethod
    def _finalize_handle(handle, destroy) -> None:
        if handle:
            destroy(handle)

    def close(self) -> None:
        if not self._closed and self._handle:
            self._destroy(self._handle)
            self._closed = True
            self._finalizer.detach()

    def _require_open(self):
        if self._closed:
            raise CClosedHandleError("C handle is already closed.")
        return self._handle

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _run_make(*targets: str) -> None:
    subprocess.run(
        [
            "make",
            "-C",
            str(CSRC_DIR),
            f"BUILD_DIR={PYTHON_BUILD_DIR}",
            "BUILD=debug",
            f"CC={PYTHON_BUILD_CC}",
            *targets,
        ],
        check=True,
    )


def _read_make_config() -> dict[str, str]:
    result = subprocess.run(
        [
            "make",
            "-C",
            str(CSRC_DIR),
            f"BUILD_DIR={PYTHON_BUILD_DIR}",
            "BUILD=debug",
            f"CC={PYTHON_BUILD_CC}",
            "print-config",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    config: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key] = value
    return config


def _compiler_identity(cc: str) -> str:
    try:
        result = subprocess.run(
            [*shlex.split(cc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return cc
    return result.stdout.strip().splitlines()[0]


def _source_tree_digest() -> str:
    digest = hashlib.sha256()
    tracked_paths = sorted((CSRC_DIR / "src").glob("*.c"))
    tracked_paths += sorted((CSRC_DIR / "include" / "bolr").glob("*.h"))
    tracked_paths.append(CSRC_DIR / "Makefile")
    config_mk = CSRC_DIR / "config.mk"
    if config_mk.exists():
        tracked_paths.append(config_mk)
    for path in tracked_paths:
        digest.update(path.relative_to(CSRC_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _desired_build_stamp(config: dict[str, str]) -> dict[str, Any]:
    return {
        "abi_version_major": 1,
        "build": config.get("BUILD", "debug"),
        "build_dir": config.get("BUILD_DIR", str(PYTHON_BUILD_DIR)),
        "out_dir": config.get("OUT_DIR", ""),
        "lib_shared": config.get("LIB_SHARED", ""),
        "cc": config.get("CC", "gcc"),
        "compiler_identity": _compiler_identity(config.get("CC", "gcc")),
        "all_cflags": config.get("ALL_CFLAGS", ""),
        "ldflags": config.get("LDFLAGS", ""),
        "all_ldlibs": config.get("ALL_LDLIBS", ""),
        "source_tree_digest": _source_tree_digest(),
    }


def _load_build_stamp() -> dict[str, Any] | None:
    if not PYTHON_BUILD_STAMP.exists():
        return None
    return json.loads(PYTHON_BUILD_STAMP.read_text(encoding="utf-8"))


def _write_build_stamp(stamp: dict[str, Any]) -> None:
    PYTHON_BUILD_STAMP.parent.mkdir(parents=True, exist_ok=True)
    temp_path = PYTHON_BUILD_STAMP.with_suffix(".tmp")
    temp_path.write_text(json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(PYTHON_BUILD_STAMP)


def _build_is_current(desired: dict[str, Any], actual: dict[str, Any] | None) -> bool:
    if actual is None or not PYTHON_LIB_PATH.exists():
        return False
    for key, value in desired.items():
        if actual.get(key) != value:
            return False
    return True


def _rebuild_python_debug_library(desired: dict[str, Any], previous_count: int) -> Path:
    _run_make("clean")
    _run_make("shared")
    stamp = dict(desired)
    stamp["build_count"] = previous_count + 1
    _write_build_stamp(stamp)
    return PYTHON_LIB_PATH


def _lock_python_build():
    PYTHON_BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = PYTHON_BUILD_LOCK.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return lock_handle


def ensure_debug_build() -> Path:
    lock_handle = _lock_python_build()
    try:
        config = _read_make_config()
        desired = _desired_build_stamp(config)
        current = _load_build_stamp()
        if _build_is_current(desired, current):
            return PYTHON_LIB_PATH
        previous_count = int(current.get("build_count", 0)) if current else 0
        return _rebuild_python_debug_library(desired, previous_count)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


@lru_cache(maxsize=1)
def load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(ensure_debug_build()))
    lib.bolr_status_string.argtypes = [ctypes.c_int32]
    lib.bolr_status_string.restype = ctypes.c_char_p
    return lib


def status_ok(lib: ctypes.CDLL, code: int, *, operation: str | None = None) -> None:
    if code == 0:
        return
    status_name = lib.bolr_status_string(code).decode("utf-8")
    prefix = f"{operation}: " if operation else ""
    exc_type = _STATUS_TO_EXCEPTION.get(code, CBackendError)
    raise exc_type(f"{prefix}{status_name}")
