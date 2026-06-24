from __future__ import annotations

import ctypes
import subprocess
import weakref
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSRC_DIR = PROJECT_ROOT / "csrc"


class BolrCError(RuntimeError):
    pass


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
            raise BolrCError("C handle is already closed.")
        return self._handle

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def ensure_debug_build() -> Path:
    lib_path = CSRC_DIR / "build" / "debug" / "libbolr.so"
    subprocess.run(["make", "-C", str(CSRC_DIR), "clean", "debug"], check=True)
    return lib_path


def load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(ensure_debug_build()))
    lib.bolr_status_string.argtypes = [ctypes.c_int32]
    lib.bolr_status_string.restype = ctypes.c_char_p
    return lib


def status_ok(lib: ctypes.CDLL, code: int) -> None:
    if code != 0:
        message = lib.bolr_status_string(code).decode("utf-8")
        raise BolrCError(message)
