from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from bolr.backend.base import NumericalBackend
from bolr.backend.c_api import CHandle, BolrCError, load_library, status_ok


class ConstVectorView(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int64), ("stride", ctypes.c_int64)]


class VectorView(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int64), ("stride", ctypes.c_int64)]


class ConstMatrixView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("rows", ctypes.c_int64),
        ("cols", ctypes.c_int64),
        ("row_stride", ctypes.c_int64),
        ("col_stride", ctypes.c_int64),
    ]


class MatrixView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("rows", ctypes.c_int64),
        ("cols", ctypes.c_int64),
        ("row_stride", ctypes.c_int64),
        ("col_stride", ctypes.c_int64),
    ]


class CholeskyDiagnostics(ctypes.Structure):
    _fields_ = [("jitter_used", ctypes.c_double), ("attempts", ctypes.c_int64), ("minimum_diagonal", ctypes.c_double)]


class TransitionConfig(ctypes.Structure):
    _fields_ = [
        ("family", ctypes.c_int32),
        ("process_noise", ConstMatrixView),
        ("global_discount", ctypes.c_double),
        ("block_discount_scales", ConstVectorView),
    ]


class PredictionDiagnostics(ctypes.Structure):
    _fields_ = [
        ("process_noise_trace", ctypes.c_double),
        ("predictive_covariance_trace", ctypes.c_double),
        ("minimum_cholesky_diagonal", ctypes.c_double),
        ("jitter_used", ctypes.c_double),
    ]


def _as_f64_vector(array: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError("Expected a contiguous float64 vector.")
    return arr


def _as_f64_matrix(array: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError("Expected a contiguous float64 matrix.")
    return arr


class CLibrary:
    def __init__(self) -> None:
        self.lib = load_library()
        self.lib.bolr_abi_version_major.restype = ctypes.c_uint32
        self.lib.bolr_abi_version_minor.restype = ctypes.c_uint32
        self.lib.bolr_abi_version_patch.restype = ctypes.c_uint32
        self.lib.bolr_matvec.argtypes = [ConstMatrixView, ConstVectorView, VectorView]
        self.lib.bolr_matvec.restype = ctypes.c_int32
        self.lib.bolr_matvec_transpose.argtypes = [ConstMatrixView, ConstVectorView, VectorView]
        self.lib.bolr_matvec_transpose.restype = ctypes.c_int32
        self.lib.bolr_candidate_a_log_factor.argtypes = [ConstVectorView, ConstVectorView, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_candidate_a_log_factor.restype = ctypes.c_int32
        self.lib.bolr_candidate_a_score_gradient.argtypes = [ConstVectorView, ConstVectorView, ctypes.c_double, VectorView]
        self.lib.bolr_candidate_a_score_gradient.restype = ctypes.c_int32
        self.lib.bolr_candidate_a_score_hvp.argtypes = [ConstVectorView, ConstVectorView, ctypes.c_double, VectorView]
        self.lib.bolr_candidate_a_score_hvp.restype = ctypes.c_int32
        self.lib.bolr_cholesky_factor.argtypes = [MatrixView, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.POINTER(CholeskyDiagnostics)]
        self.lib.bolr_cholesky_factor.restype = ctypes.c_int32
        self.lib.bolr_cholesky_solve.argtypes = [ConstMatrixView, ConstVectorView, VectorView]
        self.lib.bolr_cholesky_solve.restype = ctypes.c_int32
        self.lib.bolr_gaussian_state_create.argtypes = [ConstVectorView, ConstMatrixView, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_gaussian_state_create.restype = ctypes.c_int32
        self.lib.bolr_gaussian_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_gaussian_state_copy_mean.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_gaussian_state_copy_mean.restype = ctypes.c_int32
        self.lib.bolr_gaussian_state_copy_covariance.argtypes = [ctypes.c_void_p, MatrixView]
        self.lib.bolr_gaussian_state_copy_covariance.restype = ctypes.c_int32
        self.lib.bolr_gaussian_state_dimension.argtypes = [ctypes.c_void_p]
        self.lib.bolr_gaussian_state_dimension.restype = ctypes.c_int64
        self.lib.bolr_gaussian_state_step_index.argtypes = [ctypes.c_void_p]
        self.lib.bolr_gaussian_state_step_index.restype = ctypes.c_uint64
        self.lib.bolr_gaussian_state_export.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_gaussian_state_export.restype = ctypes.c_int32
        self.lib.bolr_gaussian_state_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_gaussian_state_import.restype = ctypes.c_int32
        self.lib.bolr_gaussian_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(TransitionConfig), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(PredictionDiagnostics)]
        self.lib.bolr_gaussian_predict.restype = ctypes.c_int32
        self.lib.bolr_checkpoint_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_checkpoint_encoded_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_checkpoint_encoded_size.restype = ctypes.c_int32
        self.lib.bolr_checkpoint_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_checkpoint_encode.restype = ctypes.c_int32
        self.lib.bolr_checkpoint_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_checkpoint_decode.restype = ctypes.c_int32
        if self.lib.bolr_abi_version_major() != 1:
            raise BolrCError("Unsupported BOLR ABI major version.")
        if self.lib.bolr_abi_version_minor() < 1:
            raise BolrCError("BOLR C backend is missing Phase L2 symbols.")

    @staticmethod
    def const_vector(array: np.ndarray) -> ConstVectorView:
        arr = _as_f64_vector(array)
        return ConstVectorView(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr.shape[0], 1)

    @staticmethod
    def vector(array: np.ndarray) -> VectorView:
        arr = _as_f64_vector(array)
        return VectorView(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr.shape[0], 1)

    @staticmethod
    def const_matrix(array: np.ndarray) -> ConstMatrixView:
        arr = _as_f64_matrix(array)
        return ConstMatrixView(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr.shape[0], arr.shape[1], arr.shape[1], 1)

    @staticmethod
    def matrix(array: np.ndarray) -> MatrixView:
        arr = _as_f64_matrix(array)
        return MatrixView(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr.shape[0], arr.shape[1], arr.shape[1], 1)


@dataclass
class CBackend(NumericalBackend):
    library: CLibrary = CLibrary()

    def block_forward(self, design: np.ndarray, state: np.ndarray) -> np.ndarray:
        design_arr = _as_f64_matrix(design)
        state_arr = _as_f64_vector(state)
        output = np.empty(design_arr.shape[0], dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_matvec(self.library.const_matrix(design_arr), self.library.const_vector(state_arr), self.library.vector(output)))
        return output

    def block_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        design_arr = _as_f64_matrix(design)
        score_arr = _as_f64_vector(score_vector)
        output = np.empty(design_arr.shape[1], dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_matvec_transpose(self.library.const_matrix(design_arr), self.library.const_vector(score_arr), self.library.vector(output)))
        return output

    def composite_forward(self, design: np.ndarray, state: np.ndarray, static_scores: np.ndarray | None = None) -> np.ndarray:
        result = self.block_forward(design, state)
        if static_scores is not None:
            result = result + _as_f64_vector(static_scores)
        return result

    def composite_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        return self.block_transpose(design, score_vector)

    def observation_value_gradient_hvp(self, scores: np.ndarray, observation: object, vector: np.ndarray | None = None):
        from bolr.targets.soft_target import Observation

        if not isinstance(observation, Observation):
            raise TypeError("CBackend Candidate A binding expects SoftTargetObservation.")
        scores_arr = _as_f64_vector(scores)
        target_arr = _as_f64_vector(observation.target_probabilities)
        vec_arr = np.zeros_like(scores_arr) if vector is None else _as_f64_vector(vector)
        gradient = np.empty_like(scores_arr)
        hvp = np.empty_like(scores_arr)
        value = ctypes.c_double()
        update_weight = float(observation.update_weight)
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_log_factor(self.library.const_vector(scores_arr), self.library.const_vector(target_arr), update_weight, ctypes.byref(value)))
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_score_gradient(self.library.const_vector(scores_arr), self.library.const_vector(target_arr), update_weight, self.library.vector(gradient)))
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_score_hvp(self.library.const_vector(scores_arr), self.library.const_vector(vec_arr), update_weight, self.library.vector(hvp)))
        return float(value.value), gradient, hvp

    def cholesky_factor(self, matrix: np.ndarray) -> np.ndarray:
        mat = _as_f64_matrix(matrix).copy()
        diagnostics = CholeskyDiagnostics()
        status_ok(self.library.lib, self.library.lib.bolr_cholesky_factor(self.library.matrix(mat), 1e-9, 10.0, 4, ctypes.byref(diagnostics)))
        return mat

    def cholesky_solve(self, factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        factor_arr = _as_f64_matrix(factor)
        rhs_arr = _as_f64_vector(rhs)
        output = np.empty_like(rhs_arr)
        status_ok(self.library.lib, self.library.lib.bolr_cholesky_solve(self.library.const_matrix(factor_arr), self.library.const_vector(rhs_arr), self.library.vector(output)))
        return output


class CWorkspace(CHandle):
    def __init__(self, score_capacity: int, state_capacity: int, context_capacity: int) -> None:
        lib = load_library()
        class Config(ctypes.Structure):
            _fields_ = [("score_capacity", ctypes.c_int64), ("state_capacity", ctypes.c_int64), ("context_capacity", ctypes.c_int64)]
        handle = ctypes.c_void_p()
        lib.bolr_workspace_create.argtypes = [ctypes.POINTER(Config), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.bolr_workspace_create.restype = ctypes.c_int32
        lib.bolr_workspace_destroy.argtypes = [ctypes.c_void_p]
        status_ok(lib, lib.bolr_workspace_create(ctypes.byref(Config(score_capacity, state_capacity, context_capacity)), None, ctypes.byref(handle)))
        super().__init__(handle, lib.bolr_workspace_destroy)


class CCheckpointState(CHandle):
    def __init__(self, handle: ctypes.c_void_p, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        super().__init__(handle, self.library.lib.bolr_checkpoint_state_destroy)

    def to_bytes(self) -> bytes:
        handle = self._require_open()
        size = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_checkpoint_encoded_size(handle, ctypes.byref(size)))
        buffer = (ctypes.c_ubyte * size.value)()
        written = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_checkpoint_encode(handle, ctypes.byref(buffer), size.value, ctypes.byref(written)))
        return bytes(buffer[: written.value])

    @classmethod
    def from_bytes(cls, payload: bytes, library: CLibrary | None = None) -> "CCheckpointState":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        raw = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        status_ok(lib.lib, lib.lib.bolr_checkpoint_decode(ctypes.byref(raw), len(payload), None, ctypes.byref(handle)))
        return cls(handle, library=lib)


class CGaussianState(CHandle):
    def __init__(
        self,
        mean: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        *,
        state_layout_hash: int | None = None,
        model_schema_hash: int | None = None,
        handle: ctypes.c_void_p | None = None,
        library: CLibrary | None = None,
    ) -> None:
        self.library = CLibrary() if library is None else library
        if handle is None:
            if mean is None or covariance is None or state_layout_hash is None or model_schema_hash is None:
                raise ValueError("Mean, covariance, and schema hashes are required when creating a new Gaussian state.")
            mean_arr = _as_f64_vector(mean)
            covariance_arr = _as_f64_matrix(covariance)
            created = ctypes.c_void_p()
            status_ok(
                self.library.lib,
                self.library.lib.bolr_gaussian_state_create(
                    self.library.const_vector(mean_arr),
                    self.library.const_matrix(covariance_arr),
                    state_layout_hash,
                    model_schema_hash,
                    None,
                    ctypes.byref(created),
                ),
            )
            handle = created
        super().__init__(handle, self.library.lib.bolr_gaussian_state_destroy)

    @property
    def dimension(self) -> int:
        return int(self.library.lib.bolr_gaussian_state_dimension(self._require_open()))

    @property
    def step_index(self) -> int:
        return int(self.library.lib.bolr_gaussian_state_step_index(self._require_open()))

    def mean(self) -> np.ndarray:
        output = np.empty(self.dimension, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_state_copy_mean(self._require_open(), self.library.vector(output)))
        return output

    def covariance(self) -> np.ndarray:
        output = np.empty((self.dimension, self.dimension), dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_state_copy_covariance(self._require_open(), self.library.matrix(output)))
        return output

    def export_checkpoint(self) -> CCheckpointState:
        checkpoint = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_state_export(self._require_open(), None, ctypes.byref(checkpoint)))
        return CCheckpointState(checkpoint, library=self.library)

    @classmethod
    def import_checkpoint(cls, checkpoint: CCheckpointState, library: CLibrary | None = None) -> "CGaussianState":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        status_ok(lib.lib, lib.lib.bolr_gaussian_state_import(checkpoint._require_open(), None, ctypes.byref(handle)))
        return cls(handle=handle, library=lib)

    def predict_additive(self, process_noise: np.ndarray) -> tuple["CGaussianState", PredictionDiagnostics]:
        process_noise_arr = _as_f64_matrix(process_noise)
        config = TransitionConfig(
            1,
            self.library.const_matrix(process_noise_arr),
            0.0,
            ConstVectorView(ctypes.POINTER(ctypes.c_double)(), 0, 1),
        )
        diagnostics = PredictionDiagnostics()
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_predict(self._require_open(), ctypes.byref(config), None, ctypes.byref(handle), ctypes.byref(diagnostics)))
        return CGaussianState(handle=handle, library=self.library), diagnostics
