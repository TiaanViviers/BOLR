from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from bolr.backend.base import NumericalBackend
from bolr.backend.c_api import CBackendError, CHandle, load_library, status_ok
from bolr.posterior.state import GaussianPosterior


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


class PredictionDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("process_noise_trace", ctypes.c_double),
        ("predictive_covariance_trace", ctypes.c_double),
        ("minimum_cholesky_diagonal", ctypes.c_double),
        ("jitter_used", ctypes.c_double),
    ]


class NewtonConfigStruct(ctypes.Structure):
    _fields_ = [
        ("maximum_iterations", ctypes.c_int64),
        ("gradient_tolerance", ctypes.c_double),
        ("step_tolerance", ctypes.c_double),
        ("objective_tolerance", ctypes.c_double),
        ("initial_damping", ctypes.c_double),
        ("damping_multiplier", ctypes.c_double),
        ("maximum_damping", ctypes.c_double),
        ("armijo_constant", ctypes.c_double),
        ("line_search_reduction", ctypes.c_double),
        ("maximum_line_search_steps", ctypes.c_int64),
        ("cholesky_initial_jitter", ctypes.c_double),
        ("cholesky_jitter_multiplier", ctypes.c_double),
        ("maximum_cholesky_attempts", ctypes.c_int64),
    ]


class NewtonDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("iterations", ctypes.c_int64),
        ("objective_evaluations", ctypes.c_int64),
        ("gradient_evaluations", ctypes.c_int64),
        ("hessian_evaluations", ctypes.c_int64),
        ("line_search_evaluations", ctypes.c_int64),
        ("initial_objective", ctypes.c_double),
        ("final_objective", ctypes.c_double),
        ("final_gradient_norm", ctypes.c_double),
        ("final_step_norm", ctypes.c_double),
        ("final_damping", ctypes.c_double),
        ("maximum_jitter_used", ctypes.c_double),
        ("converged", ctypes.c_int),
        ("used_damping", ctypes.c_int),
        ("used_jitter", ctypes.c_int),
    ]


class LaplaceDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("newton", NewtonDiagnosticsStruct),
        ("prior_covariance_trace", ctypes.c_double),
        ("posterior_covariance_trace", ctypes.c_double),
        ("posterior_covariance_log_determinant", ctypes.c_double),
        ("posterior_condition_estimate", ctypes.c_double),
        ("mean_update_norm", ctypes.c_double),
        ("mahalanobis_update_norm", ctypes.c_double),
        ("log_factor_at_predictive_mean", ctypes.c_double),
        ("log_factor_at_posterior_mode", ctypes.c_double),
        ("objective_improvement", ctypes.c_double),
        ("score_mean_min", ctypes.c_double),
        ("score_mean_max", ctypes.c_double),
        ("gradient_sum_diagnostic", ctypes.c_double),
        ("curvature_null_direction_diagnostic", ctypes.c_double),
    ]


class StateBlockSpecStruct(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("start", ctypes.c_int64),
        ("stop", ctypes.c_int64),
        ("rows", ctypes.c_int64),
        ("cols", ctypes.c_int64),
        ("dynamic", ctypes.c_int),
        ("vectorization_order", ctypes.c_char),
    ]


ObservationValueFn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ConstVectorView, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
ObservationGradientFn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ConstVectorView, VectorView, ctypes.c_void_p)
ObservationCurvatureHvpFn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ConstVectorView, ConstVectorView, VectorView, ctypes.c_void_p)


class ObservationOperatorStruct(ctypes.Structure):
    _fields_ = [
        ("value", ObservationValueFn),
        ("gradient", ObservationGradientFn),
        ("curvature_hvp", ObservationCurvatureHvpFn),
        ("context", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class PredictionDiagnostics:
    process_noise_trace: float
    predictive_covariance_trace: float
    minimum_cholesky_diagonal: float
    jitter_used: float

    @classmethod
    def from_c(cls, struct: PredictionDiagnosticsStruct) -> "PredictionDiagnostics":
        return cls(
            process_noise_trace=float(struct.process_noise_trace),
            predictive_covariance_trace=float(struct.predictive_covariance_trace),
            minimum_cholesky_diagonal=float(struct.minimum_cholesky_diagonal),
            jitter_used=float(struct.jitter_used),
        )


@dataclass(frozen=True)
class CNewtonConfig:
    maximum_iterations: int = 25
    gradient_tolerance: float = 1e-8
    step_tolerance: float = 1e-8
    objective_tolerance: float = 0.0
    initial_damping: float = 0.0
    damping_multiplier: float = 10.0
    maximum_damping: float = 1e12
    armijo_constant: float = 1e-4
    line_search_reduction: float = 0.5
    maximum_line_search_steps: int = 12
    initial_jitter: float = 1e-10
    jitter_multiplier: float = 10.0
    maximum_cholesky_attempts: int = 8

    @classmethod
    def from_python_options(cls, options: Any) -> "CNewtonConfig":
        return cls(
            maximum_iterations=int(options.max_iterations),
            gradient_tolerance=float(options.gradient_tolerance),
            step_tolerance=float(options.step_tolerance),
            initial_damping=float(options.initial_damping),
            line_search_reduction=float(options.line_search_shrinkage),
            maximum_line_search_steps=int(options.max_backtracking_steps),
            initial_jitter=float(options.initial_jitter),
            maximum_cholesky_attempts=int(options.max_jitter_attempts),
        )

    def to_c(self) -> NewtonConfigStruct:
        return NewtonConfigStruct(
            self.maximum_iterations,
            self.gradient_tolerance,
            self.step_tolerance,
            self.objective_tolerance,
            self.initial_damping,
            self.damping_multiplier,
            self.maximum_damping,
            self.armijo_constant,
            self.line_search_reduction,
            self.maximum_line_search_steps,
            self.initial_jitter,
            self.jitter_multiplier,
            self.maximum_cholesky_attempts,
        )


@dataclass(frozen=True)
class CNewtonDiagnostics:
    status: int
    iterations: int
    objective_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    line_search_evaluations: int
    initial_objective: float
    final_objective: float
    final_gradient_norm: float
    final_step_norm: float
    final_damping: float
    maximum_jitter_used: float
    converged: bool
    used_damping: bool
    used_jitter: bool

    @classmethod
    def from_c(cls, struct: NewtonDiagnosticsStruct) -> "CNewtonDiagnostics":
        return cls(
            status=int(struct.status),
            iterations=int(struct.iterations),
            objective_evaluations=int(struct.objective_evaluations),
            gradient_evaluations=int(struct.gradient_evaluations),
            hessian_evaluations=int(struct.hessian_evaluations),
            line_search_evaluations=int(struct.line_search_evaluations),
            initial_objective=float(struct.initial_objective),
            final_objective=float(struct.final_objective),
            final_gradient_norm=float(struct.final_gradient_norm),
            final_step_norm=float(struct.final_step_norm),
            final_damping=float(struct.final_damping),
            maximum_jitter_used=float(struct.maximum_jitter_used),
            converged=bool(struct.converged),
            used_damping=bool(struct.used_damping),
            used_jitter=bool(struct.used_jitter),
        )


@dataclass(frozen=True)
class CLaplaceDiagnostics:
    newton: CNewtonDiagnostics
    prior_covariance_trace: float
    posterior_covariance_trace: float
    posterior_covariance_log_determinant: float
    posterior_condition_estimate: float
    mean_update_norm: float
    mahalanobis_update_norm: float
    log_factor_at_predictive_mean: float
    log_factor_at_posterior_mode: float
    objective_improvement: float
    score_mean_min: float
    score_mean_max: float
    gradient_sum_diagnostic: float
    curvature_null_direction_diagnostic: float

    @classmethod
    def from_c(cls, struct: LaplaceDiagnosticsStruct) -> "CLaplaceDiagnostics":
        return cls(
            newton=CNewtonDiagnostics.from_c(struct.newton),
            prior_covariance_trace=float(struct.prior_covariance_trace),
            posterior_covariance_trace=float(struct.posterior_covariance_trace),
            posterior_covariance_log_determinant=float(struct.posterior_covariance_log_determinant),
            posterior_condition_estimate=float(struct.posterior_condition_estimate),
            mean_update_norm=float(struct.mean_update_norm),
            mahalanobis_update_norm=float(struct.mahalanobis_update_norm),
            log_factor_at_predictive_mean=float(struct.log_factor_at_predictive_mean),
            log_factor_at_posterior_mode=float(struct.log_factor_at_posterior_mode),
            objective_improvement=float(struct.objective_improvement),
            score_mean_min=float(struct.score_mean_min),
            score_mean_max=float(struct.score_mean_max),
            gradient_sum_diagnostic=float(struct.gradient_sum_diagnostic),
            curvature_null_direction_diagnostic=float(struct.curvature_null_direction_diagnostic),
        )


def _as_f64_vector(array: np.ndarray | list[float]) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError("Expected a contiguous float64 vector.")
    return arr


def _as_f64_matrix(array: np.ndarray | list[list[float]]) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError("Expected a contiguous float64 matrix.")
    return arr


def _null_const_vector() -> ConstVectorView:
    return ConstVectorView(ctypes.POINTER(ctypes.c_double)(), 0, 1)


def _block_rows_cols(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 1:
        return int(shape[0]), 1
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported block shape {shape!r} for C state layout.")


class CLibrary:
    def __init__(self) -> None:
        self.lib = load_library()
        self.lib.bolr_abi_version_major.restype = ctypes.c_uint32
        self.lib.bolr_abi_version_minor.restype = ctypes.c_uint32
        self.lib.bolr_abi_version_patch.restype = ctypes.c_uint32
        self.lib.bolr_status_string.argtypes = [ctypes.c_int32]
        self.lib.bolr_status_string.restype = ctypes.c_char_p

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

        self.lib.bolr_state_layout_create.argtypes = [ctypes.POINTER(StateBlockSpecStruct), ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_state_layout_create.restype = ctypes.c_int32
        self.lib.bolr_state_layout_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_state_layout_schema_hash.argtypes = [ctypes.c_void_p]
        self.lib.bolr_state_layout_schema_hash.restype = ctypes.c_uint64
        self.lib.bolr_state_layout_total_dimension.argtypes = [ctypes.c_void_p]
        self.lib.bolr_state_layout_total_dimension.restype = ctypes.c_int64

        self.lib.bolr_model_create.argtypes = [ctypes.c_void_p, ConstVectorView, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_model_create.restype = ctypes.c_int32
        self.lib.bolr_model_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_model_add_dense_block_copy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ConstMatrixView]
        self.lib.bolr_model_add_dense_block_copy.restype = ctypes.c_int32
        self.lib.bolr_model_add_context_block_copy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ConstMatrixView, ctypes.c_int64]
        self.lib.bolr_model_add_context_block_copy.restype = ctypes.c_int32
        self.lib.bolr_model_add_graph_block_copy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ConstMatrixView]
        self.lib.bolr_model_add_graph_block_copy.restype = ctypes.c_int32
        self.lib.bolr_model_forward.argtypes = [ctypes.c_void_p, ConstVectorView, ConstVectorView, VectorView, ctypes.c_void_p]
        self.lib.bolr_model_forward.restype = ctypes.c_int32
        self.lib.bolr_model_dynamic_forward.argtypes = [ctypes.c_void_p, ConstVectorView, ConstVectorView, VectorView, ctypes.c_void_p]
        self.lib.bolr_model_dynamic_forward.restype = ctypes.c_int32
        self.lib.bolr_model_transpose.argtypes = [ctypes.c_void_p, ConstVectorView, ConstVectorView, VectorView, ctypes.c_void_p]
        self.lib.bolr_model_transpose.restype = ctypes.c_int32
        self.lib.bolr_model_score_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_model_score_count.restype = ctypes.c_int64
        self.lib.bolr_model_state_dim.argtypes = [ctypes.c_void_p]
        self.lib.bolr_model_state_dim.restype = ctypes.c_int64
        self.lib.bolr_model_schema_hash.argtypes = [ctypes.c_void_p]
        self.lib.bolr_model_schema_hash.restype = ctypes.c_uint64
        self.lib.bolr_model_state_layout_hash.argtypes = [ctypes.c_void_p]
        self.lib.bolr_model_state_layout_hash.restype = ctypes.c_uint64
        self.lib.bolr_model_copy_static_scores.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_model_copy_static_scores.restype = ctypes.c_int32

        self.lib.bolr_workspace_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_workspace_create.restype = ctypes.c_int32
        self.lib.bolr_workspace_destroy.argtypes = [ctypes.c_void_p]

        self.lib.bolr_inference_workspace_create.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_inference_workspace_create.restype = ctypes.c_int32
        self.lib.bolr_inference_workspace_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_inference_workspace_state_dimension.argtypes = [ctypes.c_void_p]
        self.lib.bolr_inference_workspace_state_dimension.restype = ctypes.c_int64
        self.lib.bolr_inference_workspace_candidate_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_inference_workspace_candidate_count.restype = ctypes.c_int64

        self.lib.bolr_candidate_a_observation_create.argtypes = [ConstVectorView, ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_candidate_a_observation_create.restype = ctypes.c_int32
        self.lib.bolr_candidate_a_observation_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_candidate_a_observation_operator.argtypes = [ctypes.c_void_p, ctypes.POINTER(ObservationOperatorStruct)]
        self.lib.bolr_candidate_a_observation_operator.restype = ctypes.c_int32

        self.lib.bolr_laplace_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ConstVectorView, ctypes.POINTER(ObservationOperatorStruct), ctypes.POINTER(NewtonConfigStruct), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(LaplaceDiagnosticsStruct)]
        self.lib.bolr_laplace_update.restype = ctypes.c_int32

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
        self.lib.bolr_gaussian_state_state_layout_hash.argtypes = [ctypes.c_void_p]
        self.lib.bolr_gaussian_state_state_layout_hash.restype = ctypes.c_uint64
        self.lib.bolr_gaussian_state_model_schema_hash.argtypes = [ctypes.c_void_p]
        self.lib.bolr_gaussian_state_model_schema_hash.restype = ctypes.c_uint64
        self.lib.bolr_gaussian_state_export.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_gaussian_state_export.restype = ctypes.c_int32
        self.lib.bolr_gaussian_state_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_gaussian_state_import.restype = ctypes.c_int32
        self.lib.bolr_gaussian_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(TransitionConfig), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(PredictionDiagnosticsStruct)]
        self.lib.bolr_gaussian_predict.restype = ctypes.c_int32

        self.lib.bolr_checkpoint_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_checkpoint_encoded_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_checkpoint_encoded_size.restype = ctypes.c_int32
        self.lib.bolr_checkpoint_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_checkpoint_encode.restype = ctypes.c_int32
        self.lib.bolr_checkpoint_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_checkpoint_decode.restype = ctypes.c_int32

        self.lib.bolr_candidate_a_static_dataset_create.argtypes = [ConstMatrixView, ctypes.POINTER(ConstVectorView), ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_candidate_a_static_dataset_create.restype = ctypes.c_int32
        self.lib.bolr_candidate_a_static_dataset_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_candidate_a_static_fit.argtypes = [ctypes.c_void_p, ConstVectorView, ConstMatrixView, ctypes.POINTER(NewtonConfigStruct), ctypes.c_void_p, VectorView, VectorView, ctypes.c_void_p]
        self.lib.bolr_candidate_a_static_fit.restype = ctypes.c_int32

        if self.lib.bolr_abi_version_major() != 1:
            raise CBackendError("Unsupported BOLR ABI major version.")
        if self.lib.bolr_abi_version_minor() < 1:
            raise CBackendError("BOLR C backend is missing Phase L2 symbols.")

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


class CWorkspace(CHandle):
    def __init__(self, score_capacity: int, state_capacity: int, context_capacity: int, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        class Config(ctypes.Structure):
            _fields_ = [("score_capacity", ctypes.c_int64), ("state_capacity", ctypes.c_int64), ("context_capacity", ctypes.c_int64)]
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_workspace_create(ctypes.byref(Config(score_capacity, state_capacity, context_capacity)), None, ctypes.byref(handle)),
            operation="bolr_workspace_create",
        )
        super().__init__(handle, self.library.lib.bolr_workspace_destroy)


class CStateLayout(CHandle):
    def __init__(self, state_layout: Any, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.python_layout = state_layout
        self._name_buffers = [block.name.encode("utf-8") for block in state_layout.blocks]
        specs = []
        for idx, block in enumerate(state_layout.blocks):
            rows, cols = _block_rows_cols(block.shape)
            specs.append(
                StateBlockSpecStruct(
                    self._name_buffers[idx],
                    block.start,
                    block.stop,
                    rows,
                    cols,
                    int(block.dynamic),
                    block.vectorization_order.encode("ascii"),
                )
            )
        self._spec_array = (StateBlockSpecStruct * len(specs))(*specs)
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_state_layout_create(self._spec_array, len(specs), None, ctypes.byref(handle)),
            operation="bolr_state_layout_create",
        )
        super().__init__(handle, self.library.lib.bolr_state_layout_destroy)

    @property
    def hash(self) -> int:
        return int(self.library.lib.bolr_state_layout_schema_hash(self._require_open()))

    @property
    def total_dimension(self) -> int:
        return int(self.library.lib.bolr_state_layout_total_dimension(self._require_open()))


class CScoreContext:
    def __init__(self, context_vector: np.ndarray | None = None, *, owners: tuple[object, ...] = ()) -> None:
        self.context_vector = np.zeros(0, dtype=np.float64) if context_vector is None else _as_f64_vector(context_vector)
        self._owners = owners

    @classmethod
    def from_composite(cls, model: Any, batch: object) -> "CScoreContext":
        from bolr.model.score_blocks import ContextInteractionBlock

        keys = {block.context_key for block in model.dynamic_blocks if isinstance(block, ContextInteractionBlock)}
        if not keys:
            return cls()
        if len(keys) != 1:
            raise ValueError("C backend currently supports one shared context vector across context blocks.")
        key = next(iter(keys))
        if isinstance(batch, dict):
            vector = batch[key]
        else:
            vector = getattr(batch, key)
        return cls(vector, owners=(batch,))

    def const_view(self, library: CLibrary) -> ConstVectorView:
        return _null_const_vector() if self.context_vector.size == 0 else library.const_vector(self.context_vector)


class CInferenceWorkspace(CHandle):
    def __init__(self, state_dimension: int, candidate_count: int, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_inference_workspace_create(state_dimension, candidate_count, None, ctypes.byref(handle)),
            operation="bolr_inference_workspace_create",
        )
        super().__init__(handle, self.library.lib.bolr_inference_workspace_destroy)

    @property
    def state_dimension(self) -> int:
        return int(self.library.lib.bolr_inference_workspace_state_dimension(self._require_open()))

    @property
    def candidate_count(self) -> int:
        return int(self.library.lib.bolr_inference_workspace_candidate_count(self._require_open()))


class CCheckpointState(CHandle):
    def __init__(self, handle: ctypes.c_void_p, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        super().__init__(handle, self.library.lib.bolr_checkpoint_state_destroy)

    def to_bytes(self) -> bytes:
        size = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_checkpoint_encoded_size(self._require_open(), ctypes.byref(size)), operation="bolr_checkpoint_encoded_size")
        buffer = (ctypes.c_ubyte * size.value)()
        written = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_checkpoint_encode(self._require_open(), ctypes.byref(buffer), size.value, ctypes.byref(written)), operation="bolr_checkpoint_encode")
        return bytes(buffer[: written.value])

    @classmethod
    def from_bytes(cls, payload: bytes, *, library: CLibrary | None = None) -> "CCheckpointState":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        raw = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        status_ok(lib.lib, lib.lib.bolr_checkpoint_decode(ctypes.byref(raw), len(payload), None, ctypes.byref(handle)), operation="bolr_checkpoint_decode")
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
        owner_refs: tuple[object, ...] = (),
    ) -> None:
        self.library = CLibrary() if library is None else library
        self._owners = owner_refs
        if handle is None:
            if mean is None or covariance is None or state_layout_hash is None or model_schema_hash is None:
                raise ValueError("Mean, covariance, and schema hashes are required when creating a Gaussian state.")
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
                operation="bolr_gaussian_state_create",
            )
            handle = created
        super().__init__(handle, self.library.lib.bolr_gaussian_state_destroy)

    @property
    def dimension(self) -> int:
        return int(self.library.lib.bolr_gaussian_state_dimension(self._require_open()))

    @property
    def step_index(self) -> int:
        return int(self.library.lib.bolr_gaussian_state_step_index(self._require_open()))

    @property
    def state_layout_hash(self) -> int:
        return int(self.library.lib.bolr_gaussian_state_state_layout_hash(self._require_open()))

    @property
    def model_schema_hash(self) -> int:
        return int(self.library.lib.bolr_gaussian_state_model_schema_hash(self._require_open()))

    def mean(self) -> np.ndarray:
        output = np.empty(self.dimension, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_state_copy_mean(self._require_open(), self.library.vector(output)), operation="bolr_gaussian_state_copy_mean")
        return output

    def covariance(self) -> np.ndarray:
        output = np.empty((self.dimension, self.dimension), dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_state_copy_covariance(self._require_open(), self.library.matrix(output)), operation="bolr_gaussian_state_copy_covariance")
        return output

    def to_posterior(self, *, state_layout: dict[str, Any] | None = None, timestamp: str | None = None, version: str = "c_backend_l2") -> GaussianPosterior:
        return GaussianPosterior(mean=self.mean(), covariance=self.covariance(), state_layout=state_layout, timestamp=timestamp, version=version)

    def export_checkpoint(self) -> CCheckpointState:
        checkpoint = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_state_export(self._require_open(), None, ctypes.byref(checkpoint)), operation="bolr_gaussian_state_export")
        return CCheckpointState(checkpoint, library=self.library)

    @classmethod
    def import_checkpoint(cls, checkpoint: CCheckpointState, *, library: CLibrary | None = None) -> "CGaussianState":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        status_ok(lib.lib, lib.lib.bolr_gaussian_state_import(checkpoint._require_open(), None, ctypes.byref(handle)), operation="bolr_gaussian_state_import")
        return cls(handle=handle, library=lib)

    def predict_additive(self, process_noise: np.ndarray) -> tuple["CGaussianState", PredictionDiagnostics]:
        process_noise_arr = _as_f64_matrix(process_noise)
        config = TransitionConfig(1, self.library.const_matrix(process_noise_arr), 0.0, _null_const_vector())
        diagnostics = PredictionDiagnosticsStruct()
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_predict(self._require_open(), ctypes.byref(config), None, ctypes.byref(handle), ctypes.byref(diagnostics)), operation="bolr_gaussian_predict")
        return CGaussianState(handle=handle, library=self.library), PredictionDiagnostics.from_c(diagnostics)

    def predict_global_discount(self, discount: float) -> tuple["CGaussianState", PredictionDiagnostics]:
        config = TransitionConfig(2, ConstMatrixView(), float(discount), _null_const_vector())
        diagnostics = PredictionDiagnosticsStruct()
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_gaussian_predict(self._require_open(), ctypes.byref(config), None, ctypes.byref(handle), ctypes.byref(diagnostics)), operation="bolr_gaussian_predict")
        return CGaussianState(handle=handle, library=self.library), PredictionDiagnostics.from_c(diagnostics)


class CCandidateAObservation(CHandle):
    def __init__(self, target: np.ndarray, eta: float, update_weight: float, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.target = _as_f64_vector(target)
        self.eta = float(eta)
        self.update_weight = float(update_weight)
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_candidate_a_observation_create(
                self.library.const_vector(self.target),
                self.eta,
                self.update_weight,
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_candidate_a_observation_create",
        )
        super().__init__(handle, self.library.lib.bolr_candidate_a_observation_destroy)

    @property
    def dimension(self) -> int:
        return int(self.target.size)

    @property
    def effective_strength(self) -> float:
        return self.eta * self.update_weight

    def operator(self) -> ObservationOperatorStruct:
        operator = ObservationOperatorStruct()
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_observation_operator(self._require_open(), ctypes.byref(operator)), operation="bolr_candidate_a_observation_operator")
        return operator


class CStaticFitDataset(CHandle):
    def __init__(self, design: np.ndarray, observations: list[object], *, eta: float = 1.0, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.design = _as_f64_matrix(design)
        self.targets = [_as_f64_vector(observation.target_probabilities) for observation in observations]
        self.weights = np.ascontiguousarray(np.asarray([eta * float(observation.update_weight) for observation in observations], dtype=np.float64))
        views = [self.library.const_vector(target) for target in self.targets]
        self._target_view_array = (ConstVectorView * len(views))(*views)
        self._weights_ptr = self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_candidate_a_static_dataset_create(
                self.library.const_matrix(self.design),
                self._target_view_array,
                self._weights_ptr,
                len(views),
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_candidate_a_static_dataset_create",
        )
        super().__init__(handle, self.library.lib.bolr_candidate_a_static_dataset_destroy)


class CModelArtifacts(CHandle):
    def __init__(self, composite_model: Any, batch: object, *, library: CLibrary | None = None) -> None:
        from bolr.model.score_blocks import ContextInteractionBlock, DynamicSurfaceBlock, GraphResidualBlock, LinearDesignBlock

        self.library = CLibrary() if library is None else library
        self.composite_model = composite_model
        self.batch = batch
        self.layout = CStateLayout(composite_model.layout, library=self.library)
        self.static_scores = _as_f64_vector(composite_model.static_scores(batch))
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_model_create(self.layout._require_open(), self.library.const_vector(self.static_scores), None, ctypes.byref(handle)),
            operation="bolr_model_create",
        )
        self._name_buffers: list[bytes] = []
        self._owners: tuple[object, ...] = (batch, composite_model, self.layout)
        try:
            for block in composite_model.dynamic_blocks:
                name = block.name.encode("utf-8")
                self._name_buffers.append(name)
                if block.name in composite_model.fixed_blocks:
                    design = np.zeros_like(block.design_matrix(batch), dtype=np.float64)
                    status_ok(self.library.lib, self.library.lib.bolr_model_add_dense_block_copy(handle, name, self.library.const_matrix(design)), operation=f"bolr_model_add_dense_block_copy[{block.name}]")
                    continue
                if isinstance(block, DynamicSurfaceBlock):
                    design = _as_f64_matrix(block.candidate_basis)
                    status_ok(self.library.lib, self.library.lib.bolr_model_add_dense_block_copy(handle, name, self.library.const_matrix(design)), operation=f"bolr_model_add_dense_block_copy[{block.name}]")
                elif isinstance(block, LinearDesignBlock):
                    design = _as_f64_matrix(block.design_matrix(batch))
                    status_ok(self.library.lib, self.library.lib.bolr_model_add_dense_block_copy(handle, name, self.library.const_matrix(design)), operation=f"bolr_model_add_dense_block_copy[{block.name}]")
                elif isinstance(block, ContextInteractionBlock):
                    basis = _as_f64_matrix(block.candidate_basis)
                    context = CScoreContext.from_composite(composite_model, batch).context_vector
                    status_ok(self.library.lib, self.library.lib.bolr_model_add_context_block_copy(handle, name, self.library.const_matrix(basis), int(context.size)), operation=f"bolr_model_add_context_block_copy[{block.name}]")
                elif isinstance(block, GraphResidualBlock):
                    basis = _as_f64_matrix(block.residual_basis)
                    status_ok(self.library.lib, self.library.lib.bolr_model_add_graph_block_copy(handle, name, self.library.const_matrix(basis)), operation=f"bolr_model_add_graph_block_copy[{block.name}]")
                else:
                    raise TypeError(f"Unsupported block type for C backend: {type(block).__name__}")
        except Exception:
            self.library.lib.bolr_model_destroy(handle)
            self.layout.close()
            raise
        super().__init__(handle, self.library.lib.bolr_model_destroy)

    @property
    def candidate_count(self) -> int:
        return int(self.library.lib.bolr_model_score_count(self._require_open()))

    @property
    def state_dimension(self) -> int:
        return int(self.library.lib.bolr_model_state_dim(self._require_open()))

    @property
    def model_schema_hash(self) -> int:
        return int(self.library.lib.bolr_model_schema_hash(self._require_open()))

    @property
    def state_layout_hash(self) -> int:
        return int(self.library.lib.bolr_model_state_layout_hash(self._require_open()))

    def close(self) -> None:
        super().close()
        self.layout.close()

    def copy_static_scores(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_model_copy_static_scores(self._require_open(), self.library.vector(output)), operation="bolr_model_copy_static_scores")
        return output

    def state_from_posterior(self, posterior: GaussianPosterior) -> CGaussianState:
        return CGaussianState(
            posterior.mean,
            posterior.covariance,
            state_layout_hash=self.state_layout_hash,
            model_schema_hash=self.model_schema_hash,
            library=self.library,
            owner_refs=(self,),
        )

    def scores(self, state: np.ndarray, daily_context: CScoreContext | None = None, *, dynamic_only: bool = False) -> np.ndarray:
        daily_context = CScoreContext() if daily_context is None else daily_context
        state_arr = _as_f64_vector(state)
        output = np.empty(self.candidate_count, dtype=np.float64)
        workspace = CWorkspace(self.candidate_count, self.state_dimension, max(self.candidate_count, daily_context.context_vector.size), library=self.library)
        try:
            fn = self.library.lib.bolr_model_dynamic_forward if dynamic_only else self.library.lib.bolr_model_forward
            status_ok(
                self.library.lib,
                fn(
                    self._require_open(),
                    self.library.const_vector(state_arr),
                    daily_context.const_view(self.library),
                    self.library.vector(output),
                    workspace._require_open(),
                ),
                operation="bolr_model_forward" if not dynamic_only else "bolr_model_dynamic_forward",
            )
        finally:
            workspace.close()
        return output

    def transpose(self, score_vector: np.ndarray, daily_context: CScoreContext | None = None) -> np.ndarray:
        daily_context = CScoreContext() if daily_context is None else daily_context
        score_arr = _as_f64_vector(score_vector)
        output = np.empty(self.state_dimension, dtype=np.float64)
        workspace = CWorkspace(self.candidate_count, self.state_dimension, max(self.candidate_count, daily_context.context_vector.size), library=self.library)
        try:
            status_ok(
                self.library.lib,
                self.library.lib.bolr_model_transpose(
                    self._require_open(),
                    self.library.const_vector(score_arr),
                    daily_context.const_view(self.library),
                    self.library.vector(output),
                    workspace._require_open(),
                ),
                operation="bolr_model_transpose",
            )
        finally:
            workspace.close()
        return output


@dataclass
class CBackend(NumericalBackend):
    library: CLibrary = CLibrary()

    def block_forward(self, design: np.ndarray, state: np.ndarray) -> np.ndarray:
        design_arr = _as_f64_matrix(design)
        state_arr = _as_f64_vector(state)
        output = np.empty(design_arr.shape[0], dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_matvec(self.library.const_matrix(design_arr), self.library.const_vector(state_arr), self.library.vector(output)), operation="bolr_matvec")
        return output

    def block_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        design_arr = _as_f64_matrix(design)
        score_arr = _as_f64_vector(score_vector)
        output = np.empty(design_arr.shape[1], dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_matvec_transpose(self.library.const_matrix(design_arr), self.library.const_vector(score_arr), self.library.vector(output)), operation="bolr_matvec_transpose")
        return output

    def composite_forward(self, design: np.ndarray, state: np.ndarray, static_scores: np.ndarray | None = None) -> np.ndarray:
        result = self.block_forward(design, state)
        if static_scores is not None:
            result = result + _as_f64_vector(static_scores)
        return result

    def composite_transpose(self, design: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        return self.block_transpose(design, score_vector)

    def observation_value_gradient_hvp(self, scores: np.ndarray, observation: object, vector: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        from bolr.targets.soft_target import Observation

        if not isinstance(observation, Observation):
            raise TypeError("CBackend Candidate A binding expects SoftTargetObservation.")
        scores_arr = _as_f64_vector(scores)
        target_arr = _as_f64_vector(observation.target_probabilities)
        vec_arr = np.zeros_like(scores_arr) if vector is None else _as_f64_vector(vector)
        gradient = np.empty_like(scores_arr)
        hvp = np.empty_like(scores_arr)
        value = ctypes.c_double()
        eta_effective = float(observation.update_weight)
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_log_factor(self.library.const_vector(scores_arr), self.library.const_vector(target_arr), eta_effective, ctypes.byref(value)), operation="bolr_candidate_a_log_factor")
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_score_gradient(self.library.const_vector(scores_arr), self.library.const_vector(target_arr), eta_effective, self.library.vector(gradient)), operation="bolr_candidate_a_score_gradient")
        status_ok(self.library.lib, self.library.lib.bolr_candidate_a_score_hvp(self.library.const_vector(scores_arr), self.library.const_vector(vec_arr), eta_effective, self.library.vector(hvp)), operation="bolr_candidate_a_score_hvp")
        return float(value.value), gradient, hvp

    def cholesky_factor(self, matrix: np.ndarray) -> np.ndarray:
        mat = _as_f64_matrix(matrix).copy()
        diagnostics = CholeskyDiagnostics()
        status_ok(self.library.lib, self.library.lib.bolr_cholesky_factor(self.library.matrix(mat), 1e-9, 10.0, 4, ctypes.byref(diagnostics)), operation="bolr_cholesky_factor")
        return mat

    def cholesky_solve(self, factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        factor_arr = _as_f64_matrix(factor)
        rhs_arr = _as_f64_vector(rhs)
        output = np.empty_like(rhs_arr)
        status_ok(self.library.lib, self.library.lib.bolr_cholesky_solve(self.library.const_matrix(factor_arr), self.library.const_vector(rhs_arr), self.library.vector(output)), operation="bolr_cholesky_solve")
        return output

    def model_artifacts(self, composite_model: Any, batch: object) -> CModelArtifacts:
        return CModelArtifacts(composite_model, batch, library=self.library)

    def score_context(self, composite_model: Any, batch: object) -> CScoreContext:
        return CScoreContext.from_composite(composite_model, batch)

    def candidate_a_observation(self, observation: object, *, eta: float = 1.0) -> CCandidateAObservation:
        from bolr.targets.soft_target import Observation

        if not isinstance(observation, Observation):
            raise TypeError("Expected a SoftTarget Observation.")
        return CCandidateAObservation(observation.target_probabilities, eta=eta, update_weight=float(observation.update_weight), library=self.library)

    def laplace_update(
        self,
        predictive_state: CGaussianState,
        model: CModelArtifacts,
        daily_context: CScoreContext,
        observation: CCandidateAObservation,
        workspace: CInferenceWorkspace,
        config: CNewtonConfig | None = None,
    ) -> tuple[CGaussianState, CLaplaceDiagnostics]:
        config = CNewtonConfig() if config is None else config
        if predictive_state.state_layout_hash != model.state_layout_hash:
            raise ValueError("Predictive state layout hash does not match model artifacts.")
        if predictive_state.model_schema_hash != model.model_schema_hash:
            raise ValueError("Predictive state model schema hash does not match model artifacts.")
        if workspace.state_dimension != model.state_dimension or workspace.candidate_count != model.candidate_count:
            raise ValueError("Inference workspace dimensions do not match model artifacts.")
        operator = observation.operator()
        diagnostics = LaplaceDiagnosticsStruct()
        new_state = ctypes.c_void_p()
        config_struct = config.to_c()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_laplace_update(
                predictive_state._require_open(),
                model._require_open(),
                daily_context.const_view(self.library),
                ctypes.byref(operator),
                ctypes.byref(config_struct),
                workspace._require_open(),
                ctypes.byref(new_state),
                ctypes.byref(diagnostics),
            ),
            operation="bolr_laplace_update",
        )
        return CGaussianState(handle=new_state, library=self.library, owner_refs=(model, workspace, predictive_state)), CLaplaceDiagnostics.from_c(diagnostics)

    def static_fit(
        self,
        dataset: CStaticFitDataset,
        prior_mean: np.ndarray,
        prior_precision: np.ndarray,
        workspace: CInferenceWorkspace,
        config: CNewtonConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        config = CNewtonConfig() if config is None else config
        prior_mean_arr = _as_f64_vector(prior_mean)
        prior_precision_arr = _as_f64_matrix(prior_precision)
        coefficients = np.empty_like(prior_mean_arr)
        scores = np.empty(dataset.design.shape[0], dtype=np.float64)
        config_struct = config.to_c()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_candidate_a_static_fit(
                dataset._require_open(),
                self.library.const_vector(prior_mean_arr),
                self.library.const_matrix(prior_precision_arr),
                ctypes.byref(config_struct),
                workspace._require_open(),
                self.library.vector(coefficients),
                self.library.vector(scores),
                None,
            ),
            operation="bolr_candidate_a_static_fit",
        )
        return coefficients, scores

    def objective(self, predictive: GaussianPosterior, theta: np.ndarray, model: CModelArtifacts, daily_context: CScoreContext, observation: object) -> float:
        theta_arr = _as_f64_vector(theta)
        delta = theta_arr - predictive.mean
        prior_precision = np.linalg.inv(np.asarray(predictive.covariance, dtype=float))
        scores = model.scores(theta_arr, daily_context)
        value, _, _ = self.observation_value_gradient_hvp(scores, observation, np.zeros_like(scores))
        return float(0.5 * delta @ prior_precision @ delta - value)

    def gradient(self, predictive: GaussianPosterior, theta: np.ndarray, model: CModelArtifacts, daily_context: CScoreContext, observation: object) -> np.ndarray:
        theta_arr = _as_f64_vector(theta)
        delta = theta_arr - predictive.mean
        prior_precision = np.linalg.inv(np.asarray(predictive.covariance, dtype=float))
        scores = model.scores(theta_arr, daily_context)
        _, score_grad, _ = self.observation_value_gradient_hvp(scores, observation, np.zeros_like(scores))
        return prior_precision @ delta - model.transpose(score_grad, daily_context)

    def dense_hessian(self, predictive: GaussianPosterior, theta: np.ndarray, model: CModelArtifacts, daily_context: CScoreContext, observation: object) -> np.ndarray:
        theta_arr = _as_f64_vector(theta)
        prior_precision = np.linalg.inv(np.asarray(predictive.covariance, dtype=float))
        scores = model.scores(theta_arr, daily_context)
        identity = np.eye(theta_arr.size, dtype=np.float64)
        cols = []
        for idx in range(theta_arr.size):
            score_direction = model.scores(identity[:, idx], daily_context, dynamic_only=True)
            _, _, score_hvp = self.observation_value_gradient_hvp(scores, observation, score_direction)
            cols.append(prior_precision[:, idx] + model.transpose(score_hvp, daily_context))
        hessian = np.column_stack(cols)
        return 0.5 * (hessian + hessian.T)
