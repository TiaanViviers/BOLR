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


class CandidateATargetConfigStruct(ctypes.Structure):
    _fields_ = [
        ("kappa", ctypes.c_double),
        ("eta", ctypes.c_double),
        ("clip", ctypes.c_double),
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("min_scale", ctypes.c_double),
        ("no_update_if_degenerate", ctypes.c_int),
    ]


class CandidateATargetDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("candidate_count", ctypes.c_int64),
        ("informative", ctypes.c_int),
        ("target_sum", ctypes.c_double),
        ("target_minimum", ctypes.c_double),
        ("target_maximum", ctypes.c_double),
        ("target_entropy", ctypes.c_double),
        ("positive_candidate_count", ctypes.c_int64),
        ("highest_group_count", ctypes.c_int64),
        ("effective_temperature", ctypes.c_double),
        ("fallback_used", ctypes.c_int),
        ("tolerance_group_count", ctypes.c_int64),
        ("utility_scale", ctypes.c_double),
        ("clipping_fraction", ctypes.c_double),
        ("all_irrelevant", ctypes.c_int),
    ]


class OrderedPartitionToleranceConfigStruct(ctypes.Structure):
    _fields_ = [
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("execution_tolerance", ctypes.c_double),
        ("robust_scale_mode", ctypes.c_int),
        ("scale_floor", ctypes.c_double),
    ]


class OrderedPartitionConfigStruct(ctypes.Structure):
    _fields_ = [
        ("tolerance", OrderedPartitionToleranceConfigStruct),
        ("positive_threshold", ctypes.c_double),
        ("all_irrelevant_policy", ctypes.c_int),
        ("reduced_weight", ctypes.c_double),
    ]


class OrderedPartitionDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("candidate_count", ctypes.c_int64),
        ("group_count", ctypes.c_int64),
        ("tolerance", ctypes.c_double),
        ("utility_maximum", ctypes.c_double),
        ("utility_median", ctypes.c_double),
        ("utility_scale", ctypes.c_double),
        ("all_irrelevant", ctypes.c_int),
        ("update_weight", ctypes.c_double),
        ("possible_pair_count", ctypes.c_int64),
        ("largest_upper_partition", ctypes.c_int64),
        ("partition_complexity_proxy", ctypes.c_int64),
        ("high_group_size", ctypes.c_int64),
        ("middle_group_size", ctypes.c_int64),
        ("low_group_size", ctypes.c_int64),
    ]


class CandidateBDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("candidate_count", ctypes.c_int64),
        ("possible_pair_count", ctypes.c_int64),
        ("used_pair_count", ctypes.c_int64),
        ("duplicate_sample_count", ctypes.c_int64),
        ("update_weight", ctypes.c_double),
        ("normalize_pair_losses", ctypes.c_int),
    ]


class StandardizerConfigStruct(ctypes.Structure):
    _fields_ = [
        ("decay", ctypes.c_double),
        ("variance_floor", ctypes.c_double),
        ("warmup_count", ctypes.c_int64),
        ("clip_z", ctypes.c_double),
        ("clip_enabled", ctypes.c_int),
    ]


class StandardizerStateStruct(ctypes.Structure):
    _fields_ = [
        ("schema_version", ctypes.c_uint32),
        ("count", ctypes.c_uint64),
        ("mean", ctypes.c_double),
        ("variance", ctypes.c_double),
        ("last_z", ctypes.c_double),
        ("last_z_present", ctypes.c_int),
    ]


class StandardizerDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_double),
        ("z_score", ctypes.c_double),
        ("mean_before", ctypes.c_double),
        ("scale_before", ctypes.c_double),
        ("missing", ctypes.c_int),
        ("z_score_present", ctypes.c_int),
    ]


class BOCPDConfigStruct(ctypes.Structure):
    _fields_ = [
        ("hazard", ctypes.c_double),
        ("max_run_length", ctypes.c_int64),
        ("prior_mean", ctypes.c_double),
        ("prior_kappa", ctypes.c_double),
        ("prior_alpha", ctypes.c_double),
        ("prior_beta", ctypes.c_double),
        ("missing_policy", ctypes.c_int),
    ]


class BOCPDDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("change_probability", ctypes.c_double),
        ("map_run_length", ctypes.c_double),
        ("expected_run_length", ctypes.c_double),
        ("run_length_entropy", ctypes.c_double),
        ("predictive_log_density", ctypes.c_double),
        ("truncation_mass", ctypes.c_double),
        ("hazard", ctypes.c_double),
        ("informative", ctypes.c_int),
        ("predictive_log_density_present", ctypes.c_int),
        ("missing_policy", ctypes.c_int),
    ]


class SurpriseInputStruct(ctypes.Structure):
    _fields_ = [
        ("informative", ctypes.c_int),
        ("log_factor_at_predictive_mean", ctypes.c_double),
        ("log_factor_at_posterior_mode", ctypes.c_double),
        ("effective_strength", ctypes.c_double),
        ("information_size", ctypes.c_double),
        ("mahalanobis_update", ctypes.c_double),
        ("gaussian_kl", ctypes.c_double),
        ("objective_improvement", ctypes.c_double),
    ]


class AdaptiveBlockConfigStruct(ctypes.Structure):
    _fields_ = [
        ("block_name", ctypes.c_char_p),
        ("transition_family", ctypes.c_int),
        ("maximum_multiplier", ctypes.c_double),
        ("minimum_multiplier", ctypes.c_double),
        ("decay", ctypes.c_double),
        ("attribution_floor", ctypes.c_double),
        ("minimum_discount", ctypes.c_double),
        ("minimum_discount_present", ctypes.c_int),
        ("reset_enabled", ctypes.c_int),
        ("reset_threshold", ctypes.c_double),
        ("reset_threshold_present", ctypes.c_int),
        ("reset_strength", ctypes.c_double),
        ("reset_strength_present", ctypes.c_int),
        ("reset_cooldown", ctypes.c_int),
        ("amplitude", ctypes.c_double),
        ("adaptive_enabled", ctypes.c_int),
    ]


class AdaptivePolicyConfigStruct(ctypes.Structure):
    _fields_ = [
        ("surprise_mode", ctypes.c_int32),
        ("standardizer", StandardizerConfigStruct),
        ("detector", BOCPDConfigStruct),
        ("activation_beta", ctypes.c_double),
        ("activation_z0", ctypes.c_double),
        ("attribution_epsilon", ctypes.c_double),
    ]


class AdaptationDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("raw_surprise", ctypes.c_double),
        ("normalized_surprise", ctypes.c_double),
        ("information_normalized_surprise", ctypes.c_double),
        ("standardised_surprise", ctypes.c_double),
        ("standardizer_mean_before", ctypes.c_double),
        ("standardizer_scale_before", ctypes.c_double),
        ("change_probability", ctypes.c_double),
        ("map_run_length", ctypes.c_double),
        ("expected_run_length", ctypes.c_double),
        ("run_length_entropy", ctypes.c_double),
        ("predictive_log_density", ctypes.c_double),
        ("truncation_mass", ctypes.c_double),
        ("activation_value", ctypes.c_double),
        ("informative", ctypes.c_int),
        ("predictive_log_density_present", ctypes.c_int),
        ("block_count", ctypes.c_int64),
        ("euclidean_update_energy", ctypes.POINTER(ctypes.c_double)),
        ("mahalanobis_update_energy", ctypes.POINTER(ctypes.c_double)),
        ("attribution_weight", ctypes.POINTER(ctypes.c_double)),
        ("process_noise_multiplier", ctypes.POINTER(ctypes.c_double)),
        ("target_multiplier", ctypes.POINTER(ctypes.c_double)),
        ("active_discount", ctypes.POINTER(ctypes.c_double)),
        ("reset_strength", ctypes.POINTER(ctypes.c_double)),
        ("reset_scheduled", ctypes.POINTER(ctypes.c_int32)),
        ("reset_applied", ctypes.POINTER(ctypes.c_int32)),
        ("days_since_reset", ctypes.POINTER(ctypes.c_int64)),
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


class RNGMetadataStruct(ctypes.Structure):
    _fields_ = [
        ("schema_version", ctypes.c_uint32),
        ("algorithm_family", ctypes.c_uint32),
        ("algorithm_version", ctypes.c_uint32),
        ("pcg_variant", ctypes.c_uint32),
        ("ziggurat_layers", ctypes.c_uint32),
        ("table_hash", ctypes.c_uint64),
        ("seed", ctypes.c_uint64),
        ("stream", ctypes.c_uint64),
        ("u32_draw_count", ctypes.c_uint64),
        ("uniform_draw_count", ctypes.c_uint64),
        ("normal_draw_count", ctypes.c_uint64),
    ]


class SamplingDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("sample_count", ctypes.c_int64),
        ("state_dimension", ctypes.c_int64),
        ("antithetic", ctypes.c_int),
        ("normal_draw_count", ctypes.c_uint64),
        ("cholesky_jitter", ctypes.c_double),
        ("minimum_cholesky_diagonal", ctypes.c_double),
    ]


class ScoreSamplingDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("sample_count", ctypes.c_int64),
        ("candidate_count", ctypes.c_int64),
        ("state_dimension", ctypes.c_int64),
    ]


class RNGSeedStruct(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_uint64), ("stream", ctypes.c_uint64)]


class PosteriorPredictionDiagnosticsStruct(ctypes.Structure):
    _fields_ = [
        ("score_mean_norm", ctypes.c_double),
        ("score_variance_sum", ctypes.c_double),
        ("explicit_design_frobenius_norm", ctypes.c_double),
    ]


class PairwiseProbabilityStruct(ctypes.Structure):
    _fields_ = [
        ("left_probability", ctypes.c_double),
        ("mean_difference", ctypes.c_double),
        ("variance_difference", ctypes.c_double),
    ]


class RegionConfigStruct(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int64),
        ("top_fraction", ctypes.c_double),
        ("inclusion_threshold", ctypes.c_double),
        ("consensus_family", ctypes.c_int32),
    ]


class RegionSummaryStruct(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_int64),
        ("candidate_offset", ctypes.c_int64),
        ("candidate_count", ctypes.c_int64),
        ("inclusion_mass", ctypes.c_double),
        ("probability_best_mass", ctypes.c_double),
        ("maximum_score_mean", ctypes.c_double),
        ("average_score_mean", ctypes.c_double),
        ("inclusion_weighted_score_mean", ctypes.c_double),
        ("average_score_variance", ctypes.c_double),
        ("maximum_score_variance", ctypes.c_double),
        ("inclusion_weighted_variance", ctypes.c_double),
        ("entry_index_min", ctypes.c_int64),
        ("entry_index_max", ctypes.c_int64),
        ("stop_index_min", ctypes.c_int64),
        ("stop_index_max", ctypes.c_int64),
        ("graph_diameter", ctypes.c_double),
        ("boundary_edge_count", ctypes.c_int64),
        ("compactness", ctypes.c_double),
        ("representative_medoid_index", ctypes.c_int64),
    ]


class DecisionPolicyConfigStruct(ctypes.Structure):
    _fields_ = [
        ("family", ctypes.c_int32),
        ("top_k", ctypes.c_int64),
        ("region_selection_statistic", ctypes.c_int32),
        ("representative_policy", ctypes.c_int32),
    ]


class DecisionStruct(ctypes.Structure):
    _fields_ = [
        ("selected_index", ctypes.c_int64),
        ("selected", ctypes.c_int32),
        ("abstained", ctypes.c_int32),
        ("selected_score_mean", ctypes.c_double),
        ("selected_score_variance", ctypes.c_double),
        ("selected_probability_best", ctypes.c_double),
        ("selected_expected_rank", ctypes.c_double),
        ("selected_region_id", ctypes.c_int64),
        ("selected_region_mass", ctypes.c_double),
        ("selected_region_probability_best", ctypes.c_double),
        ("tie_flags", ctypes.c_uint32),
        ("reason_code", ctypes.c_uint32),
    ]


class DecisionDiagnosticsCStruct(ctypes.Structure):
    _fields_ = [
        ("tie_occurred", ctypes.c_int32),
        ("tie_break_stage_count", ctypes.c_int32),
        ("tie_break_stages", ctypes.c_int32 * 4),
        ("selected_region_candidate_count", ctypes.c_int64),
        ("medoid_objective", ctypes.c_double),
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
class CPosteriorPredictionDiagnostics:
    score_mean_norm: float
    score_variance_sum: float
    explicit_design_frobenius_norm: float

    @classmethod
    def from_c(cls, struct: PosteriorPredictionDiagnosticsStruct) -> "CPosteriorPredictionDiagnostics":
        return cls(
            score_mean_norm=float(struct.score_mean_norm),
            score_variance_sum=float(struct.score_variance_sum),
            explicit_design_frobenius_norm=float(struct.explicit_design_frobenius_norm),
        )


@dataclass(frozen=True)
class CPairwiseProbability:
    left_probability: float
    mean_difference: float
    variance_difference: float

    @classmethod
    def from_c(cls, struct: PairwiseProbabilityStruct) -> "CPairwiseProbability":
        return cls(
            left_probability=float(struct.left_probability),
            mean_difference=float(struct.mean_difference),
            variance_difference=float(struct.variance_difference),
        )


@dataclass(frozen=True)
class CRegionSummary:
    region_id: int
    candidate_offset: int
    candidate_count: int
    inclusion_mass: float
    probability_best_mass: float
    maximum_score_mean: float
    average_score_mean: float
    inclusion_weighted_score_mean: float
    average_score_variance: float
    maximum_score_variance: float
    inclusion_weighted_variance: float
    entry_index_min: int
    entry_index_max: int
    stop_index_min: int
    stop_index_max: int
    graph_diameter: float
    boundary_edge_count: int
    compactness: float
    representative_medoid_index: int

    @classmethod
    def from_c(cls, struct: RegionSummaryStruct) -> "CRegionSummary":
        return cls(
            region_id=int(struct.region_id),
            candidate_offset=int(struct.candidate_offset),
            candidate_count=int(struct.candidate_count),
            inclusion_mass=float(struct.inclusion_mass),
            probability_best_mass=float(struct.probability_best_mass),
            maximum_score_mean=float(struct.maximum_score_mean),
            average_score_mean=float(struct.average_score_mean),
            inclusion_weighted_score_mean=float(struct.inclusion_weighted_score_mean),
            average_score_variance=float(struct.average_score_variance),
            maximum_score_variance=float(struct.maximum_score_variance),
            inclusion_weighted_variance=float(struct.inclusion_weighted_variance),
            entry_index_min=int(struct.entry_index_min),
            entry_index_max=int(struct.entry_index_max),
            stop_index_min=int(struct.stop_index_min),
            stop_index_max=int(struct.stop_index_max),
            graph_diameter=float(struct.graph_diameter),
            boundary_edge_count=int(struct.boundary_edge_count),
            compactness=float(struct.compactness),
            representative_medoid_index=int(struct.representative_medoid_index),
        )


@dataclass(frozen=True)
class CDecisionResult:
    selected_index: int
    selected: bool
    abstained: bool
    selected_score_mean: float
    selected_score_variance: float
    selected_probability_best: float
    selected_expected_rank: float
    selected_region_id: int
    selected_region_mass: float
    selected_region_probability_best: float
    tie_flags: int
    reason_code: int

    @classmethod
    def from_c(cls, struct: DecisionStruct) -> "CDecisionResult":
        return cls(
            selected_index=int(struct.selected_index),
            selected=bool(struct.selected),
            abstained=bool(struct.abstained),
            selected_score_mean=float(struct.selected_score_mean),
            selected_score_variance=float(struct.selected_score_variance),
            selected_probability_best=float(struct.selected_probability_best),
            selected_expected_rank=float(struct.selected_expected_rank),
            selected_region_id=int(struct.selected_region_id),
            selected_region_mass=float(struct.selected_region_mass),
            selected_region_probability_best=float(struct.selected_region_probability_best),
            tie_flags=int(struct.tie_flags),
            reason_code=int(struct.reason_code),
        )


@dataclass(frozen=True)
class CDecisionDiagnostics:
    tie_occurred: bool
    tie_break_stages: tuple[int, ...]
    selected_region_candidate_count: int
    medoid_objective: float

    @classmethod
    def from_c(cls, struct: DecisionDiagnosticsCStruct) -> "CDecisionDiagnostics":
        return cls(
            tie_occurred=bool(struct.tie_occurred),
            tie_break_stages=tuple(int(struct.tie_break_stages[i]) for i in range(int(struct.tie_break_stage_count))),
            selected_region_candidate_count=int(struct.selected_region_candidate_count),
            medoid_objective=float(struct.medoid_objective),
        )


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
class CRNGMetadata:
    schema_version: int
    algorithm_family: int
    algorithm_version: int
    pcg_variant: int
    ziggurat_layers: int
    table_hash: int
    seed: int
    stream: int
    u32_draw_count: int
    uniform_draw_count: int
    normal_draw_count: int

    @classmethod
    def from_c(cls, struct: RNGMetadataStruct) -> "CRNGMetadata":
        return cls(
            schema_version=int(struct.schema_version),
            algorithm_family=int(struct.algorithm_family),
            algorithm_version=int(struct.algorithm_version),
            pcg_variant=int(struct.pcg_variant),
            ziggurat_layers=int(struct.ziggurat_layers),
            table_hash=int(struct.table_hash),
            seed=int(struct.seed),
            stream=int(struct.stream),
            u32_draw_count=int(struct.u32_draw_count),
            uniform_draw_count=int(struct.uniform_draw_count),
            normal_draw_count=int(struct.normal_draw_count),
        )


@dataclass(frozen=True)
class CSamplingDiagnostics:
    sample_count: int
    state_dimension: int
    antithetic: bool
    normal_draw_count: int
    cholesky_jitter: float
    minimum_cholesky_diagonal: float

    @classmethod
    def from_c(cls, struct: SamplingDiagnosticsStruct) -> "CSamplingDiagnostics":
        return cls(
            sample_count=int(struct.sample_count),
            state_dimension=int(struct.state_dimension),
            antithetic=bool(struct.antithetic),
            normal_draw_count=int(struct.normal_draw_count),
            cholesky_jitter=float(struct.cholesky_jitter),
            minimum_cholesky_diagonal=float(struct.minimum_cholesky_diagonal),
        )


@dataclass(frozen=True)
class CScoreSamplingDiagnostics:
    sample_count: int
    candidate_count: int
    state_dimension: int

    @classmethod
    def from_c(cls, struct: ScoreSamplingDiagnosticsStruct) -> "CScoreSamplingDiagnostics":
        return cls(
            sample_count=int(struct.sample_count),
            candidate_count=int(struct.candidate_count),
            state_dimension=int(struct.state_dimension),
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


@dataclass(frozen=True)
class CCandidateATargetDiagnostics:
    candidate_count: int
    informative: bool
    target_sum: float
    target_minimum: float
    target_maximum: float
    target_entropy: float
    positive_candidate_count: int
    highest_group_count: int
    effective_temperature: float
    fallback_used: bool
    tolerance_group_count: int
    utility_scale: float
    clipping_fraction: float
    all_irrelevant: bool

    @classmethod
    def from_c(cls, struct: CandidateATargetDiagnosticsStruct) -> "CCandidateATargetDiagnostics":
        return cls(
            candidate_count=int(struct.candidate_count),
            informative=bool(struct.informative),
            target_sum=float(struct.target_sum),
            target_minimum=float(struct.target_minimum),
            target_maximum=float(struct.target_maximum),
            target_entropy=float(struct.target_entropy),
            positive_candidate_count=int(struct.positive_candidate_count),
            highest_group_count=int(struct.highest_group_count),
            effective_temperature=float(struct.effective_temperature),
            fallback_used=bool(struct.fallback_used),
            tolerance_group_count=int(struct.tolerance_group_count),
            utility_scale=float(struct.utility_scale),
            clipping_fraction=float(struct.clipping_fraction),
            all_irrelevant=bool(struct.all_irrelevant),
        )


@dataclass(frozen=True)
class COrderedPartitionDiagnostics:
    candidate_count: int
    group_count: int
    tolerance: float
    utility_maximum: float
    utility_median: float
    utility_scale: float
    all_irrelevant: bool
    update_weight: float
    possible_pair_count: int
    largest_upper_partition: int
    partition_complexity_proxy: int
    high_group_size: int
    middle_group_size: int
    low_group_size: int

    @classmethod
    def from_c(cls, struct: OrderedPartitionDiagnosticsStruct) -> "COrderedPartitionDiagnostics":
        return cls(
            candidate_count=int(struct.candidate_count),
            group_count=int(struct.group_count),
            tolerance=float(struct.tolerance),
            utility_maximum=float(struct.utility_maximum),
            utility_median=float(struct.utility_median),
            utility_scale=float(struct.utility_scale),
            all_irrelevant=bool(struct.all_irrelevant),
            update_weight=float(struct.update_weight),
            possible_pair_count=int(struct.possible_pair_count),
            largest_upper_partition=int(struct.largest_upper_partition),
            partition_complexity_proxy=int(struct.partition_complexity_proxy),
            high_group_size=int(struct.high_group_size),
            middle_group_size=int(struct.middle_group_size),
            low_group_size=int(struct.low_group_size),
        )


@dataclass(frozen=True)
class CCandidateBDiagnostics:
    candidate_count: int
    possible_pair_count: int
    used_pair_count: int
    duplicate_sample_count: int
    update_weight: float
    normalize_pair_losses: bool

    @classmethod
    def from_c(cls, struct: CandidateBDiagnosticsStruct) -> "CCandidateBDiagnostics":
        return cls(
            candidate_count=int(struct.candidate_count),
            possible_pair_count=int(struct.possible_pair_count),
            used_pair_count=int(struct.used_pair_count),
            duplicate_sample_count=int(struct.duplicate_sample_count),
            update_weight=float(struct.update_weight),
            normalize_pair_losses=bool(struct.normalize_pair_losses),
        )


@dataclass(frozen=True)
class CBOCPDDiagnostics:
    change_probability: float
    map_run_length: float
    expected_run_length: float
    run_length_entropy: float
    predictive_log_density: float | None
    truncation_mass: float
    hazard: float
    informative: bool
    missing_policy: int

    @classmethod
    def from_c(cls, struct: BOCPDDiagnosticsStruct) -> "CBOCPDDiagnostics":
        return cls(
            change_probability=float(struct.change_probability),
            map_run_length=float(struct.map_run_length),
            expected_run_length=float(struct.expected_run_length),
            run_length_entropy=float(struct.run_length_entropy),
            predictive_log_density=float(struct.predictive_log_density) if bool(struct.predictive_log_density_present) else None,
            truncation_mass=float(struct.truncation_mass),
            hazard=float(struct.hazard),
            informative=bool(struct.informative),
            missing_policy=int(struct.missing_policy),
        )


@dataclass(frozen=True)
class CAdaptationDiagnostics:
    raw_surprise: float
    normalized_surprise: float
    information_normalized_surprise: float
    standardised_surprise: float
    standardizer_mean_before: float
    standardizer_scale_before: float
    change_probability: float
    map_run_length: float
    expected_run_length: float
    run_length_entropy: float
    predictive_log_density: float | None
    truncation_mass: float
    activation_value: float
    informative: bool
    euclidean_update_energy: np.ndarray
    mahalanobis_update_energy: np.ndarray
    attribution_weight: np.ndarray
    process_noise_multiplier: np.ndarray
    target_multiplier: np.ndarray
    active_discount: np.ndarray
    reset_strength: np.ndarray
    reset_scheduled: np.ndarray
    reset_applied: np.ndarray
    days_since_reset: np.ndarray

    @classmethod
    def from_c(cls, struct: AdaptationDiagnosticsStruct, arrays: dict[str, np.ndarray]) -> "CAdaptationDiagnostics":
        return cls(
            raw_surprise=float(struct.raw_surprise),
            normalized_surprise=float(struct.normalized_surprise),
            information_normalized_surprise=float(struct.information_normalized_surprise),
            standardised_surprise=float(struct.standardised_surprise),
            standardizer_mean_before=float(struct.standardizer_mean_before),
            standardizer_scale_before=float(struct.standardizer_scale_before),
            change_probability=float(struct.change_probability),
            map_run_length=float(struct.map_run_length),
            expected_run_length=float(struct.expected_run_length),
            run_length_entropy=float(struct.run_length_entropy),
            predictive_log_density=float(struct.predictive_log_density) if bool(struct.predictive_log_density_present) else None,
            truncation_mass=float(struct.truncation_mass),
            activation_value=float(struct.activation_value),
            informative=bool(struct.informative),
            euclidean_update_energy=arrays["euclidean_update_energy"],
            mahalanobis_update_energy=arrays["mahalanobis_update_energy"],
            attribution_weight=arrays["attribution_weight"],
            process_noise_multiplier=arrays["process_noise_multiplier"],
            target_multiplier=arrays["target_multiplier"],
            active_discount=arrays["active_discount"],
            reset_strength=arrays["reset_strength"],
            reset_scheduled=arrays["reset_scheduled"],
            reset_applied=arrays["reset_applied"],
            days_since_reset=arrays["days_since_reset"],
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


def _as_i64_vector(array: np.ndarray | list[int]) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.int64))
    if arr.ndim != 1:
        raise ValueError("Expected a contiguous int64 vector.")
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
        self.lib.bolr_candidate_a_target_build.argtypes = [ctypes.POINTER(CandidateATargetConfigStruct), ConstVectorView, VectorView, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(CandidateATargetDiagnosticsStruct)]
        self.lib.bolr_candidate_a_target_build.restype = ctypes.c_int32
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
        self.lib.bolr_ordered_partition_build.argtypes = [ctypes.POINTER(OrderedPartitionConfigStruct), ConstVectorView, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_ordered_partition_build.restype = ctypes.c_int32
        self.lib.bolr_ordered_partition_create_copy.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.bolr_ordered_partition_create_copy.restype = ctypes.c_int32
        self.lib.bolr_ordered_partition_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_ordered_partition_candidate_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_ordered_partition_candidate_count.restype = ctypes.c_int64
        self.lib.bolr_ordered_partition_group_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_ordered_partition_group_count.restype = ctypes.c_int64
        self.lib.bolr_ordered_partition_copy_candidate_to_group.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.bolr_ordered_partition_copy_candidate_to_group.restype = ctypes.c_int32
        self.lib.bolr_ordered_partition_copy_group_offsets.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.bolr_ordered_partition_copy_group_offsets.restype = ctypes.c_int32
        self.lib.bolr_ordered_partition_copy_group_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.bolr_ordered_partition_copy_group_indices.restype = ctypes.c_int32
        self.lib.bolr_ordered_partition_get_diagnostics.argtypes = [ctypes.c_void_p, ctypes.POINTER(OrderedPartitionDiagnosticsStruct)]
        self.lib.bolr_ordered_partition_get_diagnostics.restype = ctypes.c_int32
        self.lib.bolr_candidate_b_exact_observation_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_candidate_b_exact_observation_create.restype = ctypes.c_int32
        self.lib.bolr_candidate_b_exact_observation_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_candidate_b_exact_observation_operator.argtypes = [ctypes.c_void_p, ctypes.POINTER(ObservationOperatorStruct)]
        self.lib.bolr_candidate_b_exact_observation_operator.restype = ctypes.c_int32
        self.lib.bolr_candidate_b_exact_observation_diagnostics.argtypes = [ctypes.c_void_p, ctypes.POINTER(CandidateBDiagnosticsStruct)]
        self.lib.bolr_candidate_b_exact_observation_diagnostics.restype = ctypes.c_int32
        self.lib.bolr_candidate_b_sampled_observation_create.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.bolr_candidate_b_sampled_observation_create.restype = ctypes.c_int32
        self.lib.bolr_candidate_b_sampled_observation_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_candidate_b_sampled_observation_operator.argtypes = [ctypes.c_void_p, ctypes.POINTER(ObservationOperatorStruct)]
        self.lib.bolr_candidate_b_sampled_observation_operator.restype = ctypes.c_int32
        self.lib.bolr_candidate_b_sampled_observation_diagnostics.argtypes = [ctypes.c_void_p, ctypes.POINTER(CandidateBDiagnosticsStruct)]
        self.lib.bolr_candidate_b_sampled_observation_diagnostics.restype = ctypes.c_int32

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
        self.lib.bolr_gaussian_state_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, MatrixView, ctypes.POINTER(SamplingDiagnosticsStruct), ctypes.c_void_p]
        self.lib.bolr_gaussian_state_sample.restype = ctypes.c_int32
        self.lib.bolr_composite_score_samples.argtypes = [ctypes.c_void_p, ConstVectorView, ConstMatrixView, MatrixView, ctypes.c_void_p, ctypes.POINTER(ScoreSamplingDiagnosticsStruct)]
        self.lib.bolr_composite_score_samples.restype = ctypes.c_int32
        self.lib.bolr_posterior_score_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ConstVectorView, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, MatrixView, ctypes.POINTER(SamplingDiagnosticsStruct), ctypes.c_void_p]
        self.lib.bolr_posterior_score_sample.restype = ctypes.c_int32

        self.lib.bolr_rng_create.argtypes = [RNGSeedStruct, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_rng_create.restype = ctypes.c_int32
        self.lib.bolr_rng_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_rng_clone.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_rng_clone.restype = ctypes.c_int32
        self.lib.bolr_rng_u32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self.lib.bolr_rng_u32.restype = ctypes.c_int32
        self.lib.bolr_rng_uniform_open01.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_rng_uniform_open01.restype = ctypes.c_int32
        self.lib.bolr_rng_standard_normal.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_rng_standard_normal.restype = ctypes.c_int32
        self.lib.bolr_rng_fill_uniform_open01.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_rng_fill_uniform_open01.restype = ctypes.c_int32
        self.lib.bolr_rng_fill_standard_normal.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_rng_fill_standard_normal.restype = ctypes.c_int32
        self.lib.bolr_rng_metadata_copy.argtypes = [ctypes.c_void_p, ctypes.POINTER(RNGMetadataStruct)]
        self.lib.bolr_rng_metadata_copy.restype = ctypes.c_int32
        self.lib.bolr_rng_export.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_rng_export.restype = ctypes.c_int32
        self.lib.bolr_rng_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_rng_import.restype = ctypes.c_int32
        self.lib.bolr_rng_checkpoint_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_rng_checkpoint_metadata_copy.argtypes = [ctypes.c_void_p, ctypes.POINTER(RNGMetadataStruct)]
        self.lib.bolr_rng_checkpoint_metadata_copy.restype = ctypes.c_int32
        self.lib.bolr_rng_checkpoint_encoded_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_rng_checkpoint_encoded_size.restype = ctypes.c_int32
        self.lib.bolr_rng_checkpoint_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_rng_checkpoint_encode.restype = ctypes.c_int32
        self.lib.bolr_rng_checkpoint_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_rng_checkpoint_decode.restype = ctypes.c_int32

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

        self.lib.bolr_bocpd_state_create.argtypes = [ctypes.POINTER(BOCPDConfigStruct), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_bocpd_state_create.restype = ctypes.c_int32
        self.lib.bolr_bocpd_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_bocpd_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.POINTER(BOCPDDiagnosticsStruct)]
        self.lib.bolr_bocpd_step.restype = ctypes.c_int32
        self.lib.bolr_bocpd_copy_run_length_posterior.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_bocpd_copy_run_length_posterior.restype = ctypes.c_int32
        self.lib.bolr_bocpd_max_run_length.argtypes = [ctypes.c_void_p]
        self.lib.bolr_bocpd_max_run_length.restype = ctypes.c_int64
        self.lib.bolr_bocpd_encoded_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_bocpd_encoded_size.restype = ctypes.c_int32
        self.lib.bolr_bocpd_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_bocpd_encode.restype = ctypes.c_int32
        self.lib.bolr_bocpd_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_bocpd_decode.restype = ctypes.c_int32

        self.lib.bolr_adaptive_policy_create.argtypes = [ctypes.c_void_p, ConstMatrixView, ctypes.POINTER(AdaptivePolicyConfigStruct), ctypes.POINTER(AdaptiveBlockConfigStruct), ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_adaptive_policy_create.restype = ctypes.c_int32
        self.lib.bolr_adaptive_policy_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_adaptive_policy_configuration_hash.argtypes = [ctypes.c_void_p]
        self.lib.bolr_adaptive_policy_configuration_hash.restype = ctypes.c_uint64
        self.lib.bolr_adaptive_policy_block_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_adaptive_policy_block_count.restype = ctypes.c_int64
        self.lib.bolr_adaptive_state_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_adaptive_state_create.restype = ctypes.c_int32
        self.lib.bolr_adaptive_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_adaptive_state_step_index.argtypes = [ctypes.c_void_p]
        self.lib.bolr_adaptive_state_step_index.restype = ctypes.c_uint64
        self.lib.bolr_adaptive_state_copy_block_multipliers.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_adaptive_state_copy_block_multipliers.restype = ctypes.c_int32
        self.lib.bolr_adaptive_state_copy_block_discounts.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_adaptive_state_copy_block_discounts.restype = ctypes.c_int32
        self.lib.bolr_adaptive_state_copy_run_length_posterior.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_adaptive_state_copy_run_length_posterior.restype = ctypes.c_int32
        self.lib.bolr_adaptive_policy_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(AdaptationDiagnosticsStruct)]
        self.lib.bolr_adaptive_policy_predict.restype = ctypes.c_int32
        self.lib.bolr_adaptive_policy_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(SurpriseInputStruct), ctypes.POINTER(AdaptationDiagnosticsStruct)]
        self.lib.bolr_adaptive_policy_observe.restype = ctypes.c_int32
        self.lib.bolr_adaptive_state_encoded_size.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_adaptive_state_encoded_size.restype = ctypes.c_int32
        self.lib.bolr_adaptive_state_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.bolr_adaptive_state_encode.restype = ctypes.c_int32
        self.lib.bolr_adaptive_state_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_adaptive_state_decode.restype = ctypes.c_int32

        if self.lib.bolr_abi_version_major() != 1:
            raise CBackendError("Unsupported BOLR ABI major version.")
        self.lib.bolr_posterior_prediction_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ConstVectorView, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(PosteriorPredictionDiagnosticsStruct)]
        self.lib.bolr_posterior_prediction_create.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_posterior_prediction_candidate_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_posterior_prediction_candidate_count.restype = ctypes.c_int64
        self.lib.bolr_posterior_prediction_state_dim.argtypes = [ctypes.c_void_p]
        self.lib.bolr_posterior_prediction_state_dim.restype = ctypes.c_int64
        self.lib.bolr_posterior_prediction_copy_score_mean.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_posterior_prediction_copy_score_mean.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_copy_score_variance.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_posterior_prediction_copy_score_variance.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_copy_state_mean.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_posterior_prediction_copy_state_mean.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_copy_state_covariance.argtypes = [ctypes.c_void_p, MatrixView]
        self.lib.bolr_posterior_prediction_copy_state_covariance.restype = ctypes.c_int32
        self.lib.bolr_selected_score_covariance.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, MatrixView]
        self.lib.bolr_selected_score_covariance.restype = ctypes.c_int32
        self.lib.bolr_pairwise_probability.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(PairwiseProbabilityStruct)]
        self.lib.bolr_pairwise_probability.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_set_probability_best.argtypes = [ctypes.c_void_p, ConstVectorView]
        self.lib.bolr_posterior_prediction_set_probability_best.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_set_probability_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int64, ConstVectorView]
        self.lib.bolr_posterior_prediction_set_probability_top_k.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_set_expected_rank.argtypes = [ctypes.c_void_p, ConstVectorView]
        self.lib.bolr_posterior_prediction_set_expected_rank.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_copy_probability_best.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_posterior_prediction_copy_probability_best.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_copy_probability_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int64, VectorView]
        self.lib.bolr_posterior_prediction_copy_probability_top_k.restype = ctypes.c_int32
        self.lib.bolr_posterior_prediction_copy_expected_rank.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_posterior_prediction_copy_expected_rank.restype = ctypes.c_int32
        self.lib.bolr_probability_entropy.argtypes = [ConstVectorView, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_probability_entropy.restype = ctypes.c_int32
        self.lib.bolr_grid_graph_create.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_grid_graph_create.restype = ctypes.c_int32
        self.lib.bolr_grid_graph_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_grid_graph_node_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_grid_graph_node_count.restype = ctypes.c_int64
        self.lib.bolr_grid_graph_edge_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_grid_graph_edge_count.restype = ctypes.c_int64
        self.lib.bolr_region_set_build.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RegionConfigStruct), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_region_set_build.restype = ctypes.c_int32
        self.lib.bolr_region_set_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_region_set_region_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_region_set_region_count.restype = ctypes.c_int64
        self.lib.bolr_region_set_top_k.argtypes = [ctypes.c_void_p]
        self.lib.bolr_region_set_top_k.restype = ctypes.c_int64
        self.lib.bolr_region_set_copy_inclusion_probability.argtypes = [ctypes.c_void_p, VectorView]
        self.lib.bolr_region_set_copy_inclusion_probability.restype = ctypes.c_int32
        self.lib.bolr_region_set_copy_consensus_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.bolr_region_set_copy_consensus_indices.restype = ctypes.c_int32
        self.lib.bolr_region_set_consensus_count.argtypes = [ctypes.c_void_p]
        self.lib.bolr_region_set_consensus_count.restype = ctypes.c_int64
        self.lib.bolr_region_set_empty_consensus.argtypes = [ctypes.c_void_p]
        self.lib.bolr_region_set_empty_consensus.restype = ctypes.c_int32
        self.lib.bolr_region_set_summary.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(RegionSummaryStruct)]
        self.lib.bolr_region_set_summary.restype = ctypes.c_int32
        self.lib.bolr_region_set_copy_region_candidates.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.lib.bolr_region_set_copy_region_candidates.restype = ctypes.c_int32
        self.lib.bolr_weighted_graph_medoid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ConstVectorView, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_weighted_graph_medoid.restype = ctypes.c_int32
        self.lib.bolr_decision_policy_create.argtypes = [ctypes.POINTER(DecisionPolicyConfigStruct), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.bolr_decision_policy_create.restype = ctypes.c_int32
        self.lib.bolr_decision_policy_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bolr_decision_policy_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(DecisionStruct), ctypes.POINTER(DecisionDiagnosticsCStruct)]
        self.lib.bolr_decision_policy_apply.restype = ctypes.c_int32
        self.lib.bolr_realized_best_distribution.argtypes = [ConstVectorView, ctypes.c_double, VectorView]
        self.lib.bolr_realized_best_distribution.restype = ctypes.c_int32
        self.lib.bolr_realized_top_k_indicator.argtypes = [ConstVectorView, ctypes.c_int64, VectorView]
        self.lib.bolr_realized_top_k_indicator.restype = ctypes.c_int32
        self.lib.bolr_probability_best_brier.argtypes = [ConstVectorView, ConstVectorView, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_probability_best_brier.restype = ctypes.c_int32
        self.lib.bolr_top_k_brier.argtypes = [ConstVectorView, ConstVectorView, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
        self.lib.bolr_top_k_brier.restype = ctypes.c_int32
        self.lib.bolr_region_coverage.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ConstVectorView, ctypes.c_double, ctypes.POINTER(ctypes.c_int32)]
        self.lib.bolr_region_coverage.restype = ctypes.c_int32

        if self.lib.bolr_abi_version_minor() < 5:
            raise CBackendError("BOLR C backend is missing Phase L4B1 symbols.")

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


class CRNGCheckpoint(CHandle):
    def __init__(self, handle: ctypes.c_void_p, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        super().__init__(handle, self.library.lib.bolr_rng_checkpoint_destroy)

    def metadata(self) -> CRNGMetadata:
        metadata = RNGMetadataStruct()
        status_ok(self.library.lib, self.library.lib.bolr_rng_checkpoint_metadata_copy(self._require_open(), ctypes.byref(metadata)), operation="bolr_rng_checkpoint_metadata_copy")
        return CRNGMetadata.from_c(metadata)

    def to_bytes(self) -> bytes:
        size = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_rng_checkpoint_encoded_size(self._require_open(), ctypes.byref(size)), operation="bolr_rng_checkpoint_encoded_size")
        buffer = (ctypes.c_ubyte * size.value)()
        written = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_rng_checkpoint_encode(self._require_open(), ctypes.byref(buffer), size.value, ctypes.byref(written)), operation="bolr_rng_checkpoint_encode")
        return bytes(buffer[: written.value])

    @classmethod
    def from_bytes(cls, payload: bytes, *, library: CLibrary | None = None) -> "CRNGCheckpoint":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        raw = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        status_ok(lib.lib, lib.lib.bolr_rng_checkpoint_decode(ctypes.byref(raw), len(payload), None, ctypes.byref(handle)), operation="bolr_rng_checkpoint_decode")
        return cls(handle, library=lib)


class CRNG(CHandle):
    def __init__(self, seed: int | None = None, stream: int = 0, *, handle: ctypes.c_void_p | None = None, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        if handle is None:
            if seed is None:
                raise ValueError("Seed is required when creating a native RNG.")
            created = ctypes.c_void_p()
            status_ok(
                self.library.lib,
                self.library.lib.bolr_rng_create(RNGSeedStruct(int(seed), int(stream)), None, ctypes.byref(created)),
                operation="bolr_rng_create",
            )
            handle = created
        super().__init__(handle, self.library.lib.bolr_rng_destroy)

    def metadata(self) -> CRNGMetadata:
        metadata = RNGMetadataStruct()
        status_ok(self.library.lib, self.library.lib.bolr_rng_metadata_copy(self._require_open(), ctypes.byref(metadata)), operation="bolr_rng_metadata_copy")
        return CRNGMetadata.from_c(metadata)

    def u32(self) -> int:
        value = ctypes.c_uint32()
        status_ok(self.library.lib, self.library.lib.bolr_rng_u32(self._require_open(), ctypes.byref(value)), operation="bolr_rng_u32")
        return int(value.value)

    def uniform(self) -> float:
        value = ctypes.c_double()
        status_ok(self.library.lib, self.library.lib.bolr_rng_uniform_open01(self._require_open(), ctypes.byref(value)), operation="bolr_rng_uniform_open01")
        return float(value.value)

    def normal(self) -> float:
        value = ctypes.c_double()
        status_ok(self.library.lib, self.library.lib.bolr_rng_standard_normal(self._require_open(), ctypes.byref(value)), operation="bolr_rng_standard_normal")
        return float(value.value)

    def fill_uniform(self, count: int) -> np.ndarray:
        output = np.empty(int(count), dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_rng_fill_uniform_open01(self._require_open(), self.library.vector(output)), operation="bolr_rng_fill_uniform_open01")
        return output

    def fill_standard_normal(self, count: int) -> np.ndarray:
        output = np.empty(int(count), dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_rng_fill_standard_normal(self._require_open(), self.library.vector(output)), operation="bolr_rng_fill_standard_normal")
        return output

    def clone(self) -> "CRNG":
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_rng_clone(self._require_open(), None, ctypes.byref(handle)), operation="bolr_rng_clone")
        return CRNG(handle=handle, library=self.library)

    def export_checkpoint(self) -> CRNGCheckpoint:
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_rng_export(self._require_open(), None, ctypes.byref(handle)), operation="bolr_rng_export")
        return CRNGCheckpoint(handle, library=self.library)

    @classmethod
    def import_checkpoint(cls, checkpoint: CRNGCheckpoint, *, library: CLibrary | None = None) -> "CRNG":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        status_ok(lib.lib, lib.lib.bolr_rng_import(checkpoint._require_open(), None, ctypes.byref(handle)), operation="bolr_rng_import")
        return cls(handle=handle, library=lib)


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

    def to_posterior(self, *, state_layout: dict[str, Any] | None = None, timestamp: str | None = None, version: str = "c_backend_l4b1") -> GaussianPosterior:
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


class CBOCPDState(CHandle):
    def __init__(self, config: Any, *, handle: ctypes.c_void_p | None = None, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.config = config
        if handle is None:
            handle = ctypes.c_void_p()
            cfg = BOCPDConfigStruct(
                float(config.hazard),
                int(config.max_run_length),
                float(config.prior_mean),
                float(config.prior_kappa),
                float(config.prior_alpha),
                float(config.prior_beta),
                1 if config.missing_policy == "hold" else 2,
            )
            status_ok(self.library.lib, self.library.lib.bolr_bocpd_state_create(ctypes.byref(cfg), None, ctypes.byref(handle)), operation="bolr_bocpd_state_create")
        super().__init__(handle, self.library.lib.bolr_bocpd_state_destroy)

    @property
    def max_run_length(self) -> int:
        return int(self.library.lib.bolr_bocpd_max_run_length(self._require_open()))

    def step(self, value: float | None) -> CBOCPDDiagnostics:
        diagnostics = BOCPDDiagnosticsStruct()
        present = int(value is not None)
        status_ok(self.library.lib, self.library.lib.bolr_bocpd_step(self._require_open(), 0.0 if value is None else float(value), present, ctypes.byref(diagnostics)), operation="bolr_bocpd_step")
        return CBOCPDDiagnostics.from_c(diagnostics)

    def run_length_posterior(self) -> np.ndarray:
        output = np.empty(self.max_run_length + 1, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_bocpd_copy_run_length_posterior(self._require_open(), self.library.vector(output)), operation="bolr_bocpd_copy_run_length_posterior")
        return output

    def to_bytes(self) -> bytes:
        size = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_bocpd_encoded_size(self._require_open(), ctypes.byref(size)), operation="bolr_bocpd_encoded_size")
        buffer = (ctypes.c_ubyte * size.value)()
        written = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_bocpd_encode(self._require_open(), ctypes.byref(buffer), size.value, ctypes.byref(written)), operation="bolr_bocpd_encode")
        return bytes(buffer[: written.value])

    @classmethod
    def from_bytes(cls, payload: bytes, config: Any, *, library: CLibrary | None = None) -> "CBOCPDState":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        raw = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        status_ok(lib.lib, lib.lib.bolr_bocpd_decode(ctypes.byref(raw), len(payload), None, ctypes.byref(handle)), operation="bolr_bocpd_decode")
        return cls(config, handle=handle, library=lib)


def _surprise_mode_from_name(name: str) -> int:
    return {
        "generalized_predictive_loss": 2,
        "posterior_mahalanobis": 4,
        "posterior_kl": 5,
    }.get(name, 2)


class CAdaptivePolicy(CHandle):
    def __init__(self, python_policy: Any, layout: Any, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.python_policy = python_policy
        self.layout = CStateLayout(layout, library=self.library)
        self.base_process_noise = _as_f64_matrix(np.asarray(python_policy.base_process_noise, dtype=np.float64))
        self._block_name_buffers = [block.name.encode("utf-8") for block in layout.blocks]
        blocks = []
        for idx, block in enumerate(layout.blocks):
            cfg = python_policy.block_configs.get(block.name)
            if cfg is None:
                from bolr.config.foundation import BlockAdaptationConfig

                cfg = BlockAdaptationConfig(block_name=block.name, transition_family="fixed", adaptive_enabled=False)
            blocks.append(
                AdaptiveBlockConfigStruct(
                    self._block_name_buffers[idx],
                    {"fixed": 0, "additive": 1, "discount": 2, "zero_noise": 3, "frozen": 4}[cfg.transition_family],
                    float(cfg.maximum_multiplier),
                    float(cfg.minimum_multiplier),
                    float(cfg.decay),
                    float(cfg.attribution_floor),
                    float(cfg.minimum_discount or 0.0),
                    int(cfg.minimum_discount is not None),
                    int(bool(cfg.reset_enabled)),
                    float(cfg.reset_threshold or 0.0),
                    int(cfg.reset_threshold is not None),
                    float(cfg.reset_strength or 0.0),
                    int(cfg.reset_strength is not None),
                    int(cfg.reset_cooldown),
                    float(cfg.amplitude),
                    int(bool(cfg.adaptive_enabled)),
                )
            )
        self._block_array = (AdaptiveBlockConfigStruct * len(blocks))(*blocks)
        policy_cfg = AdaptivePolicyConfigStruct(
            _surprise_mode_from_name(python_policy.config.surprise_signal),
            StandardizerConfigStruct(
                float(python_policy.config.standardizer.decay),
                float(python_policy.config.standardizer.variance_floor),
                int(python_policy.config.standardizer.warmup_count),
                float(python_policy.config.standardizer.clip_z or 0.0),
                int(python_policy.config.standardizer.clip_z is not None),
            ),
            BOCPDConfigStruct(
                float(python_policy.config.detector.hazard),
                int(python_policy.config.detector.max_run_length),
                float(python_policy.config.detector.prior_mean),
                float(python_policy.config.detector.prior_kappa),
                float(python_policy.config.detector.prior_alpha),
                float(python_policy.config.detector.prior_beta),
                1 if python_policy.config.detector.missing_policy == "hold" else 2,
            ),
            float((python_policy.config.activation_parameters or {"beta": 1.0}).get("beta", 1.0)),
            float((python_policy.config.activation_parameters or {"z0": 2.0}).get("z0", 2.0)),
            1e-8,
        )
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_adaptive_policy_create(
                self.layout._require_open(),
                self.library.const_matrix(self.base_process_noise),
                ctypes.byref(policy_cfg),
                self._block_array,
                len(blocks),
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_adaptive_policy_create",
        )
        super().__init__(handle, self.library.lib.bolr_adaptive_policy_destroy)

    @property
    def block_count(self) -> int:
        return int(self.library.lib.bolr_adaptive_policy_block_count(self._require_open()))

    def close(self) -> None:
        super().close()
        self.layout.close()


class CAdaptiveState(CHandle):
    def __init__(self, policy: CAdaptivePolicy, *, handle: ctypes.c_void_p | None = None, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.policy = policy
        if handle is None:
            handle = ctypes.c_void_p()
            status_ok(self.library.lib, self.library.lib.bolr_adaptive_state_create(policy._require_open(), None, ctypes.byref(handle)), operation="bolr_adaptive_state_create")
        super().__init__(handle, self.library.lib.bolr_adaptive_state_destroy)

    @property
    def step_index(self) -> int:
        return int(self.library.lib.bolr_adaptive_state_step_index(self._require_open()))

    def block_multipliers(self) -> np.ndarray:
        output = np.empty(self.policy.block_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_adaptive_state_copy_block_multipliers(self._require_open(), self.library.vector(output)), operation="bolr_adaptive_state_copy_block_multipliers")
        return output

    def block_discounts(self) -> np.ndarray:
        output = np.empty(self.policy.block_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_adaptive_state_copy_block_discounts(self._require_open(), self.library.vector(output)), operation="bolr_adaptive_state_copy_block_discounts")
        return output

    def run_length_posterior(self) -> np.ndarray:
        output = np.empty(self.policy.python_policy.config.detector.max_run_length + 1, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_adaptive_state_copy_run_length_posterior(self._require_open(), self.library.vector(output)), operation="bolr_adaptive_state_copy_run_length_posterior")
        return output

    def to_bytes(self) -> bytes:
        size = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_adaptive_state_encoded_size(self.policy._require_open(), self._require_open(), ctypes.byref(size)), operation="bolr_adaptive_state_encoded_size")
        buffer = (ctypes.c_ubyte * size.value)()
        written = ctypes.c_size_t()
        status_ok(self.library.lib, self.library.lib.bolr_adaptive_state_encode(self.policy._require_open(), self._require_open(), ctypes.byref(buffer), size.value, ctypes.byref(written)), operation="bolr_adaptive_state_encode")
        return bytes(buffer[: written.value])

    @classmethod
    def from_bytes(cls, policy: CAdaptivePolicy, payload: bytes, *, library: CLibrary | None = None) -> "CAdaptiveState":
        lib = CLibrary() if library is None else library
        handle = ctypes.c_void_p()
        raw = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        status_ok(lib.lib, lib.lib.bolr_adaptive_state_decode(policy._require_open(), ctypes.byref(raw), len(payload), None, ctypes.byref(handle)), operation="bolr_adaptive_state_decode")
        return cls(policy, handle=handle, library=lib)


def _alloc_adaptation_struct(block_count: int) -> tuple[AdaptationDiagnosticsStruct, dict[str, np.ndarray]]:
    arrays = {
        "euclidean_update_energy": np.empty(block_count, dtype=np.float64),
        "mahalanobis_update_energy": np.empty(block_count, dtype=np.float64),
        "attribution_weight": np.empty(block_count, dtype=np.float64),
        "process_noise_multiplier": np.empty(block_count, dtype=np.float64),
        "target_multiplier": np.empty(block_count, dtype=np.float64),
        "active_discount": np.empty(block_count, dtype=np.float64),
        "reset_strength": np.empty(block_count, dtype=np.float64),
        "reset_scheduled": np.empty(block_count, dtype=np.int32),
        "reset_applied": np.empty(block_count, dtype=np.int32),
        "days_since_reset": np.empty(block_count, dtype=np.int64),
    }
    struct = AdaptationDiagnosticsStruct(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, block_count,
        arrays["euclidean_update_energy"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["mahalanobis_update_energy"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["attribution_weight"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["process_noise_multiplier"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["target_multiplier"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["active_discount"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["reset_strength"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        arrays["reset_scheduled"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        arrays["reset_applied"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        arrays["days_since_reset"].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return struct, arrays


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


class COrderedPartition(CHandle):
    def __init__(self, handle: ctypes.c_void_p, *, library: CLibrary | None = None, owner_refs: tuple[object, ...] = ()) -> None:
        self.library = CLibrary() if library is None else library
        self._owner_refs = owner_refs
        super().__init__(handle, self.library.lib.bolr_ordered_partition_destroy)

    @classmethod
    def from_utilities(cls, utilities: np.ndarray, config: Any, *, library: CLibrary | None = None) -> "COrderedPartition":
        lib = CLibrary() if library is None else library
        utilities_arr = _as_f64_vector(utilities)
        handle = ctypes.c_void_p()
        cfg = OrderedPartitionConfigStruct(
            OrderedPartitionToleranceConfigStruct(
                float(config.tolerance.absolute_tolerance),
                float(config.tolerance.relative_tolerance),
                float(config.tolerance.execution_tolerance),
                {"mad": 1, "iqr": 2, "max": 3}[config.tolerance.robust_scale],
                float(config.tolerance.scale_floor),
            ),
            float(config.positive_threshold),
            {"always_relative": 1, "no_update": 2, "reduced_weight": 3}[config.all_irrelevant_policy],
            float(config.reduced_weight),
        )
        status_ok(lib.lib, lib.lib.bolr_ordered_partition_build(ctypes.byref(cfg), lib.const_vector(utilities_arr), None, ctypes.byref(handle)), operation="bolr_ordered_partition_build")
        return cls(handle, library=lib)

    @classmethod
    def from_observation(cls, observation: Any, *, library: CLibrary | None = None) -> "COrderedPartition":
        lib = CLibrary() if library is None else library
        candidate_to_group = _as_i64_vector(observation.candidate_to_group)
        group_sizes = [int(len(group)) for group in observation.ordered_groups]
        group_offsets = np.empty(len(group_sizes) + 1, dtype=np.int64)
        group_offsets[0] = 0
        np.cumsum(np.asarray(group_sizes, dtype=np.int64), out=group_offsets[1:])
        group_indices = _as_i64_vector(np.concatenate([np.asarray(group, dtype=np.int64) for group in observation.ordered_groups]))
        handle = ctypes.c_void_p()
        status_ok(
            lib.lib,
            lib.lib.bolr_ordered_partition_create_copy(
                group_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                group_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                int(len(observation.ordered_groups)),
                candidate_to_group.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                int(candidate_to_group.size),
                float(observation.tolerance),
                float(observation.utility_maximum),
                float(observation.utility_median),
                float(observation.utility_scale),
                int(bool(observation.all_irrelevant)),
                float(observation.update_weight),
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_ordered_partition_create_copy",
        )
        instance = cls(handle, library=lib)
        instance._group_offsets_owner = group_offsets
        instance._group_indices_owner = group_indices
        instance._candidate_to_group_owner = candidate_to_group
        return instance

    @property
    def candidate_count(self) -> int:
        return int(self.library.lib.bolr_ordered_partition_candidate_count(self._require_open()))

    @property
    def group_count(self) -> int:
        return int(self.library.lib.bolr_ordered_partition_group_count(self._require_open()))

    def candidate_to_group(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.int64)
        status_ok(self.library.lib, self.library.lib.bolr_ordered_partition_copy_candidate_to_group(self._require_open(), output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), int(output.size)), operation="bolr_ordered_partition_copy_candidate_to_group")
        return output

    def group_offsets(self) -> np.ndarray:
        output = np.empty(self.group_count + 1, dtype=np.int64)
        status_ok(self.library.lib, self.library.lib.bolr_ordered_partition_copy_group_offsets(self._require_open(), output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), int(output.size)), operation="bolr_ordered_partition_copy_group_offsets")
        return output

    def group_indices(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.int64)
        status_ok(self.library.lib, self.library.lib.bolr_ordered_partition_copy_group_indices(self._require_open(), output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), int(output.size)), operation="bolr_ordered_partition_copy_group_indices")
        return output

    def diagnostics(self) -> COrderedPartitionDiagnostics:
        diagnostics = OrderedPartitionDiagnosticsStruct()
        status_ok(self.library.lib, self.library.lib.bolr_ordered_partition_get_diagnostics(self._require_open(), ctypes.byref(diagnostics)), operation="bolr_ordered_partition_get_diagnostics")
        return COrderedPartitionDiagnostics.from_c(diagnostics)


class CCandidateBExactObservation(CHandle):
    def __init__(self, partition: COrderedPartition, *, normalize_pair_losses: bool = True, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.partition = partition
        self.normalize_pair_losses = bool(normalize_pair_losses)
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_candidate_b_exact_observation_create(partition._require_open(), int(self.normalize_pair_losses), None, ctypes.byref(handle)), operation="bolr_candidate_b_exact_observation_create")
        super().__init__(handle, self.library.lib.bolr_candidate_b_exact_observation_destroy)

    def operator(self) -> ObservationOperatorStruct:
        operator = ObservationOperatorStruct()
        status_ok(self.library.lib, self.library.lib.bolr_candidate_b_exact_observation_operator(self._require_open(), ctypes.byref(operator)), operation="bolr_candidate_b_exact_observation_operator")
        return operator

    def diagnostics(self) -> CCandidateBDiagnostics:
        diagnostics = CandidateBDiagnosticsStruct()
        status_ok(self.library.lib, self.library.lib.bolr_candidate_b_exact_observation_diagnostics(self._require_open(), ctypes.byref(diagnostics)), operation="bolr_candidate_b_exact_observation_diagnostics")
        return CCandidateBDiagnostics.from_c(diagnostics)


class CCandidateBSampledObservation(CHandle):
    def __init__(
        self,
        candidate_count: int,
        winner_indices: np.ndarray,
        loser_indices: np.ndarray,
        pair_weights: np.ndarray,
        *,
        update_weight: float,
        possible_pair_count: int,
        duplicate_sample_count: int,
        normalize_pair_losses: bool,
        library: CLibrary | None = None,
    ) -> None:
        self.library = CLibrary() if library is None else library
        self.winner_indices = _as_i64_vector(winner_indices)
        self.loser_indices = _as_i64_vector(loser_indices)
        self.pair_weights = _as_f64_vector(pair_weights)
        if not (self.winner_indices.size == self.loser_indices.size == self.pair_weights.size):
            raise ValueError("Sampled pair arrays must have the same length.")
        self.candidate_count = int(candidate_count)
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_candidate_b_sampled_observation_create(
                self.candidate_count,
                self.winner_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                self.loser_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                self.pair_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(self.pair_weights.size),
                float(update_weight),
                int(possible_pair_count),
                int(duplicate_sample_count),
                int(bool(normalize_pair_losses)),
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_candidate_b_sampled_observation_create",
        )
        super().__init__(handle, self.library.lib.bolr_candidate_b_sampled_observation_destroy)

    def operator(self) -> ObservationOperatorStruct:
        operator = ObservationOperatorStruct()
        status_ok(self.library.lib, self.library.lib.bolr_candidate_b_sampled_observation_operator(self._require_open(), ctypes.byref(operator)), operation="bolr_candidate_b_sampled_observation_operator")
        return operator

    def diagnostics(self) -> CCandidateBDiagnostics:
        diagnostics = CandidateBDiagnosticsStruct()
        status_ok(self.library.lib, self.library.lib.bolr_candidate_b_sampled_observation_diagnostics(self._require_open(), ctypes.byref(diagnostics)), operation="bolr_candidate_b_sampled_observation_diagnostics")
        return CCandidateBDiagnostics.from_c(diagnostics)


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


_CONSENSUS_FAMILY = {"threshold": 1, "top_count": 2, "cumulative_mass": 3}
_DECISION_FAMILY = {
    "posterior_mean_argmax": 1,
    "maximum_probability_best": 2,
    "maximum_probability_top_k": 3,
    "minimum_expected_rank": 4,
    "highest_mass_region": 5,
}
_REGION_STATISTIC = {"probability_best": 1, "inclusion_mass": 2}
_REGION_REPRESENTATIVE = {"posterior_mean": 1, "probability_best": 2, "probability_top_k": 3, "weighted_medoid": 4}


class CPosteriorPrediction(CHandle):
    def __init__(
        self,
        predictive_state: CGaussianState,
        model: "CModelArtifacts",
        daily_context: CScoreContext | None = None,
        *,
        library: CLibrary | None = None,
    ) -> None:
        self.library = CLibrary() if library is None else library
        self.predictive_state = predictive_state
        self.model = model
        self.daily_context = CScoreContext() if daily_context is None else daily_context
        handle = ctypes.c_void_p()
        diagnostics = PosteriorPredictionDiagnosticsStruct()
        workspace = CWorkspace(model.candidate_count, model.state_dimension, max(model.candidate_count, self.daily_context.context_vector.size), library=self.library)
        try:
            status_ok(
                self.library.lib,
                self.library.lib.bolr_posterior_prediction_create(
                    predictive_state._require_open(),
                    model._require_open(),
                    self.daily_context.const_view(self.library),
                    workspace._require_open(),
                    None,
                    ctypes.byref(handle),
                    ctypes.byref(diagnostics),
                ),
                operation="bolr_posterior_prediction_create",
            )
        finally:
            workspace.close()
        self.diagnostics = CPosteriorPredictionDiagnostics.from_c(diagnostics)
        super().__init__(handle, self.library.lib.bolr_posterior_prediction_destroy)

    @property
    def candidate_count(self) -> int:
        return int(self.library.lib.bolr_posterior_prediction_candidate_count(self._require_open()))

    @property
    def state_dimension(self) -> int:
        return int(self.library.lib.bolr_posterior_prediction_state_dim(self._require_open()))

    def score_mean(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_score_mean(self._require_open(), self.library.vector(output)), operation="bolr_posterior_prediction_copy_score_mean")
        return output

    def score_variance(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_score_variance(self._require_open(), self.library.vector(output)), operation="bolr_posterior_prediction_copy_score_variance")
        return output

    def state_mean(self) -> np.ndarray:
        output = np.empty(self.state_dimension, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_state_mean(self._require_open(), self.library.vector(output)), operation="bolr_posterior_prediction_copy_state_mean")
        return output

    def state_covariance(self) -> np.ndarray:
        output = np.empty((self.state_dimension, self.state_dimension), dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_state_covariance(self._require_open(), self.library.matrix(output)), operation="bolr_posterior_prediction_copy_state_covariance")
        return output

    def selected_score_covariance(self, indices: np.ndarray) -> np.ndarray:
        indices_arr = _as_i64_vector(indices)
        output = np.empty((indices_arr.size, indices_arr.size), dtype=np.float64)
        status_ok(
            self.library.lib,
            self.library.lib.bolr_selected_score_covariance(
                self._require_open(),
                indices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                int(indices_arr.size),
                self.library.matrix(output),
            ),
            operation="bolr_selected_score_covariance",
        )
        return output

    def pairwise_probabilities(self, left_indices: np.ndarray, right_indices: np.ndarray) -> tuple[CPairwiseProbability, ...]:
        left = _as_i64_vector(left_indices)
        right = _as_i64_vector(right_indices)
        if left.shape != right.shape:
            raise ValueError("left_indices and right_indices must have matching shapes.")
        output = (PairwiseProbabilityStruct * int(left.size))()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_pairwise_probability(
                self._require_open(),
                left.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                right.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                int(left.size),
                output,
            ),
            operation="bolr_pairwise_probability",
        )
        return tuple(CPairwiseProbability.from_c(output[i]) for i in range(int(left.size)))

    def set_probability_best(self, probability_best: np.ndarray) -> None:
        arr = _as_f64_vector(probability_best)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_set_probability_best(self._require_open(), self.library.const_vector(arr)), operation="bolr_posterior_prediction_set_probability_best")

    def set_probability_top_k(self, top_k: int, probability_top_k: np.ndarray) -> None:
        arr = _as_f64_vector(probability_top_k)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_set_probability_top_k(self._require_open(), int(top_k), self.library.const_vector(arr)), operation="bolr_posterior_prediction_set_probability_top_k")

    def set_expected_rank(self, expected_rank: np.ndarray) -> None:
        arr = _as_f64_vector(expected_rank)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_set_expected_rank(self._require_open(), self.library.const_vector(arr)), operation="bolr_posterior_prediction_set_expected_rank")

    def probability_best(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_probability_best(self._require_open(), self.library.vector(output)), operation="bolr_posterior_prediction_copy_probability_best")
        return output

    def probability_top_k(self, top_k: int) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_probability_top_k(self._require_open(), int(top_k), self.library.vector(output)), operation="bolr_posterior_prediction_copy_probability_top_k")
        return output

    def expected_rank(self) -> np.ndarray:
        output = np.empty(self.candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_posterior_prediction_copy_expected_rank(self._require_open(), self.library.vector(output)), operation="bolr_posterior_prediction_copy_expected_rank")
        return output


class CGridGraph(CHandle):
    def __init__(self, graph: Any, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        self.edge_index = np.ascontiguousarray(np.asarray(graph.edge_index, dtype=np.int64))
        self.entry_indices = np.ascontiguousarray(np.asarray(graph.entry_indices, dtype=np.int64))
        self.stop_indices = np.ascontiguousarray(np.asarray(graph.stop_indices, dtype=np.int64))
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_grid_graph_create(
                int(graph.node_count),
                self.edge_index.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                int(graph.edge_count),
                self.entry_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                self.stop_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_grid_graph_create",
        )
        super().__init__(handle, self.library.lib.bolr_grid_graph_destroy)

    @property
    def node_count(self) -> int:
        return int(self.library.lib.bolr_grid_graph_node_count(self._require_open()))

    @property
    def edge_count(self) -> int:
        return int(self.library.lib.bolr_grid_graph_edge_count(self._require_open()))

    def weighted_medoid(self, indices: np.ndarray, weights: np.ndarray) -> tuple[int, float]:
        indices_arr = _as_i64_vector(indices)
        weights_arr = _as_f64_vector(weights)
        medoid = ctypes.c_int64()
        objective = ctypes.c_double()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_weighted_graph_medoid(
                self._require_open(),
                indices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                self.library.const_vector(weights_arr),
                int(indices_arr.size),
                ctypes.byref(medoid),
                ctypes.byref(objective),
            ),
            operation="bolr_weighted_graph_medoid",
        )
        return int(medoid.value), float(objective.value)


class CRegionSet(CHandle):
    def __init__(self, prediction: CPosteriorPrediction, graph: CGridGraph, config: Any, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        family = _CONSENSUS_FAMILY[str(config.consensus_family)]
        top_fraction = getattr(config, "top_fraction", 0.0)
        cfg = RegionConfigStruct(
            0 if getattr(config, "top_k", None) is None else int(config.top_k),
            0.0 if top_fraction is None else float(top_fraction),
            float(config.inclusion_threshold),
            family,
        )
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_region_set_build(
                prediction._require_open(),
                graph._require_open(),
                ctypes.byref(cfg),
                None,
                ctypes.byref(handle),
            ),
            operation="bolr_region_set_build",
        )
        super().__init__(handle, self.library.lib.bolr_region_set_destroy)

    @property
    def region_count(self) -> int:
        return int(self.library.lib.bolr_region_set_region_count(self._require_open()))

    @property
    def top_k(self) -> int:
        return int(self.library.lib.bolr_region_set_top_k(self._require_open()))

    @property
    def consensus_count(self) -> int:
        return int(self.library.lib.bolr_region_set_consensus_count(self._require_open()))

    @property
    def empty_consensus(self) -> bool:
        return bool(self.library.lib.bolr_region_set_empty_consensus(self._require_open()))

    def inclusion_probability(self, candidate_count: int) -> np.ndarray:
        output = np.empty(candidate_count, dtype=np.float64)
        status_ok(self.library.lib, self.library.lib.bolr_region_set_copy_inclusion_probability(self._require_open(), self.library.vector(output)), operation="bolr_region_set_copy_inclusion_probability")
        return output

    def consensus_indices(self) -> np.ndarray:
        output = np.empty(self.consensus_count, dtype=np.int64)
        status_ok(self.library.lib, self.library.lib.bolr_region_set_copy_consensus_indices(self._require_open(), output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), int(output.size)), operation="bolr_region_set_copy_consensus_indices")
        return output

    def summary(self, region_index: int) -> CRegionSummary:
        struct = RegionSummaryStruct()
        status_ok(self.library.lib, self.library.lib.bolr_region_set_summary(self._require_open(), int(region_index), ctypes.byref(struct)), operation="bolr_region_set_summary")
        return CRegionSummary.from_c(struct)

    def region_candidates(self, region_index: int) -> np.ndarray:
        summary = self.summary(region_index)
        output = np.empty(summary.candidate_count, dtype=np.int64)
        status_ok(self.library.lib, self.library.lib.bolr_region_set_copy_region_candidates(self._require_open(), int(region_index), output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), int(output.size)), operation="bolr_region_set_copy_region_candidates")
        return output


class CDecisionPolicy(CHandle):
    def __init__(self, config: Any, *, library: CLibrary | None = None) -> None:
        self.library = CLibrary() if library is None else library
        cfg = DecisionPolicyConfigStruct(
            _DECISION_FAMILY[str(config.family)],
            0 if getattr(config, "top_k", None) is None else int(config.top_k),
            _REGION_STATISTIC.get(str(getattr(config, "region_selection_statistic", "probability_best")), 1),
            _REGION_REPRESENTATIVE.get(str(getattr(config, "representative_policy", "posterior_mean")), 1),
        )
        handle = ctypes.c_void_p()
        status_ok(self.library.lib, self.library.lib.bolr_decision_policy_create(ctypes.byref(cfg), None, ctypes.byref(handle)), operation="bolr_decision_policy_create")
        super().__init__(handle, self.library.lib.bolr_decision_policy_destroy)

    def apply(self, prediction: CPosteriorPrediction, regions: CRegionSet | None = None, graph: CGridGraph | None = None) -> tuple[CDecisionResult, CDecisionDiagnostics]:
        decision = DecisionStruct()
        diagnostics = DecisionDiagnosticsCStruct()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_decision_policy_apply(
                self._require_open(),
                prediction._require_open(),
                None if regions is None else regions._require_open(),
                None if graph is None else graph._require_open(),
                ctypes.byref(decision),
                ctypes.byref(diagnostics),
            ),
            operation="bolr_decision_policy_apply",
        )
        return CDecisionResult.from_c(decision), CDecisionDiagnostics.from_c(diagnostics)


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

    def posterior_prediction(self, predictive_state: CGaussianState, model: CModelArtifacts, daily_context: CScoreContext | None = None) -> CPosteriorPrediction:
        return CPosteriorPrediction(predictive_state, model, daily_context, library=self.library)

    def grid_graph(self, graph: Any) -> CGridGraph:
        return CGridGraph(graph, library=self.library)

    def region_set(self, prediction: CPosteriorPrediction, graph: CGridGraph, config: Any) -> CRegionSet:
        return CRegionSet(prediction, graph, config, library=self.library)

    def decision_policy(self, config: Any) -> CDecisionPolicy:
        return CDecisionPolicy(config, library=self.library)

    def rng(self, seed: int, stream: int = 0) -> CRNG:
        return CRNG(seed=seed, stream=stream, library=self.library)

    def sample_gaussian_state(
        self,
        state: CGaussianState,
        rng: CRNG,
        sample_count: int,
        *,
        antithetic: bool = False,
        workspace: CWorkspace | None = None,
    ) -> tuple[np.ndarray, CSamplingDiagnostics]:
        output = np.empty((int(sample_count), state.dimension), dtype=np.float64)
        diagnostics = SamplingDiagnosticsStruct()
        owned_workspace = workspace
        if owned_workspace is None:
            owned_workspace = CWorkspace(0, state.dimension, 0, library=self.library)
        try:
            status_ok(
                self.library.lib,
                self.library.lib.bolr_gaussian_state_sample(
                    state._require_open(),
                    rng._require_open(),
                    int(sample_count),
                    int(bool(antithetic)),
                    self.library.matrix(output),
                    ctypes.byref(diagnostics),
                    owned_workspace._require_open(),
                ),
                operation="bolr_gaussian_state_sample",
            )
        finally:
            if workspace is None:
                owned_workspace.close()
        return output, CSamplingDiagnostics.from_c(diagnostics)

    def score_samples_from_state_samples(
        self,
        model: CModelArtifacts,
        state_samples: np.ndarray,
        context: CScoreContext | None = None,
        *,
        workspace: CWorkspace | None = None,
    ) -> tuple[np.ndarray, CScoreSamplingDiagnostics]:
        states = _as_f64_matrix(state_samples)
        context = CScoreContext() if context is None else context
        output = np.empty((states.shape[0], model.candidate_count), dtype=np.float64)
        diagnostics = ScoreSamplingDiagnosticsStruct()
        owned_workspace = workspace
        if owned_workspace is None:
            owned_workspace = CWorkspace(model.candidate_count, model.state_dimension, max(0, context.context_vector.size), library=self.library)
        try:
            status_ok(
                self.library.lib,
                self.library.lib.bolr_composite_score_samples(
                    model._require_open(),
                    context.const_view(self.library),
                    self.library.const_matrix(states),
                    self.library.matrix(output),
                    owned_workspace._require_open(),
                    ctypes.byref(diagnostics),
                ),
                operation="bolr_composite_score_samples",
            )
        finally:
            if workspace is None:
                owned_workspace.close()
        return output, CScoreSamplingDiagnostics.from_c(diagnostics)

    def sample_posterior_scores(
        self,
        state: CGaussianState,
        model: CModelArtifacts,
        context: CScoreContext | None,
        rng: CRNG,
        sample_count: int,
        *,
        antithetic: bool = False,
        workspace: CWorkspace | None = None,
    ) -> tuple[np.ndarray, CSamplingDiagnostics]:
        context = CScoreContext() if context is None else context
        output = np.empty((int(sample_count), model.candidate_count), dtype=np.float64)
        diagnostics = SamplingDiagnosticsStruct()
        owned_workspace = workspace
        if owned_workspace is None:
            owned_workspace = CWorkspace(model.candidate_count, model.state_dimension, max(0, context.context_vector.size), library=self.library)
        try:
            status_ok(
                self.library.lib,
                self.library.lib.bolr_posterior_score_sample(
                    state._require_open(),
                    model._require_open(),
                    context.const_view(self.library),
                    rng._require_open(),
                    int(sample_count),
                    int(bool(antithetic)),
                    self.library.matrix(output),
                    ctypes.byref(diagnostics),
                    owned_workspace._require_open(),
                ),
                operation="bolr_posterior_score_sample",
            )
        finally:
            if workspace is None:
                owned_workspace.close()
        return output, CSamplingDiagnostics.from_c(diagnostics)

    def probability_entropy(self, probabilities: np.ndarray) -> tuple[float, float, float]:
        probs = _as_f64_vector(probabilities)
        entropy = ctypes.c_double()
        effective_count = ctypes.c_double()
        maximum = ctypes.c_double()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_probability_entropy(self.library.const_vector(probs), ctypes.byref(entropy), ctypes.byref(effective_count), ctypes.byref(maximum)),
            operation="bolr_probability_entropy",
        )
        return float(entropy.value), float(effective_count.value), float(maximum.value)

    def realized_best_distribution(self, utilities: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        utilities_arr = _as_f64_vector(utilities)
        output = np.empty_like(utilities_arr)
        status_ok(self.library.lib, self.library.lib.bolr_realized_best_distribution(self.library.const_vector(utilities_arr), float(tolerance), self.library.vector(output)), operation="bolr_realized_best_distribution")
        return output

    def realized_top_k_indicator(self, utilities: np.ndarray, top_k: int) -> np.ndarray:
        utilities_arr = _as_f64_vector(utilities)
        output = np.empty_like(utilities_arr)
        status_ok(self.library.lib, self.library.lib.bolr_realized_top_k_indicator(self.library.const_vector(utilities_arr), int(top_k), self.library.vector(output)), operation="bolr_realized_top_k_indicator")
        return output

    def probability_best_brier(self, probability_best: np.ndarray, utilities: np.ndarray, tolerance: float = 0.0) -> float:
        probability_best_arr = _as_f64_vector(probability_best)
        utilities_arr = _as_f64_vector(utilities)
        value = ctypes.c_double()
        status_ok(self.library.lib, self.library.lib.bolr_probability_best_brier(self.library.const_vector(probability_best_arr), self.library.const_vector(utilities_arr), float(tolerance), ctypes.byref(value)), operation="bolr_probability_best_brier")
        return float(value.value)

    def top_k_brier(self, probability_top_k: np.ndarray, utilities: np.ndarray, top_k: int) -> float:
        probability_top_k_arr = _as_f64_vector(probability_top_k)
        utilities_arr = _as_f64_vector(utilities)
        value = ctypes.c_double()
        status_ok(self.library.lib, self.library.lib.bolr_top_k_brier(self.library.const_vector(probability_top_k_arr), self.library.const_vector(utilities_arr), int(top_k), ctypes.byref(value)), operation="bolr_top_k_brier")
        return float(value.value)

    def region_coverage(self, region_indices: np.ndarray, utilities: np.ndarray, tolerance: float = 0.0) -> bool:
        region_indices_arr = _as_i64_vector(region_indices)
        utilities_arr = _as_f64_vector(utilities)
        covered = ctypes.c_int32()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_region_coverage(
                region_indices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                int(region_indices_arr.size),
                self.library.const_vector(utilities_arr),
                float(tolerance),
                ctypes.byref(covered),
            ),
            operation="bolr_region_coverage",
        )
        return bool(covered.value)

    def bocpd_state(self, config: Any) -> CBOCPDState:
        return CBOCPDState(config, library=self.library)

    def adaptive_policy(self, transition_policy: Any, layout: Any) -> CAdaptivePolicy:
        return CAdaptivePolicy(transition_policy, layout, library=self.library)

    def adaptive_state(self, policy: CAdaptivePolicy) -> CAdaptiveState:
        return CAdaptiveState(policy, library=self.library)

    def build_candidate_a_target(self, utilities: np.ndarray, config: Any | None = None) -> tuple[np.ndarray, float, CCandidateATargetDiagnostics]:
        from bolr.config.foundation import SoftTargetConfig

        cfg = config or SoftTargetConfig()
        utilities_arr = _as_f64_vector(utilities)
        target = np.empty_like(utilities_arr)
        diagnostics = CandidateATargetDiagnosticsStruct()
        update_weight = ctypes.c_double()
        config_struct = CandidateATargetConfigStruct(
            float(cfg.kappa),
            float(cfg.eta),
            float(cfg.clip),
            float(cfg.absolute_tolerance),
            float(cfg.relative_tolerance),
            float(cfg.min_scale),
            int(bool(cfg.no_update_if_degenerate)),
        )
        status_ok(
            self.library.lib,
            self.library.lib.bolr_candidate_a_target_build(
                ctypes.byref(config_struct),
                self.library.const_vector(utilities_arr),
                self.library.vector(target),
                ctypes.byref(update_weight),
                ctypes.byref(diagnostics),
            ),
            operation="bolr_candidate_a_target_build",
        )
        return target, float(update_weight.value), CCandidateATargetDiagnostics.from_c(diagnostics)

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

    def ordered_partition(self, utilities: np.ndarray, config: Any | None = None) -> COrderedPartition:
        from bolr.config.foundation import OrderedPartitionConfig

        return COrderedPartition.from_utilities(utilities, config or OrderedPartitionConfig(), library=self.library)

    def ordered_partition_from_observation(self, observation: object) -> COrderedPartition:
        return COrderedPartition.from_observation(observation, library=self.library)

    def candidate_b_exact_observation(self, observation: object, *, normalize_pair_losses: bool = True) -> CCandidateBExactObservation:
        partition = self.ordered_partition_from_observation(observation)
        return CCandidateBExactObservation(partition, normalize_pair_losses=normalize_pair_losses, library=self.library)

    def candidate_b_sampled_observation(self, observation: object, config: Any) -> CCandidateBSampledObservation:
        materialized = self.materialize_candidate_b_pairs(observation, config)
        return CCandidateBSampledObservation(
            int(materialized["candidate_count"]),
            materialized["winner_indices"],
            materialized["loser_indices"],
            materialized["pair_weights"],
            update_weight=float(materialized["update_weight"]),
            possible_pair_count=int(materialized["possible_pair_count"]),
            duplicate_sample_count=int(materialized["duplicate_sample_count"]),
            normalize_pair_losses=bool(materialized["normalize_pair_losses"]),
            library=self.library,
        )

    def candidate_b_value_gradient_hvp(self, scores: np.ndarray, observation: object, config: Any, vector: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        from bolr.config.foundation import CrossGroupLogisticConfig

        if not isinstance(config, CrossGroupLogisticConfig):
            raise TypeError("Expected CrossGroupLogisticConfig.")
        handle: CCandidateBExactObservation | CCandidateBSampledObservation
        if config.sampled_pair_budget is None:
            handle = self.candidate_b_exact_observation(observation, normalize_pair_losses=bool(config.normalize_pair_losses))
        else:
            handle = self.candidate_b_sampled_observation(observation, config)
        try:
            return self._evaluate_operator(handle.operator(), scores, vector)
        finally:
            handle.close()

    @staticmethod
    def materialize_candidate_b_pairs(observation: object, config: Any) -> dict[str, Any]:
        from bolr.targets.ordered_partition import deterministic_sampling_seed

        ordered_groups = observation.ordered_groups
        active_pairs = [(a, b) for a in range(len(ordered_groups)) for b in range(a + 1, len(ordered_groups))]
        weights = np.full(len(active_pairs), 1.0 / len(active_pairs), dtype=np.float64) if active_pairs else np.empty(0, dtype=np.float64)
        possible_pair_count = int(observation.metadata["possible_pair_count"])
        duplicate_sample_count = 0
        used_pair_count = 0
        sampling_seed = None
        pair_budget = config.sampled_pair_budget
        rng = None
        if pair_budget is not None:
            sampling_seed = deterministic_sampling_seed(observation.metadata.get("date"), config.sampling_seed)
            rng = np.random.default_rng(sampling_seed)
        winners: list[np.ndarray] = []
        losers: list[np.ndarray] = []
        coeffs: list[np.ndarray] = []
        group_pair_allocations: dict[str, int] = {}
        for pair_idx, (a, b) in enumerate(active_pairs):
            group_a = np.asarray(ordered_groups[a], dtype=np.int64)
            group_b = np.asarray(ordered_groups[b], dtype=np.int64)
            all_pairs = np.array([(int(i), int(j)) for i in group_a for j in group_b], dtype=np.int64)
            possible = int(all_pairs.shape[0])
            if pair_budget is not None:
                alloc = max(1, pair_budget // len(active_pairs))
                alloc = min(alloc, possible) if not config.sampled_with_replacement else alloc
                sample_indices = rng.choice(possible, size=alloc, replace=config.sampled_with_replacement)
                sampled_pairs = all_pairs[sample_indices]
                duplicate_sample_count += int(sampled_pairs.shape[0] - np.unique(sample_indices).size)
            else:
                sampled_pairs = all_pairs
                alloc = possible
            used_pair_count += int(sampled_pairs.shape[0])
            group_pair_allocations[f"{a}>{b}"] = int(alloc)
            normalizer = float(sampled_pairs.shape[0]) if config.normalize_pair_losses and sampled_pairs.shape[0] > 0 else 1.0
            coeff = float(weights[pair_idx] / normalizer) if sampled_pairs.shape[0] > 0 else 0.0
            if sampled_pairs.shape[0] > 0:
                winners.append(sampled_pairs[:, 0].copy())
                losers.append(sampled_pairs[:, 1].copy())
                coeffs.append(np.full(sampled_pairs.shape[0], coeff, dtype=np.float64))
        winner_indices = np.concatenate(winners) if winners else np.empty(0, dtype=np.int64)
        loser_indices = np.concatenate(losers) if losers else np.empty(0, dtype=np.int64)
        pair_weights = np.concatenate(coeffs) if coeffs else np.empty(0, dtype=np.float64)
        return {
            "candidate_count": int(observation.candidate_to_group.size),
            "winner_indices": winner_indices,
            "loser_indices": loser_indices,
            "pair_weights": pair_weights,
            "possible_pair_count": possible_pair_count,
            "used_pair_count": used_pair_count,
            "duplicate_sample_count": duplicate_sample_count,
            "update_weight": float(observation.update_weight),
            "normalize_pair_losses": bool(config.normalize_pair_losses),
            "sampling_seed": sampling_seed,
            "pair_budget": pair_budget,
            "group_pair_allocations": group_pair_allocations,
        }

    def _evaluate_operator(self, operator: ObservationOperatorStruct, scores: np.ndarray, vector: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        scores_arr = _as_f64_vector(scores)
        vec_arr = np.zeros_like(scores_arr) if vector is None else _as_f64_vector(vector)
        gradient = np.empty_like(scores_arr)
        hvp = np.empty_like(scores_arr)
        value = ctypes.c_double()
        status_ok(self.library.lib, operator.value(operator.context, self.library.const_vector(scores_arr), ctypes.byref(value), None), operation="observation.value")
        status_ok(self.library.lib, operator.gradient(operator.context, self.library.const_vector(scores_arr), self.library.vector(gradient), None), operation="observation.gradient")
        status_ok(self.library.lib, operator.curvature_hvp(operator.context, self.library.const_vector(scores_arr), self.library.const_vector(vec_arr), self.library.vector(hvp), None), operation="observation.curvature_hvp")
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
        observation: Any,
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

    def adaptive_predict(self, policy: CAdaptivePolicy, state: CAdaptiveState, posterior: CGaussianState) -> tuple[CGaussianState, CAdaptationDiagnostics]:
        diagnostics, arrays = _alloc_adaptation_struct(policy.block_count)
        handle = ctypes.c_void_p()
        status_ok(
            self.library.lib,
            self.library.lib.bolr_adaptive_policy_predict(
                policy._require_open(),
                state._require_open(),
                posterior._require_open(),
                None,
                ctypes.byref(handle),
                ctypes.byref(diagnostics),
            ),
            operation="bolr_adaptive_policy_predict",
        )
        return CGaussianState(handle=handle, library=self.library, owner_refs=(policy, state, posterior)), CAdaptationDiagnostics.from_c(diagnostics, arrays)

    def adaptive_observe(
        self,
        policy: CAdaptivePolicy,
        state: CAdaptiveState,
        predictive: CGaussianState,
        posterior: CGaussianState,
        *,
        log_factor_at_predictive_mean: float,
        log_factor_at_posterior_mode: float,
        effective_strength: float,
        information_size: float,
        mahalanobis_update: float,
        gaussian_kl: float,
        objective_improvement: float,
        informative: bool = True,
    ) -> CAdaptationDiagnostics:
        diagnostics, arrays = _alloc_adaptation_struct(policy.block_count)
        surprise = SurpriseInputStruct(
            int(bool(informative)),
            float(log_factor_at_predictive_mean),
            float(log_factor_at_posterior_mode),
            float(effective_strength),
            float(information_size),
            float(mahalanobis_update),
            float(gaussian_kl),
            float(objective_improvement),
        )
        status_ok(
            self.library.lib,
            self.library.lib.bolr_adaptive_policy_observe(
                policy._require_open(),
                state._require_open(),
                predictive._require_open(),
                posterior._require_open(),
                ctypes.byref(surprise),
                ctypes.byref(diagnostics),
            ),
            operation="bolr_adaptive_policy_observe",
        )
        return CAdaptationDiagnostics.from_c(diagnostics, arrays)

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
