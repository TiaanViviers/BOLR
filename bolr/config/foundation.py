from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateGridConfig:
    config_id_column: str = "config_id"
    date_column: str = "date"
    entry_column: str = "entry_percentage"
    stop_column: str = "sl_trail_percentage"
    utility_column: str = "pnl"
    expected_entry_count: int = 34
    expected_stop_count: int = 42

    def __post_init__(self) -> None:
        for name in (
            self.config_id_column,
            self.date_column,
            self.entry_column,
            self.stop_column,
            self.utility_column,
        ):
            if not name:
                raise ValueError("Column names must be non-empty.")
        if self.expected_entry_count <= 0 or self.expected_stop_count <= 0:
            raise ValueError("Expected grid dimensions must be positive.")


@dataclass(frozen=True)
class CoordinateTransformConfig:
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


@dataclass(frozen=True)
class SplineAxisConfig:
    n_basis: int
    degree: int = 3

    def __post_init__(self) -> None:
        if self.n_basis <= 1:
            raise ValueError("n_basis must be at least 2.")
        if self.degree < 0:
            raise ValueError("degree must be non-negative.")
        if self.n_basis <= self.degree:
            raise ValueError("n_basis must exceed degree.")


@dataclass(frozen=True)
class TensorBasisConfig:
    entry_basis: SplineAxisConfig
    stop_basis: SplineAxisConfig
    center: bool = True
    rank_tol: float = 1e-10

    def __post_init__(self) -> None:
        if self.rank_tol <= 0.0:
            raise ValueError("rank_tol must be positive.")


@dataclass(frozen=True)
class SelectedColumnsContextConfig:
    columns: tuple[str, ...]
    add_intercept: bool = True
    scale: bool = True

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("At least one context column is required.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Context columns must be unique.")


@dataclass(frozen=True)
class SoftTargetConfig:
    kappa: float = 1.0
    eta: float = 1.0
    clip: float = 4.0
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    min_scale: float = 1e-6
    no_update_if_degenerate: bool = True

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.eta < 0.0:
            raise ValueError("eta must be non-negative.")
        if self.clip <= 0.0:
            raise ValueError("clip must be positive.")
        if self.absolute_tolerance < 0.0 or self.relative_tolerance < 0.0:
            raise ValueError("Tolerances must be non-negative.")
        if self.min_scale <= 0.0:
            raise ValueError("min_scale must be positive.")


@dataclass(frozen=True)
class HistoricalDatasetConfig:
    date_column: str = "date"
    config_id_column: str = "config_id"
    entry_column: str = "entry_percentage"
    stop_column: str = "sl_trail_percentage"
    utility_column: str = "pnl"
    expected_rows: int = 1_362_312
    expected_columns: int = 126
    expected_dates: int = 954
    expected_rows_per_date: int = 1428
    expected_start_date: str = "2021-01-29"
    expected_end_date: str = "2024-10-08"


@dataclass(frozen=True)
class StaticSurfaceConfig:
    regularization: float = 1.0
    max_iterations: int = 50
    gradient_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.gradient_tolerance <= 0.0:
            raise ValueError("gradient_tolerance must be positive.")


@dataclass(frozen=True)
class HistoricalRunConfig:
    warm_up_days: int = 504
    sigma0: float = 1.0
    random_walk_variance: float = 0.05
    target: SoftTargetConfig = SoftTargetConfig()
    static_surface: StaticSurfaceConfig = StaticSurfaceConfig()
    outputs_dir: str = "outputs/historical_candidate_a"

    def __post_init__(self) -> None:
        if self.warm_up_days <= 0:
            raise ValueError("warm_up_days must be positive.")
        if self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive.")
        if self.random_walk_variance < 0.0:
            raise ValueError("random_walk_variance must be non-negative.")


@dataclass(frozen=True)
class OrderedPartitionToleranceConfig:
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    execution_tolerance: float = 0.0
    robust_scale: str = "mad"
    scale_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0.0 or self.relative_tolerance < 0.0 or self.execution_tolerance < 0.0:
            raise ValueError("Tolerance components must be non-negative.")
        if self.robust_scale not in {"mad", "iqr", "max"}:
            raise ValueError("robust_scale must be one of {'mad', 'iqr', 'max'}.")
        if self.scale_floor <= 0.0:
            raise ValueError("scale_floor must be positive.")


@dataclass(frozen=True)
class OrderedPartitionConfig:
    tolerance: OrderedPartitionToleranceConfig = OrderedPartitionToleranceConfig()
    positive_threshold: float = 0.0
    all_irrelevant_policy: str = "no_update"
    reduced_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.all_irrelevant_policy not in {"always_relative", "no_update", "reduced_weight"}:
            raise ValueError("Unsupported all_irrelevant_policy.")
        if not (0.0 <= self.reduced_weight <= 1.0):
            raise ValueError("reduced_weight must be between 0 and 1.")


@dataclass(frozen=True)
class CrossGroupLogisticConfig:
    normalize_pair_losses: bool = True
    sampled_pair_budget: int | None = None
    sampled_with_replacement: bool = False
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        if self.sampled_pair_budget is not None and self.sampled_pair_budget <= 0:
            raise ValueError("sampled_pair_budget must be positive when provided.")


@dataclass(frozen=True)
class PenaltyConfig:
    family: str
    weight: float = 1.0
    ridge: float = 0.0
    difference_order: int | None = None
    axis_weights: tuple[float, ...] | None = None
    source_name: str | None = None

    def __post_init__(self) -> None:
        if self.family not in {"identity", "diagonal", "difference", "tensor_surface", "context_matrix", "named"}:
            raise ValueError("Unsupported penalty family.")
        if self.weight < 0.0:
            raise ValueError("Penalty weight must be non-negative.")
        if self.ridge < 0.0:
            raise ValueError("Penalty ridge must be non-negative.")
        if self.difference_order is not None and self.difference_order not in {1, 2}:
            raise ValueError("difference_order must be 1 or 2.")
        if self.axis_weights is not None and any(weight < 0.0 for weight in self.axis_weights):
            raise ValueError("axis_weights must be non-negative.")


@dataclass(frozen=True)
class BlockPriorConfig:
    block_name: str
    family: str
    mean: tuple[float, ...] | None = None
    isotropic_scale: float | None = None
    diagonal: tuple[float, ...] | None = None
    penalty: PenaltyConfig | None = None

    def __post_init__(self) -> None:
        if not self.block_name:
            raise ValueError("block_name must be non-empty.")
        if self.family not in {"isotropic_gaussian", "diagonal_gaussian", "structured_gaussian"}:
            raise ValueError("Unsupported block prior family.")
        if self.family == "isotropic_gaussian" and self.isotropic_scale is None:
            raise ValueError("isotropic_gaussian prior requires isotropic_scale.")
        if self.family == "diagonal_gaussian" and self.diagonal is None:
            raise ValueError("diagonal_gaussian prior requires diagonal entries.")
        if self.isotropic_scale is not None and self.isotropic_scale <= 0.0:
            raise ValueError("isotropic_scale must be positive.")
        if self.diagonal is not None and any(value <= 0.0 for value in self.diagonal):
            raise ValueError("diagonal entries must be positive.")
        if self.family == "structured_gaussian" and self.penalty is None:
            raise ValueError("structured_gaussian prior requires a penalty config.")


@dataclass(frozen=True)
class BlockDynamicsConfig:
    block_name: str
    family: str
    scale: float = 0.0
    properization: float | None = None
    diagonal: tuple[float, ...] | None = None
    frozen: bool = False

    def __post_init__(self) -> None:
        if not self.block_name:
            raise ValueError("block_name must be non-empty.")
        if self.family not in {"frozen", "isotropic_random_walk", "diagonal_random_walk", "penalty_shaped_random_walk", "covariance_discount"}:
            raise ValueError("Unsupported block dynamics family.")
        if self.family == "diagonal_random_walk" and self.diagonal is None:
            raise ValueError("diagonal_random_walk requires diagonal entries.")
        if self.family == "penalty_shaped_random_walk" and self.properization is None:
            raise ValueError("penalty_shaped_random_walk requires properization.")
        if self.scale < 0.0:
            raise ValueError("scale must be non-negative.")
        if self.properization is not None and self.properization <= 0.0:
            raise ValueError("properization must be positive when provided.")
        if self.diagonal is not None and any(value < 0.0 for value in self.diagonal):
            raise ValueError("diagonal entries must be non-negative.")


@dataclass(frozen=True)
class SurpriseStandardizerConfig:
    decay: float = 0.1
    variance_floor: float = 1e-6
    warmup_count: int = 1
    clip_z: float | None = 8.0

    def __post_init__(self) -> None:
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must lie in (0, 1].")
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive.")
        if self.warmup_count < 0:
            raise ValueError("warmup_count must be non-negative.")
        if self.clip_z is not None and self.clip_z <= 0.0:
            raise ValueError("clip_z must be positive when provided.")


@dataclass(frozen=True)
class BOCPDConfig:
    hazard: float = 0.05
    max_run_length: int = 64
    prior_mean: float = 0.0
    prior_kappa: float = 1.0
    prior_alpha: float = 2.0
    prior_beta: float = 1.0
    missing_policy: str = "hold"

    def __post_init__(self) -> None:
        if not (0.0 < self.hazard < 1.0):
            raise ValueError("hazard must lie in (0, 1).")
        if self.max_run_length <= 0:
            raise ValueError("max_run_length must be positive.")
        if self.prior_kappa <= 0.0 or self.prior_alpha <= 0.0 or self.prior_beta <= 0.0:
            raise ValueError("BOCPD prior parameters must be positive.")
        if self.missing_policy not in {"hold", "hazard_only"}:
            raise ValueError("missing_policy must be 'hold' or 'hazard_only'.")


@dataclass(frozen=True)
class BlockAdaptationConfig:
    block_name: str
    transition_family: str
    maximum_multiplier: float = 4.0
    minimum_multiplier: float = 1.0
    decay: float = 0.5
    attribution_floor: float = 1e-6
    minimum_discount: float | None = None
    reset_enabled: bool = False
    reset_threshold: float | None = None
    reset_strength: float | None = None
    reset_anchor: str | None = None
    reset_cooldown: int = 0
    amplitude: float = 1.0
    adaptive_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.block_name:
            raise ValueError("block_name must be non-empty.")
        if self.transition_family not in {"fixed", "additive", "discount", "zero_noise", "frozen"}:
            raise ValueError("Unsupported transition_family.")
        if self.maximum_multiplier < self.minimum_multiplier or self.minimum_multiplier <= 0.0:
            raise ValueError("Multiplier bounds are invalid.")
        if not (0.0 <= self.decay < 1.0):
            raise ValueError("decay must lie in [0, 1).")
        if self.attribution_floor < 0.0:
            raise ValueError("attribution_floor must be non-negative.")
        if self.minimum_discount is not None and not (0.0 < self.minimum_discount <= 1.0):
            raise ValueError("minimum_discount must lie in (0, 1].")
        if self.reset_strength is not None and not (0.0 <= self.reset_strength <= 1.0):
            raise ValueError("reset_strength must lie in [0, 1].")
        if self.reset_cooldown < 0:
            raise ValueError("reset_cooldown must be non-negative.")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative.")


@dataclass(frozen=True)
class AdaptiveTransitionConfig:
    surprise_signal: str = "generalized_predictive_loss"
    standardizer: SurpriseStandardizerConfig = SurpriseStandardizerConfig()
    detector: BOCPDConfig = BOCPDConfig()
    blocks: tuple[BlockAdaptationConfig, ...] = tuple()
    activation_family: str = "max_change_or_sigmoid"
    activation_parameters: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.activation_family not in {"max_change_or_sigmoid"}:
            raise ValueError("Unsupported activation_family.")
        block_names = [block.block_name for block in self.blocks]
        if len(set(block_names)) != len(block_names):
            raise ValueError("Adaptive block configurations must be unique.")


@dataclass(frozen=True)
class PosteriorSamplingConfig:
    sample_count: int = 0
    seed: int = 0
    antithetic: bool = False
    retain_score_samples: bool = False

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative.")


@dataclass(frozen=True)
class RegionDefinitionConfig:
    top_k: int | None = None
    top_fraction: float | None = None
    inclusion_threshold: float = 0.5
    consensus_family: str = "threshold"
    edge_comembership_enabled: bool = False

    def __post_init__(self) -> None:
        if self.top_k is None and self.top_fraction is None:
            raise ValueError("Either top_k or top_fraction must be provided.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")
        if self.top_fraction is not None and not (0.0 < self.top_fraction <= 1.0):
            raise ValueError("top_fraction must lie in (0, 1].")
        if self.consensus_family not in {"threshold", "top_count", "cumulative_mass"}:
            raise ValueError("Unsupported consensus_family.")
        if self.consensus_family == "threshold":
            if not (0.0 <= self.inclusion_threshold <= 1.0):
                raise ValueError("Threshold consensus requires inclusion_threshold in [0, 1].")
        elif self.inclusion_threshold <= 0.0:
            raise ValueError("Non-threshold consensus requires a positive inclusion_threshold.")


@dataclass(frozen=True)
class DecisionPolicyConfig:
    family: str = "posterior_mean_argmax"
    top_k: int | None = None
    region_selection_statistic: str | None = None
    representative_policy: str | None = None
    outside_option_provider: str | None = None

    def __post_init__(self) -> None:
        valid_families = {
            "posterior_mean_argmax",
            "maximum_probability_best",
            "maximum_probability_top_k",
            "minimum_expected_rank",
            "thompson",
            "highest_mass_region",
            "outside_option",
        }
        if self.family not in valid_families:
            raise ValueError("Unsupported decision policy family.")
        if self.family == "maximum_probability_top_k" and (self.top_k is None or self.top_k <= 0):
            raise ValueError("maximum_probability_top_k requires a positive top_k.")
        if self.family == "highest_mass_region":
            if self.region_selection_statistic not in {"probability_best", "inclusion_mass"}:
                raise ValueError("Region policies require a valid region_selection_statistic.")
            if self.representative_policy not in {
                "posterior_mean",
                "probability_best",
                "probability_top_k",
                "weighted_medoid",
            }:
                raise ValueError("Region policies require a valid representative_policy.")
        if self.family == "outside_option" and not self.outside_option_provider:
            raise ValueError("outside_option policy requires an outside_option_provider.")
