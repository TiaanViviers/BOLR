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
