"""Backend contracts and C bindings."""

from bolr.backend.base import NumpyBackend, NumericalBackend
from bolr.backend.c_backend import (
    CBackend,
    CCheckpointState,
    CGaussianState,
    CInferenceWorkspace,
    CLaplaceDiagnostics,
    CLibrary,
    CModelArtifacts,
    CNewtonConfig,
    CScoreContext,
    CStateLayout,
    CStaticFitDataset,
    CCandidateAObservation,
    CWorkspace,
)

__all__ = [
    "NumericalBackend",
    "NumpyBackend",
    "CBackend",
    "CLibrary",
    "CWorkspace",
    "CGaussianState",
    "CCheckpointState",
    "CStateLayout",
    "CModelArtifacts",
    "CScoreContext",
    "CInferenceWorkspace",
    "CCandidateAObservation",
    "CNewtonConfig",
    "CLaplaceDiagnostics",
    "CStaticFitDataset",
]
