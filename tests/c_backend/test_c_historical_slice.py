from __future__ import annotations

from pathlib import Path

import numpy as np

from bolr.backend.c_backend import CBackend, CInferenceWorkspace, CNewtonConfig
from bolr.config.foundation import CandidateGridConfig, SoftTargetConfig, SplineAxisConfig, TensorBasisConfig
from bolr.data.candidate_grid import load_candidate_grid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.initialization.static_surface import fit_static_surface
from bolr.inference.laplace import laplace_update_composite
from bolr.model.composite import CompositeScoreModel
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.posterior.state import GaussianPosterior
from bolr.representation.coordinates import LogCoordinateTransform
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.targets.soft_target import SoftTargetBuilder


def test_c_historical_slice_matches_python_reference() -> None:
    project_root = Path(__file__).resolve().parents[2]
    grid = load_candidate_grid(project_root / "data" / "YM_grid.csv", CandidateGridConfig())
    dataset = HistoricalDataset.from_parquet(project_root / "data" / "YM_full.parquet", candidate_grid=grid)
    coordinates = LogCoordinateTransform().fit(grid.entry_values, grid.stop_values).transform(grid.entry_values, grid.stop_values)
    candidate_basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=6, degree=3),
            stop_basis=SplineAxisConfig(n_basis=8, degree=3),
        )
    ).fit_transform(coordinates).reduced_basis
    warm_up_days = 504
    target_builder = SoftTargetBuilder(SoftTargetConfig())
    warm_up_observations = [target_builder.build(dataset.day_frame(date)["pnl"].to_numpy(dtype=float), date=date) for date in dataset.dates[:warm_up_days]]
    static_surface = fit_static_surface(candidate_basis, warm_up_observations, observation_model=SoftTargetObservationModel())
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", candidate_basis, static_surface.coefficients, {"fit": "historical_slice"})],
        [DynamicSurfaceBlock("surface", candidate_basis)],
        {},
    )
    process_noise = np.eye(candidate_basis.shape[1]) * 0.05
    python_posterior = make_initial_dynamic_prior(candidate_basis.shape[1], sigma0=1.0)
    backend = CBackend()

    with backend.model_artifacts(model, {}) as artifacts:
        context_handle = backend.score_context(model, {})
        c_state = artifacts.state_from_posterior(python_posterior)
        try:
            with CInferenceWorkspace(artifacts.state_dimension, artifacts.candidate_count, library=backend.library) as workspace:
                for date in dataset.dates[warm_up_days : warm_up_days + 10]:
                    dataset.get_predictors(date)
                    observation = target_builder.build(dataset.reveal_outcomes(date).pnl, date=date)
                    python_predictive = GaussianPosterior(
                        mean=python_posterior.mean.copy(),
                        covariance=python_posterior.covariance + process_noise,
                        state_layout=python_posterior.state_layout,
                    )
                    c_predictive, _ = c_state.predict_additive(process_noise)
                    python_scores = model.scores({}, python_predictive.mean)
                    with c_predictive:
                        c_scores = artifacts.scores(c_predictive.mean(), context_handle)
                        assert np.allclose(c_scores, python_scores, atol=1e-9, rtol=1e-8), date
                        python_result = laplace_update_composite(
                            python_predictive,
                            model,
                            {},
                            observation,
                            observation_model=SoftTargetObservationModel(),
                        )
                        with backend.candidate_a_observation(observation) as obs_handle:
                            c_posterior_handle, _ = backend.laplace_update(
                                c_predictive,
                                artifacts,
                                context_handle,
                                obs_handle,
                                workspace,
                                CNewtonConfig(),
                            )
                            with c_posterior_handle:
                                c_posterior = c_posterior_handle.to_posterior(state_layout=model.layout.metadata(), timestamp=date)
                                assert np.allclose(c_posterior.mean, python_result.posterior.mean, atol=1e-8, rtol=1e-8), date
                                assert np.allclose(c_posterior.covariance, python_result.posterior.covariance, atol=1e-8, rtol=1e-8), date
                        c_state.close()
                        c_state = artifacts.state_from_posterior(c_posterior)
                    python_posterior = python_result.posterior
        finally:
            c_state.close()
