import json
from pathlib import Path

import numpy as np

from bolr.adaptation.attribution import block_innovation_attribution
from bolr.adaptation.bocpd import BOCPDDetector
from bolr.adaptation.policy import AdaptiveAdditiveTransitionPolicy, HeterogeneousDiscountTransitionPolicy
from bolr.adaptation.reset import PendingReset, apply_partial_reset
from bolr.adaptation.standardizer import EWStandardizer
from bolr.inference.laplace import laplace_update_composite
from bolr.initialization.prior import make_initial_dynamic_prior
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph import build_canonical_grid_graph, graph_energy
from bolr.model.graph_residual import (
    ProjectedLaplacianOperator,
    build_graph_residual_basis,
    build_smooth_subspace_projector,
    graph_penalty_shaped_process_noise,
    graph_residual_prior,
)
from bolr.model.diagnostics import innovation_diagnostics
from bolr.model.penalties import context_matrix_penalty, difference_matrix, difference_penalty, project_penalty, tensor_product_penalty
from bolr.model.priors import BlockDynamicsSpec, BlockPriorSpec, assemble_block_prior, assemble_block_process_noise
from bolr.model.score_blocks import DynamicSurfaceBlock, GraphResidualBlock, StaticBaselineBlock
from bolr.model.structured import penalty_shaped_process_noise, prior_from_penalty
from bolr.data.candidate_grid import CandidateGrid
from bolr.representation.tensor_basis import TensorProductBasis
from bolr.config.foundation import AdaptiveTransitionConfig, BlockAdaptationConfig, BOCPDConfig, SplineAxisConfig, SurpriseStandardizerConfig, TensorBasisConfig
from bolr.config.foundation import DecisionPolicyConfig, PosteriorSamplingConfig, RegionDefinitionConfig
from bolr.decision.metrics import probability_best_brier, top_k_brier
from bolr.decision.policies import HighestMassRegionDecisionPolicy, MaximumProbabilityBestDecisionPolicy, PosteriorMeanDecisionPolicy
from bolr.decision.prediction import build_posterior_prediction
from bolr.model.state_layout import make_state_layout
from bolr.posterior.state import GaussianPosterior
from bolr.observations.soft_target_gibbs import SoftTargetObservationModel
from bolr.targets.soft_target import SoftTargetBuilder


def test_composite_golden_fixture_regenerates_exactly() -> None:
    base = Path("tests/fixtures/golden")
    arrays = np.load(base / "composite_reference.npz")
    metadata = json.loads((base / "composite_reference.json").read_text())
    phi = arrays["phi"]
    alpha = arrays["alpha"]
    model = CompositeScoreModel.from_blocks(
        [StaticBaselineBlock("baseline", phi, alpha, {"fit": "golden"})],
        [DynamicSurfaceBlock("surface", phi)],
        {},
    )
    prior = make_initial_dynamic_prior(2, sigma0=1.0)
    observation = SoftTargetBuilder().build(np.array([1.0, 0.2, -0.4], dtype=float))
    result = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    assert metadata["schema_version"] == "phase_g_v1"
    assert np.allclose(model.static_scores({}), arrays["static_scores"])
    assert np.allclose(model.explicit_design_matrix({}), arrays["design"])
    assert np.allclose(result.posterior.mean, arrays["posterior_mean"])
    assert np.allclose(result.posterior.covariance, arrays["posterior_covariance"])


def test_structured_prior_golden_fixture_regenerates_exactly() -> None:
    base = Path("tests/fixtures/golden")
    arrays = np.load(base / "structured_prior_reference.npz")
    metadata = json.loads((base / "structured_prior_reference.json").read_text())

    assert metadata["schema_version"] == "phase_h_v1"

    diff1 = difference_matrix(5, 1)
    diff2 = difference_matrix(5, 2)
    assert np.allclose(diff1, arrays["difference_matrix_order_1"])
    assert np.allclose(diff2, arrays["difference_matrix_order_2"])

    coordinates = arrays["coordinates"]
    basis = TensorProductBasis(
        TensorBasisConfig(
            entry_basis=SplineAxisConfig(n_basis=3, degree=1),
            stop_basis=SplineAxisConfig(n_basis=4, degree=1),
            center=True,
            rank_tol=1e-10,
        )
    ).fit_transform(coordinates)
    assert np.allclose(basis.lift_matrix, arrays["lift_matrix"])

    raw_penalty = tensor_product_penalty(3, 4, entry_order=2, stop_order=2, entry_weight=1.5, stop_weight=0.7, ridge=0.2)
    reduced_penalty = project_penalty(raw_penalty, basis.lift_matrix)
    assert np.allclose(raw_penalty.matrix, arrays["raw_tensor_penalty"])
    assert np.allclose(reduced_penalty.matrix, arrays["reduced_penalty"])

    surface_penalty = difference_penalty(3, 2)
    surface_prior = prior_from_penalty(surface_penalty, smooth_weight=2.0, ridge=0.3)
    context_penalty = context_matrix_penalty(surface_penalty, np.diag([2.0, 0.5]), candidate_weight=1.2, context_weight=0.8, ridge=0.1)
    process_noise = penalty_shaped_process_noise(surface_penalty, scale=0.05, properization=0.2)
    assert np.allclose(surface_prior.precision, arrays["surface_prior_precision"])
    assert np.allclose(context_penalty.matrix, arrays["context_prior_precision"])
    assert np.allclose(process_noise.covariance, arrays["penalty_shaped_q"])

    phi = arrays["phi"]
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi)], {})
    prior = assemble_block_prior(
        model.layout,
        [BlockPriorSpec("surface", family="structured_gaussian", mean=surface_prior.mean, covariance=surface_prior.covariance)],
    )
    blockwise_q = assemble_block_process_noise(
        model.layout,
        [BlockDynamicsSpec("surface", family="penalty_shaped_random_walk", process_noise=process_noise)],
    )
    assert np.allclose(blockwise_q, arrays["blockwise_q"])

    observation = SoftTargetBuilder().build(arrays["utilities"])
    result = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())
    diagnostics = innovation_diagnostics(
        prior.mean,
        prior.covariance,
        result.posterior.mean,
        result.posterior.covariance,
        SoftTargetObservationModel().log_factor(model.scores({}, prior.mean), observation),
        SoftTargetObservationModel().log_factor(model.scores({}, result.posterior.mean), observation),
    )

    assert np.allclose(result.posterior.mean, arrays["posterior_mean"])
    assert np.allclose(result.posterior.covariance, arrays["posterior_covariance"])
    assert np.allclose(np.array([diagnostics["state_update_l2"], diagnostics["state_update_mahalanobis"], diagnostics["gaussian_kl"]]), arrays["innovation_summary"])


def test_graph_residual_golden_fixture_regenerates_exactly() -> None:
    base = Path("tests/fixtures/golden")
    arrays = np.load(base / "graph_residual_reference.npz")
    metadata = json.loads((base / "graph_residual_reference.json").read_text())
    assert metadata["schema_version"] == "phase_i_v1"

    grid = CandidateGrid(
        config_ids=np.arange(9),
        entry_values=np.repeat(np.array([0.1, 0.2, 0.3]), 3),
        stop_values=np.tile(np.array([0.1, 0.2, 0.3]), 3),
        pair_to_id={(float(e), float(s)): idx for idx, (e, s) in enumerate(zip(np.repeat([0.1, 0.2, 0.3], 3), np.tile([0.1, 0.2, 0.3], 3), strict=True))},
        grid_shape=(3, 3),
    )
    phi = arrays["phi"]
    graph = build_canonical_grid_graph(grid, entry_weight=1.2, stop_weight=0.8)
    projector = build_smooth_subspace_projector(phi)
    projected_operator = ProjectedLaplacianOperator(graph, projector)
    residual = build_graph_residual_basis(graph, phi, residual_dimension=3)
    residual_prior = graph_residual_prior(residual, graph_energy_weight=2.5, ridge_weight=0.4)
    process_noise = graph_penalty_shaped_process_noise(residual, scale=0.05, properization=0.3)
    model = CompositeScoreModel.from_blocks([], [DynamicSurfaceBlock("surface", phi), GraphResidualBlock("local", residual.basis)], {})
    prior = assemble_block_prior(
        model.layout,
        [
            BlockPriorSpec("surface", isotropic_scale=1.0),
            BlockPriorSpec("local", covariance=residual_prior.covariance, mean=residual_prior.mean),
        ],
    )
    observation = SoftTargetBuilder().build(arrays["utilities"])
    result = laplace_update_composite(prior, model, {}, observation, observation_model=SoftTargetObservationModel())

    assert np.array_equal(graph.edge_index, arrays["edge_index"])
    assert np.allclose(graph.degree, arrays["degree"])
    assert np.allclose(graph.laplacian @ arrays["test_vector"], arrays["laplacian_action"])
    assert np.allclose(projector.project(arrays["test_vector"]), arrays["projector_action"])
    assert np.allclose(projected_operator.apply(arrays["test_vector"]), arrays["projected_laplacian_action"])
    assert np.allclose(residual.subspace_projector(), arrays["residual_subspace_projector"], atol=1e-8)
    assert np.allclose(residual.eigenvalues, arrays["residual_eigenvalues"])
    assert np.allclose(residual.forward(arrays["residual_coefficients"]), arrays["residual_scores"])
    assert np.allclose(residual.transpose(arrays["test_vector"]), arrays["transpose_scores"])
    assert np.allclose(residual_prior.precision, arrays["graph_prior_precision"])
    assert np.allclose(process_noise.covariance, arrays["graph_process_noise"])
    assert np.allclose(model.scores({}, arrays["joint_state"]), arrays["composite_scores"])
    assert np.allclose(result.posterior.mean, arrays["posterior_mean"])
    assert np.allclose(result.posterior.covariance, arrays["posterior_covariance"])
    energy = graph_energy(arrays["residual_scores"], graph)
    assert np.isclose(energy["total_graph_energy"], arrays["residual_graph_energy"][0])


def test_adaptive_dynamics_golden_fixture_regenerates_exactly() -> None:
    base = Path("tests/fixtures/golden")
    arrays = np.load(base / "adaptive_dynamics_reference.npz")
    metadata = json.loads((base / "adaptive_dynamics_reference.json").read_text())
    assert metadata["schema_version"] == "phase_j_v1"

    standardizer = EWStandardizer(SurpriseStandardizerConfig(decay=0.5, variance_floor=1e-4, warmup_count=0, clip_z=None))
    s_state = standardizer.initial_state()
    s_state, first = standardizer.step(2.0, s_state)
    s_state, second = standardizer.step(4.0, s_state)
    assert np.allclose(np.array([first["z_score"], second["z_score"], s_state.mean, s_state.variance]), arrays["ew_summary"])

    detector = BOCPDDetector(BOCPDConfig(hazard=0.1, max_run_length=8, prior_mean=0.0, prior_kappa=1.0, prior_alpha=2.0, prior_beta=1.0))
    d_state = detector.initial_state()
    d_state, one_step = detector.step(0.5, d_state)
    assert np.allclose(one_step["run_length_posterior"], arrays["bocpd_one_step"])
    for value in arrays["bocpd_sequence"]:
        d_state, multi = detector.step(float(value), d_state)
    assert np.allclose(multi["run_length_posterior"], arrays["bocpd_multi_step"])

    layout = make_state_layout([{"name": "surface", "shape": (1,)}, {"name": "residual", "shape": (1,)}])
    posterior = GaussianPosterior(mean=np.zeros(2), covariance=np.array([[2.0, 0.5], [0.5, 1.0]]))
    discount_policy = HeterogeneousDiscountTransitionPolicy({"surface": 0.5, "residual": 0.8})
    discount_state = discount_policy.initial_state(layout=layout)
    discounted, _, _ = discount_policy.predict(posterior, discount_state, layout=layout)
    assert np.allclose(discounted.covariance, arrays["heterogeneous_discount_covariance"])

    attribution = block_innovation_attribution(layout, np.zeros(2), np.eye(2), np.array([1.0, 0.0]))
    assert np.allclose(np.array([attribution["surface"]["attribution_weight"], attribution["residual"]["attribution_weight"]]), arrays["block_attribution"])

    reset = PendingReset("surface", 0.5, np.zeros(1), np.eye(1))
    reset_applied = apply_partial_reset(GaussianPosterior(mean=np.array([2.0, 0.5]), covariance=np.array([[2.0, 0.4], [0.4, 1.0]])), layout, reset)
    assert np.allclose(reset_applied.mean, arrays["partial_reset_mean"])
    assert np.allclose(reset_applied.covariance, arrays["partial_reset_covariance"])

    adaptive = AdaptiveAdditiveTransitionPolicy(
        np.diag([0.2, 0.1]),
        AdaptiveTransitionConfig(
            standardizer=SurpriseStandardizerConfig(warmup_count=0),
            detector=BOCPDConfig(hazard=0.2, max_run_length=8),
            blocks=(
                BlockAdaptationConfig(block_name="surface", transition_family="additive", amplitude=2.0, decay=0.0),
                BlockAdaptationConfig(block_name="residual", transition_family="additive", amplitude=1.0, decay=0.0),
            ),
        ),
    )
    adaptive_state = adaptive.initial_state(layout=layout)
    predicted, adaptive_state, _ = adaptive.predict(GaussianPosterior(mean=np.zeros(2), covariance=np.eye(2)), adaptive_state, layout=layout)
    adaptive_state, adaptive_diag = adaptive.observe_update(
        predictive_posterior=predicted,
        posterior=GaussianPosterior(mean=np.array([1.0, 0.0]), covariance=np.array([[0.8, 0.0], [0.0, 0.9]])),
        observation_diagnostics={},
        block_diagnostics={},
        policy_state=adaptive_state,
        layout=layout,
        predictive_scores=np.array([0.0]),
        posterior_scores=np.array([1.0]),
        observation=type("Obs", (), {"update_weight": 1.0, "metadata": {}})(),
        observation_model=type("ObsModel", (), {"log_factor": lambda self, scores, obs: -2.0})(),
        date="2026-01-01",
    )
    assert np.allclose(np.array([adaptive_state.block_multipliers["surface"], adaptive_state.block_multipliers["residual"]]), arrays["adaptive_multipliers"])
    assert np.allclose(np.array([adaptive_diag["activation_value"], adaptive_diag["change_probability"]]), arrays["adaptive_cycle_summary"])


def test_posterior_decision_golden_fixture_regenerates_exactly() -> None:
    base = Path("tests/fixtures/golden")
    arrays = np.load(base / "posterior_decision_reference.npz")
    metadata = json.loads((base / "posterior_decision_reference.json").read_text())
    assert metadata["schema_version"] == "phase_k_v1"

    grid = CandidateGrid(
        config_ids=np.arange(6),
        entry_values=np.repeat(np.array([0.1, 0.2]), 3),
        stop_values=np.tile(np.array([0.1, 0.2, 0.3]), 2),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.1, 0.3): 2, (0.2, 0.1): 3, (0.2, 0.2): 4, (0.2, 0.3): 5},
        grid_shape=(2, 3),
    )
    graph = build_canonical_grid_graph(grid)
    prediction = build_posterior_prediction(
        date="2026-01-01",
        state_mean=arrays["state_mean"],
        state_covariance=arrays["state_covariance"],
        score_mean=arrays["score_mean"],
        design_matrix=arrays["design_matrix"],
        sampling_config=PosteriorSamplingConfig(sample_count=int(arrays["score_samples"].shape[0]), seed=11, retain_score_samples=True),
        top_k_values=(1, 2, 3),
        selected_covariance_indices=arrays["selected_covariance_indices"],
        pairwise_pairs=((0, 1), (0, 3)),
        candidate_grid=grid,
        graph=graph,
        region_config=RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold", edge_comembership_enabled=True),
        seed=11,
    )
    assert np.allclose(prediction.score_variance, arrays["score_variance"])
    assert np.allclose(prediction.selected_score_covariance, arrays["selected_score_covariance"])
    assert np.allclose(prediction.state_samples, arrays["state_samples"])
    assert np.allclose(prediction.score_samples, arrays["score_samples"])
    assert np.allclose(prediction.probability_best, arrays["probability_best"])
    assert np.allclose(prediction.probability_top_k[2], arrays["probability_top_2"])
    assert np.allclose(prediction.expected_rank, arrays["expected_rank"])
    assert np.allclose(np.array([item.analytic_probability_left_wins for item in prediction.pairwise_probabilities]), arrays["pairwise_analytic"])
    assert np.array_equal(prediction.region_summary.consensus_indices, arrays["consensus_indices"])
    assert np.allclose(np.array([component.inclusion_mass for component in prediction.region_summary.components]), arrays["region_masses"])

    mean_decision = PosteriorMeanDecisionPolicy().decide(prediction, candidate_grid=grid, graph=graph)
    prob_decision = MaximumProbabilityBestDecisionPolicy().decide(prediction, candidate_grid=grid, graph=graph)
    region_decision = HighestMassRegionDecisionPolicy(
        DecisionPolicyConfig(
            family="highest_mass_region",
            region_selection_statistic="probability_best",
            representative_policy="weighted_medoid",
        )
    ).decide(prediction, candidate_grid=grid, graph=graph)
    assert np.array_equal(np.array([mean_decision.selected_index, prob_decision.selected_index, region_decision.selected_index]), arrays["decision_indices"])

    utilities = arrays["utilities"]
    assert np.isclose(probability_best_brier(prediction.probability_best, utilities), arrays["probability_best_brier"][0])
    assert np.isclose(top_k_brier(prediction.probability_top_k[2], utilities, k=2), arrays["top_2_brier"][0])
