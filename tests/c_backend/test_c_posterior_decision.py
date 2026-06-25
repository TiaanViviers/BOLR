from __future__ import annotations

import numpy as np

from bolr.backend.c_backend import CBackend
from bolr.config.foundation import DecisionPolicyConfig, PosteriorSamplingConfig, RegionDefinitionConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.decision.metrics import probability_best_brier, region_coverage, top_k_brier
from bolr.decision.policies import (
    HighestMassRegionDecisionPolicy,
    MaximumProbabilityBestDecisionPolicy,
    MaximumProbabilityTopKDecisionPolicy,
    MinimumExpectedRankDecisionPolicy,
    PosteriorMeanDecisionPolicy,
)
from bolr.decision.prediction import build_posterior_prediction, pairwise_win_probabilities
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.posterior.state import GaussianPosterior


def _grid() -> CandidateGrid:
    return CandidateGrid(
        config_ids=np.arange(6),
        entry_values=np.repeat(np.array([0.1, 0.2]), 3),
        stop_values=np.tile(np.array([0.1, 0.2, 0.3]), 2),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.1, 0.3): 2, (0.2, 0.1): 3, (0.2, 0.2): 4, (0.2, 0.3): 5},
        grid_shape=(2, 3),
    )


def _artifacts(backend: CBackend, design: np.ndarray, static_scores: np.ndarray | None = None) -> tuple[object, object]:
    static_blocks = []
    if static_scores is not None:
        static_blocks.append(
            StaticBaselineBlock(
                "baseline",
                np.eye(int(np.asarray(static_scores).size), dtype=float),
                np.asarray(static_scores, dtype=float),
                {"fit": "test"},
            )
        )
    model = CompositeScoreModel.from_blocks(static_blocks, [DynamicSurfaceBlock("surface", np.asarray(design, dtype=float))], {})
    artifacts = backend.model_artifacts(model, {})
    return model, artifacts


def test_c_posterior_prediction_matches_dense_reference() -> None:
    backend = CBackend()
    arrays = np.load("tests/fixtures/golden/posterior_decision_reference.npz")
    static_scores = arrays["score_mean"] - arrays["design_matrix"] @ arrays["state_mean"]
    _, artifacts = _artifacts(backend, arrays["design_matrix"], static_scores)
    try:
        state = artifacts.state_from_posterior(GaussianPosterior(mean=arrays["state_mean"], covariance=arrays["state_covariance"]))
        with state:
            with backend.posterior_prediction(state, artifacts) as prediction:
                assert np.allclose(prediction.score_mean(), arrays["score_mean"])
                assert np.allclose(prediction.score_variance(), arrays["score_variance"])
                assert np.allclose(prediction.selected_score_covariance(arrays["selected_covariance_indices"]), arrays["selected_score_covariance"])
                pairwise = prediction.pairwise_probabilities(np.array([0, 0]), np.array([1, 2]))
                expected = pairwise_win_probabilities(
                    arrays["score_mean"],
                    arrays["design_matrix"],
                    arrays["state_covariance"],
                    ((0, 1), (0, 2)),
                )
                assert np.allclose([item.left_probability for item in pairwise], [item.analytic_probability_left_wins for item in expected])
    finally:
        artifacts.close()


def test_c_region_and_decision_policies_match_python_reference() -> None:
    backend = CBackend()
    grid = _grid()
    graph = build_canonical_grid_graph(grid)
    design = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.0],
            [0.8, 0.0],
            [0.0, 1.0],
            [0.0, 0.9],
            [0.0, 0.8],
        ]
    )
    state_mean = np.zeros(2)
    state_covariance = np.eye(2) * 0.1
    score_mean = np.array([1.0, 0.99, 0.98, 0.2, 0.1, 0.0])
    python_prediction = build_posterior_prediction(
        date=None,
        state_mean=state_mean,
        state_covariance=state_covariance,
        score_mean=score_mean,
        design_matrix=design,
        sampling_config=PosteriorSamplingConfig(sample_count=128, seed=3, retain_score_samples=True),
        top_k_values=(1, 2),
        candidate_grid=grid,
        graph=graph,
        region_config=RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold"),
        seed=3,
    )
    _, artifacts = _artifacts(backend, design, score_mean)
    try:
        state = artifacts.state_from_posterior(GaussianPosterior(mean=state_mean, covariance=state_covariance))
        with state:
            with backend.posterior_prediction(state, artifacts) as prediction:
                prediction.set_probability_best(python_prediction.probability_best)
                prediction.set_probability_top_k(2, python_prediction.probability_top_k[2])
                prediction.set_expected_rank(python_prediction.expected_rank)
                with backend.grid_graph(graph) as c_graph:
                    with backend.region_set(prediction, c_graph, RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold")) as regions:
                        assert np.array_equal(regions.consensus_indices(), python_prediction.region_summary.consensus_indices)
                        assert np.isclose(regions.summary(0).inclusion_mass, python_prediction.region_summary.components[0].inclusion_mass)

                        mean_policy = backend.decision_policy(DecisionPolicyConfig(family="posterior_mean_argmax"))
                        prob_best_policy = backend.decision_policy(DecisionPolicyConfig(family="maximum_probability_best"))
                        top_k_policy = backend.decision_policy(DecisionPolicyConfig(family="maximum_probability_top_k", top_k=2))
                        rank_policy = backend.decision_policy(DecisionPolicyConfig(family="minimum_expected_rank"))
                        region_policy = backend.decision_policy(
                            DecisionPolicyConfig(
                                family="highest_mass_region",
                                region_selection_statistic="probability_best",
                                representative_policy="weighted_medoid",
                            )
                        )
                        with mean_policy, prob_best_policy, top_k_policy, rank_policy, region_policy:
                            assert mean_policy.apply(prediction)[0].selected_index == PosteriorMeanDecisionPolicy().decide(python_prediction, candidate_grid=grid).selected_index
                            assert prob_best_policy.apply(prediction)[0].selected_index == MaximumProbabilityBestDecisionPolicy().decide(python_prediction, candidate_grid=grid).selected_index
                            assert top_k_policy.apply(prediction)[0].selected_index == MaximumProbabilityTopKDecisionPolicy(DecisionPolicyConfig(family="maximum_probability_top_k", top_k=2)).decide(python_prediction, candidate_grid=grid).selected_index
                            assert rank_policy.apply(prediction)[0].selected_index == MinimumExpectedRankDecisionPolicy().decide(python_prediction, candidate_grid=grid).selected_index
                            region_result, region_diag = region_policy.apply(prediction, regions, c_graph)
                            python_region = HighestMassRegionDecisionPolicy(
                                DecisionPolicyConfig(
                                    family="highest_mass_region",
                                    region_selection_statistic="probability_best",
                                    representative_policy="weighted_medoid",
                                )
                            ).decide(python_prediction, candidate_grid=grid, graph=graph)
                            assert region_result.selected_index == python_region.selected_index
                            assert region_result.selected_region_id == int(python_region.selected_region_id)
                            assert region_diag.selected_region_candidate_count == python_prediction.region_summary.components[0].candidate_count
    finally:
        artifacts.close()


def test_c_decision_calibration_matches_python_metrics() -> None:
    backend = CBackend()
    utilities = np.array([1.0, 1.0, -0.5])
    probability_best = np.array([0.6, 0.4, 0.0])
    probability_top_2 = np.array([1.0, 1.0, 0.0])
    assert np.allclose(backend.realized_best_distribution(utilities), np.array([0.5, 0.5, 0.0]))
    assert np.isclose(backend.probability_best_brier(probability_best, utilities), probability_best_brier(probability_best, utilities))
    assert np.isclose(backend.top_k_brier(probability_top_2, utilities, 2), top_k_brier(probability_top_2, utilities, k=2))
    assert backend.region_coverage(np.array([0, 2]), utilities) == region_coverage(np.array([0, 2]), utilities)
