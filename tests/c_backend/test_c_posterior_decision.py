from __future__ import annotations

from types import SimpleNamespace

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
    ThompsonDecisionPolicy,
)
from bolr.decision.prediction import PosteriorPrediction, build_posterior_prediction, monte_carlo_rank_summaries, pairwise_win_probabilities
from bolr.model.composite import CompositeScoreModel
from bolr.model.graph import build_canonical_grid_graph
from bolr.model.score_blocks import DynamicSurfaceBlock, StaticBaselineBlock
from bolr.posterior.state import GaussianPosterior
from bolr.targets.soft_target import Observation as SoftTargetObservation


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


def test_c_monte_carlo_ranking_and_thompson_match_python_semantics() -> None:
    backend = CBackend()
    design = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    state_mean = np.array([0.2, -0.1], dtype=float)
    state_covariance = np.array([[0.3, 0.05], [0.05, 0.2]], dtype=float)
    score_mean = np.array([0.5, 0.45, 0.1, -0.05], dtype=float)
    _, artifacts = _artifacts(backend, design, score_mean)
    try:
        state = artifacts.state_from_posterior(GaussianPosterior(mean=state_mean, covariance=state_covariance))
        with state:
            with backend.posterior_prediction(state, artifacts) as prediction:
                with backend.rng(seed=17, stream=3) as rng:
                    diagnostics = prediction.monte_carlo_rank(
                        rng,
                        96,
                        top_k_values=(1, 2, 3),
                        antithetic=True,
                        retain_score_samples=True,
                    )
                assert diagnostics.sample_count == 96
                assert diagnostics.retained_score_sample_count == 96
                assert prediction.score_sample_count == 96
                score_samples = np.vstack([prediction.score_sample(i) for i in range(prediction.score_sample_count)])
                rank_stats = monte_carlo_rank_summaries(score_samples, top_k_values=(1, 2, 3))
                assert np.allclose(prediction.probability_best(), rank_stats["probability_best"])
                assert np.allclose(prediction.probability_top_k(2), rank_stats["probability_top_k"][2])
                assert np.allclose(prediction.expected_rank(), rank_stats["expected_rank"])
                assert np.allclose(prediction.rank_stddev(), rank_stats["rank_stddev"])
                assert diagnostics.tie_count == int(rank_stats["tie_count"])

                python_prediction = PosteriorPrediction(
                    date=None,
                    score_mean=prediction.score_mean(),
                    score_variance=prediction.score_variance(),
                    state_mean=prediction.state_mean(),
                    state_covariance=prediction.state_covariance(),
                    probability_best=prediction.probability_best(),
                    probability_top_k={1: prediction.probability_top_k(1), 2: prediction.probability_top_k(2), 3: prediction.probability_top_k(3)},
                    expected_rank=prediction.expected_rank(),
                    rank_stddev=prediction.rank_stddev(),
                    score_samples=score_samples,
                )
                with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as policy:
                    c_decision, _ = policy.apply(prediction)
                py_decision = ThompsonDecisionPolicy().decide(python_prediction, candidate_grid=None)
                assert c_decision.selected_index == py_decision.selected_index
    finally:
        artifacts.close()


def test_c_streaming_monte_carlo_ranking_matches_retained_reference() -> None:
    backend = CBackend()
    design = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    state_mean = np.array([0.2, -0.1], dtype=float)
    state_covariance = np.array([[0.3, 0.05], [0.05, 0.2]], dtype=float)
    score_mean = np.array([0.5, 0.45, 0.1, -0.05], dtype=float)
    _, artifacts = _artifacts(backend, design, score_mean)
    try:
        state = artifacts.state_from_posterior(GaussianPosterior(mean=state_mean, covariance=state_covariance))
        with state:
            with backend.posterior_prediction(state, artifacts) as retained_prediction:
                with backend.rng(seed=23, stream=5) as retained_rng:
                    retained_diag = retained_prediction.monte_carlo_rank(
                        retained_rng,
                        96,
                        top_k_values=(1, 2, 3),
                        antithetic=True,
                        retain_score_samples=True,
                    )

                with backend.posterior_prediction(state, artifacts) as streaming_prediction:
                    with backend.rng(seed=23, stream=5) as streaming_rng:
                        streaming_diag = streaming_prediction.monte_carlo_rank_streaming(
                            streaming_rng,
                            96,
                            chunk_size=11,
                            top_k_values=(1, 2, 3),
                            antithetic=True,
                            retention="sample_zero",
                        )

                    assert streaming_diag.sample_count == retained_diag.sample_count
                    assert streaming_diag.tie_count == retained_diag.tie_count
                    assert streaming_diag.retained_score_sample_count == 1
                    assert streaming_prediction.score_sample_count == 1
                    assert np.allclose(streaming_prediction.probability_best(), retained_prediction.probability_best())
                    assert np.allclose(streaming_prediction.probability_top_k(2), retained_prediction.probability_top_k(2))
                    assert np.allclose(streaming_prediction.expected_rank(), retained_prediction.expected_rank())
                    assert np.allclose(streaming_prediction.rank_stddev(), retained_prediction.rank_stddev())
                    assert np.allclose(streaming_prediction.score_sample(0), retained_prediction.score_sample(0))

                    with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as policy:
                        retained_decision, _ = policy.apply(retained_prediction)
                        streaming_decision, _ = policy.apply(streaming_prediction)
                    assert streaming_decision.selected_index == retained_decision.selected_index
    finally:
        artifacts.close()


def test_c_replay_engine_checkpoint_resume_matches_uninterrupted_sequence() -> None:
    backend = CBackend()
    design = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    state_mean = np.zeros(2, dtype=float)
    state_covariance = np.eye(2, dtype=float) * 0.2
    score_mean = np.array([0.4, 0.2, -0.1], dtype=float)
    target = np.array([0.6, 0.3, 0.1], dtype=float)
    _, artifacts = _artifacts(backend, design, score_mean)
    transition = SimpleNamespace(
        family="additive",
        process_noise=np.eye(2, dtype=float) * 0.01,
        global_discount=0.0,
        block_discount_scales=None,
    )
    try:
        posterior_a = artifacts.state_from_posterior(GaussianPosterior(mean=state_mean, covariance=state_covariance))
        posterior_b = artifacts.state_from_posterior(GaussianPosterior(mean=state_mean, covariance=state_covariance))
        with posterior_a, posterior_b:
            with backend.rng(seed=7, stream=2) as rng_a, backend.rng(seed=7, stream=2) as rng_b:
                with backend.replay_engine_fixed(posterior_a, transition, rng_a) as resumed_engine:
                    with backend.replay_engine_fixed(posterior_b, transition, rng_b) as direct_engine:
                        with backend.decision_policy(DecisionPolicyConfig(family="thompson")) as policy:
                            resumed_decision, resumed_begin = resumed_engine.begin_day(
                                artifacts,
                                policy,
                                ranking_sample_count=16,
                                chunk_size=5,
                                top_k_values=(1, 2),
                                antithetic=True,
                                retention="sample_zero",
                            )
                            assert resumed_engine.phase == 2
                            assert resumed_begin.phase == 2
                            assert resumed_engine.pending_selected_index == resumed_decision.selected_index

                            with resumed_engine.export_checkpoint() as checkpoint:
                                assert checkpoint.phase == 2
                                assert checkpoint.pending_selected_index == resumed_decision.selected_index
                                with backend.replay_engine_import_fixed(checkpoint) as restored_engine:
                                    with backend.candidate_a_observation(
                                        SoftTargetObservation(
                                            type="SOFT_TARGET",
                                            utility_values=target.copy(),
                                            transformed_values=target.copy(),
                                            target_probabilities=target,
                                            tolerance=0.0,
                                            update_weight=1.0,
                                            metadata={},
                                        )
                                    ) as observation:
                                        restored_laplace, restored_finish = restored_engine.finish_day(
                                            artifacts,
                                            observation,
                                            effective_strength=1.0,
                                            information_size=1.0,
                                            informative=True,
                                        )
                                        direct_decision, direct_begin = direct_engine.begin_day(
                                            artifacts,
                                            policy,
                                            ranking_sample_count=16,
                                            chunk_size=5,
                                            top_k_values=(1, 2),
                                            antithetic=True,
                                            retention="sample_zero",
                                        )
                                        direct_laplace, direct_finish = direct_engine.finish_day(
                                            artifacts,
                                            observation,
                                            effective_strength=1.0,
                                            information_size=1.0,
                                            informative=True,
                                        )

                                    assert direct_engine.phase == 1
                                    assert restored_engine.phase == 1
                                    assert direct_begin.phase == 2
                                    assert direct_finish.phase_after == 1
                                    assert restored_finish.phase_after == 1
                                    assert direct_decision.selected_index == resumed_decision.selected_index
                                    assert np.allclose(
                                        restored_engine.posterior_mean(artifacts.state_dimension),
                                        direct_engine.posterior_mean(artifacts.state_dimension),
                                    )
                                    assert np.allclose(
                                        restored_engine.posterior_covariance(artifacts.state_dimension),
                                        direct_engine.posterior_covariance(artifacts.state_dimension),
                                    )
                                    assert np.isclose(restored_laplace.objective_improvement, direct_laplace.objective_improvement)
                                    assert np.isclose(restored_finish.posterior_trace, direct_finish.posterior_trace)
    finally:
        artifacts.close()
