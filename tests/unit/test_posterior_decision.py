import numpy as np

from bolr.config.foundation import DecisionPolicyConfig, PosteriorSamplingConfig, RegionDefinitionConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.decision.metrics import probability_best_brier, realized_best_distribution, region_coverage, top_k_brier
from bolr.decision.policies import (
    HighestMassRegionDecisionPolicy,
    MaximumProbabilityBestDecisionPolicy,
    MaximumProbabilityTopKDecisionPolicy,
    MinimumExpectedRankDecisionPolicy,
    PosteriorMeanDecisionPolicy,
    ThompsonDecisionPolicy,
)
from bolr.decision.prediction import (
    build_posterior_prediction,
    gaussian_state_samples,
    pairwise_win_probabilities,
    score_variance_diagonal,
    selected_score_covariance,
)
from bolr.model.graph import build_canonical_grid_graph


def _grid() -> CandidateGrid:
    return CandidateGrid(
        config_ids=np.arange(6),
        entry_values=np.repeat(np.array([0.1, 0.2]), 3),
        stop_values=np.tile(np.array([0.1, 0.2, 0.3]), 2),
        pair_to_id={(0.1, 0.1): 0, (0.1, 0.2): 1, (0.1, 0.3): 2, (0.2, 0.1): 3, (0.2, 0.2): 4, (0.2, 0.3): 5},
        grid_shape=(2, 3),
    )


def test_score_variance_and_selected_covariance_match_dense_formula() -> None:
    design = np.array([[1.0, 0.0], [0.5, -0.2], [0.0, 1.0]])
    covariance = np.array([[2.0, 0.3], [0.3, 1.5]])
    variance = score_variance_diagonal(design, covariance)
    selected = selected_score_covariance(design, covariance, np.array([0, 2]))
    assert np.allclose(variance, np.diag(design @ covariance @ design.T))
    assert np.allclose(selected, (design[[0, 2]] @ covariance @ design[[0, 2]].T))


def test_gaussian_state_sampling_is_deterministic_and_antithetic() -> None:
    config = PosteriorSamplingConfig(sample_count=4, seed=7, antithetic=True, retain_score_samples=True)
    samples = gaussian_state_samples(np.array([1.0, -1.0]), np.eye(2), config=config)
    samples_again = gaussian_state_samples(np.array([1.0, -1.0]), np.eye(2), config=config)
    assert np.allclose(samples, samples_again)
    centered = samples - np.array([1.0, -1.0])
    assert np.allclose(centered[0], -centered[2])
    assert np.allclose(centered[1], -centered[3])


def test_pairwise_probabilities_handle_zero_variance_and_match_monte_carlo() -> None:
    score_mean = np.array([2.0, 1.0, 2.0])
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    covariance = np.diag([0.5, 0.5])
    score_samples = np.array([[2.2, 1.0, 2.2], [1.9, 1.4, 1.9]])
    pairs = pairwise_win_probabilities(score_mean, design, covariance, ((0, 1), (0, 2)), score_samples=score_samples)
    assert pairs[0].analytic_probability_left_wins > 0.5
    assert pairs[0].monte_carlo_probability_left_wins == 1.0
    assert pairs[1].variance_difference == 0.0
    assert pairs[1].analytic_probability_left_wins == 1.0


def test_prediction_builds_probabilities_ranks_and_regions() -> None:
    grid = _grid()
    graph = build_canonical_grid_graph(grid)
    state_mean = np.array([0.0, 0.0])
    state_covariance = np.array([[0.5, 0.0], [0.0, 0.5]])
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
    score_mean = np.array([1.2, 1.15, 1.1, 0.4, 0.35, 0.3])
    prediction = build_posterior_prediction(
        date="2026-01-01",
        state_mean=state_mean,
        state_covariance=state_covariance,
        score_mean=score_mean,
        design_matrix=design,
        sampling_config=PosteriorSamplingConfig(sample_count=256, seed=4, retain_score_samples=True),
        top_k_values=(1, 2, 3),
        candidate_grid=grid,
        graph=graph,
        region_config=RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold", edge_comembership_enabled=True),
        seed=4,
    )
    assert prediction.probability_best is not None
    assert np.isclose(prediction.probability_best.sum(), 1.0)
    assert np.isclose(prediction.probability_top_k[2].sum(), 2.0, atol=1e-10)
    assert prediction.expected_rank.shape == (6,)
    assert prediction.region_summary is not None
    assert prediction.region_summary.consensus_indices.size >= 1
    assert len(prediction.region_summary.components) >= 1


def test_reference_decision_policies_are_deterministic() -> None:
    grid = _grid()
    graph = build_canonical_grid_graph(grid)
    prediction = build_posterior_prediction(
        date=None,
        state_mean=np.zeros(2),
        state_covariance=np.eye(2) * 0.1,
        score_mean=np.array([1.0, 0.99, 0.98, 0.2, 0.1, 0.0]),
        design_matrix=np.array([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.0, 1.0], [0.0, 0.9], [0.0, 0.8]]),
        sampling_config=PosteriorSamplingConfig(sample_count=128, seed=3, retain_score_samples=True),
        top_k_values=(1, 2),
        candidate_grid=grid,
        graph=graph,
        region_config=RegionDefinitionConfig(top_k=2, inclusion_threshold=0.2, consensus_family="threshold"),
        seed=3,
    )
    assert PosteriorMeanDecisionPolicy().decide(prediction, candidate_grid=grid).selected_index == 0
    assert MaximumProbabilityBestDecisionPolicy().decide(prediction, candidate_grid=grid).selected_index is not None
    assert MaximumProbabilityTopKDecisionPolicy(DecisionPolicyConfig(family="maximum_probability_top_k", top_k=2)).decide(prediction, candidate_grid=grid).selected_index is not None
    assert MinimumExpectedRankDecisionPolicy().decide(prediction, candidate_grid=grid).selected_index is not None
    assert ThompsonDecisionPolicy().decide(prediction, candidate_grid=grid).selected_index is not None
    region_policy = HighestMassRegionDecisionPolicy(
        DecisionPolicyConfig(
            family="highest_mass_region",
            region_selection_statistic="probability_best",
            representative_policy="weighted_medoid",
        )
    )
    region_decision = region_policy.decide(prediction, candidate_grid=grid, graph=graph)
    assert region_decision.selected_region_id is not None
    assert region_decision.selected_index is not None


def test_calibration_metrics_support_ties_and_region_coverage() -> None:
    prob_best = np.array([0.6, 0.4, 0.0])
    utilities = np.array([1.0, 1.0, -0.5])
    assert np.allclose(realized_best_distribution(utilities), np.array([0.5, 0.5, 0.0]))
    assert probability_best_brier(prob_best, utilities) >= 0.0
    assert top_k_brier(np.array([1.0, 1.0, 0.0]), utilities, k=2) == 0.0
    assert region_coverage(np.array([0, 2]), utilities)
