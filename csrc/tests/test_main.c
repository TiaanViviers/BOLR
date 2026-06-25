#include "test_suite.h"

#include <stdio.h>

int main(void) {
    int failures = 0;
    if (test_math() != 0) { fprintf(stderr, "test_math failed\n"); failures += 1; }
    if (test_linalg() != 0) { fprintf(stderr, "test_linalg failed\n"); failures += 1; }
    if (test_ownership() != 0) { fprintf(stderr, "test_ownership failed\n"); failures += 1; }
    if (test_state_layout() != 0) { fprintf(stderr, "test_state_layout failed\n"); failures += 1; }
    if (test_score() != 0) { fprintf(stderr, "test_score failed\n"); failures += 1; }
    if (test_candidate_a() != 0) { fprintf(stderr, "test_candidate_a failed\n"); failures += 1; }
    if (test_candidate_a_target() != 0) { fprintf(stderr, "test_candidate_a_target failed\n"); failures += 1; }
    if (test_ordered_partition() != 0) { fprintf(stderr, "test_ordered_partition failed\n"); failures += 1; }
    if (test_candidate_b_exact() != 0) { fprintf(stderr, "test_candidate_b_exact failed\n"); failures += 1; }
    if (test_candidate_b_sampled() != 0) { fprintf(stderr, "test_candidate_b_sampled failed\n"); failures += 1; }
    if (test_candidate_b_laplace() != 0) { fprintf(stderr, "test_candidate_b_laplace failed\n"); failures += 1; }
    if (test_standardizer() != 0) { fprintf(stderr, "test_standardizer failed\n"); failures += 1; }
    if (test_bocpd() != 0) { fprintf(stderr, "test_bocpd failed\n"); failures += 1; }
    if (test_attribution() != 0) { fprintf(stderr, "test_attribution failed\n"); failures += 1; }
    if (test_partial_reset() != 0) { fprintf(stderr, "test_partial_reset failed\n"); failures += 1; }
    if (test_adaptive_policy() != 0) { fprintf(stderr, "test_adaptive_policy failed\n"); failures += 1; }
    if (test_adaptive_checkpoint() != 0) { fprintf(stderr, "test_adaptive_checkpoint failed\n"); failures += 1; }
    if (test_score_uncertainty() != 0) { fprintf(stderr, "test_score_uncertainty failed\n"); failures += 1; }
    if (test_pairwise_probability() != 0) { fprintf(stderr, "test_pairwise_probability failed\n"); failures += 1; }
    if (test_decision_policy() != 0) { fprintf(stderr, "test_decision_policy failed\n"); failures += 1; }
    if (test_consensus_region() != 0) { fprintf(stderr, "test_consensus_region failed\n"); failures += 1; }
    if (test_connected_components() != 0) { fprintf(stderr, "test_connected_components failed\n"); failures += 1; }
    if (test_region_summary() != 0) { fprintf(stderr, "test_region_summary failed\n"); failures += 1; }
    if (test_graph_medoid() != 0) { fprintf(stderr, "test_graph_medoid failed\n"); failures += 1; }
    if (test_decision_calibration() != 0) { fprintf(stderr, "test_decision_calibration failed\n"); failures += 1; }
    if (test_penalty() != 0) { fprintf(stderr, "test_penalty failed\n"); failures += 1; }
    if (test_dynamics() != 0) { fprintf(stderr, "test_dynamics failed\n"); failures += 1; }
    if (test_golden() != 0) { fprintf(stderr, "test_golden failed\n"); failures += 1; }
    if (test_gaussian_state() != 0) { fprintf(stderr, "test_gaussian_state failed\n"); failures += 1; }
    if (test_gaussian_prediction() != 0) { fprintf(stderr, "test_gaussian_prediction failed\n"); failures += 1; }
    if (test_posterior_objective() != 0) { fprintf(stderr, "test_posterior_objective failed\n"); failures += 1; }
    if (test_newton() != 0) { fprintf(stderr, "test_newton failed\n"); failures += 1; }
    if (test_laplace() != 0) { fprintf(stderr, "test_laplace failed\n"); failures += 1; }
    if (test_static_fit() != 0) { fprintf(stderr, "test_static_fit failed\n"); failures += 1; }
    if (test_inference_ownership() != 0) { fprintf(stderr, "test_inference_ownership failed\n"); failures += 1; }
    if (failures != 0) {
        fprintf(stderr, "bolr_tests: %d failures\n", failures);
        return 1;
    }
    return 0;
}
