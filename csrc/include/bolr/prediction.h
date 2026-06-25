#ifndef BOLR_PREDICTION_H
#define BOLR_PREDICTION_H

#include "bolr/gaussian.h"
#include "bolr/score.h"
#include "bolr/workspace.h"

typedef struct bolr_posterior_prediction bolr_posterior_prediction;

typedef struct {
    bolr_real left_probability;
    bolr_real mean_difference;
    bolr_real variance_difference;
} bolr_pairwise_probability_result;

typedef struct {
    bolr_real score_mean_norm;
    bolr_real score_variance_sum;
    bolr_real explicit_design_frobenius_norm;
} bolr_posterior_prediction_diagnostics;

bolr_status bolr_posterior_prediction_create(
    const bolr_gaussian_state *predictive,
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_workspace *workspace,
    const bolr_allocator *allocator,
    bolr_posterior_prediction **out_prediction,
    bolr_posterior_prediction_diagnostics *diagnostics
);
void bolr_posterior_prediction_destroy(bolr_posterior_prediction *prediction);
bolr_index bolr_posterior_prediction_candidate_count(const bolr_posterior_prediction *prediction);
bolr_index bolr_posterior_prediction_state_dim(const bolr_posterior_prediction *prediction);
uint64_t bolr_posterior_prediction_model_schema_hash(const bolr_posterior_prediction *prediction);
bolr_status bolr_posterior_prediction_copy_score_mean(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_score_variance(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_state_mean(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_state_covariance(const bolr_posterior_prediction *prediction, bolr_matrix_view output);
bolr_status bolr_selected_score_covariance(
    const bolr_posterior_prediction *prediction,
    const bolr_index *indices,
    bolr_index count,
    bolr_matrix_view output
);
bolr_status bolr_pairwise_probability(
    const bolr_posterior_prediction *prediction,
    const bolr_index *left_indices,
    const bolr_index *right_indices,
    bolr_index count,
    bolr_pairwise_probability_result *output
);
bolr_status bolr_posterior_prediction_set_probability_best(bolr_posterior_prediction *prediction, bolr_const_vector_view probability_best);
bolr_status bolr_posterior_prediction_set_probability_top_k(bolr_posterior_prediction *prediction, bolr_index top_k, bolr_const_vector_view probability_top_k);
bolr_status bolr_posterior_prediction_set_expected_rank(bolr_posterior_prediction *prediction, bolr_const_vector_view expected_rank);
bolr_status bolr_posterior_prediction_copy_probability_best(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_probability_top_k(const bolr_posterior_prediction *prediction, bolr_index top_k, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_expected_rank(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_probability_entropy(bolr_const_vector_view probabilities, bolr_real *out_entropy, bolr_real *out_effective_count, bolr_real *out_maximum);

#endif
