#ifndef BOLR_PREDICTION_H
#define BOLR_PREDICTION_H

#include "bolr/gaussian.h"
#include "bolr/rng.h"
#include "bolr/score.h"
#include "bolr/workspace.h"

typedef struct bolr_posterior_prediction bolr_posterior_prediction;
typedef struct bolr_rank_accumulator bolr_rank_accumulator;

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

typedef struct {
    bolr_index sample_count;
    bolr_index top_k_count;
    bolr_index retained_score_sample_count;
    bolr_index retained_state_sample_count;
    bolr_index tie_count;
    bolr_real winner_entropy;
    bolr_real effective_winner_count;
} bolr_monte_carlo_ranking_diagnostics;

typedef enum {
    BOLR_SCORE_RETENTION_NONE = 0,
    BOLR_SCORE_RETENTION_SAMPLE_ZERO = 1,
    BOLR_SCORE_RETENTION_ALL = 2
} bolr_score_retention;

bolr_status bolr_rank_accumulator_create(
    bolr_index candidate_count,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    const bolr_allocator *allocator,
    bolr_rank_accumulator **out_accumulator
);
void bolr_rank_accumulator_destroy(bolr_rank_accumulator *accumulator);
bolr_status bolr_rank_accumulator_reset(bolr_rank_accumulator *accumulator);
bolr_status bolr_rank_accumulator_accumulate_scores(bolr_rank_accumulator *accumulator, bolr_const_matrix_view score_samples);
bolr_status bolr_rank_accumulator_merge(bolr_rank_accumulator *destination, const bolr_rank_accumulator *source);
bolr_status bolr_rank_accumulator_copy_probability_best(const bolr_rank_accumulator *accumulator, bolr_vector_view output);
bolr_status bolr_rank_accumulator_copy_probability_top_k(const bolr_rank_accumulator *accumulator, bolr_index top_k, bolr_vector_view output);
bolr_status bolr_rank_accumulator_copy_expected_rank(const bolr_rank_accumulator *accumulator, bolr_vector_view output);
bolr_status bolr_rank_accumulator_copy_rank_stddev(const bolr_rank_accumulator *accumulator, bolr_vector_view output);
uint64_t bolr_rank_accumulator_sample_count(const bolr_rank_accumulator *accumulator);
bolr_index bolr_rank_accumulator_tie_count(const bolr_rank_accumulator *accumulator);

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
bolr_status bolr_posterior_prediction_set_rank_stddev(bolr_posterior_prediction *prediction, bolr_const_vector_view rank_stddev);
bolr_status bolr_posterior_prediction_copy_probability_best(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_probability_top_k(const bolr_posterior_prediction *prediction, bolr_index top_k, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_expected_rank(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_rank_stddev(const bolr_posterior_prediction *prediction, bolr_vector_view output);
bolr_status bolr_posterior_prediction_copy_score_sample(const bolr_posterior_prediction *prediction, bolr_index sample_index, bolr_vector_view output);
bolr_index bolr_posterior_prediction_score_sample_count(const bolr_posterior_prediction *prediction);
bolr_status bolr_posterior_prediction_monte_carlo_rank(
    bolr_posterior_prediction *prediction,
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_rng *rng,
    bolr_index sample_count,
    int antithetic,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    int retain_score_samples,
    int retain_state_samples,
    bolr_workspace *workspace,
    bolr_monte_carlo_ranking_diagnostics *diagnostics
);
bolr_status bolr_posterior_prediction_monte_carlo_rank_streaming(
    bolr_posterior_prediction *prediction,
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_rng *rng,
    bolr_index sample_count,
    bolr_index chunk_size,
    int antithetic,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    bolr_score_retention retention,
    bolr_workspace *workspace,
    bolr_monte_carlo_ranking_diagnostics *diagnostics
);
bolr_status bolr_probability_entropy(bolr_const_vector_view probabilities, bolr_real *out_entropy, bolr_real *out_effective_count, bolr_real *out_maximum);

#endif
