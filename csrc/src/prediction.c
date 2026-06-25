#include "bolr/prediction.h"

#include "bolr/linalg.h"
#include "bolr/math.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    bolr_real value;
    bolr_index index;
} bolr_score_index_pair;

static bolr_status alloc_real_copy(const bolr_allocator *allocator, bolr_index count, const bolr_real *source, bolr_real **out) {
    bolr_real *data;
    size_t bytes;
    if (count < 0) return BOLR_INVALID_ARGUMENT;
    if (bolr_checked_size_mul((size_t) count, sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    data = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    memcpy(data, source, bytes);
    *out = data;
    return BOLR_OK;
}

static bolr_status alloc_real_zero(const bolr_allocator *allocator, bolr_index count, bolr_real **out) {
    bolr_real *data = (bolr_real *) bolr_allocator_calloc(allocator, (size_t) count, sizeof(bolr_real));
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    *out = data;
    return BOLR_OK;
}

static bolr_status alloc_u64_zero(const bolr_allocator *allocator, bolr_index count, uint64_t **out) {
    uint64_t *data = (uint64_t *) bolr_allocator_calloc(allocator, (size_t) count, sizeof(uint64_t));
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    *out = data;
    return BOLR_OK;
}

static int compare_score_index_pair(const void *left, const void *right) {
    const bolr_score_index_pair *a = (const bolr_score_index_pair *) left;
    const bolr_score_index_pair *b = (const bolr_score_index_pair *) right;
    if (a->value > b->value) return -1;
    if (a->value < b->value) return 1;
    if (a->index < b->index) return -1;
    if (a->index > b->index) return 1;
    return 0;
}

static bolr_index rank_accumulator_top_k_slot(const struct bolr_rank_accumulator *accumulator, bolr_index top_k) {
    bolr_index i;
    for (i = 0; i < accumulator->top_k_count; ++i) if (accumulator->top_k_keys[i] == top_k) return i;
    return -1;
}

static bolr_status compute_rank_order(
    bolr_const_vector_view scores,
    bolr_score_index_pair *pairs,
    bolr_index *ranks,
    bolr_index *out_winner,
    int *out_tied_winner
) {
    bolr_index i;
    if ((pairs == NULL) || (ranks == NULL) || (out_winner == NULL) || (out_tied_winner == NULL)) return BOLR_INVALID_ARGUMENT;
    if (bolr_vector_view_validate(scores) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < scores.length; ++i) {
        bolr_real value = scores.data[i * scores.stride];
        if (!isfinite(value)) return BOLR_NONFINITE_INPUT;
        pairs[i].value = value;
        pairs[i].index = i;
    }
    qsort(pairs, (size_t) scores.length, sizeof(*pairs), compare_score_index_pair);
    for (i = 0; i < scores.length; ++i) ranks[pairs[i].index] = i + 1;
    *out_winner = pairs[0].index;
    *out_tied_winner = ((scores.length > 1) && (pairs[1].value == pairs[0].value)) ? 1 : 0;
    return BOLR_OK;
}

static bolr_status replace_prediction_samples(
    struct bolr_posterior_prediction *prediction,
    bolr_real *score_samples,
    bolr_index score_sample_count,
    bolr_real *state_samples
) {
    bolr_allocator_free(prediction->allocator, prediction->score_samples);
    bolr_allocator_free(prediction->allocator, prediction->state_samples);
    prediction->score_samples = score_samples;
    prediction->state_samples = state_samples;
    prediction->score_sample_count = score_sample_count;
    return BOLR_OK;
}

static void clear_prediction_rank_outputs(struct bolr_posterior_prediction *prediction) {
    bolr_index i;
    bolr_allocator_free(prediction->allocator, prediction->probability_best);
    bolr_allocator_free(prediction->allocator, prediction->expected_rank);
    bolr_allocator_free(prediction->allocator, prediction->rank_stddev);
    if (prediction->probability_top_k_values != NULL) {
        for (i = 0; i < prediction->probability_top_k_count; ++i) bolr_allocator_free(prediction->allocator, prediction->probability_top_k_values[i]);
    }
    bolr_allocator_free(prediction->allocator, prediction->probability_top_k_values);
    bolr_allocator_free(prediction->allocator, prediction->probability_top_k_keys);
    prediction->probability_best = NULL;
    prediction->expected_rank = NULL;
    prediction->rank_stddev = NULL;
    prediction->probability_top_k_values = NULL;
    prediction->probability_top_k_keys = NULL;
    prediction->probability_top_k_count = 0;
}

static bolr_status build_explicit_design(
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_workspace *workspace,
    bolr_index candidate_count,
    bolr_index state_dim,
    bolr_real **out_design
) {
    bolr_real *design;
    bolr_vector_view score_buffer;
    bolr_vector_view state_buffer;
    bolr_index i, j;
    size_t bytes;
    if (bolr_checked_size_mul((size_t) (candidate_count * state_dim), sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    design = (bolr_real *) malloc(bytes);
    if (design == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_workspace_score_buffer(workspace, candidate_count, &score_buffer) != BOLR_OK) { free(design); return BOLR_INVALID_SHAPE; }
    if (bolr_workspace_state_buffer(workspace, state_dim, &state_buffer) != BOLR_OK) { free(design); return BOLR_INVALID_SHAPE; }
    for (i = 0; i < candidate_count; ++i) {
        for (j = 0; j < candidate_count; ++j) score_buffer.data[j] = 0.0;
        for (j = 0; j < state_dim; ++j) state_buffer.data[j] = 0.0;
        score_buffer.data[i] = 1.0;
        if (bolr_model_transpose(model, (bolr_const_vector_view){score_buffer.data, candidate_count, 1}, context, state_buffer, workspace) != BOLR_OK) {
            free(design);
            return BOLR_NUMERICAL_FAILURE;
        }
        for (j = 0; j < state_dim; ++j) design[i * state_dim + j] = state_buffer.data[j];
    }
    *out_design = design;
    return BOLR_OK;
}

static bolr_status validate_probability_vector(bolr_const_vector_view values, int require_unit_sum, bolr_real require_sum, int require_positive) {
    bolr_real sum = 0.0;
    bolr_index i;
    if (bolr_vector_view_validate(values) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < values.length; ++i) {
        bolr_real value = values.data[i * values.stride];
        if (!isfinite(value)) return BOLR_NONFINITE_INPUT;
        if ((value < 0.0) || (value > 1.0 + 1e-10)) return BOLR_INVALID_ARGUMENT;
        sum += value;
    }
    if (require_positive && (sum <= 0.0)) return BOLR_INVALID_ARGUMENT;
    if (require_unit_sum && (fabs(sum - require_sum) > 1e-8)) return BOLR_INVALID_ARGUMENT;
    return BOLR_OK;
}

static bolr_index top_k_slot(const struct bolr_posterior_prediction *prediction, bolr_index top_k) {
    bolr_index i;
    for (i = 0; i < prediction->probability_top_k_count; ++i) if (prediction->probability_top_k_keys[i] == top_k) return i;
    return -1;
}

bolr_status bolr_posterior_prediction_create(
    const bolr_gaussian_state *predictive,
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_workspace *workspace,
    const bolr_allocator *allocator,
    bolr_posterior_prediction **out_prediction,
    bolr_posterior_prediction_diagnostics *diagnostics
) {
    struct bolr_posterior_prediction *prediction;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index candidate_count;
    bolr_index state_dim;
    bolr_workspace *local_workspace = NULL;
    bolr_index i, j;
    bolr_real frob = 0.0;
    if ((predictive == NULL) || (model == NULL) || (out_prediction == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((predictive->state_layout_hash != bolr_model_state_layout_hash(model)) || (predictive->model_schema_hash != bolr_model_schema_hash(model))) return BOLR_SCHEMA_MISMATCH;
    *out_prediction = NULL;
    candidate_count = bolr_model_score_count(model);
    state_dim = bolr_model_state_dim(model);
    if (workspace == NULL) {
        bolr_workspace_config config = {candidate_count, state_dim, context.length};
        if (bolr_workspace_create(&config, active, &local_workspace) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
        workspace = local_workspace;
    }
    prediction = (struct bolr_posterior_prediction *) bolr_allocator_calloc(active, 1U, sizeof(*prediction));
    if (prediction == NULL) { bolr_workspace_destroy(local_workspace); return BOLR_ALLOCATION_FAILED; }
    prediction->allocator = active;
    prediction->candidate_count = candidate_count;
    prediction->state_dim = state_dim;
    prediction->model_schema_hash = predictive->model_schema_hash;
    prediction->state_layout_hash = predictive->state_layout_hash;
    if ((alloc_real_copy(active, state_dim, predictive->mean, &prediction->state_mean) != BOLR_OK) ||
        (alloc_real_copy(active, state_dim * state_dim, predictive->covariance, &prediction->state_covariance) != BOLR_OK) ||
        (alloc_real_zero(active, candidate_count, &prediction->score_mean) != BOLR_OK) ||
        (alloc_real_zero(active, candidate_count, &prediction->score_variance) != BOLR_OK)) {
        bolr_posterior_prediction_destroy(prediction);
        bolr_workspace_destroy(local_workspace);
        return BOLR_ALLOCATION_FAILED;
    }
    if (build_explicit_design(model, context, workspace, candidate_count, state_dim, &prediction->design_matrix) != BOLR_OK) {
        bolr_posterior_prediction_destroy(prediction);
        bolr_workspace_destroy(local_workspace);
        return BOLR_ALLOCATION_FAILED;
    }
    for (i = 0; i < candidate_count; ++i) prediction->score_mean[i] = 0.0;
    if (bolr_model_forward(model, (bolr_const_vector_view){predictive->mean, state_dim, 1}, context, (bolr_vector_view){prediction->score_mean, candidate_count, 1}, workspace) != BOLR_OK) {
        bolr_posterior_prediction_destroy(prediction);
        bolr_workspace_destroy(local_workspace);
        return BOLR_NUMERICAL_FAILURE;
    }
    for (i = 0; i < candidate_count; ++i) {
        bolr_real variance = 0.0;
        for (j = 0; j < state_dim; ++j) {
            bolr_real left = 0.0;
            bolr_index k;
            frob += prediction->design_matrix[i * state_dim + j] * prediction->design_matrix[i * state_dim + j];
            for (k = 0; k < state_dim; ++k) {
                left += prediction->design_matrix[i * state_dim + k] * predictive->covariance[k * state_dim + j];
            }
            variance += left * prediction->design_matrix[i * state_dim + j];
        }
        if (variance < -1e-10) {
            bolr_posterior_prediction_destroy(prediction);
            bolr_workspace_destroy(local_workspace);
            return BOLR_NUMERICAL_FAILURE;
        }
        prediction->score_variance[i] = (variance < 0.0) ? 0.0 : variance;
    }
    if (diagnostics != NULL) {
        bolr_real mean_norm = 0.0;
        bolr_real variance_sum = 0.0;
        for (i = 0; i < candidate_count; ++i) {
            mean_norm += prediction->score_mean[i] * prediction->score_mean[i];
            variance_sum += prediction->score_variance[i];
        }
        diagnostics->score_mean_norm = sqrt(mean_norm);
        diagnostics->score_variance_sum = variance_sum;
        diagnostics->explicit_design_frobenius_norm = sqrt(frob);
    }
    bolr_workspace_destroy(local_workspace);
    *out_prediction = prediction;
    return BOLR_OK;
}

void bolr_posterior_prediction_destroy(bolr_posterior_prediction *opaque) {
    struct bolr_posterior_prediction *prediction = opaque;
    bolr_index i;
    if (prediction == NULL) return;
    bolr_allocator_free(prediction->allocator, prediction->score_mean);
    bolr_allocator_free(prediction->allocator, prediction->score_variance);
    bolr_allocator_free(prediction->allocator, prediction->state_mean);
    bolr_allocator_free(prediction->allocator, prediction->state_covariance);
    free(prediction->design_matrix);
    bolr_allocator_free(prediction->allocator, prediction->probability_best);
    if (prediction->probability_top_k_values != NULL) {
        for (i = 0; i < prediction->probability_top_k_count; ++i) bolr_allocator_free(prediction->allocator, prediction->probability_top_k_values[i]);
    }
    bolr_allocator_free(prediction->allocator, prediction->probability_top_k_values);
    bolr_allocator_free(prediction->allocator, prediction->probability_top_k_keys);
    bolr_allocator_free(prediction->allocator, prediction->expected_rank);
    bolr_allocator_free(prediction->allocator, prediction->rank_stddev);
    bolr_allocator_free(prediction->allocator, prediction->score_samples);
    bolr_allocator_free(prediction->allocator, prediction->state_samples);
    bolr_allocator_free(prediction->allocator, prediction);
}

bolr_index bolr_posterior_prediction_candidate_count(const bolr_posterior_prediction *opaque) { const struct bolr_posterior_prediction *prediction = opaque; return (prediction == NULL) ? -1 : prediction->candidate_count; }
bolr_index bolr_posterior_prediction_state_dim(const bolr_posterior_prediction *opaque) { const struct bolr_posterior_prediction *prediction = opaque; return (prediction == NULL) ? -1 : prediction->state_dim; }
uint64_t bolr_posterior_prediction_model_schema_hash(const bolr_posterior_prediction *opaque) { const struct bolr_posterior_prediction *prediction = opaque; return (prediction == NULL) ? 0ULL : prediction->model_schema_hash; }

bolr_status bolr_posterior_prediction_copy_score_mean(const bolr_posterior_prediction *opaque, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    return (prediction == NULL) ? BOLR_INVALID_ARGUMENT : bolr_copy((bolr_const_vector_view){prediction->score_mean, prediction->candidate_count, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_score_variance(const bolr_posterior_prediction *opaque, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    return (prediction == NULL) ? BOLR_INVALID_ARGUMENT : bolr_copy((bolr_const_vector_view){prediction->score_variance, prediction->candidate_count, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_state_mean(const bolr_posterior_prediction *opaque, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    return (prediction == NULL) ? BOLR_INVALID_ARGUMENT : bolr_copy((bolr_const_vector_view){prediction->state_mean, prediction->state_dim, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_state_covariance(const bolr_posterior_prediction *opaque, bolr_matrix_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    bolr_index r, c;
    if ((prediction == NULL) || (output.rows != prediction->state_dim) || (output.cols != prediction->state_dim)) return BOLR_INVALID_ARGUMENT;
    for (r = 0; r < prediction->state_dim; ++r) for (c = 0; c < prediction->state_dim; ++c) output.data[r * output.row_stride + c * output.col_stride] = prediction->state_covariance[r * prediction->state_dim + c];
    return BOLR_OK;
}

bolr_status bolr_selected_score_covariance(const bolr_posterior_prediction *opaque, const bolr_index *indices, bolr_index count, bolr_matrix_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    bolr_index r, c, j, k;
    if ((prediction == NULL) || (indices == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((count != output.rows) || (count != output.cols)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < count; ++r) {
        if ((indices[r] < 0) || (indices[r] >= prediction->candidate_count)) return BOLR_INVALID_ARGUMENT;
        for (c = 0; c < count; ++c) {
            bolr_real value = 0.0;
            for (j = 0; j < prediction->state_dim; ++j) {
                bolr_real left = 0.0;
                for (k = 0; k < prediction->state_dim; ++k) left += prediction->design_matrix[indices[r] * prediction->state_dim + k] * prediction->state_covariance[k * prediction->state_dim + j];
                value += left * prediction->design_matrix[indices[c] * prediction->state_dim + j];
            }
            output.data[r * output.row_stride + c * output.col_stride] = value;
        }
    }
    return BOLR_OK;
}

bolr_status bolr_pairwise_probability(const bolr_posterior_prediction *opaque, const bolr_index *left_indices, const bolr_index *right_indices, bolr_index count, bolr_pairwise_probability_result *output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    bolr_index i, j, k;
    if ((prediction == NULL) || (left_indices == NULL) || (right_indices == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < count; ++i) {
        bolr_real mean_diff;
        bolr_real variance = 0.0;
        if ((left_indices[i] < 0) || (left_indices[i] >= prediction->candidate_count) || (right_indices[i] < 0) || (right_indices[i] >= prediction->candidate_count)) return BOLR_INVALID_ARGUMENT;
        mean_diff = prediction->score_mean[left_indices[i]] - prediction->score_mean[right_indices[i]];
        for (j = 0; j < prediction->state_dim; ++j) {
            bolr_real left = 0.0;
            for (k = 0; k < prediction->state_dim; ++k) {
                bolr_real diff_k = prediction->design_matrix[left_indices[i] * prediction->state_dim + k] - prediction->design_matrix[right_indices[i] * prediction->state_dim + k];
                left += diff_k * prediction->state_covariance[k * prediction->state_dim + j];
            }
            variance += left * (prediction->design_matrix[left_indices[i] * prediction->state_dim + j] - prediction->design_matrix[right_indices[i] * prediction->state_dim + j]);
        }
        if (variance < -1e-10) return BOLR_NUMERICAL_FAILURE;
        output[i].mean_difference = mean_diff;
        output[i].variance_difference = (variance < 0.0) ? 0.0 : variance;
        if (output[i].variance_difference <= 0.0) {
            output[i].left_probability = (mean_diff > 0.0) ? 1.0 : ((mean_diff < 0.0) ? 0.0 : 1.0);
        } else {
            output[i].left_probability = 0.5 * (1.0 + erf(mean_diff / sqrt(2.0 * output[i].variance_difference)));
        }
    }
    return BOLR_OK;
}

bolr_status bolr_posterior_prediction_set_probability_best(bolr_posterior_prediction *opaque, bolr_const_vector_view probability_best) {
    struct bolr_posterior_prediction *prediction = opaque;
    bolr_real *copy;
    if (prediction == NULL) return BOLR_INVALID_ARGUMENT;
    if ((probability_best.length != prediction->candidate_count) || (validate_probability_vector(probability_best, 1, 1.0, 1) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &copy) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    if (bolr_copy(probability_best, (bolr_vector_view){copy, prediction->candidate_count, 1}) != BOLR_OK) { bolr_allocator_free(prediction->allocator, copy); return BOLR_INVALID_ARGUMENT; }
    bolr_allocator_free(prediction->allocator, prediction->probability_best);
    prediction->probability_best = copy;
    return BOLR_OK;
}

bolr_status bolr_posterior_prediction_set_probability_top_k(bolr_posterior_prediction *opaque, bolr_index top_k, bolr_const_vector_view probability_top_k) {
    struct bolr_posterior_prediction *prediction = opaque;
    bolr_real *copy;
    bolr_index slot;
    if (prediction == NULL) return BOLR_INVALID_ARGUMENT;
    if ((top_k <= 0) || (probability_top_k.length != prediction->candidate_count) || (validate_probability_vector(probability_top_k, 1, (bolr_real) top_k, 1) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &copy) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    if (bolr_copy(probability_top_k, (bolr_vector_view){copy, prediction->candidate_count, 1}) != BOLR_OK) { bolr_allocator_free(prediction->allocator, copy); return BOLR_INVALID_ARGUMENT; }
    slot = top_k_slot(prediction, top_k);
    if (slot >= 0) {
        bolr_allocator_free(prediction->allocator, prediction->probability_top_k_values[slot]);
        prediction->probability_top_k_values[slot] = copy;
        return BOLR_OK;
    }
    {
        bolr_real **new_values = (bolr_real **) bolr_allocator_calloc(prediction->allocator, (size_t) (prediction->probability_top_k_count + 1), sizeof(bolr_real *));
        bolr_index *new_keys = (bolr_index *) bolr_allocator_calloc(prediction->allocator, (size_t) (prediction->probability_top_k_count + 1), sizeof(bolr_index));
        bolr_index i;
        if ((new_values == NULL) || (new_keys == NULL)) {
            bolr_allocator_free(prediction->allocator, copy);
            bolr_allocator_free(prediction->allocator, new_values);
            bolr_allocator_free(prediction->allocator, new_keys);
            return BOLR_ALLOCATION_FAILED;
        }
        for (i = 0; i < prediction->probability_top_k_count; ++i) {
            new_values[i] = prediction->probability_top_k_values[i];
            new_keys[i] = prediction->probability_top_k_keys[i];
        }
        new_values[prediction->probability_top_k_count] = copy;
        new_keys[prediction->probability_top_k_count] = top_k;
        bolr_allocator_free(prediction->allocator, prediction->probability_top_k_values);
        bolr_allocator_free(prediction->allocator, prediction->probability_top_k_keys);
        prediction->probability_top_k_values = new_values;
        prediction->probability_top_k_keys = new_keys;
        prediction->probability_top_k_count += 1;
    }
    return BOLR_OK;
}

bolr_status bolr_posterior_prediction_set_expected_rank(bolr_posterior_prediction *opaque, bolr_const_vector_view expected_rank) {
    struct bolr_posterior_prediction *prediction = opaque;
    bolr_real *copy;
    bolr_index i;
    if ((prediction == NULL) || (expected_rank.length != prediction->candidate_count)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < expected_rank.length; ++i) {
        bolr_real value = expected_rank.data[i * expected_rank.stride];
        if (!isfinite(value) || (value <= 0.0)) return BOLR_INVALID_ARGUMENT;
    }
    if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &copy) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    if (bolr_copy(expected_rank, (bolr_vector_view){copy, prediction->candidate_count, 1}) != BOLR_OK) { bolr_allocator_free(prediction->allocator, copy); return BOLR_INVALID_ARGUMENT; }
    bolr_allocator_free(prediction->allocator, prediction->expected_rank);
    prediction->expected_rank = copy;
    return BOLR_OK;
}

bolr_status bolr_posterior_prediction_set_rank_stddev(bolr_posterior_prediction *opaque, bolr_const_vector_view rank_stddev) {
    struct bolr_posterior_prediction *prediction = opaque;
    bolr_real *copy;
    bolr_index i;
    if ((prediction == NULL) || (rank_stddev.length != prediction->candidate_count)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < rank_stddev.length; ++i) {
        bolr_real value = rank_stddev.data[i * rank_stddev.stride];
        if (!isfinite(value) || (value < 0.0)) return BOLR_INVALID_ARGUMENT;
    }
    if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &copy) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    if (bolr_copy(rank_stddev, (bolr_vector_view){copy, prediction->candidate_count, 1}) != BOLR_OK) { bolr_allocator_free(prediction->allocator, copy); return BOLR_INVALID_ARGUMENT; }
    bolr_allocator_free(prediction->allocator, prediction->rank_stddev);
    prediction->rank_stddev = copy;
    return BOLR_OK;
}

bolr_status bolr_posterior_prediction_copy_probability_best(const bolr_posterior_prediction *opaque, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    if ((prediction == NULL) || (prediction->probability_best == NULL)) return BOLR_INVALID_ARGUMENT;
    return bolr_copy((bolr_const_vector_view){prediction->probability_best, prediction->candidate_count, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_probability_top_k(const bolr_posterior_prediction *opaque, bolr_index top_k, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    bolr_index slot;
    if (prediction == NULL) return BOLR_INVALID_ARGUMENT;
    slot = top_k_slot(prediction, top_k);
    if (slot < 0) return BOLR_INVALID_ARGUMENT;
    return bolr_copy((bolr_const_vector_view){prediction->probability_top_k_values[slot], prediction->candidate_count, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_expected_rank(const bolr_posterior_prediction *opaque, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    if ((prediction == NULL) || (prediction->expected_rank == NULL)) return BOLR_INVALID_ARGUMENT;
    return bolr_copy((bolr_const_vector_view){prediction->expected_rank, prediction->candidate_count, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_rank_stddev(const bolr_posterior_prediction *opaque, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    if ((prediction == NULL) || (prediction->rank_stddev == NULL)) return BOLR_INVALID_ARGUMENT;
    return bolr_copy((bolr_const_vector_view){prediction->rank_stddev, prediction->candidate_count, 1}, output);
}

bolr_status bolr_posterior_prediction_copy_score_sample(const bolr_posterior_prediction *opaque, bolr_index sample_index, bolr_vector_view output) {
    const struct bolr_posterior_prediction *prediction = opaque;
    if ((prediction == NULL) || (prediction->score_samples == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((sample_index < 0) || (sample_index >= prediction->score_sample_count) || (output.length != prediction->candidate_count)) return BOLR_INVALID_ARGUMENT;
    return bolr_copy(
        (bolr_const_vector_view){prediction->score_samples + sample_index * prediction->candidate_count, prediction->candidate_count, 1},
        output
    );
}

bolr_index bolr_posterior_prediction_score_sample_count(const bolr_posterior_prediction *opaque) {
    const struct bolr_posterior_prediction *prediction = opaque;
    return (prediction == NULL) ? -1 : prediction->score_sample_count;
}

bolr_status bolr_rank_accumulator_create(
    bolr_index candidate_count,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    const bolr_allocator *allocator,
    bolr_rank_accumulator **out_accumulator
) {
    struct bolr_rank_accumulator *accumulator;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index i;
    if ((candidate_count <= 0) || (top_k_count < 0) || (out_accumulator == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((top_k_count > 0) && (top_k_values == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_accumulator = NULL;
    accumulator = (struct bolr_rank_accumulator *) bolr_allocator_calloc(active, 1U, sizeof(*accumulator));
    if (accumulator == NULL) return BOLR_ALLOCATION_FAILED;
    accumulator->allocator = active;
    accumulator->candidate_count = candidate_count;
    accumulator->top_k_count = top_k_count;
    if ((alloc_u64_zero(active, candidate_count, &accumulator->best_counts) != BOLR_OK) ||
        (alloc_u64_zero(active, candidate_count, &accumulator->rank_sums) != BOLR_OK) ||
        (alloc_u64_zero(active, candidate_count, &accumulator->rank_squared_sums) != BOLR_OK)) {
        bolr_rank_accumulator_destroy(accumulator);
        return BOLR_ALLOCATION_FAILED;
    }
    if (top_k_count > 0) {
        accumulator->top_k_keys = (bolr_index *) bolr_allocator_calloc(active, (size_t) top_k_count, sizeof(bolr_index));
        accumulator->top_k_counts = (uint64_t **) bolr_allocator_calloc(active, (size_t) top_k_count, sizeof(uint64_t *));
        if ((accumulator->top_k_keys == NULL) || (accumulator->top_k_counts == NULL)) {
            bolr_rank_accumulator_destroy(accumulator);
            return BOLR_ALLOCATION_FAILED;
        }
        for (i = 0; i < top_k_count; ++i) {
            if ((top_k_values[i] <= 0) || (top_k_values[i] > candidate_count)) {
                bolr_rank_accumulator_destroy(accumulator);
                return BOLR_INVALID_ARGUMENT;
            }
            accumulator->top_k_keys[i] = top_k_values[i];
            if (alloc_u64_zero(active, candidate_count, &accumulator->top_k_counts[i]) != BOLR_OK) {
                bolr_rank_accumulator_destroy(accumulator);
                return BOLR_ALLOCATION_FAILED;
            }
        }
    }
    *out_accumulator = accumulator;
    return BOLR_OK;
}

void bolr_rank_accumulator_destroy(bolr_rank_accumulator *opaque) {
    struct bolr_rank_accumulator *accumulator = opaque;
    bolr_index i;
    if (accumulator == NULL) return;
    if (accumulator->top_k_counts != NULL) {
        for (i = 0; i < accumulator->top_k_count; ++i) bolr_allocator_free(accumulator->allocator, accumulator->top_k_counts[i]);
    }
    bolr_allocator_free(accumulator->allocator, accumulator->top_k_counts);
    bolr_allocator_free(accumulator->allocator, accumulator->top_k_keys);
    bolr_allocator_free(accumulator->allocator, accumulator->best_counts);
    bolr_allocator_free(accumulator->allocator, accumulator->rank_sums);
    bolr_allocator_free(accumulator->allocator, accumulator->rank_squared_sums);
    bolr_allocator_free(accumulator->allocator, accumulator);
}

bolr_status bolr_rank_accumulator_reset(bolr_rank_accumulator *opaque) {
    struct bolr_rank_accumulator *accumulator = opaque;
    bolr_index i;
    if (accumulator == NULL) return BOLR_INVALID_ARGUMENT;
    memset(accumulator->best_counts, 0, (size_t) accumulator->candidate_count * sizeof(uint64_t));
    memset(accumulator->rank_sums, 0, (size_t) accumulator->candidate_count * sizeof(uint64_t));
    memset(accumulator->rank_squared_sums, 0, (size_t) accumulator->candidate_count * sizeof(uint64_t));
    for (i = 0; i < accumulator->top_k_count; ++i) memset(accumulator->top_k_counts[i], 0, (size_t) accumulator->candidate_count * sizeof(uint64_t));
    accumulator->sample_count = 0ULL;
    accumulator->tie_count = 0ULL;
    return BOLR_OK;
}

bolr_status bolr_rank_accumulator_accumulate_scores(bolr_rank_accumulator *opaque, bolr_const_matrix_view score_samples) {
    struct bolr_rank_accumulator *accumulator = opaque;
    bolr_score_index_pair *pairs;
    bolr_index *ranks;
    bolr_index row;
    bolr_index i;
    bolr_status status;
    if ((accumulator == NULL) || (bolr_matrix_view_validate(score_samples) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if (score_samples.cols != accumulator->candidate_count) return BOLR_INVALID_SHAPE;
    if (score_samples.rows == 0) return BOLR_OK;
    pairs = (bolr_score_index_pair *) malloc((size_t) accumulator->candidate_count * sizeof(*pairs));
    ranks = (bolr_index *) malloc((size_t) accumulator->candidate_count * sizeof(*ranks));
    if ((pairs == NULL) || (ranks == NULL)) {
        free(pairs);
        free(ranks);
        return BOLR_ALLOCATION_FAILED;
    }
    for (row = 0; row < score_samples.rows; ++row) {
        bolr_const_vector_view row_view = {score_samples.data + row * score_samples.row_stride, score_samples.cols, score_samples.col_stride};
        bolr_index winner;
        int tied_winner = 0;
        status = compute_rank_order(row_view, pairs, ranks, &winner, &tied_winner);
        if (status != BOLR_OK) {
            free(pairs);
            free(ranks);
            return status;
        }
        accumulator->best_counts[winner] += 1ULL;
        accumulator->sample_count += 1ULL;
        accumulator->tie_count += (uint64_t) tied_winner;
        for (i = 0; i < accumulator->candidate_count; ++i) {
            uint64_t rank = (uint64_t) ranks[i];
            bolr_index slot;
            accumulator->rank_sums[i] += rank;
            accumulator->rank_squared_sums[i] += rank * rank;
            for (slot = 0; slot < accumulator->top_k_count; ++slot) {
                if (ranks[i] <= accumulator->top_k_keys[slot]) accumulator->top_k_counts[slot][i] += 1ULL;
            }
        }
    }
    free(pairs);
    free(ranks);
    return BOLR_OK;
}

bolr_status bolr_rank_accumulator_merge(bolr_rank_accumulator *destination_opaque, const bolr_rank_accumulator *source_opaque) {
    struct bolr_rank_accumulator *destination = destination_opaque;
    const struct bolr_rank_accumulator *source = source_opaque;
    bolr_index i, slot;
    if ((destination == NULL) || (source == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((destination->candidate_count != source->candidate_count) || (destination->top_k_count != source->top_k_count)) return BOLR_INVALID_ARGUMENT;
    for (slot = 0; slot < destination->top_k_count; ++slot) if (destination->top_k_keys[slot] != source->top_k_keys[slot]) return BOLR_INVALID_ARGUMENT;
    destination->sample_count += source->sample_count;
    destination->tie_count += source->tie_count;
    for (i = 0; i < destination->candidate_count; ++i) {
        destination->best_counts[i] += source->best_counts[i];
        destination->rank_sums[i] += source->rank_sums[i];
        destination->rank_squared_sums[i] += source->rank_squared_sums[i];
    }
    for (slot = 0; slot < destination->top_k_count; ++slot) {
        for (i = 0; i < destination->candidate_count; ++i) destination->top_k_counts[slot][i] += source->top_k_counts[slot][i];
    }
    return BOLR_OK;
}

static bolr_status copy_rank_metric_probabilities(
    const struct bolr_rank_accumulator *accumulator,
    const uint64_t *counts,
    bolr_real scale,
    bolr_vector_view output
) {
    bolr_index i;
    if ((accumulator == NULL) || (counts == NULL) || (output.length != accumulator->candidate_count)) return BOLR_INVALID_ARGUMENT;
    if (output.stride <= 0) return BOLR_INVALID_ARGUMENT;
    if (accumulator->sample_count == 0ULL) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < accumulator->candidate_count; ++i) output.data[i * output.stride] = ((bolr_real) counts[i]) * scale;
    return BOLR_OK;
}

bolr_status bolr_rank_accumulator_copy_probability_best(const bolr_rank_accumulator *opaque, bolr_vector_view output) {
    const struct bolr_rank_accumulator *accumulator = opaque;
    if ((accumulator == NULL) || (accumulator->sample_count == 0ULL)) return BOLR_INVALID_ARGUMENT;
    return copy_rank_metric_probabilities(accumulator, accumulator->best_counts, 1.0 / (bolr_real) accumulator->sample_count, output);
}

bolr_status bolr_rank_accumulator_copy_probability_top_k(const bolr_rank_accumulator *opaque, bolr_index top_k, bolr_vector_view output) {
    const struct bolr_rank_accumulator *accumulator = opaque;
    bolr_index slot;
    if ((accumulator == NULL) || (accumulator->sample_count == 0ULL)) return BOLR_INVALID_ARGUMENT;
    slot = rank_accumulator_top_k_slot(accumulator, top_k);
    if (slot < 0) return BOLR_INVALID_ARGUMENT;
    return copy_rank_metric_probabilities(accumulator, accumulator->top_k_counts[slot], 1.0 / (bolr_real) accumulator->sample_count, output);
}

bolr_status bolr_rank_accumulator_copy_expected_rank(const bolr_rank_accumulator *opaque, bolr_vector_view output) {
    const struct bolr_rank_accumulator *accumulator = opaque;
    bolr_index i;
    if ((accumulator == NULL) || (output.length != accumulator->candidate_count) || (accumulator->sample_count == 0ULL)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < accumulator->candidate_count; ++i) output.data[i * output.stride] = ((bolr_real) accumulator->rank_sums[i]) / (bolr_real) accumulator->sample_count;
    return BOLR_OK;
}

bolr_status bolr_rank_accumulator_copy_rank_stddev(const bolr_rank_accumulator *opaque, bolr_vector_view output) {
    const struct bolr_rank_accumulator *accumulator = opaque;
    bolr_index i;
    if ((accumulator == NULL) || (output.length != accumulator->candidate_count) || (accumulator->sample_count == 0ULL)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < accumulator->candidate_count; ++i) {
        bolr_real mean = ((bolr_real) accumulator->rank_sums[i]) / (bolr_real) accumulator->sample_count;
        bolr_real mean_sq = ((bolr_real) accumulator->rank_squared_sums[i]) / (bolr_real) accumulator->sample_count;
        bolr_real variance = mean_sq - mean * mean;
        output.data[i * output.stride] = (variance <= 0.0) ? 0.0 : sqrt(variance);
    }
    return BOLR_OK;
}

uint64_t bolr_rank_accumulator_sample_count(const bolr_rank_accumulator *opaque) {
    const struct bolr_rank_accumulator *accumulator = opaque;
    return (accumulator == NULL) ? 0ULL : accumulator->sample_count;
}

bolr_index bolr_rank_accumulator_tie_count(const bolr_rank_accumulator *opaque) {
    const struct bolr_rank_accumulator *accumulator = opaque;
    return (accumulator == NULL) ? -1 : (bolr_index) accumulator->tie_count;
}

bolr_status bolr_posterior_prediction_monte_carlo_rank(
    bolr_posterior_prediction *opaque,
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
) {
    struct bolr_posterior_prediction *prediction = opaque;
    struct bolr_rank_accumulator *accumulator = NULL;
    struct bolr_gaussian_state temporary_state;
    bolr_real *state_samples = NULL;
    bolr_real *score_samples = NULL;
    bolr_real *probability_best = NULL;
    bolr_real *expected_rank = NULL;
    bolr_real *rank_stddev = NULL;
    bolr_real **probability_top_k_values = NULL;
    bolr_index *probability_top_k_keys = NULL;
    bolr_sampling_diagnostics sampling_diagnostics;
    bolr_status status = BOLR_OK;
    bolr_index slot;
    int state_samples_with_allocator;
    bolr_real winner_maximum = 0.0;
    if ((prediction == NULL) || (model == NULL) || (rng == NULL) || (sample_count < 0) || (top_k_count < 0)) return BOLR_INVALID_ARGUMENT;
    if ((top_k_count > 0) && (top_k_values == NULL)) return BOLR_INVALID_ARGUMENT;
    if (sample_count == 0) {
        status = replace_prediction_samples(prediction, NULL, 0, NULL);
        if (status != BOLR_OK) return status;
        clear_prediction_rank_outputs(prediction);
        if (diagnostics != NULL) memset(diagnostics, 0, sizeof(*diagnostics));
        return BOLR_OK;
    }
    status = bolr_rank_accumulator_create(prediction->candidate_count, top_k_values, top_k_count, prediction->allocator, &accumulator);
    if (status != BOLR_OK) return status;
    state_samples_with_allocator = (retain_state_samples || retain_score_samples) ? 1 : 0;
    {
        size_t state_bytes;
        if (bolr_checked_size_mul((size_t) (sample_count * prediction->state_dim), sizeof(bolr_real), &state_bytes) != BOLR_OK) {
            bolr_rank_accumulator_destroy(accumulator);
            return BOLR_DIMENSION_OVERFLOW;
        }
        state_samples = (bolr_real *) (state_samples_with_allocator
            ? bolr_allocator_malloc(prediction->allocator, state_bytes)
            : malloc(state_bytes));
        if (state_samples == NULL) {
            bolr_rank_accumulator_destroy(accumulator);
            return BOLR_ALLOCATION_FAILED;
        }
    }
    {
        size_t score_bytes;
        if (bolr_checked_size_mul((size_t) (sample_count * prediction->candidate_count), sizeof(bolr_real), &score_bytes) != BOLR_OK) {
            if (retain_state_samples || retain_score_samples) bolr_allocator_free(prediction->allocator, state_samples);
            else free(state_samples);
            bolr_rank_accumulator_destroy(accumulator);
            return BOLR_DIMENSION_OVERFLOW;
        }
        score_samples = (bolr_real *) (retain_score_samples
            ? bolr_allocator_malloc(prediction->allocator, score_bytes)
            : malloc(score_bytes));
        if (score_samples == NULL) {
            if (state_samples_with_allocator) bolr_allocator_free(prediction->allocator, state_samples);
            else free(state_samples);
            bolr_rank_accumulator_destroy(accumulator);
            return BOLR_ALLOCATION_FAILED;
        }
    }
    memset(&temporary_state, 0, sizeof(temporary_state));
    temporary_state.allocator = prediction->allocator;
    temporary_state.mean = prediction->state_mean;
    temporary_state.covariance = prediction->state_covariance;
    temporary_state.dimension = prediction->state_dim;
    temporary_state.state_layout_hash = prediction->state_layout_hash;
    temporary_state.model_schema_hash = prediction->model_schema_hash;
    memset(&sampling_diagnostics, 0, sizeof(sampling_diagnostics));
    status = bolr_gaussian_state_sample(
        &temporary_state,
        rng,
        sample_count,
        antithetic,
        (bolr_matrix_view){state_samples, sample_count, prediction->state_dim, prediction->state_dim, 1},
        &sampling_diagnostics,
        workspace
    );
    if (status != BOLR_OK) goto cleanup;
    status = bolr_composite_score_samples(
        model,
        context,
        (bolr_const_matrix_view){state_samples, sample_count, prediction->state_dim, prediction->state_dim, 1},
        (bolr_matrix_view){score_samples, sample_count, prediction->candidate_count, prediction->candidate_count, 1},
        workspace,
        NULL
    );
    if (status != BOLR_OK) goto cleanup;
    status = bolr_rank_accumulator_accumulate_scores(accumulator, (bolr_const_matrix_view){score_samples, sample_count, prediction->candidate_count, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &probability_best) != BOLR_OK ||
        alloc_real_zero(prediction->allocator, prediction->candidate_count, &expected_rank) != BOLR_OK ||
        alloc_real_zero(prediction->allocator, prediction->candidate_count, &rank_stddev) != BOLR_OK) {
        status = BOLR_ALLOCATION_FAILED;
        goto cleanup;
    }
    status = bolr_rank_accumulator_copy_probability_best(accumulator, (bolr_vector_view){probability_best, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    status = bolr_rank_accumulator_copy_expected_rank(accumulator, (bolr_vector_view){expected_rank, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    status = bolr_rank_accumulator_copy_rank_stddev(accumulator, (bolr_vector_view){rank_stddev, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    if (top_k_count > 0) {
        probability_top_k_values = (bolr_real **) bolr_allocator_calloc(prediction->allocator, (size_t) top_k_count, sizeof(bolr_real *));
        probability_top_k_keys = (bolr_index *) bolr_allocator_calloc(prediction->allocator, (size_t) top_k_count, sizeof(bolr_index));
        if ((probability_top_k_values == NULL) || (probability_top_k_keys == NULL)) {
            status = BOLR_ALLOCATION_FAILED;
            goto cleanup;
        }
        for (slot = 0; slot < top_k_count; ++slot) {
            probability_top_k_keys[slot] = top_k_values[slot];
            if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &probability_top_k_values[slot]) != BOLR_OK) {
                status = BOLR_ALLOCATION_FAILED;
                goto cleanup;
            }
            status = bolr_rank_accumulator_copy_probability_top_k(
                accumulator,
                top_k_values[slot],
                (bolr_vector_view){probability_top_k_values[slot], prediction->candidate_count, 1}
            );
            if (status != BOLR_OK) goto cleanup;
        }
    }
    clear_prediction_rank_outputs(prediction);
    prediction->probability_best = probability_best;
    prediction->expected_rank = expected_rank;
    prediction->rank_stddev = rank_stddev;
    prediction->probability_top_k_values = probability_top_k_values;
    prediction->probability_top_k_keys = probability_top_k_keys;
    prediction->probability_top_k_count = top_k_count;
    probability_best = NULL;
    expected_rank = NULL;
    rank_stddev = NULL;
    probability_top_k_values = NULL;
    probability_top_k_keys = NULL;
    if (!retain_score_samples) {
        free(score_samples);
        score_samples = NULL;
    }
    if (!retain_state_samples) {
        if (state_samples_with_allocator) bolr_allocator_free(prediction->allocator, state_samples);
        else free(state_samples);
        state_samples = NULL;
    }
    status = replace_prediction_samples(prediction, score_samples, retain_score_samples ? sample_count : 0, state_samples);
    if (status != BOLR_OK) goto cleanup;
    score_samples = NULL;
    state_samples = NULL;
    if (diagnostics != NULL) {
        memset(diagnostics, 0, sizeof(*diagnostics));
        diagnostics->sample_count = sample_count;
        diagnostics->top_k_count = top_k_count;
        diagnostics->retained_score_sample_count = retain_score_samples ? sample_count : 0;
        diagnostics->retained_state_sample_count = retain_state_samples ? sample_count : 0;
        diagnostics->tie_count = (bolr_index) accumulator->tie_count;
        status = bolr_probability_entropy((bolr_const_vector_view){prediction->probability_best, prediction->candidate_count, 1}, &diagnostics->winner_entropy, &diagnostics->effective_winner_count, &winner_maximum);
        if (status != BOLR_OK) goto cleanup;
    }
cleanup:
    if (probability_top_k_values != NULL) {
        for (slot = 0; slot < top_k_count; ++slot) bolr_allocator_free(prediction->allocator, probability_top_k_values[slot]);
    }
    bolr_allocator_free(prediction->allocator, probability_top_k_values);
    bolr_allocator_free(prediction->allocator, probability_top_k_keys);
    bolr_allocator_free(prediction->allocator, probability_best);
    bolr_allocator_free(prediction->allocator, expected_rank);
    bolr_allocator_free(prediction->allocator, rank_stddev);
    if (score_samples != NULL) {
        if (retain_score_samples) bolr_allocator_free(prediction->allocator, score_samples);
        else free(score_samples);
    }
    if (state_samples != NULL) {
        if (state_samples_with_allocator) bolr_allocator_free(prediction->allocator, state_samples);
        else free(state_samples);
    }
    bolr_rank_accumulator_destroy(accumulator);
    return status;
}

bolr_status bolr_posterior_prediction_monte_carlo_rank_streaming(
    bolr_posterior_prediction *opaque,
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
) {
    struct bolr_posterior_prediction *prediction = opaque;
    struct bolr_rank_accumulator *accumulator = NULL;
    struct bolr_gaussian_state temporary_state;
    bolr_real *state_chunk = NULL;
    bolr_real *score_chunk = NULL;
    bolr_real *mirror_chunk = NULL;
    bolr_real *captured_scores = NULL;
    bolr_real *probability_best = NULL;
    bolr_real *expected_rank = NULL;
    bolr_real *rank_stddev = NULL;
    bolr_real **probability_top_k_values = NULL;
    bolr_index *probability_top_k_keys = NULL;
    bolr_sampling_diagnostics sampling_diagnostics;
    bolr_status status = BOLR_OK;
    bolr_index positive_total;
    bolr_index positive_processed = 0;
    bolr_index slot;
    bolr_index i;
    bolr_real winner_maximum = 0.0;
    if ((prediction == NULL) || (model == NULL) || (rng == NULL) || (sample_count < 0) || (chunk_size <= 0) || (top_k_count < 0)) return BOLR_INVALID_ARGUMENT;
    if ((top_k_count > 0) && (top_k_values == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((retention != BOLR_SCORE_RETENTION_NONE) && (retention != BOLR_SCORE_RETENTION_SAMPLE_ZERO)) return BOLR_UNSUPPORTED_OPERATION;
    if (sample_count == 0) {
        status = replace_prediction_samples(prediction, NULL, 0, NULL);
        if (status != BOLR_OK) return status;
        clear_prediction_rank_outputs(prediction);
        if (diagnostics != NULL) memset(diagnostics, 0, sizeof(*diagnostics));
        return BOLR_OK;
    }
    positive_total = antithetic ? ((sample_count + 1) / 2) : sample_count;
    if (chunk_size > positive_total) chunk_size = positive_total;
    status = bolr_rank_accumulator_create(prediction->candidate_count, top_k_values, top_k_count, prediction->allocator, &accumulator);
    if (status != BOLR_OK) return status;
    {
        size_t state_bytes;
        size_t score_bytes;
        if (bolr_checked_size_mul((size_t) (chunk_size * prediction->state_dim), sizeof(bolr_real), &state_bytes) != BOLR_OK) { status = BOLR_DIMENSION_OVERFLOW; goto cleanup; }
        if (bolr_checked_size_mul((size_t) (chunk_size * prediction->candidate_count), sizeof(bolr_real), &score_bytes) != BOLR_OK) { status = BOLR_DIMENSION_OVERFLOW; goto cleanup; }
        state_chunk = (bolr_real *) malloc(state_bytes);
        score_chunk = (bolr_real *) malloc(score_bytes);
        mirror_chunk = antithetic ? (bolr_real *) malloc(score_bytes) : NULL;
        if ((state_chunk == NULL) || (score_chunk == NULL) || (antithetic && (mirror_chunk == NULL))) { status = BOLR_ALLOCATION_FAILED; goto cleanup; }
    }
    if (retention == BOLR_SCORE_RETENTION_SAMPLE_ZERO) {
        captured_scores = (bolr_real *) bolr_allocator_calloc(prediction->allocator, (size_t) prediction->candidate_count, sizeof(bolr_real));
        if (captured_scores == NULL) { status = BOLR_ALLOCATION_FAILED; goto cleanup; }
    }
    memset(&temporary_state, 0, sizeof(temporary_state));
    temporary_state.allocator = prediction->allocator;
    temporary_state.mean = prediction->state_mean;
    temporary_state.covariance = prediction->state_covariance;
    temporary_state.dimension = prediction->state_dim;
    temporary_state.state_layout_hash = prediction->state_layout_hash;
    temporary_state.model_schema_hash = prediction->model_schema_hash;
    while (positive_processed < positive_total) {
        bolr_index positive_chunk = positive_total - positive_processed;
        if (positive_chunk > chunk_size) positive_chunk = chunk_size;
        memset(&sampling_diagnostics, 0, sizeof(sampling_diagnostics));
        status = bolr_gaussian_state_sample(
            &temporary_state,
            rng,
            positive_chunk,
            0,
            (bolr_matrix_view){state_chunk, positive_chunk, prediction->state_dim, prediction->state_dim, 1},
            &sampling_diagnostics,
            workspace
        );
        if (status != BOLR_OK) goto cleanup;
        status = bolr_composite_score_samples(
            model,
            context,
            (bolr_const_matrix_view){state_chunk, positive_chunk, prediction->state_dim, prediction->state_dim, 1},
            (bolr_matrix_view){score_chunk, positive_chunk, prediction->candidate_count, prediction->candidate_count, 1},
            workspace,
            NULL
        );
        if (status != BOLR_OK) goto cleanup;
        status = bolr_rank_accumulator_accumulate_scores(accumulator, (bolr_const_matrix_view){score_chunk, positive_chunk, prediction->candidate_count, prediction->candidate_count, 1});
        if (status != BOLR_OK) goto cleanup;
        if ((retention == BOLR_SCORE_RETENTION_SAMPLE_ZERO) && (positive_processed == 0)) {
            for (i = 0; i < prediction->candidate_count; ++i) captured_scores[i] = score_chunk[i];
        }
        if (antithetic) {
            bolr_index mirrored_rows = sample_count - positive_total;
            bolr_index local_rows = mirrored_rows - positive_processed;
            if (local_rows > positive_chunk) local_rows = positive_chunk;
            if (local_rows > 0) {
                for (i = 0; i < local_rows; ++i) {
                    bolr_index j;
                    for (j = 0; j < prediction->candidate_count; ++j) {
                        mirror_chunk[i * prediction->candidate_count + j] =
                            (2.0 * prediction->score_mean[j]) - score_chunk[i * prediction->candidate_count + j];
                    }
                }
                status = bolr_rank_accumulator_accumulate_scores(accumulator, (bolr_const_matrix_view){mirror_chunk, local_rows, prediction->candidate_count, prediction->candidate_count, 1});
                if (status != BOLR_OK) goto cleanup;
            }
        }
        positive_processed += positive_chunk;
    }
    if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &probability_best) != BOLR_OK ||
        alloc_real_zero(prediction->allocator, prediction->candidate_count, &expected_rank) != BOLR_OK ||
        alloc_real_zero(prediction->allocator, prediction->candidate_count, &rank_stddev) != BOLR_OK) {
        status = BOLR_ALLOCATION_FAILED;
        goto cleanup;
    }
    status = bolr_rank_accumulator_copy_probability_best(accumulator, (bolr_vector_view){probability_best, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    status = bolr_rank_accumulator_copy_expected_rank(accumulator, (bolr_vector_view){expected_rank, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    status = bolr_rank_accumulator_copy_rank_stddev(accumulator, (bolr_vector_view){rank_stddev, prediction->candidate_count, 1});
    if (status != BOLR_OK) goto cleanup;
    if (top_k_count > 0) {
        probability_top_k_values = (bolr_real **) bolr_allocator_calloc(prediction->allocator, (size_t) top_k_count, sizeof(bolr_real *));
        probability_top_k_keys = (bolr_index *) bolr_allocator_calloc(prediction->allocator, (size_t) top_k_count, sizeof(bolr_index));
        if ((probability_top_k_values == NULL) || (probability_top_k_keys == NULL)) { status = BOLR_ALLOCATION_FAILED; goto cleanup; }
        for (slot = 0; slot < top_k_count; ++slot) {
            probability_top_k_keys[slot] = top_k_values[slot];
            if (alloc_real_zero(prediction->allocator, prediction->candidate_count, &probability_top_k_values[slot]) != BOLR_OK) { status = BOLR_ALLOCATION_FAILED; goto cleanup; }
            status = bolr_rank_accumulator_copy_probability_top_k(accumulator, top_k_values[slot], (bolr_vector_view){probability_top_k_values[slot], prediction->candidate_count, 1});
            if (status != BOLR_OK) goto cleanup;
        }
    }
    clear_prediction_rank_outputs(prediction);
    prediction->probability_best = probability_best;
    prediction->expected_rank = expected_rank;
    prediction->rank_stddev = rank_stddev;
    prediction->probability_top_k_values = probability_top_k_values;
    prediction->probability_top_k_keys = probability_top_k_keys;
    prediction->probability_top_k_count = top_k_count;
    probability_best = NULL;
    expected_rank = NULL;
    rank_stddev = NULL;
    probability_top_k_values = NULL;
    probability_top_k_keys = NULL;
    status = replace_prediction_samples(prediction, captured_scores, (retention == BOLR_SCORE_RETENTION_SAMPLE_ZERO) ? 1 : 0, NULL);
    if (status != BOLR_OK) goto cleanup;
    captured_scores = NULL;
    if (diagnostics != NULL) {
        memset(diagnostics, 0, sizeof(*diagnostics));
        diagnostics->sample_count = sample_count;
        diagnostics->top_k_count = top_k_count;
        diagnostics->retained_score_sample_count = (retention == BOLR_SCORE_RETENTION_SAMPLE_ZERO) ? 1 : 0;
        diagnostics->retained_state_sample_count = 0;
        diagnostics->tie_count = (bolr_index) accumulator->tie_count;
        status = bolr_probability_entropy((bolr_const_vector_view){prediction->probability_best, prediction->candidate_count, 1}, &diagnostics->winner_entropy, &diagnostics->effective_winner_count, &winner_maximum);
        if (status != BOLR_OK) goto cleanup;
    }
cleanup:
    if (probability_top_k_values != NULL) {
        for (slot = 0; slot < top_k_count; ++slot) bolr_allocator_free(prediction->allocator, probability_top_k_values[slot]);
    }
    bolr_allocator_free(prediction->allocator, probability_top_k_values);
    bolr_allocator_free(prediction->allocator, probability_top_k_keys);
    bolr_allocator_free(prediction->allocator, probability_best);
    bolr_allocator_free(prediction->allocator, expected_rank);
    bolr_allocator_free(prediction->allocator, rank_stddev);
    bolr_allocator_free(prediction->allocator, captured_scores);
    free(state_chunk);
    free(score_chunk);
    free(mirror_chunk);
    bolr_rank_accumulator_destroy(accumulator);
    return status;
}

bolr_status bolr_probability_entropy(bolr_const_vector_view probabilities, bolr_real *out_entropy, bolr_real *out_effective_count, bolr_real *out_maximum) {
    bolr_real sum = 0.0;
    bolr_real entropy = 0.0;
    bolr_real maximum = 0.0;
    bolr_index i;
    if ((out_entropy == NULL) || (out_effective_count == NULL) || (out_maximum == NULL)) return BOLR_INVALID_ARGUMENT;
    if (validate_probability_vector(probabilities, 0, 0.0, 0) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < probabilities.length; ++i) {
        bolr_real value = probabilities.data[i * probabilities.stride];
        sum += value;
        if (value > maximum) maximum = value;
    }
    if (sum <= 0.0) {
        *out_entropy = 0.0;
        *out_effective_count = 0.0;
        *out_maximum = 0.0;
        return BOLR_OK;
    }
    for (i = 0; i < probabilities.length; ++i) {
        bolr_real value = probabilities.data[i * probabilities.stride] / sum;
        if (value > 0.0) entropy -= value * log(value);
    }
    *out_entropy = entropy;
    *out_effective_count = exp(entropy);
    *out_maximum = maximum / sum;
    return BOLR_OK;
}
