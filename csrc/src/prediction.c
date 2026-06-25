#include "bolr/prediction.h"

#include "bolr/linalg.h"
#include "bolr/math.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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
