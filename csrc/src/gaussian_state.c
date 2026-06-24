#include "bolr/gaussian.h"
#include "bolr/checkpoint.h"
#include "bolr/linalg.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static bolr_status validate_finite_vector(bolr_const_vector_view view) {
    bolr_index i;
    bolr_status status = bolr_vector_view_validate(view);
    if (status != BOLR_OK) return status;
    for (i = 0; i < view.length; ++i) if (!isfinite(view.data[i * view.stride])) return BOLR_NONFINITE_INPUT;
    return BOLR_OK;
}

static bolr_status validate_covariance(bolr_const_matrix_view covariance) {
    bolr_index r, c;
    bolr_status status = bolr_matrix_view_validate(covariance);
    if (status != BOLR_OK) return status;
    if (covariance.rows != covariance.cols) return BOLR_INVALID_SHAPE;
    for (r = 0; r < covariance.rows; ++r) {
        for (c = 0; c < covariance.cols; ++c) {
            bolr_real value = covariance.data[r * covariance.row_stride + c * covariance.col_stride];
            if (!isfinite(value)) return BOLR_NONFINITE_INPUT;
            if (fabs(value - covariance.data[c * covariance.row_stride + r * covariance.col_stride]) > 1e-8) return BOLR_INVALID_ARGUMENT;
        }
    }
    return BOLR_OK;
}

static bolr_status allocate_copy_vector(bolr_const_vector_view source, const bolr_allocator *allocator, bolr_real **out_data) {
    size_t bytes;
    bolr_index i;
    bolr_real *data;
    bolr_status status = bolr_checked_size_mul((size_t) source.length, sizeof(bolr_real), &bytes);
    if (status != BOLR_OK) return status;
    data = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < source.length; ++i) data[i] = source.data[i * source.stride];
    *out_data = data;
    return BOLR_OK;
}

static bolr_status allocate_copy_matrix(bolr_const_matrix_view source, const bolr_allocator *allocator, bolr_real **out_data) {
    size_t bytes;
    bolr_index r, c;
    bolr_real *data;
    bolr_status status = bolr_checked_size_mul((size_t) (source.rows * source.cols), sizeof(bolr_real), &bytes);
    if (status != BOLR_OK) return status;
    data = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    for (r = 0; r < source.rows; ++r) for (c = 0; c < source.cols; ++c) data[r * source.cols + c] = source.data[r * source.row_stride + c * source.col_stride];
    *out_data = data;
    return BOLR_OK;
}

static bolr_status validate_positive_definite_copy(const bolr_real *covariance, bolr_index dim) {
    bolr_real *copy;
    bolr_cholesky_diagnostics diagnostics;
    copy = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
    if (copy == NULL) return BOLR_ALLOCATION_FAILED;
    memcpy(copy, covariance, (size_t) (dim * dim) * sizeof(bolr_real));
    if (bolr_cholesky_factor((bolr_matrix_view){copy, dim, dim, dim, 1}, 1e-10, 10.0, 8, &diagnostics) != BOLR_OK) {
        free(copy);
        return BOLR_NOT_POSITIVE_DEFINITE;
    }
    free(copy);
    return BOLR_OK;
}

bolr_status bolr_gaussian_state_create(
    bolr_const_vector_view mean,
    bolr_const_matrix_view covariance,
    uint64_t state_layout_hash,
    uint64_t model_schema_hash,
    const bolr_allocator *allocator,
    bolr_gaussian_state **out_state
) {
    bolr_gaussian_state *state;
    bolr_status status;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((out_state == NULL) || (state_layout_hash == 0ULL) || (model_schema_hash == 0ULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    status = validate_finite_vector(mean); if (status != BOLR_OK) return status;
    status = validate_covariance(covariance); if (status != BOLR_OK) return status;
    if (mean.length != covariance.rows) return BOLR_INVALID_SHAPE;
    state = (bolr_gaussian_state *) bolr_allocator_calloc(active, 1U, sizeof(*state));
    if (state == NULL) return BOLR_ALLOCATION_FAILED;
    state->allocator = active;
    state->dimension = mean.length;
    state->state_layout_hash = state_layout_hash;
    state->model_schema_hash = model_schema_hash;
    state->schema_version = 1U;
    status = allocate_copy_vector(mean, active, &state->mean); if (status != BOLR_OK) { bolr_gaussian_state_destroy(state); return status; }
    status = allocate_copy_matrix(covariance, active, &state->covariance); if (status != BOLR_OK) { bolr_gaussian_state_destroy(state); return status; }
    status = validate_positive_definite_copy(state->covariance, state->dimension); if (status != BOLR_OK) { bolr_gaussian_state_destroy(state); return status; }
    *out_state = state;
    return BOLR_OK;
}

void bolr_gaussian_state_destroy(bolr_gaussian_state *state) {
    if (state == NULL) return;
    bolr_allocator_free(state->allocator, state->mean);
    bolr_allocator_free(state->allocator, state->covariance);
    bolr_allocator_free(state->allocator, state);
}

bolr_status bolr_gaussian_state_clone(const bolr_gaussian_state *source, const bolr_allocator *allocator, bolr_gaussian_state **out_clone) {
    bolr_status status;
    if ((source == NULL) || (out_clone == NULL)) return BOLR_INVALID_ARGUMENT;
    status = bolr_gaussian_state_create((bolr_const_vector_view){source->mean, source->dimension, 1}, (bolr_const_matrix_view){source->covariance, source->dimension, source->dimension, source->dimension, 1}, source->state_layout_hash, source->model_schema_hash, allocator, out_clone);
    if (status == BOLR_OK) (*out_clone)->step_index = source->step_index;
    return status;
}

bolr_status bolr_gaussian_state_set(bolr_gaussian_state *state, bolr_const_vector_view mean, bolr_const_matrix_view covariance, uint64_t step_index) {
    bolr_real *new_mean = NULL;
    bolr_real *new_covariance = NULL;
    bolr_status status;
    if (state == NULL) return BOLR_INVALID_ARGUMENT;
    status = validate_finite_vector(mean); if (status != BOLR_OK) return status;
    status = validate_covariance(covariance); if (status != BOLR_OK) return status;
    if ((mean.length != state->dimension) || (covariance.rows != state->dimension)) return BOLR_INVALID_SHAPE;
    status = allocate_copy_vector(mean, state->allocator, &new_mean); if (status != BOLR_OK) return status;
    status = allocate_copy_matrix(covariance, state->allocator, &new_covariance); if (status != BOLR_OK) { bolr_allocator_free(state->allocator, new_mean); return status; }
    status = validate_positive_definite_copy(new_covariance, state->dimension); if (status != BOLR_OK) { bolr_allocator_free(state->allocator, new_mean); bolr_allocator_free(state->allocator, new_covariance); return status; }
    bolr_allocator_free(state->allocator, state->mean);
    bolr_allocator_free(state->allocator, state->covariance);
    state->mean = new_mean;
    state->covariance = new_covariance;
    state->step_index = step_index;
    return BOLR_OK;
}

bolr_index bolr_gaussian_state_dimension(const bolr_gaussian_state *state) { return (state == NULL) ? -1 : state->dimension; }
uint64_t bolr_gaussian_state_step_index(const bolr_gaussian_state *state) { return (state == NULL) ? 0ULL : state->step_index; }
uint64_t bolr_gaussian_state_state_layout_hash(const bolr_gaussian_state *state) { return (state == NULL) ? 0ULL : state->state_layout_hash; }
uint64_t bolr_gaussian_state_model_schema_hash(const bolr_gaussian_state *state) { return (state == NULL) ? 0ULL : state->model_schema_hash; }
uint32_t bolr_gaussian_state_schema_version(const bolr_gaussian_state *state) { return (state == NULL) ? 0U : state->schema_version; }
bolr_status bolr_gaussian_state_copy_mean(const bolr_gaussian_state *state, bolr_vector_view output) { return (state == NULL) ? BOLR_INVALID_ARGUMENT : bolr_copy((bolr_const_vector_view){state->mean, state->dimension, 1}, output); }
bolr_status bolr_gaussian_state_copy_covariance(const bolr_gaussian_state *state, bolr_matrix_view output) {
    bolr_index r, c;
    if ((state == NULL) || (output.rows != state->dimension) || (output.cols != state->dimension)) return BOLR_INVALID_ARGUMENT;
    for (r = 0; r < state->dimension; ++r) for (c = 0; c < state->dimension; ++c) output.data[r * output.row_stride + c * output.col_stride] = state->covariance[r * state->dimension + c];
    return BOLR_OK;
}

bolr_status bolr_gaussian_state_export(const bolr_gaussian_state *state, const bolr_allocator *allocator, bolr_checkpoint_state **out_checkpoint) {
    bolr_checkpoint_state *checkpoint;
    bolr_checkpoint_header header;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((state == NULL) || (out_checkpoint == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_checkpoint = NULL;
    memset(&header, 0, sizeof(header));
    header.magic = 0x424F4C52U;
    header.format_major = 1U;
    header.format_minor = 0U;
    header.abi_major = 1U;
    header.abi_minor = 1U;
    header.model_schema_hash = state->model_schema_hash;
    header.state_layout_hash = state->state_layout_hash;
    header.payload_size = (uint64_t) ((state->dimension + (state->dimension * state->dimension)) * (bolr_index) sizeof(bolr_real));
    if (bolr_checkpoint_state_create(&header, active, &checkpoint) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    checkpoint->dimension = state->dimension;
    checkpoint->step_index = state->step_index;
    checkpoint->state_layout_hash = state->state_layout_hash;
    checkpoint->model_schema_hash = state->model_schema_hash;
    checkpoint->gaussian_state_schema_version = state->schema_version;
    if (allocate_copy_vector((bolr_const_vector_view){state->mean, state->dimension, 1}, active, &checkpoint->mean) != BOLR_OK) {
        bolr_checkpoint_state_destroy(checkpoint);
        return BOLR_ALLOCATION_FAILED;
    }
    if (allocate_copy_matrix((bolr_const_matrix_view){state->covariance, state->dimension, state->dimension, state->dimension, 1}, active, &checkpoint->covariance) != BOLR_OK) {
        bolr_checkpoint_state_destroy(checkpoint);
        return BOLR_ALLOCATION_FAILED;
    }
    *out_checkpoint = checkpoint;
    return BOLR_OK;
}

bolr_status bolr_gaussian_state_import(const bolr_checkpoint_state *checkpoint, const bolr_allocator *allocator, bolr_gaussian_state **out_state) {
    bolr_gaussian_state *state;
    bolr_status status;
    if ((checkpoint == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((checkpoint->mean == NULL) || (checkpoint->covariance == NULL) || (checkpoint->dimension <= 0)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    status = bolr_gaussian_state_create(
        (bolr_const_vector_view){checkpoint->mean, checkpoint->dimension, 1},
        (bolr_const_matrix_view){checkpoint->covariance, checkpoint->dimension, checkpoint->dimension, checkpoint->dimension, 1},
        checkpoint->state_layout_hash,
        checkpoint->model_schema_hash,
        allocator,
        &state
    );
    if (status != BOLR_OK) return status;
    state->step_index = checkpoint->step_index;
    state->schema_version = checkpoint->gaussian_state_schema_version;
    *out_state = state;
    return BOLR_OK;
}
