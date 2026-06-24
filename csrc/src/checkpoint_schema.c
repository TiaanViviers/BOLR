#include "bolr/checkpoint.h"

#include "bolr/linalg.h"
#include "bolr/math.h"
#include "bolr/version.h"
#include "internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    bolr_checkpoint_header header;
    int64_t dimension;
    uint64_t step_index;
    uint64_t state_layout_hash;
    uint64_t model_schema_hash;
    uint32_t gaussian_state_schema_version;
    uint32_t reserved;
} bolr_checkpoint_wire_header;

static bolr_status alloc_dense_copy(const bolr_allocator *allocator, size_t count, const bolr_real *source, bolr_real **out) {
    size_t bytes;
    bolr_real *copy;
    bolr_status status = bolr_checked_size_mul(count, sizeof(bolr_real), &bytes);
    if (status != BOLR_OK) return status;
    copy = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (copy == NULL) return BOLR_ALLOCATION_FAILED;
    memcpy(copy, source, bytes);
    *out = copy;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_state_create(const bolr_checkpoint_header *header, const bolr_allocator *allocator, bolr_checkpoint_state **out_state) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_checkpoint_state *state;
    if ((header == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    state = (bolr_checkpoint_state *) bolr_allocator_calloc(active, 1U, sizeof(*state));
    if (state == NULL) return BOLR_ALLOCATION_FAILED;
    state->allocator = active;
    state->header = *header;
    *out_state = state;
    return BOLR_OK;
}

void bolr_checkpoint_state_destroy(bolr_checkpoint_state *state) {
    if (state == NULL) return;
    bolr_allocator_free(state->allocator, state->mean);
    bolr_allocator_free(state->allocator, state->covariance);
    bolr_allocator_free(state->allocator, state);
}

bolr_status bolr_checkpoint_state_header(const bolr_checkpoint_state *state, bolr_checkpoint_header *out_header) {
    if ((state == NULL) || (out_header == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_header = state->header;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_state_copy_mean(const bolr_checkpoint_state *state, bolr_vector_view output) {
    return ((state == NULL) || (state->mean == NULL)) ? BOLR_INVALID_ARGUMENT : bolr_copy((bolr_const_vector_view){state->mean, state->dimension, 1}, output);
}

bolr_status bolr_checkpoint_state_copy_covariance(const bolr_checkpoint_state *state, bolr_matrix_view output) {
    bolr_index r, c;
    if ((state == NULL) || (state->covariance == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((output.rows != state->dimension) || (output.cols != state->dimension)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < state->dimension; ++r) {
        for (c = 0; c < state->dimension; ++c) {
            output.data[r * output.row_stride + c * output.col_stride] = state->covariance[r * state->dimension + c];
        }
    }
    return BOLR_OK;
}

uint64_t bolr_checkpoint_state_step_index(const bolr_checkpoint_state *state) { return (state == NULL) ? 0ULL : state->step_index; }
uint64_t bolr_checkpoint_state_state_layout_hash(const bolr_checkpoint_state *state) { return (state == NULL) ? 0ULL : state->state_layout_hash; }
uint64_t bolr_checkpoint_state_model_schema_hash(const bolr_checkpoint_state *state) { return (state == NULL) ? 0ULL : state->model_schema_hash; }

bolr_status bolr_checkpoint_encoded_size(const bolr_checkpoint_state *state, size_t *out_size) {
    size_t dense_count;
    size_t dense_bytes;
    size_t total;
    bolr_status status;
    if ((state == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    dense_count = (size_t) state->dimension * (size_t) (state->dimension + 1);
    status = bolr_checked_size_mul(dense_count, sizeof(bolr_real), &dense_bytes);
    if (status != BOLR_OK) return status;
    status = bolr_checked_size_add(sizeof(bolr_checkpoint_wire_header), dense_bytes, &total);
    if (status != BOLR_OK) return status;
    *out_size = total;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_encode(const bolr_checkpoint_state *state, void *output, size_t output_size, size_t *out_written) {
    bolr_checkpoint_wire_header wire;
    size_t expected;
    size_t offset;
    if (out_written != NULL) *out_written = 0U;
    if ((state == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    if (bolr_checkpoint_encoded_size(state, &expected) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    if (output_size < expected) return BOLR_INVALID_SHAPE;
    memset(&wire, 0, sizeof(wire));
    wire.header = state->header;
    wire.dimension = state->dimension;
    wire.step_index = state->step_index;
    wire.state_layout_hash = state->state_layout_hash;
    wire.model_schema_hash = state->model_schema_hash;
    wire.gaussian_state_schema_version = state->gaussian_state_schema_version;
    memcpy(output, &wire, sizeof(wire));
    offset = sizeof(wire);
    memcpy((unsigned char *) output + offset, state->mean, (size_t) state->dimension * sizeof(bolr_real));
    offset += (size_t) state->dimension * sizeof(bolr_real);
    memcpy((unsigned char *) output + offset, state->covariance, (size_t) (state->dimension * state->dimension) * sizeof(bolr_real));
    if (out_written != NULL) *out_written = expected;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_decode(const void *data, size_t data_size, const bolr_allocator *allocator, bolr_checkpoint_state **out_state) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    const bolr_checkpoint_wire_header *wire;
    bolr_checkpoint_state *state;
    size_t expected;
    size_t offset;
    if (out_state != NULL) *out_state = NULL;
    if ((data == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    if (data_size < sizeof(bolr_checkpoint_wire_header)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    wire = (const bolr_checkpoint_wire_header *) data;
    if ((wire->dimension <= 0) || (wire->gaussian_state_schema_version == 0U)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    expected = sizeof(bolr_checkpoint_wire_header) +
        (size_t) wire->dimension * sizeof(bolr_real) +
        (size_t) (wire->dimension * wire->dimension) * sizeof(bolr_real);
    if (data_size < expected) return BOLR_INCOMPATIBLE_CHECKPOINT;
    state = (bolr_checkpoint_state *) bolr_allocator_calloc(active, 1U, sizeof(*state));
    if (state == NULL) return BOLR_ALLOCATION_FAILED;
    state->allocator = active;
    state->header = wire->header;
    state->dimension = (bolr_index) wire->dimension;
    state->step_index = wire->step_index;
    state->state_layout_hash = wire->state_layout_hash;
    state->model_schema_hash = wire->model_schema_hash;
    state->gaussian_state_schema_version = wire->gaussian_state_schema_version;
    offset = sizeof(*wire);
    if (alloc_dense_copy(active, (size_t) state->dimension, (const bolr_real *) ((const unsigned char *) data + offset), &state->mean) != BOLR_OK) {
        bolr_checkpoint_state_destroy(state);
        return BOLR_ALLOCATION_FAILED;
    }
    offset += (size_t) state->dimension * sizeof(bolr_real);
    if (alloc_dense_copy(active, (size_t) (state->dimension * state->dimension), (const bolr_real *) ((const unsigned char *) data + offset), &state->covariance) != BOLR_OK) {
        bolr_checkpoint_state_destroy(state);
        return BOLR_ALLOCATION_FAILED;
    }
    *out_state = state;
    return BOLR_OK;
}
