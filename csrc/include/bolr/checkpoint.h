#ifndef BOLR_CHECKPOINT_H
#define BOLR_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/allocator.h"
#include "bolr/array.h"

typedef struct bolr_checkpoint_state bolr_checkpoint_state;

typedef struct {
    uint32_t magic;
    uint16_t format_major;
    uint16_t format_minor;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint64_t model_schema_hash;
    uint64_t state_layout_hash;
    uint64_t configuration_hash;
    uint64_t payload_size;
    uint64_t flags;
} bolr_checkpoint_header;

bolr_status bolr_checkpoint_state_create(const bolr_checkpoint_header *header, const bolr_allocator *allocator, bolr_checkpoint_state **out_state);
void bolr_checkpoint_state_destroy(bolr_checkpoint_state *state);
bolr_status bolr_checkpoint_state_header(const bolr_checkpoint_state *state, bolr_checkpoint_header *out_header);
bolr_status bolr_checkpoint_state_copy_mean(const bolr_checkpoint_state *state, bolr_vector_view output);
bolr_status bolr_checkpoint_state_copy_covariance(const bolr_checkpoint_state *state, bolr_matrix_view output);
uint64_t bolr_checkpoint_state_step_index(const bolr_checkpoint_state *state);
uint64_t bolr_checkpoint_state_state_layout_hash(const bolr_checkpoint_state *state);
uint64_t bolr_checkpoint_state_model_schema_hash(const bolr_checkpoint_state *state);
bolr_status bolr_checkpoint_encoded_size(const bolr_checkpoint_state *state, size_t *out_size);
bolr_status bolr_checkpoint_encode(const bolr_checkpoint_state *state, void *output, size_t output_size, size_t *out_written);
bolr_status bolr_checkpoint_decode(const void *data, size_t data_size, const bolr_allocator *allocator, bolr_checkpoint_state **out_state);

#endif
