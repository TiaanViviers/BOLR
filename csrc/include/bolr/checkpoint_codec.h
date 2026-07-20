#ifndef BOLR_CHECKPOINT_CODEC_H
#define BOLR_CHECKPOINT_CODEC_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/adaptation.h"
#include "bolr/allocator.h"
#include "bolr/replay.h"
#include "bolr/types.h"

typedef struct {
    uint64_t maximum_checkpoint_bytes;
    uint64_t maximum_section_count;
    uint64_t maximum_candidates;
    uint64_t maximum_state_dimension;
    uint64_t maximum_blocks;
    uint64_t maximum_run_length;
    uint64_t maximum_context_values;
} bolr_checkpoint_limits;

bolr_checkpoint_limits bolr_checkpoint_limits_default(void);

typedef struct {
    const bolr_adaptive_policy *adaptive_policy;
    uint64_t expected_model_schema_hash;
    uint64_t expected_state_layout_hash;
    uint64_t expected_grid_hash;
    uint64_t expected_replay_config_hash;
    uint64_t expected_adaptive_policy_hash;
    bolr_checkpoint_limits limits;
} bolr_replay_restore_context;

bolr_status bolr_replay_checkpoint_encoded_size(const bolr_replay_engine *engine, size_t *out_size);
bolr_status bolr_replay_checkpoint_encode(const bolr_replay_engine *engine, void *output, size_t output_size, size_t *out_written);
bolr_status bolr_replay_checkpoint_encode_buffer(const bolr_replay_engine *engine, const bolr_allocator *allocator, void **out_bytes, size_t *out_size);
bolr_status bolr_replay_checkpoint_decode(const void *data, size_t data_size, const bolr_replay_restore_context *context, const bolr_allocator *allocator, bolr_replay_engine **out_engine);

typedef struct {
    size_t total_bytes;
    size_t header_bytes;
    size_t directory_bytes;
    size_t payload_bytes;
    bolr_index candidate_count;
    bolr_index state_dimension;
} bolr_checkpoint_size_report;

bolr_status bolr_replay_checkpoint_size_report(const bolr_replay_engine *engine, bolr_checkpoint_size_report *out);

#endif
