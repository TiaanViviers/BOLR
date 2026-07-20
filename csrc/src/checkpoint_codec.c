#include "bolr/checkpoint_codec.h"

#include "bolr/checksum.h"
#include "bolr/endian.h"
#include "bolr/version.h"
#include "checkpoint_internal.h"
#include "checkpoint_sections.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

#define BOLR_REPLAY_METADATA_PAYLOAD_SIZE 104U
#define BOLR_PENDING_DAY_METADATA_SIZE 36U
#define BOLR_DECISION_CONFIG_SIZE 20U
#define BOLR_PENDING_DECISION_SIZE 80U
#define BOLR_PROVENANCE_SIZE 35U
#define BOLR_CHECKPOINT_MAX_SECTIONS 32U

typedef struct {
    uint32_t section_type;
    uint32_t section_flags;
    uint64_t element_count;
    size_t payload_size;
    unsigned char *payload;
} bolr_checkpoint_built_section;

static uint64_t fnv1a_update(uint64_t state, const unsigned char *data, size_t size) {
    size_t i;
    for (i = 0U; i < size; ++i) {
        state ^= (uint64_t) data[i];
        state *= 1099511628211ULL;
    }
    return state;
}

bolr_checkpoint_limits bolr_checkpoint_limits_default(void) {
    bolr_checkpoint_limits limits;
    limits.maximum_checkpoint_bytes = 256ULL * 1024ULL * 1024ULL;
    limits.maximum_section_count = 64ULL;
    limits.maximum_candidates = 1000000ULL;
    limits.maximum_state_dimension = 100000ULL;
    limits.maximum_blocks = 10000ULL;
    limits.maximum_run_length = 1000000ULL;
    limits.maximum_context_values = 1000000ULL;
    return limits;
}

static void free_built_sections(const bolr_allocator *allocator, bolr_checkpoint_built_section *sections, size_t count) {
    size_t i;
    for (i = 0U; i < count; ++i) bolr_allocator_free(allocator, sections[i].payload);
}

static int section_compare(const void *a, const void *b) {
    const bolr_checkpoint_built_section *lhs = (const bolr_checkpoint_built_section *) a;
    const bolr_checkpoint_built_section *rhs = (const bolr_checkpoint_built_section *) b;
    if (lhs->section_type < rhs->section_type) return -1;
    if (lhs->section_type > rhs->section_type) return 1;
    return 0;
}

static size_t pending_rank_summary_payload_size(const struct bolr_replay_engine *engine) {
    bolr_index n = engine->pending_candidate_count;
    size_t size = (size_t) (8U + 8U + 4U + (size_t) n * sizeof(bolr_real));
    size = (size_t) (size + 4U + (size_t) n * sizeof(bolr_real));
    size = (size_t) (size + 4U + (size_t) n * sizeof(bolr_real));
    size = (size_t) (size + sizeof(bolr_real) + sizeof(bolr_real) + 8U);
    if ((engine->pending_probability_top_k != NULL) && (engine->pending_rank_top_k > 0)) {
        size = (size_t) (size + 4U + (size_t) n * sizeof(bolr_real));
    } else {
        size += 4U;
    }
    return size;
}

static size_t pending_region_set_payload_size(const struct bolr_replay_engine *engine) {
    return (size_t) (8U + 8U + 4U + (size_t) engine->pending_consensus_count * sizeof(bolr_index) + (size_t) engine->pending_region_count * (8U + 8U + 8U));
}

static bolr_status build_section_payload(
    const bolr_allocator *allocator,
    uint32_t section_type,
    uint32_t section_flags,
    uint64_t element_count,
    size_t payload_size,
    bolr_checkpoint_built_section *out_section
) {
    out_section->section_type = section_type;
    out_section->section_flags = section_flags;
    out_section->element_count = element_count;
    out_section->payload_size = payload_size;
    out_section->payload = NULL;
    if (payload_size == 0U) return BOLR_OK;
    out_section->payload = (unsigned char *) bolr_allocator_malloc(allocator, payload_size);
    if (out_section->payload == NULL) return BOLR_ALLOCATION_FAILED;
    memset(out_section->payload, 0, payload_size);
    return BOLR_OK;
}

static bolr_status encode_replay_metadata_payload(const struct bolr_replay_engine *engine, unsigned char *buf, size_t cap, size_t *written) {
    const struct bolr_gaussian_state *posterior = (const struct bolr_gaussian_state *) engine->posterior;
    bolr_index block_count = 0;
    bolr_index candidate_count = engine->pending_candidate_count;
    size_t cursor = 0U;
    bolr_status status;
    if (engine->adaptive_enabled && (engine->adaptive_policy != NULL)) {
        block_count = bolr_adaptive_policy_block_count(engine->adaptive_policy);
    } else {
        block_count = engine->transition.block_discount_scales.length;
    }
    status = bolr_encode_u32_le(buf, cap, &cursor, (uint32_t) engine->phase);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, engine->completed_day_index);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, &cursor, engine->adaptive_enabled ? 1U : 0U);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, &cursor, engine->adaptive_enabled ? 0U : (uint32_t) engine->transition.family);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, candidate_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, posterior->dimension);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, block_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, posterior->model_schema_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, posterior->state_layout_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, engine->graph_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, bolr_checkpoint_hash_transition_config(&engine->transition, posterior->dimension));
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, bolr_checkpoint_hash_decision_config(&engine->pending_decision_config));
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, bolr_checkpoint_hash_monte_carlo_config(&engine->pending_ranking, engine->pending_top_k, engine->pending_top_k_count));
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, engine->adaptive_enabled ? bolr_adaptive_policy_configuration_hash(engine->adaptive_policy) : 0ULL);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, &cursor, 1U);
    if (status != BOLR_OK) return status;
    *written = cursor;
    return BOLR_OK;
}

static bolr_status encode_pending_day_metadata_payload(const struct bolr_replay_engine *engine, unsigned char *buf, size_t cap, size_t *written) {
    size_t cursor = 0U;
    bolr_status status = bolr_encode_u64_le(buf, cap, &cursor, engine->pending_decision_id);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, &cursor, (uint32_t) engine->pending_decision_config.family);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, engine->pending_decision.selected_index);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, engine->pending_decision.selected_region_id);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, &cursor, engine->graph_hash);
    if (status != BOLR_OK) return status;
    *written = cursor;
    return BOLR_OK;
}

static bolr_status encode_pending_rank_summary_payload(const struct bolr_replay_engine *engine, unsigned char *buf, size_t cap, size_t *written) {
    const bolr_monte_carlo_ranking_diagnostics *diag = &engine->pending_rank_diagnostics;
    bolr_index n = engine->pending_candidate_count;
    size_t cursor = 0U;
    bolr_status status = bolr_encode_i64_le(buf, cap, &cursor, diag->sample_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, engine->pending_rank_top_k);
    if (status != BOLR_OK) return status;
    status = bolr_checkpoint_section_encode_real_array(
        buf,
        cap,
        &cursor,
        engine->pending_probability_best,
        (engine->pending_probability_best != NULL) ? n : 0
    );
    if (status != BOLR_OK) return status;
    status = bolr_checkpoint_section_encode_real_array(
        buf,
        cap,
        &cursor,
        engine->pending_expected_rank,
        (engine->pending_expected_rank != NULL) ? n : 0
    );
    if (status != BOLR_OK) return status;
    status = bolr_checkpoint_section_encode_real_array(
        buf,
        cap,
        &cursor,
        engine->pending_rank_stddev,
        (engine->pending_rank_stddev != NULL) ? n : 0
    );
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, &cursor, diag->winner_entropy);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, &cursor, diag->effective_winner_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, diag->tie_count);
    if (status != BOLR_OK) return status;
    if ((engine->pending_probability_top_k != NULL) && (engine->pending_rank_top_k > 0)) {
        status = bolr_checkpoint_section_encode_real_array(buf, cap, &cursor, engine->pending_probability_top_k, n);
        if (status != BOLR_OK) return status;
    } else {
        status = bolr_encode_u32_le(buf, cap, &cursor, 0U);
        if (status != BOLR_OK) return status;
    }
    *written = cursor;
    return BOLR_OK;
}

static bolr_status encode_pending_region_set_payload(const struct bolr_replay_engine *engine, unsigned char *buf, size_t cap, size_t *written) {
    bolr_index i;
    size_t cursor = 0U;
    bolr_status status = bolr_encode_i64_le(buf, cap, &cursor, engine->pending_region_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, &cursor, engine->pending_consensus_count);
    if (status != BOLR_OK) return status;
    status = bolr_checkpoint_section_encode_index_array(buf, cap, &cursor, engine->pending_consensus_indices, engine->pending_consensus_count);
    if (status != BOLR_OK) return status;
    for (i = 0; i < engine->pending_region_count; ++i) {
        const bolr_region_summary *summary = &engine->pending_region_summaries[i];
        status = bolr_encode_i64_le(buf, cap, &cursor, summary->region_id);
        if (status != BOLR_OK) return status;
        status = bolr_encode_f64_le(buf, cap, &cursor, summary->inclusion_mass);
        if (status != BOLR_OK) return status;
        status = bolr_encode_i64_le(buf, cap, &cursor, summary->candidate_count);
        if (status != BOLR_OK) return status;
    }
    *written = cursor;
    return BOLR_OK;
}

static bolr_status build_sections_from_engine(
    const struct bolr_replay_engine *engine,
    const bolr_allocator *allocator,
    bolr_checkpoint_built_section *sections,
    size_t *out_count,
    uint32_t *out_header_flags
) {
    const struct bolr_gaussian_state *posterior = (const struct bolr_gaussian_state *) engine->posterior;
    size_t count = 0U;
    uint32_t header_flags = BOLR_CHECKPOINT_HEADER_FLAG_RNG_STATEFUL;
    bolr_status status;
    size_t payload_size;
    size_t written;
    int awaiting = (engine->phase == BOLR_REPLAY_PHASE_AWAITING_OUTCOME);
    if (engine->adaptive_enabled) header_flags |= BOLR_CHECKPOINT_HEADER_FLAG_ADAPTIVE_PRESENT;
    if (awaiting) header_flags |= BOLR_CHECKPOINT_HEADER_FLAG_PENDING_DAY;
    if (engine->pending_rank_diagnostics.sample_count > 0) header_flags |= BOLR_CHECKPOINT_HEADER_FLAG_RANK_SUMMARY;
    if (engine->pending_region_count > 0) header_flags |= BOLR_CHECKPOINT_HEADER_FLAG_REGION_SUMMARY;
    if (engine->pending_decision_config.family == BOLR_DECISION_THOMPSON) header_flags |= BOLR_CHECKPOINT_HEADER_FLAG_THOMPSON_DECISION;

    status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_REPLAY_METADATA, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, BOLR_REPLAY_METADATA_PAYLOAD_SIZE, &sections[count]);
    if (status != BOLR_OK) return status;
    written = 0U;
    status = encode_replay_metadata_payload(engine, sections[count].payload, BOLR_REPLAY_METADATA_PAYLOAD_SIZE, &written);
    if (status != BOLR_OK) return status;
    ++count;

    status = bolr_checkpoint_section_size_gaussian(posterior, &payload_size);
    if (status != BOLR_OK) return status;
    status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_GAUSSIAN_POSTERIOR, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, payload_size, &sections[count]);
    if (status != BOLR_OK) return status;
  {
        size_t cursor = 0U;
        status = bolr_checkpoint_section_encode_gaussian(sections[count].payload, payload_size, &cursor, posterior);
        if (status != BOLR_OK) return status;
    }
    ++count;

    if (engine->adaptive_enabled) {
        status = bolr_checkpoint_section_size_adaptive(engine->adaptive_policy, engine->adaptive_state, &payload_size);
        if (status != BOLR_OK) return status;
        status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_ADAPTIVE_STATE, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, payload_size, &sections[count]);
        if (status != BOLR_OK) return status;
        {
            size_t cursor = 0U;
            status = bolr_checkpoint_section_encode_adaptive(sections[count].payload, payload_size, &cursor, engine->adaptive_policy, engine->adaptive_state);
            if (status != BOLR_OK) return status;
        }
        ++count;
    } else {
        status = bolr_checkpoint_section_size_transition(&engine->transition, posterior->dimension, &payload_size);
        if (status != BOLR_OK) return status;
        status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_TRANSITION_CONFIG, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, payload_size, &sections[count]);
        if (status != BOLR_OK) return status;
        {
            size_t cursor = 0U;
            status = bolr_checkpoint_section_encode_transition(sections[count].payload, payload_size, &cursor, &engine->transition, posterior->dimension);
            if (status != BOLR_OK) return status;
        }
        ++count;
    }

    status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_DECISION_CONFIG, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, BOLR_DECISION_CONFIG_SIZE, &sections[count]);
    if (status != BOLR_OK) return status;
    {
        size_t cursor = 0U;
        status = bolr_checkpoint_section_encode_decision_config(sections[count].payload, BOLR_DECISION_CONFIG_SIZE, &cursor, &engine->pending_decision_config);
        if (status != BOLR_OK) return status;
    }
    ++count;

    payload_size = (size_t) (8U + 8U + 4U + 4U + 4U + (size_t) engine->pending_top_k_count * 8U);
    status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_MONTE_CARLO_CONFIG, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, payload_size, &sections[count]);
    if (status != BOLR_OK) return status;
    {
        size_t cursor = 0U;
        status = bolr_checkpoint_section_encode_monte_carlo(sections[count].payload, payload_size, &cursor, &engine->pending_ranking, engine->pending_top_k, engine->pending_top_k_count);
        if (status != BOLR_OK) return status;
    }
    ++count;

    status = bolr_checkpoint_section_size_rng((const struct bolr_rng *) engine->rng, &payload_size);
    if (status != BOLR_OK) return status;
    status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_RNG_STATE, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, payload_size, &sections[count]);
    if (status != BOLR_OK) return status;
    {
        size_t cursor = 0U;
        status = bolr_checkpoint_section_encode_rng(sections[count].payload, payload_size, &cursor, (const struct bolr_rng *) engine->rng);
        if (status != BOLR_OK) return status;
    }
    ++count;

    status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PROVENANCE, BOLR_CHECKPOINT_SECTION_FLAG_OPTIONAL, 1U, BOLR_PROVENANCE_SIZE, &sections[count]);
    if (status != BOLR_OK) return status;
    {
        size_t cursor = 0U;
        status = bolr_checkpoint_section_encode_provenance(sections[count].payload, BOLR_PROVENANCE_SIZE, &cursor);
        if (status != BOLR_OK) return status;
    }
    ++count;

    if (awaiting) {
        status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_DAY_METADATA, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, BOLR_PENDING_DAY_METADATA_SIZE, &sections[count]);
        if (status != BOLR_OK) return status;
        {
            status = encode_pending_day_metadata_payload(engine, sections[count].payload, BOLR_PENDING_DAY_METADATA_SIZE, &written);
            if (status != BOLR_OK) return status;
        }
        ++count;

        status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_SCORE_CONTEXT, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, (uint64_t) engine->pending_context_length, 0U, &sections[count]);
        if (status != BOLR_OK) return status;
        payload_size = (size_t) (4U + (size_t) engine->pending_context_length * sizeof(bolr_real));
        sections[count].payload_size = payload_size;
        sections[count].payload = (unsigned char *) bolr_allocator_malloc(allocator, payload_size);
        if (sections[count].payload == NULL) return BOLR_ALLOCATION_FAILED;
        {
            size_t cursor = 0U;
            status = bolr_checkpoint_section_encode_real_array(sections[count].payload, payload_size, &cursor, engine->pending_context, engine->pending_context_length);
            if (status != BOLR_OK) return status;
        }
        ++count;

        if (engine->pending_predictive != NULL) {
            status = bolr_checkpoint_section_size_gaussian((const struct bolr_gaussian_state *) engine->pending_predictive, &payload_size);
            if (status != BOLR_OK) return status;
            status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_PREDICTIVE_GAUSSIAN, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, payload_size, &sections[count]);
            if (status != BOLR_OK) return status;
            {
                size_t cursor = 0U;
                status = bolr_checkpoint_section_encode_gaussian(sections[count].payload, payload_size, &cursor, (const struct bolr_gaussian_state *) engine->pending_predictive);
                if (status != BOLR_OK) return status;
            }
            ++count;
        }

        payload_size = (size_t) (8U + (size_t) engine->pending_candidate_count * sizeof(bolr_real) * 2U);
        status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_POSTERIOR_PREDICTION, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, (uint64_t) engine->pending_candidate_count, payload_size, &sections[count]);
        if (status != BOLR_OK) return status;
        {
            bolr_index i;
            size_t cursor = 0U;
            status = bolr_encode_i64_le(sections[count].payload, payload_size, &cursor, engine->pending_candidate_count);
            if (status != BOLR_OK) return status;
            for (i = 0; i < engine->pending_candidate_count; ++i) {
                status = bolr_encode_f64_le(sections[count].payload, payload_size, &cursor, engine->pending_score_mean[i]);
                if (status != BOLR_OK) return status;
            }
            for (i = 0; i < engine->pending_candidate_count; ++i) {
                status = bolr_encode_f64_le(sections[count].payload, payload_size, &cursor, engine->pending_score_variance[i]);
                if (status != BOLR_OK) return status;
            }
        }
        ++count;

        if (engine->pending_rank_diagnostics.sample_count > 0) {
            payload_size = pending_rank_summary_payload_size(engine);
            status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_RANK_SUMMARY, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, (uint64_t) engine->pending_candidate_count, payload_size, &sections[count]);
            if (status != BOLR_OK) return status;
            {
                status = encode_pending_rank_summary_payload(engine, sections[count].payload, payload_size, &written);
                if (status != BOLR_OK) return status;
            }
            ++count;
        }

        if (engine->pending_region_count > 0) {
            payload_size = pending_region_set_payload_size(engine);
            status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_REGION_SET, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, (uint64_t) engine->pending_region_count, payload_size, &sections[count]);
            if (status != BOLR_OK) return status;
            {
                status = encode_pending_region_set_payload(engine, sections[count].payload, payload_size, &written);
                if (status != BOLR_OK) return status;
            }
            ++count;
        }

        status = build_section_payload(allocator, BOLR_CHECKPOINT_SECTION_PENDING_DECISION, BOLR_CHECKPOINT_SECTION_FLAG_REQUIRED, 1U, BOLR_PENDING_DECISION_SIZE, &sections[count]);
        if (status != BOLR_OK) return status;
        {
            size_t cursor = 0U;
            status = bolr_checkpoint_section_encode_pending_decision(sections[count].payload, BOLR_PENDING_DECISION_SIZE, &cursor, &engine->pending_decision);
            if (status != BOLR_OK) return status;
        }
        ++count;
    }

    qsort(sections, count, sizeof(sections[0]), section_compare);
    *out_count = count;
    *out_header_flags = header_flags;
    return BOLR_OK;
}

static bolr_status compute_total_size(size_t section_count, size_t payload_size, size_t *out_total) {
    size_t directory_bytes;
    if (bolr_checked_size_mul(section_count, (size_t) BOLR_CHECKPOINT_DIRECTORY_ENTRY_SIZE, &directory_bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    if (bolr_checked_size_add(BOLR_CHECKPOINT_HEADER_SIZE, directory_bytes, out_total) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    return bolr_checked_size_add(*out_total, payload_size, out_total);
}

static bolr_status write_header(
    unsigned char *output,
    uint32_t section_count,
    uint32_t header_flags,
    uint64_t total_file_size,
    uint64_t payload_size,
    uint32_t payload_crc32,
    const struct bolr_replay_engine *engine,
    uint64_t checkpoint_id
) {
    const struct bolr_gaussian_state *posterior = (const struct bolr_gaussian_state *) engine->posterior;
    size_t cursor = 0U;
    bolr_status status;
    uint32_t header_crc_placeholder = 0U;
    status = bolr_encode_bytes(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, BOLR_CHECKPOINT_MAGIC, 8U);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u16_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint16_t) BOLR_CHECKPOINT_FORMAT_MAJOR);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u16_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint16_t) BOLR_CHECKPOINT_FORMAT_MINOR);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint32_t) BOLR_CHECKPOINT_HEADER_SIZE);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint32_t) BOLR_CHECKPOINT_DIRECTORY_ENTRY_SIZE);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, section_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, header_flags);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, total_file_size);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint64_t) BOLR_CHECKPOINT_HEADER_SIZE);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint64_t) BOLR_CHECKPOINT_HEADER_SIZE + (uint64_t) section_count * (uint64_t) BOLR_CHECKPOINT_DIRECTORY_ENTRY_SIZE);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, payload_size);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, payload_crc32);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, header_crc_placeholder);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u16_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint16_t) bolr_abi_version_major());
    if (status != BOLR_OK) return status;
    status = bolr_encode_u16_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint16_t) bolr_abi_version_minor());
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, (uint32_t) engine->phase);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, engine->completed_day_index);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, posterior->model_schema_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, posterior->state_layout_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, engine->graph_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, bolr_checkpoint_hash_transition_config(&engine->transition, posterior->dimension));
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, bolr_checkpoint_hash_decision_config(&engine->pending_decision_config));
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, bolr_checkpoint_hash_monte_carlo_config(&engine->pending_ranking, engine->pending_top_k, engine->pending_top_k_count));
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, engine->adaptive_enabled ? bolr_adaptive_policy_configuration_hash(engine->adaptive_policy) : 0ULL);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(output, BOLR_CHECKPOINT_HEADER_SIZE, &cursor, checkpoint_id);
    if (status != BOLR_OK) return status;
    memset(output + cursor, 0, 32U);
    cursor += 32U;
    return (cursor == BOLR_CHECKPOINT_HEADER_SIZE) ? BOLR_OK : BOLR_INVALID_SHAPE;
}

static bolr_status finalize_header_crc(unsigned char *output) {
    uint32_t crc;
    memset(output + 64U, 0, sizeof(uint32_t));
    crc = bolr_crc32(output, BOLR_CHECKPOINT_HEADER_SIZE);
    memcpy(output + 64U, &crc, sizeof(crc));
    return BOLR_OK;
}

static bolr_status compute_encoded_size(const struct bolr_replay_engine *engine, const bolr_allocator *allocator, size_t *out_size) {
    bolr_checkpoint_built_section sections[BOLR_CHECKPOINT_MAX_SECTIONS];
    size_t section_count = 0U;
    size_t payload_total = 0U;
    size_t i;
    uint32_t header_flags = 0U;
    bolr_status status;
    memset(sections, 0, sizeof(sections));
    status = build_sections_from_engine(engine, allocator, sections, &section_count, &header_flags);
    if (status != BOLR_OK) goto cleanup;
    for (i = 0U; i < section_count; ++i) {
        if (bolr_checked_size_add(payload_total, sections[i].payload_size, &payload_total) != BOLR_OK) { status = BOLR_DIMENSION_OVERFLOW; goto cleanup; }
    }
    status = compute_total_size(section_count, payload_total, out_size);
cleanup:
    free_built_sections(allocator, sections, section_count);
    return status;
}

static bolr_status encode_checkpoint_bytes(
    const struct bolr_replay_engine *engine,
    const bolr_allocator *allocator,
    void *output,
    size_t output_size,
    size_t *out_written
) {
    bolr_checkpoint_built_section sections[BOLR_CHECKPOINT_MAX_SECTIONS];
    size_t section_count = 0U;
    size_t payload_total = 0U;
    size_t total_size = 0U;
    size_t payload_offset;
    size_t i;
    uint32_t header_flags = 0U;
    uint32_t payload_crc32 = 0U;
    uint64_t checkpoint_id;
    bolr_status status;
    memset(sections, 0, sizeof(sections));
    status = bolr_platform_validate();
    if (status != BOLR_OK) return status;
    status = build_sections_from_engine(engine, allocator, sections, &section_count, &header_flags);
    if (status != BOLR_OK) goto cleanup;
    for (i = 0U; i < section_count; ++i) {
        if (bolr_checked_size_add(payload_total, sections[i].payload_size, &payload_total) != BOLR_OK) { status = BOLR_DIMENSION_OVERFLOW; goto cleanup; }
    }
    status = compute_total_size(section_count, payload_total, &total_size);
    if (status != BOLR_OK) goto cleanup;
    if (output_size < total_size) { status = BOLR_INVALID_SHAPE; goto cleanup; }
    payload_offset = (size_t) BOLR_CHECKPOINT_HEADER_SIZE + section_count * (size_t) BOLR_CHECKPOINT_DIRECTORY_ENTRY_SIZE;
    memset(output, 0, total_size);
    checkpoint_id = fnv1a_update(14695981039346656037ULL, (const unsigned char *) engine->posterior->mean, (size_t) engine->posterior->dimension * sizeof(bolr_real));
    checkpoint_id = fnv1a_update(checkpoint_id, (const unsigned char *) &engine->completed_day_index, sizeof(engine->completed_day_index));
    /* Placeholder payload CRC; overwritten after payloads are copied. */
    status = write_header(output, (uint32_t) section_count, header_flags, (uint64_t) total_size, (uint64_t) payload_total, payload_crc32, engine, checkpoint_id);
    if (status != BOLR_OK) goto cleanup;
    {
        size_t cursor = BOLR_CHECKPOINT_HEADER_SIZE;
        size_t payload_cursor = payload_offset;
        for (i = 0U; i < section_count; ++i) {
            uint32_t section_crc = (sections[i].payload_size > 0U) ? bolr_crc32(sections[i].payload, sections[i].payload_size) : 0U;
            status = bolr_encode_u32_le(output, total_size, &cursor, sections[i].section_type);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u16_le(output, total_size, &cursor, BOLR_CHECKPOINT_SECTION_SCHEMA_MAJOR);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u16_le(output, total_size, &cursor, BOLR_CHECKPOINT_SECTION_SCHEMA_MINOR);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u32_le(output, total_size, &cursor, sections[i].section_flags);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u64_le(output, total_size, &cursor, (uint64_t) payload_cursor);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u64_le(output, total_size, &cursor, (uint64_t) sections[i].payload_size);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u64_le(output, total_size, &cursor, sections[i].element_count);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u32_le(output, total_size, &cursor, section_crc);
            if (status != BOLR_OK) goto cleanup;
            status = bolr_encode_u32_le(output, total_size, &cursor, 0U);
            if (status != BOLR_OK) goto cleanup;
            if (sections[i].payload_size > 0U) {
                memcpy((unsigned char *) output + payload_cursor, sections[i].payload, sections[i].payload_size);
                payload_cursor += sections[i].payload_size;
            }
        }
        if (payload_total > 0U) {
            size_t crc_cursor = 60U;
            uint32_t actual_payload_crc = bolr_crc32((const unsigned char *) output + payload_offset, payload_total);
            status = bolr_encode_u32_le(output, total_size, &crc_cursor, actual_payload_crc);
            if (status != BOLR_OK) goto cleanup;
        }
    }
    status = finalize_header_crc((unsigned char *) output);
    if (status != BOLR_OK) goto cleanup;
    if (out_written != NULL) *out_written = total_size;
cleanup:
    free_built_sections(allocator, sections, section_count);
    return status;
}

bolr_status bolr_replay_checkpoint_encoded_size(const bolr_replay_engine *engine, size_t *out_size) {
    const bolr_allocator *allocator;
    if ((engine == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    allocator = ((const struct bolr_replay_engine *) engine)->allocator;
    return compute_encoded_size((const struct bolr_replay_engine *) engine, allocator, out_size);
}

bolr_status bolr_replay_checkpoint_encode(const bolr_replay_engine *engine, void *output, size_t output_size, size_t *out_written) {
    const bolr_allocator *allocator;
    size_t needed = 0U;
    bolr_status status;
    if ((engine == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    allocator = ((const struct bolr_replay_engine *) engine)->allocator;
    status = bolr_replay_checkpoint_encoded_size(engine, &needed);
    if (status != BOLR_OK) return status;
    if (output_size < needed) return BOLR_INVALID_SHAPE;
    return encode_checkpoint_bytes((const struct bolr_replay_engine *) engine, allocator, output, output_size, out_written);
}

bolr_status bolr_replay_checkpoint_encode_buffer(const bolr_replay_engine *engine, const bolr_allocator *allocator, void **out_bytes, size_t *out_size) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    size_t needed = 0U;
    void *buffer;
    bolr_status status;
    if ((engine == NULL) || (out_bytes == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_bytes = NULL;
    status = bolr_replay_checkpoint_encoded_size(engine, &needed);
    if (status != BOLR_OK) return status;
    buffer = bolr_allocator_malloc(active, needed);
    if (buffer == NULL) return BOLR_ALLOCATION_FAILED;
    status = bolr_replay_checkpoint_encode(engine, buffer, needed, out_size);
    if (status != BOLR_OK) {
        bolr_allocator_free(active, buffer);
        return status;
    }
    *out_bytes = buffer;
    return BOLR_OK;
}

bolr_status bolr_replay_checkpoint_size_report(const bolr_replay_engine *engine, bolr_checkpoint_size_report *out) {
    const bolr_allocator *allocator;
    bolr_checkpoint_built_section sections[BOLR_CHECKPOINT_MAX_SECTIONS];
    size_t section_count = 0U;
    size_t payload_total = 0U;
    size_t directory_bytes = 0U;
    size_t total = 0U;
    size_t i;
    uint32_t header_flags = 0U;
    bolr_status status;
    if ((engine == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    allocator = ((const struct bolr_replay_engine *) engine)->allocator;
    memset(sections, 0, sizeof(sections));
    status = build_sections_from_engine((const struct bolr_replay_engine *) engine, allocator, sections, &section_count, &header_flags);
    if (status != BOLR_OK) return status;
    for (i = 0U; i < section_count; ++i) {
        if (bolr_checked_size_add(payload_total, sections[i].payload_size, &payload_total) != BOLR_OK) {
            free_built_sections(allocator, sections, section_count);
            return BOLR_DIMENSION_OVERFLOW;
        }
    }
    status = compute_total_size(section_count, payload_total, &total);
    if (status != BOLR_OK) {
        free_built_sections(allocator, sections, section_count);
        return status;
    }
    if (bolr_checked_size_mul(section_count, (size_t) BOLR_CHECKPOINT_DIRECTORY_ENTRY_SIZE, &directory_bytes) != BOLR_OK) {
        free_built_sections(allocator, sections, section_count);
        return BOLR_DIMENSION_OVERFLOW;
    }
    memset(out, 0, sizeof(*out));
    out->total_bytes = total;
    out->header_bytes = BOLR_CHECKPOINT_HEADER_SIZE;
    out->directory_bytes = directory_bytes;
    out->payload_bytes = payload_total;
    out->candidate_count = ((const struct bolr_replay_engine *) engine)->pending_candidate_count;
    out->state_dimension = bolr_gaussian_state_dimension(((const struct bolr_replay_engine *) engine)->posterior);
    free_built_sections(allocator, sections, section_count);
    return BOLR_OK;
}

static bolr_status find_section(const bolr_checkpoint_parsed_section *sections, size_t count, uint32_t type, const bolr_checkpoint_parsed_section **out) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (sections[i].section_type == type) {
            if (out != NULL) *out = &sections[i];
            return BOLR_OK;
        }
    }
    return BOLR_CHECKPOINT_MISSING_SECTION;
}

static bolr_status parse_directory(const unsigned char *data, size_t data_size, size_t directory_offset, uint32_t section_count, bolr_checkpoint_parsed_section *sections) {
    size_t i;
    size_t cursor = directory_offset;
    for (i = 0U; i < (size_t) section_count; ++i) {
        uint64_t payload_offset = 0ULL;
        uint64_t payload_length = 0ULL;
        uint32_t section_crc = 0U;
        uint32_t reserved = 0U;
        uint16_t schema_major = 0U;
        uint16_t schema_minor = 0U;
        bolr_status status;
        status = bolr_decode_u32_le(data, data_size, &cursor, &sections[i].section_type);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u16_le(data, data_size, &cursor, &schema_major);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u16_le(data, data_size, &cursor, &schema_minor);
        if (status != BOLR_OK) return status;
        (void) schema_major;
        (void) schema_minor;
        status = bolr_decode_u32_le(data, data_size, &cursor, &sections[i].section_flags);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u64_le(data, data_size, &cursor, &payload_offset);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u64_le(data, data_size, &cursor, &payload_length);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u64_le(data, data_size, &cursor, &sections[i].element_count);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u32_le(data, data_size, &cursor, &section_crc);
        if (status != BOLR_OK) return status;
        status = bolr_decode_u32_le(data, data_size, &cursor, &reserved);
        if (status != BOLR_OK) return status;
        if (reserved != 0U) return BOLR_CHECKPOINT_INVALID_DIRECTORY;
        if ((payload_offset + payload_length) > data_size) return BOLR_CHECKPOINT_TRUNCATED;
        sections[i].payload = data + payload_offset;
        sections[i].payload_length = (size_t) payload_length;
        if (payload_length > 0U) {
            uint32_t actual_crc = bolr_crc32(sections[i].payload, sections[i].payload_length);
            if (actual_crc != section_crc) return BOLR_CHECKPOINT_CHECKSUM_MISMATCH;
        }
    }
    return BOLR_OK;
}

bolr_status bolr_replay_checkpoint_decode(const void *data, size_t data_size, const bolr_replay_restore_context *context, const bolr_allocator *allocator, bolr_replay_engine **out_engine) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    const unsigned char *bytes = (const unsigned char *) data;
    bolr_checkpoint_parsed_section sections[BOLR_CHECKPOINT_MAX_SECTIONS];
    struct bolr_replay_checkpoint *checkpoint = NULL;
    const bolr_checkpoint_parsed_section *section = NULL;
    bolr_checkpoint_limits limits;
    uint32_t section_count = 0U;
    uint32_t header_flags = 0U;
    uint32_t payload_crc = 0U;
    uint32_t header_crc = 0U;
    uint64_t total_file_size = 0ULL;
    uint64_t payload_size = 0ULL;
    uint64_t payload_offset = 0ULL;
    uint16_t format_major = 0U;
    uint16_t format_minor = 0U;
    uint32_t replay_phase = 0U;
    int64_t completed_day_index = 0;
    uint64_t model_hash = 0ULL;
    uint64_t layout_hash = 0ULL;
    uint64_t grid_hash = 0ULL;
    uint64_t replay_hash = 0ULL;
    uint64_t decision_hash = 0ULL;
    uint64_t monte_hash = 0ULL;
    uint64_t adaptive_hash = 0ULL;
    size_t cursor = 0U;
    bolr_status status;
    if ((data == NULL) || (context == NULL) || (out_engine == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_engine = NULL;
    memset(sections, 0, sizeof(sections));
    limits = context->limits.maximum_checkpoint_bytes == 0ULL ? bolr_checkpoint_limits_default() : context->limits;
    if (data_size > (size_t) limits.maximum_checkpoint_bytes) return BOLR_CHECKPOINT_LIMIT_EXCEEDED;
    status = bolr_platform_validate();
    if (status != BOLR_OK) return status;
    if (data_size < BOLR_CHECKPOINT_HEADER_SIZE) return BOLR_CHECKPOINT_TRUNCATED;
    if (memcmp(bytes, BOLR_CHECKPOINT_MAGIC, 8U) != 0) return BOLR_CHECKPOINT_BAD_MAGIC;
    cursor = 8U;
    status = bolr_decode_u16_le(bytes, data_size, &cursor, &format_major);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u16_le(bytes, data_size, &cursor, &format_minor);
    if (status != BOLR_OK) return status;
    if ((format_major != BOLR_CHECKPOINT_FORMAT_MAJOR) || (format_minor != BOLR_CHECKPOINT_FORMAT_MINOR)) return BOLR_CHECKPOINT_UNSUPPORTED_VERSION;
    cursor = 64U;
    memcpy(&header_crc, bytes + 64U, sizeof(header_crc));
    {
        unsigned char header_copy[BOLR_CHECKPOINT_HEADER_SIZE];
        uint32_t computed_crc;
        memcpy(header_copy, bytes, BOLR_CHECKPOINT_HEADER_SIZE);
        memset(header_copy + 64U, 0, sizeof(uint32_t));
        computed_crc = bolr_crc32(header_copy, BOLR_CHECKPOINT_HEADER_SIZE);
        if (computed_crc != header_crc) return BOLR_CHECKPOINT_CHECKSUM_MISMATCH;
    }
    cursor = 20U;
    status = bolr_decode_u32_le(bytes, data_size, &cursor, &section_count);
    if (status != BOLR_OK) return status;
    if (section_count > limits.maximum_section_count) return BOLR_CHECKPOINT_LIMIT_EXCEEDED;
    status = bolr_decode_u32_le(bytes, data_size, &cursor, &header_flags);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &total_file_size);
    if (status != BOLR_OK) return status;
    if (total_file_size != data_size) return BOLR_CHECKPOINT_TRUNCATED;
    cursor = 44U;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &payload_offset);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &payload_size);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(bytes, data_size, &cursor, &payload_crc);
    if (status != BOLR_OK) return status;
    if (payload_size > 0U) {
        uint32_t actual_payload_crc = bolr_crc32(bytes + payload_offset, (size_t) payload_size);
        if (actual_payload_crc != payload_crc) return BOLR_CHECKPOINT_CHECKSUM_MISMATCH;
    }
    cursor = 72U;
    status = bolr_decode_u32_le(bytes, data_size, &cursor, &replay_phase);
    if (status != BOLR_OK) return status;
    status = bolr_decode_i64_le(bytes, data_size, &cursor, &completed_day_index);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &model_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &layout_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &grid_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &replay_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &decision_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &monte_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(bytes, data_size, &cursor, &adaptive_hash);
    if (status != BOLR_OK) return status;
    if ((context->expected_model_schema_hash != 0ULL) && (context->expected_model_schema_hash != model_hash)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if ((context->expected_state_layout_hash != 0ULL) && (context->expected_state_layout_hash != layout_hash)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if ((context->expected_grid_hash != 0ULL) && (context->expected_grid_hash != grid_hash)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if ((context->expected_replay_config_hash != 0ULL) && (context->expected_replay_config_hash != replay_hash)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if ((header_flags & BOLR_CHECKPOINT_HEADER_FLAG_ADAPTIVE_PRESENT) != 0U) {
        if ((context->expected_adaptive_policy_hash != 0ULL) && (context->expected_adaptive_policy_hash != adaptive_hash)) return BOLR_INCOMPATIBLE_CHECKPOINT;
        if (context->adaptive_policy == NULL) return BOLR_INCOMPATIBLE_CHECKPOINT;
    }
    if (((header_flags & BOLR_CHECKPOINT_HEADER_FLAG_PENDING_DAY) != 0U) && (replay_phase != BOLR_REPLAY_PHASE_AWAITING_OUTCOME)) return BOLR_CHECKPOINT_INVALID_DIRECTORY;
    if (((header_flags & BOLR_CHECKPOINT_HEADER_FLAG_PENDING_DAY) == 0U) && (replay_phase == BOLR_REPLAY_PHASE_AWAITING_OUTCOME)) return BOLR_CHECKPOINT_INVALID_DIRECTORY;
    status = parse_directory(bytes, data_size, BOLR_CHECKPOINT_HEADER_SIZE, section_count, sections);
    if (status != BOLR_OK) return status;

    checkpoint = (struct bolr_replay_checkpoint *) bolr_allocator_calloc(active, 1U, sizeof(*checkpoint));
    if (checkpoint == NULL) return BOLR_ALLOCATION_FAILED;
    checkpoint->allocator = active;
    checkpoint->phase = (bolr_replay_phase) replay_phase;
    checkpoint->completed_day_index = completed_day_index;
    checkpoint->graph_hash = grid_hash;
    checkpoint->adaptive_enabled = ((header_flags & BOLR_CHECKPOINT_HEADER_FLAG_ADAPTIVE_PRESENT) != 0U) ? 1 : 0;
    checkpoint->adaptive_policy_hash = adaptive_hash;

    status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_GAUSSIAN_POSTERIOR, &section);
    if (status != BOLR_OK) goto fail_checkpoint;
    {
        size_t sc = 0U;
        bolr_gaussian_state *posterior = NULL;
        status = bolr_checkpoint_section_decode_gaussian(section->payload, section->payload_length, &sc, active, &posterior);
        if (status != BOLR_OK) goto fail_checkpoint;
        status = bolr_gaussian_state_export(posterior, active, &checkpoint->posterior_checkpoint);
        bolr_gaussian_state_destroy(posterior);
        if (status != BOLR_OK) goto fail_checkpoint;
    }

    if (checkpoint->adaptive_enabled) {
        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_ADAPTIVE_STATE, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            bolr_adaptive_state *adaptive_state = NULL;
            status = bolr_checkpoint_section_decode_adaptive(section->payload, section->payload_length, &sc, context->adaptive_policy, active, &adaptive_state, &checkpoint->adaptive_state_bytes, &checkpoint->adaptive_state_size);
            bolr_adaptive_state_destroy(adaptive_state);
            if (status != BOLR_OK) goto fail_checkpoint;
        }
    } else {
        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_TRANSITION_CONFIG, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            bolr_index dim = 0;
            status = bolr_checkpoint_section_decode_transition(section->payload, section->payload_length, &sc, active, &checkpoint->transition, &checkpoint->transition_process_noise, &checkpoint->transition_block_discount_scales, &dim);
            if (status != BOLR_OK) goto fail_checkpoint;
        }
    }

    status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_DECISION_CONFIG, &section);
    if (status != BOLR_OK) goto fail_checkpoint;
    {
        size_t sc = 0U;
        status = bolr_checkpoint_section_decode_decision_config(section->payload, section->payload_length, &sc, &checkpoint->pending_decision_config);
        if (status != BOLR_OK) goto fail_checkpoint;
    }

    status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_MONTE_CARLO_CONFIG, &section);
    if (status != BOLR_OK) goto fail_checkpoint;
    {
        size_t sc = 0U;
        status = bolr_checkpoint_section_decode_monte_carlo(section->payload, section->payload_length, &sc, active, &checkpoint->pending_ranking, &checkpoint->pending_top_k, &checkpoint->pending_top_k_count);
        if (status != BOLR_OK) goto fail_checkpoint;
    }

    status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_RNG_STATE, &section);
    if (status != BOLR_OK) goto fail_checkpoint;
    {
        size_t sc = 0U;
        bolr_rng *rng = NULL;
        status = bolr_checkpoint_section_decode_rng(section->payload, section->payload_length, &sc, active, &rng);
        if (status != BOLR_OK) goto fail_checkpoint;
        status = bolr_rng_export(rng, active, &checkpoint->rng_checkpoint);
        bolr_rng_destroy(rng);
        if (status != BOLR_OK) goto fail_checkpoint;
    }

    if (checkpoint->phase == BOLR_REPLAY_PHASE_AWAITING_OUTCOME) {
        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_DAY_METADATA, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            uint32_t family;
            status = bolr_decode_u64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_decision_id);
            if (status != BOLR_OK) goto fail_checkpoint;
            status = bolr_decode_u32_le(section->payload, section->payload_length, &sc, &family);
            if (status != BOLR_OK) goto fail_checkpoint;
            checkpoint->pending_decision_config.family = (bolr_decision_family) family;
            status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_decision.selected_index);
            if (status != BOLR_OK) goto fail_checkpoint;
            status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_decision.selected_region_id);
            if (status != BOLR_OK) goto fail_checkpoint;
        }

        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_SCORE_CONTEXT, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            status = bolr_checkpoint_section_decode_real_array(section->payload, section->payload_length, &sc, active, &checkpoint->pending_context, &checkpoint->pending_context_length);
            if (status != BOLR_OK) goto fail_checkpoint;
        }

        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_PREDICTIVE_GAUSSIAN, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            bolr_gaussian_state *predictive = NULL;
            status = bolr_checkpoint_section_decode_gaussian(section->payload, section->payload_length, &sc, active, &predictive);
            if (status != BOLR_OK) goto fail_checkpoint;
            status = bolr_gaussian_state_export(predictive, active, &checkpoint->pending_predictive_checkpoint);
            bolr_gaussian_state_destroy(predictive);
            if (status != BOLR_OK) goto fail_checkpoint;
        }

        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_POSTERIOR_PREDICTION, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            int64_t candidate_count = 0;
            bolr_index i;
            status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &candidate_count);
            if (status != BOLR_OK) goto fail_checkpoint;
            checkpoint->pending_candidate_count = candidate_count;
            if (candidate_count > 0) {
                checkpoint->pending_score_mean = (bolr_real *) bolr_allocator_malloc(active, (size_t) candidate_count * sizeof(bolr_real));
                checkpoint->pending_score_variance = (bolr_real *) bolr_allocator_malloc(active, (size_t) candidate_count * sizeof(bolr_real));
                if ((checkpoint->pending_score_mean == NULL) || (checkpoint->pending_score_variance == NULL)) {
                    status = BOLR_ALLOCATION_FAILED;
                    goto fail_checkpoint;
                }
                for (i = 0; i < candidate_count; ++i) {
                    status = bolr_decode_f64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_score_mean[i]);
                    if (status != BOLR_OK) goto fail_checkpoint;
                }
                for (i = 0; i < candidate_count; ++i) {
                    status = bolr_decode_f64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_score_variance[i]);
                    if (status != BOLR_OK) goto fail_checkpoint;
                }
            }
        }

        if ((header_flags & BOLR_CHECKPOINT_HEADER_FLAG_RANK_SUMMARY) != 0U) {
            status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_RANK_SUMMARY, &section);
            if (status != BOLR_OK) goto fail_checkpoint;
            {
                size_t sc = 0U;
                bolr_index dummy_count;
                status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_rank_diagnostics.sample_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_rank_top_k);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_checkpoint_section_decode_real_array(section->payload, section->payload_length, &sc, active, &checkpoint->pending_probability_best, &dummy_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_checkpoint_section_decode_real_array(section->payload, section->payload_length, &sc, active, &checkpoint->pending_expected_rank, &dummy_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_checkpoint_section_decode_real_array(section->payload, section->payload_length, &sc, active, &checkpoint->pending_rank_stddev, &dummy_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_decode_f64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_rank_diagnostics.winner_entropy);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_decode_f64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_rank_diagnostics.effective_winner_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_rank_diagnostics.tie_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                if (sc < section->payload_length) {
                    status = bolr_checkpoint_section_decode_real_array(section->payload, section->payload_length, &sc, active, &checkpoint->pending_probability_top_k, &dummy_count);
                    if (status != BOLR_OK) goto fail_checkpoint;
                }
            }
        }

        if ((header_flags & BOLR_CHECKPOINT_HEADER_FLAG_REGION_SUMMARY) != 0U) {
            status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_REGION_SET, &section);
            if (status != BOLR_OK) goto fail_checkpoint;
            {
                size_t sc = 0U;
                bolr_index i;
                status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_region_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_consensus_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                status = bolr_checkpoint_section_decode_index_array(section->payload, section->payload_length, &sc, active, &checkpoint->pending_consensus_indices, &checkpoint->pending_consensus_count);
                if (status != BOLR_OK) goto fail_checkpoint;
                if (checkpoint->pending_region_count > 0) {
                    checkpoint->pending_region_summaries = (bolr_region_summary *) bolr_allocator_calloc(active, (size_t) checkpoint->pending_region_count, sizeof(bolr_region_summary));
                    if (checkpoint->pending_region_summaries == NULL) { status = BOLR_ALLOCATION_FAILED; goto fail_checkpoint; }
                    for (i = 0; i < checkpoint->pending_region_count; ++i) {
                        int64_t region_id;
                        int64_t candidate_count;
                        status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &region_id);
                        if (status != BOLR_OK) goto fail_checkpoint;
                        checkpoint->pending_region_summaries[i].region_id = region_id;
                        status = bolr_decode_f64_le(section->payload, section->payload_length, &sc, &checkpoint->pending_region_summaries[i].inclusion_mass);
                        if (status != BOLR_OK) goto fail_checkpoint;
                        status = bolr_decode_i64_le(section->payload, section->payload_length, &sc, &candidate_count);
                        if (status != BOLR_OK) goto fail_checkpoint;
                        checkpoint->pending_region_summaries[i].candidate_count = candidate_count;
                    }
                }
            }
        }

        status = find_section(sections, section_count, BOLR_CHECKPOINT_SECTION_PENDING_DECISION, &section);
        if (status != BOLR_OK) goto fail_checkpoint;
        {
            size_t sc = 0U;
            status = bolr_checkpoint_section_decode_pending_decision(section->payload, section->payload_length, &sc, &checkpoint->pending_decision);
            if (status != BOLR_OK) goto fail_checkpoint;
        }
    }

    if (checkpoint->adaptive_enabled) {
        status = bolr_replay_engine_import_adaptive(checkpoint, context->adaptive_policy, active, out_engine);
    } else {
        status = bolr_replay_engine_import_fixed(checkpoint, active, out_engine);
    }
    if (status != BOLR_OK) goto fail_checkpoint;
    bolr_replay_checkpoint_destroy(checkpoint);
    return BOLR_OK;

fail_checkpoint:
    bolr_replay_checkpoint_destroy(checkpoint);
    *out_engine = NULL;
    return status;
}
