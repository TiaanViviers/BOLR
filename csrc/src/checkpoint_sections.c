#include "checkpoint_internal.h"
#include "checkpoint_sections.h"

#include "bolr/checksum.h"
#include "bolr/endian.h"
#include "bolr/version.h"

#include <string.h>

static uint64_t fnv1a_update(uint64_t state, const unsigned char *data, size_t size) {
    size_t i;
    for (i = 0U; i < size; ++i) {
        state ^= (uint64_t) data[i];
        state *= 1099511628211ULL;
    }
    return state;
}

static uint64_t fnv1a_bytes(const void *data, size_t size) {
    return fnv1a_update(14695981039346656037ULL, (const unsigned char *) data, size);
}

uint64_t bolr_checkpoint_hash_decision_config(const bolr_decision_policy_config *config) {
    return (config == NULL) ? 0ULL : fnv1a_bytes(config, sizeof(*config));
}

uint64_t bolr_checkpoint_hash_monte_carlo_config(const bolr_replay_ranking_config *ranking, const bolr_index *top_k, bolr_index top_k_count) {
    uint64_t h = 14695981039346656037ULL;
    if (ranking != NULL) h = fnv1a_update(h, (const unsigned char *) ranking, sizeof(*ranking));
    if ((top_k != NULL) && (top_k_count > 0)) h = fnv1a_update(h, (const unsigned char *) top_k, (size_t) top_k_count * sizeof(bolr_index));
    return h;
}

uint64_t bolr_checkpoint_hash_transition_config(const bolr_transition_config *transition, bolr_index dimension) {
    uint64_t h = 14695981039346656037ULL;
    bolr_index r, c;
    if (transition == NULL) return 0ULL;
    h = fnv1a_update(h, (const unsigned char *) &transition->family, sizeof(transition->family));
    h = fnv1a_update(h, (const unsigned char *) &transition->global_discount, sizeof(transition->global_discount));
    if ((transition->process_noise.data != NULL) && (dimension > 0)) {
        for (r = 0; r < dimension; ++r) {
            for (c = 0; c < dimension; ++c) {
                bolr_real v = transition->process_noise.data[r * transition->process_noise.row_stride + c * transition->process_noise.col_stride];
                h = fnv1a_update(h, (const unsigned char *) &v, sizeof(v));
            }
        }
    }
    if (transition->block_discount_scales.data != NULL) {
        for (c = 0; c < transition->block_discount_scales.length; ++c) {
            bolr_real v = transition->block_discount_scales.data[c * transition->block_discount_scales.stride];
            h = fnv1a_update(h, (const unsigned char *) &v, sizeof(v));
        }
    }
    return h;
}

uint64_t bolr_checkpoint_hash_pending_decision_id(const bolr_decision *decision, const bolr_decision_policy_config *config, uint64_t graph_hash) {
    uint64_t h = 14695981039346656037ULL;
    if (decision != NULL) h = fnv1a_update(h, (const unsigned char *) decision, sizeof(*decision));
    if (config != NULL) h = fnv1a_update(h, (const unsigned char *) config, sizeof(*config));
    h = fnv1a_update(h, (const unsigned char *) &graph_hash, sizeof(graph_hash));
    return h;
}

static bolr_status encode_real_vector(void *buf, size_t cap, size_t *cursor, const bolr_real *values, bolr_index count) {
    bolr_index i;
    bolr_status status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) count);
    if (status != BOLR_OK) return status;
    for (i = 0; i < count; ++i) {
        status = bolr_encode_f64_le(buf, cap, cursor, values[i]);
        if (status != BOLR_OK) return status;
    }
    return BOLR_OK;
}

static bolr_status decode_real_vector(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_real **out_values, bolr_index *out_count) {
    uint32_t count_u32;
    bolr_index i;
    bolr_real *values;
    bolr_status status = bolr_decode_u32_le(buf, cap, cursor, &count_u32);
    if (status != BOLR_OK) return status;
    *out_values = NULL;
    *out_count = (bolr_index) count_u32;
    if (count_u32 == 0U) return BOLR_OK;
    values = (bolr_real *) bolr_allocator_malloc(allocator, (size_t) count_u32 * sizeof(bolr_real));
    if (values == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < (bolr_index) count_u32; ++i) {
        status = bolr_decode_f64_le(buf, cap, cursor, &values[i]);
        if (status != BOLR_OK) { bolr_allocator_free(allocator, values); return status; }
    }
    *out_values = values;
    return BOLR_OK;
}

static bolr_status encode_index_vector(void *buf, size_t cap, size_t *cursor, const bolr_index *values, bolr_index count) {
    bolr_index i;
    bolr_status status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) count);
    if (status != BOLR_OK) return status;
    for (i = 0; i < count; ++i) {
        status = bolr_encode_i64_le(buf, cap, cursor, values[i]);
        if (status != BOLR_OK) return status;
    }
    return BOLR_OK;
}

static bolr_status decode_index_vector(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_index **out_values, bolr_index *out_count) {
    uint32_t count_u32;
    bolr_index i;
    bolr_index *values;
    int64_t raw;
    bolr_status status = bolr_decode_u32_le(buf, cap, cursor, &count_u32);
    if (status != BOLR_OK) return status;
    *out_values = NULL;
    *out_count = (bolr_index) count_u32;
    if (count_u32 == 0U) return BOLR_OK;
    values = (bolr_index *) bolr_allocator_malloc(allocator, (size_t) count_u32 * sizeof(bolr_index));
    if (values == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < (bolr_index) count_u32; ++i) {
        status = bolr_decode_i64_le(buf, cap, cursor, &raw);
        if (status != BOLR_OK) { bolr_allocator_free(allocator, values); return status; }
        values[i] = raw;
    }
    *out_values = values;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_size_gaussian(const struct bolr_gaussian_state *state, size_t *out_size) {
    bolr_index dim;
    if ((state == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    dim = state->dimension;
    *out_size = (size_t) (4U + 4U + 8U + 8U + 8U + 8U + (size_t) dim * sizeof(bolr_real) + (size_t) dim * (size_t) dim * sizeof(bolr_real));
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_gaussian(void *buf, size_t cap, size_t *cursor, const struct bolr_gaussian_state *state) {
    bolr_index dim;
    bolr_index r, c;
    bolr_status status;
    if ((state == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    dim = state->dimension;
    status = bolr_encode_u32_le(buf, cap, cursor, state->schema_version);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) dim);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, state->step_index);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, state->state_layout_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, state->model_schema_hash);
    if (status != BOLR_OK) return status;
    for (r = 0; r < dim; ++r) {
        status = bolr_encode_f64_le(buf, cap, cursor, state->mean[r]);
        if (status != BOLR_OK) return status;
    }
    for (r = 0; r < dim; ++r) {
        for (c = 0; c < dim; ++c) {
            status = bolr_encode_f64_le(buf, cap, cursor, state->covariance[r * dim + c]);
            if (status != BOLR_OK) return status;
        }
    }
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_decode_gaussian(
    const void *buf,
    size_t cap,
    size_t *cursor,
    const bolr_allocator *allocator,
    struct bolr_gaussian_state **out_state
) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    uint32_t schema;
    uint32_t dim_u32;
    uint64_t step_index;
    uint64_t layout_hash;
    uint64_t model_hash;
    bolr_index dim;
    bolr_real *mean = NULL;
    bolr_real *cov = NULL;
    bolr_status status;
    if ((cursor == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    status = bolr_decode_u32_le(buf, cap, cursor, &schema);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &dim_u32);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(buf, cap, cursor, &step_index);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(buf, cap, cursor, &layout_hash);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u64_le(buf, cap, cursor, &model_hash);
    if (status != BOLR_OK) return status;
    dim = (bolr_index) dim_u32;
    if (dim > 0) {
        bolr_index r, c;
        mean = (bolr_real *) bolr_allocator_malloc(active, (size_t) dim * sizeof(bolr_real));
        cov = (bolr_real *) bolr_allocator_malloc(active, (size_t) dim * (size_t) dim * sizeof(bolr_real));
        if ((mean == NULL) || (cov == NULL)) {
            bolr_allocator_free(active, mean);
            bolr_allocator_free(active, cov);
            return BOLR_ALLOCATION_FAILED;
        }
        for (r = 0; r < dim; ++r) {
            status = bolr_decode_f64_le(buf, cap, cursor, &mean[r]);
            if (status != BOLR_OK) goto fail;
        }
        for (r = 0; r < dim; ++r) {
            for (c = 0; c < dim; ++c) {
                status = bolr_decode_f64_le(buf, cap, cursor, &cov[r * dim + c]);
                if (status != BOLR_OK) goto fail;
            }
        }
    }
    status = bolr_gaussian_state_create(
        (bolr_const_vector_view){mean, dim, 1},
        (bolr_const_matrix_view){cov, dim, dim, dim, 1},
        layout_hash,
        model_hash,
        active,
        out_state
    );
    if (status == BOLR_OK) {
        status = bolr_gaussian_state_set(*out_state, (bolr_const_vector_view){mean, dim, 1}, (bolr_const_matrix_view){cov, dim, dim, dim, 1}, step_index);
    }
fail:
    bolr_allocator_free(active, mean);
    bolr_allocator_free(active, cov);
    if (status != BOLR_OK) bolr_gaussian_state_destroy(*out_state);
    return status;
}

bolr_status bolr_checkpoint_section_size_transition(const bolr_transition_config *transition, bolr_index dimension, size_t *out_size) {
    bolr_index scale_len = (transition == NULL) ? 0 : transition->block_discount_scales.length;
    if (out_size == NULL) return BOLR_INVALID_ARGUMENT;
    *out_size = (size_t) (4U + 4U + (size_t) dimension * (size_t) dimension * sizeof(bolr_real) + sizeof(bolr_real) + 4U + (size_t) scale_len * sizeof(bolr_real));
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_transition(void *buf, size_t cap, size_t *cursor, const bolr_transition_config *transition, bolr_index dimension) {
    bolr_index r, c;
    bolr_status status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) transition->family);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) dimension);
    if (status != BOLR_OK) return status;
    for (r = 0; r < dimension; ++r) {
        for (c = 0; c < dimension; ++c) {
            bolr_real v = 0.0;
            if ((transition->process_noise.data != NULL) && (r < transition->process_noise.rows) && (c < transition->process_noise.cols)) {
                v = transition->process_noise.data[r * transition->process_noise.row_stride + c * transition->process_noise.col_stride];
            }
            status = bolr_encode_f64_le(buf, cap, cursor, v);
            if (status != BOLR_OK) return status;
        }
    }
    status = bolr_encode_f64_le(buf, cap, cursor, transition->global_discount);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) transition->block_discount_scales.length);
    if (status != BOLR_OK) return status;
    for (r = 0; r < transition->block_discount_scales.length; ++r) {
        bolr_real v = transition->block_discount_scales.data[r * transition->block_discount_scales.stride];
        status = bolr_encode_f64_le(buf, cap, cursor, v);
        if (status != BOLR_OK) return status;
    }
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_decode_transition(
    const void *buf,
    size_t cap,
    size_t *cursor,
    const bolr_allocator *allocator,
    bolr_transition_config *out_transition,
    bolr_real **out_process_noise,
    bolr_real **out_block_scales,
    bolr_index *out_dimension
) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    uint32_t family;
    uint32_t dim_u32;
    uint32_t scale_len;
    bolr_index dim;
    bolr_real *process_noise = NULL;
    bolr_real *scales = NULL;
    bolr_status status;
    if ((cursor == NULL) || (out_transition == NULL) || (out_process_noise == NULL) || (out_block_scales == NULL) || (out_dimension == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_process_noise = NULL;
    *out_block_scales = NULL;
    status = bolr_decode_u32_le(buf, cap, cursor, &family);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &dim_u32);
    if (status != BOLR_OK) return status;
    dim = (bolr_index) dim_u32;
    *out_dimension = dim;
    if (dim > 0) {
        bolr_index r, c;
        size_t bytes;
        if (bolr_checked_size_mul((size_t) dim * (size_t) dim, sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
        process_noise = (bolr_real *) bolr_allocator_malloc(active, bytes);
        if (process_noise == NULL) return BOLR_ALLOCATION_FAILED;
        for (r = 0; r < dim; ++r) {
            for (c = 0; c < dim; ++c) {
                status = bolr_decode_f64_le(buf, cap, cursor, &process_noise[r * dim + c]);
                if (status != BOLR_OK) goto fail;
            }
        }
    }
    out_transition->family = (bolr_transition_family) family;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_transition->global_discount);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u32_le(buf, cap, cursor, &scale_len);
    if (status != BOLR_OK) goto fail;
    if (scale_len > 0U) {
        bolr_index i;
        scales = (bolr_real *) bolr_allocator_malloc(active, (size_t) scale_len * sizeof(bolr_real));
        if (scales == NULL) { status = BOLR_ALLOCATION_FAILED; goto fail; }
        for (i = 0; i < (bolr_index) scale_len; ++i) {
            status = bolr_decode_f64_le(buf, cap, cursor, &scales[i]);
            if (status != BOLR_OK) goto fail;
        }
    }
    out_transition->process_noise = (bolr_const_matrix_view){process_noise, dim, dim, dim, 1};
    out_transition->block_discount_scales = (bolr_const_vector_view){scales, (bolr_index) scale_len, 1};
    *out_process_noise = process_noise;
    *out_block_scales = scales;
    return BOLR_OK;
fail:
    bolr_allocator_free(active, process_noise);
    bolr_allocator_free(active, scales);
    return status;
}

bolr_status bolr_checkpoint_section_size_rng(const struct bolr_rng *rng, size_t *out_size) {
    (void) rng;
    if (out_size == NULL) return BOLR_INVALID_ARGUMENT;
    *out_size = (size_t) (8U + 8U + 4U * 5U + 8U * 6U);
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_rng(void *buf, size_t cap, size_t *cursor, const struct bolr_rng *rng) {
    bolr_rng_metadata metadata;
    bolr_status status;
    if ((rng == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    bolr_rng_metadata_copy((const bolr_rng *) rng, &metadata);
    status = bolr_encode_u64_le(buf, cap, cursor, rng->state);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, rng->increment);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, metadata.schema_version);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, metadata.algorithm_family);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, metadata.algorithm_version);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, metadata.pcg_variant);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, metadata.ziggurat_layers);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, metadata.table_hash);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, metadata.seed);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, metadata.stream);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, metadata.u32_draw_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, metadata.uniform_draw_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u64_le(buf, cap, cursor, metadata.normal_draw_count);
    return status;
}

bolr_status bolr_checkpoint_section_decode_rng(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_rng **out_rng) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    struct bolr_rng_checkpoint *checkpoint;
    bolr_status status;
    if ((cursor == NULL) || (out_rng == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_rng = NULL;
    checkpoint = (struct bolr_rng_checkpoint *) bolr_allocator_calloc(active, 1U, sizeof(*checkpoint));
    if (checkpoint == NULL) return BOLR_ALLOCATION_FAILED;
    checkpoint->allocator = active;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->state);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->increment);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u32_le(buf, cap, cursor, &checkpoint->metadata.schema_version);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u32_le(buf, cap, cursor, &checkpoint->metadata.algorithm_family);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u32_le(buf, cap, cursor, &checkpoint->metadata.algorithm_version);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u32_le(buf, cap, cursor, &checkpoint->metadata.pcg_variant);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u32_le(buf, cap, cursor, &checkpoint->metadata.ziggurat_layers);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->metadata.table_hash);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->metadata.seed);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->metadata.stream);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->metadata.u32_draw_count);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->metadata.uniform_draw_count);
    if (status != BOLR_OK) goto fail;
    status = bolr_decode_u64_le(buf, cap, cursor, &checkpoint->metadata.normal_draw_count);
    if (status != BOLR_OK) goto fail;
    status = bolr_rng_import(checkpoint, active, out_rng);
fail:
    bolr_rng_checkpoint_destroy(checkpoint);
    return status;
}

bolr_status bolr_checkpoint_section_size_adaptive(const bolr_adaptive_policy *policy, const bolr_adaptive_state *state, size_t *out_size) {
    size_t blob = 0U;
    bolr_status status;
    if ((policy == NULL) || (state == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    status = bolr_adaptive_state_encoded_size(policy, state, &blob);
    if (status != BOLR_OK) return status;
    *out_size = (size_t) (4U + 4U + blob);
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_adaptive(void *buf, size_t cap, size_t *cursor, const bolr_adaptive_policy *policy, const bolr_adaptive_state *state) {
    size_t blob_size = 0U;
    size_t written = 0U;
    unsigned char *blob;
    bolr_status status;
    if ((policy == NULL) || (state == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    status = bolr_adaptive_state_encoded_size(policy, state, &blob_size);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, BOLR_CHECKPOINT_SECTION_SCHEMA_MAJOR);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) blob_size);
    if (status != BOLR_OK) return status;
    blob = (unsigned char *) buf + *cursor;
    if ((*cursor + blob_size) > cap) return BOLR_CHECKPOINT_TRUNCATED;
    status = bolr_adaptive_state_encode(policy, state, blob, blob_size, &written);
    if (status != BOLR_OK) return status;
    *cursor += written;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_decode_adaptive(
    const void *buf,
    size_t cap,
    size_t *cursor,
    const bolr_adaptive_policy *policy,
    const bolr_allocator *allocator,
    bolr_adaptive_state **out_state,
    void **out_bytes,
    size_t *out_size
) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    uint32_t schema;
    uint32_t blob_size;
    bolr_status status;
    if ((policy == NULL) || (cursor == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    if (out_bytes != NULL) *out_bytes = NULL;
    if (out_size != NULL) *out_size = 0U;
    status = bolr_decode_u32_le(buf, cap, cursor, &schema);
    if (status != BOLR_OK) return status;
    (void) schema;
    status = bolr_decode_u32_le(buf, cap, cursor, &blob_size);
    if (status != BOLR_OK) return status;
    if ((*cursor + (size_t) blob_size) > cap) return BOLR_CHECKPOINT_TRUNCATED;
    status = bolr_adaptive_state_decode(policy, (const unsigned char *) buf + *cursor, (size_t) blob_size, active, out_state);
    if (status != BOLR_OK) return status;
    if ((out_bytes != NULL) && (out_size != NULL) && (blob_size > 0U)) {
        void *copy = bolr_allocator_malloc(active, (size_t) blob_size);
        if (copy == NULL) { bolr_adaptive_state_destroy(*out_state); *out_state = NULL; return BOLR_ALLOCATION_FAILED; }
        memcpy(copy, (const unsigned char *) buf + *cursor, (size_t) blob_size);
        *out_bytes = copy;
        *out_size = (size_t) blob_size;
    }
    *cursor += (size_t) blob_size;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_decision_config(void *buf, size_t cap, size_t *cursor, const bolr_decision_policy_config *config) {
    bolr_status status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) config->family);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, cursor, config->top_k);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) config->region_selection_statistic);
    if (status != BOLR_OK) return status;
    return bolr_encode_u32_le(buf, cap, cursor, (uint32_t) config->representative_policy);
}

bolr_status bolr_checkpoint_section_decode_decision_config(const void *buf, size_t cap, size_t *cursor, bolr_decision_policy_config *out_config) {
    uint32_t family;
    int64_t top_k;
    uint32_t region_stat;
    uint32_t rep;
    bolr_status status = bolr_decode_u32_le(buf, cap, cursor, &family);
    if (status != BOLR_OK) return status;
    status = bolr_decode_i64_le(buf, cap, cursor, &top_k);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &region_stat);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &rep);
    if (status != BOLR_OK) return status;
    out_config->family = (bolr_decision_family) family;
    out_config->top_k = top_k;
    out_config->region_selection_statistic = (bolr_region_statistic) region_stat;
    out_config->representative_policy = (bolr_region_representative) rep;
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_monte_carlo(void *buf, size_t cap, size_t *cursor, const bolr_replay_ranking_config *ranking, const bolr_index *top_k, bolr_index top_k_count) {
    bolr_index i;
    bolr_status status = bolr_encode_i64_le(buf, cap, cursor, ranking->sample_count);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, cursor, ranking->chunk_size);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, ranking->antithetic ? 1U : 0U);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) ranking->retention);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, (uint32_t) top_k_count);
    if (status != BOLR_OK) return status;
    for (i = 0; i < top_k_count; ++i) {
        status = bolr_encode_i64_le(buf, cap, cursor, top_k[i]);
        if (status != BOLR_OK) return status;
    }
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_decode_monte_carlo(
    const void *buf,
    size_t cap,
    size_t *cursor,
    const bolr_allocator *allocator,
    bolr_replay_ranking_config *out_ranking,
    bolr_index **out_top_k,
    bolr_index *out_top_k_count
) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    int64_t sample_count;
    int64_t chunk_size;
    uint32_t antithetic;
    uint32_t retention;
    uint32_t top_k_count_u32;
    bolr_status status;
    if ((cursor == NULL) || (out_ranking == NULL) || (out_top_k == NULL) || (out_top_k_count == NULL)) return BOLR_INVALID_ARGUMENT;
    status = bolr_decode_i64_le(buf, cap, cursor, &sample_count);
    if (status != BOLR_OK) return status;
    status = bolr_decode_i64_le(buf, cap, cursor, &chunk_size);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &antithetic);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &retention);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &top_k_count_u32);
    if (status != BOLR_OK) return status;
    out_ranking->sample_count = sample_count;
    out_ranking->chunk_size = chunk_size;
    out_ranking->antithetic = (antithetic != 0U) ? 1 : 0;
    out_ranking->retention = (bolr_score_retention) retention;
    *out_top_k = NULL;
    *out_top_k_count = (bolr_index) top_k_count_u32;
    if (top_k_count_u32 > 0U) {
        bolr_index i;
        bolr_index *top_k = (bolr_index *) bolr_allocator_malloc(active, (size_t) top_k_count_u32 * sizeof(bolr_index));
        if (top_k == NULL) return BOLR_ALLOCATION_FAILED;
        for (i = 0; i < (bolr_index) top_k_count_u32; ++i) {
            int64_t raw;
            status = bolr_decode_i64_le(buf, cap, cursor, &raw);
            if (status != BOLR_OK) { bolr_allocator_free(active, top_k); return status; }
            top_k[i] = raw;
        }
        *out_top_k = top_k;
    }
    return BOLR_OK;
}

bolr_status bolr_checkpoint_section_encode_pending_decision(void *buf, size_t cap, size_t *cursor, const bolr_decision *decision) {
    bolr_status status = bolr_encode_i64_le(buf, cap, cursor, decision->selected_index);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, decision->selected ? 1U : 0U);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, decision->abstained ? 1U : 0U);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, cursor, decision->selected_score_mean);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, cursor, decision->selected_score_variance);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, cursor, decision->selected_probability_best);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, cursor, decision->selected_expected_rank);
    if (status != BOLR_OK) return status;
    status = bolr_encode_i64_le(buf, cap, cursor, decision->selected_region_id);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, cursor, decision->selected_region_mass);
    if (status != BOLR_OK) return status;
    status = bolr_encode_f64_le(buf, cap, cursor, decision->selected_region_probability_best);
    if (status != BOLR_OK) return status;
    status = bolr_encode_u32_le(buf, cap, cursor, decision->tie_flags);
    if (status != BOLR_OK) return status;
    return bolr_encode_u32_le(buf, cap, cursor, decision->reason_code);
}

bolr_status bolr_checkpoint_section_decode_pending_decision(const void *buf, size_t cap, size_t *cursor, bolr_decision *out_decision) {
    int64_t selected_index;
    uint32_t selected;
    uint32_t abstained;
    int64_t region_id;
    bolr_status status = bolr_decode_i64_le(buf, cap, cursor, &selected_index);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &selected);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &abstained);
    if (status != BOLR_OK) return status;
    out_decision->selected_index = selected_index;
    out_decision->selected = (selected != 0U) ? 1 : 0;
    out_decision->abstained = (abstained != 0U) ? 1 : 0;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_decision->selected_score_mean);
    if (status != BOLR_OK) return status;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_decision->selected_score_variance);
    if (status != BOLR_OK) return status;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_decision->selected_probability_best);
    if (status != BOLR_OK) return status;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_decision->selected_expected_rank);
    if (status != BOLR_OK) return status;
    status = bolr_decode_i64_le(buf, cap, cursor, &region_id);
    if (status != BOLR_OK) return status;
    out_decision->selected_region_id = region_id;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_decision->selected_region_mass);
    if (status != BOLR_OK) return status;
    status = bolr_decode_f64_le(buf, cap, cursor, &out_decision->selected_region_probability_best);
    if (status != BOLR_OK) return status;
    status = bolr_decode_u32_le(buf, cap, cursor, &out_decision->tie_flags);
    if (status != BOLR_OK) return status;
    return bolr_decode_u32_le(buf, cap, cursor, &out_decision->reason_code);
}

bolr_status bolr_checkpoint_section_encode_provenance(void *buf, size_t cap, size_t *cursor) {
    char version[32];
    size_t version_len;
    strncpy(version, bolr_library_version(), sizeof(version) - 1U);
    version[sizeof(version) - 1U] = '\0';
    version_len = strlen(version);
    if (version_len > 31U) version_len = 31U;
    {
        bolr_status status = bolr_encode_bytes(buf, cap, cursor, version, version_len);
        if (status != BOLR_OK) return status;
    }
    return bolr_encode_u32_le(buf, cap, cursor, bolr_abi_version_major());
}

bolr_status bolr_checkpoint_section_decode_provenance(const void *buf, size_t cap, size_t *cursor) {
    char version[32];
    uint32_t abi_major;
    size_t version_len = 31U;
    bolr_status status = bolr_decode_bytes(buf, cap, cursor, version, version_len);
    if (status != BOLR_OK) return status;
    return bolr_decode_u32_le(buf, cap, cursor, &abi_major);
}

bolr_status bolr_checkpoint_section_encode_real_array(void *buf, size_t cap, size_t *cursor, const bolr_real *values, bolr_index count) {
    return encode_real_vector(buf, cap, cursor, values, count);
}

bolr_status bolr_checkpoint_section_decode_real_array(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_real **out_values, bolr_index *out_count) {
    return decode_real_vector(buf, cap, cursor, allocator, out_values, out_count);
}

bolr_status bolr_checkpoint_section_encode_index_array(void *buf, size_t cap, size_t *cursor, const bolr_index *values, bolr_index count) {
    return encode_index_vector(buf, cap, cursor, values, count);
}

bolr_status bolr_checkpoint_section_decode_index_array(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_index **out_values, bolr_index *out_count) {
    return decode_index_vector(buf, cap, cursor, allocator, out_values, out_count);
}
