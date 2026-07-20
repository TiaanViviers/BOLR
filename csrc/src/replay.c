#include "bolr/replay.h"

#include "bolr/version.h"
#include "checkpoint_internal.h"
#include "internal.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static bolr_status copy_real_array(const bolr_allocator *allocator, bolr_const_vector_view source, bolr_real **out) {
    size_t bytes;
    bolr_real *copy;
    if (source.length <= 0) {
        *out = NULL;
        return BOLR_OK;
    }
    if (bolr_checked_size_mul((size_t) source.length, sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    copy = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (copy == NULL) return BOLR_ALLOCATION_FAILED;
    for (bolr_index i = 0; i < source.length; ++i) copy[i] = source.data[i * source.stride];
    *out = copy;
    return BOLR_OK;
}

static bolr_status copy_real_matrix_row_major(const bolr_allocator *allocator, bolr_const_matrix_view source, bolr_real **out) {
    size_t bytes;
    bolr_real *copy;
    bolr_index r, c;
    if ((source.rows <= 0) || (source.cols <= 0) || (source.data == NULL)) {
        *out = NULL;
        return BOLR_OK;
    }
    if (bolr_checked_size_mul((size_t) (source.rows * source.cols), sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    copy = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (copy == NULL) return BOLR_ALLOCATION_FAILED;
    for (r = 0; r < source.rows; ++r) for (c = 0; c < source.cols; ++c) copy[r * source.cols + c] = source.data[r * source.row_stride + c * source.col_stride];
    *out = copy;
    return BOLR_OK;
}

static bolr_status copy_index_array(const bolr_allocator *allocator, const bolr_index *source, bolr_index count, bolr_index **out) {
    size_t bytes;
    bolr_index *copy;
    if (count <= 0) {
        *out = NULL;
        return BOLR_OK;
    }
    if (source == NULL) return BOLR_INVALID_ARGUMENT;
    if (bolr_checked_size_mul((size_t) count, sizeof(bolr_index), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    copy = (bolr_index *) bolr_allocator_malloc(allocator, bytes);
    if (copy == NULL) return BOLR_ALLOCATION_FAILED;
    memcpy(copy, source, bytes);
    *out = copy;
    return BOLR_OK;
}

static uint64_t fnv1a_update(uint64_t state, const unsigned char *data, size_t size) {
    size_t i;
    for (i = 0U; i < size; ++i) {
        state ^= (uint64_t) data[i];
        state *= 1099511628211ULL;
    }
    return state;
}

static uint64_t hash_pending_decision_id(const bolr_decision *decision, const bolr_decision_policy_config *config, uint64_t graph_hash) {
    uint64_t h = 14695981039346656037ULL;
    if (decision != NULL) h = fnv1a_update(h, (const unsigned char *) decision, sizeof(*decision));
    if (config != NULL) h = fnv1a_update(h, (const unsigned char *) config, sizeof(*config));
    h = fnv1a_update(h, (const unsigned char *) &graph_hash, sizeof(graph_hash));
    return h;
}

static void destroy_engine_pending_arrays(struct bolr_replay_engine *engine) {
    if (engine == NULL) return;
    bolr_allocator_free(engine->allocator, engine->pending_context);
    bolr_allocator_free(engine->allocator, engine->pending_top_k);
    bolr_allocator_free(engine->allocator, engine->pending_score_mean);
    bolr_allocator_free(engine->allocator, engine->pending_score_variance);
    bolr_allocator_free(engine->allocator, engine->pending_probability_best);
    bolr_allocator_free(engine->allocator, engine->pending_expected_rank);
    bolr_allocator_free(engine->allocator, engine->pending_rank_stddev);
    bolr_allocator_free(engine->allocator, engine->pending_probability_top_k);
    bolr_allocator_free(engine->allocator, engine->pending_consensus_indices);
    bolr_allocator_free(engine->allocator, engine->pending_region_summaries);
    engine->pending_context = NULL;
    engine->pending_top_k = NULL;
    engine->pending_score_mean = NULL;
    engine->pending_score_variance = NULL;
    engine->pending_probability_best = NULL;
    engine->pending_expected_rank = NULL;
    engine->pending_rank_stddev = NULL;
    engine->pending_probability_top_k = NULL;
    engine->pending_consensus_indices = NULL;
    engine->pending_region_summaries = NULL;
    engine->pending_context_length = 0;
    engine->pending_top_k_count = 0;
    engine->pending_candidate_count = 0;
    engine->pending_rank_top_k = 0;
    engine->pending_region_count = 0;
    engine->pending_consensus_count = 0;
    memset(&engine->pending_rank_diagnostics, 0, sizeof(engine->pending_rank_diagnostics));
}

static void destroy_checkpoint_pending_arrays(struct bolr_replay_checkpoint *checkpoint) {
    if (checkpoint == NULL) return;
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_context);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_top_k);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_score_mean);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_score_variance);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_probability_best);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_expected_rank);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_rank_stddev);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_probability_top_k);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_consensus_indices);
    bolr_allocator_free(checkpoint->allocator, checkpoint->pending_region_summaries);
    checkpoint->pending_context = NULL;
    checkpoint->pending_top_k = NULL;
    checkpoint->pending_score_mean = NULL;
    checkpoint->pending_score_variance = NULL;
    checkpoint->pending_probability_best = NULL;
    checkpoint->pending_expected_rank = NULL;
    checkpoint->pending_rank_stddev = NULL;
    checkpoint->pending_probability_top_k = NULL;
    checkpoint->pending_consensus_indices = NULL;
    checkpoint->pending_region_summaries = NULL;
}

static bolr_status copy_real_vector_owned(const bolr_allocator *allocator, const bolr_real *source, bolr_index count, bolr_real **out) {
    size_t bytes;
    bolr_real *copy;
    if (count <= 0) {
        *out = NULL;
        return BOLR_OK;
    }
    if (source == NULL) return BOLR_INVALID_ARGUMENT;
    if (bolr_checked_size_mul((size_t) count, sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    copy = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (copy == NULL) return BOLR_ALLOCATION_FAILED;
    memcpy(copy, source, bytes);
    *out = copy;
    return BOLR_OK;
}

static bolr_status capture_prediction_pending(
    struct bolr_replay_engine *engine,
    bolr_posterior_prediction *prediction,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    const bolr_monte_carlo_ranking_diagnostics *rank_diag
) {
    struct bolr_posterior_prediction *pred = (struct bolr_posterior_prediction *) prediction;
    bolr_index n = pred->candidate_count;
    bolr_index i;
    bolr_status status;
    destroy_engine_pending_arrays(engine);
    engine->pending_candidate_count = n;
    engine->pending_rank_diagnostics = *rank_diag;
    status = copy_real_vector_owned(engine->allocator, pred->score_mean, n, &engine->pending_score_mean);
    if (status != BOLR_OK) return status;
    status = copy_real_vector_owned(engine->allocator, pred->score_variance, n, &engine->pending_score_variance);
    if (status != BOLR_OK) return status;
    if (pred->probability_best != NULL) {
        status = copy_real_vector_owned(engine->allocator, pred->probability_best, n, &engine->pending_probability_best);
        if (status != BOLR_OK) return status;
    }
    if (pred->expected_rank != NULL) {
        status = copy_real_vector_owned(engine->allocator, pred->expected_rank, n, &engine->pending_expected_rank);
        if (status != BOLR_OK) return status;
    }
    if (pred->rank_stddev != NULL) {
        status = copy_real_vector_owned(engine->allocator, pred->rank_stddev, n, &engine->pending_rank_stddev);
        if (status != BOLR_OK) return status;
    }
    if ((top_k_count > 0) && (top_k_values != NULL) && (pred->probability_top_k_values != NULL) && (pred->probability_top_k_keys != NULL)) {
        engine->pending_rank_top_k = top_k_values[0];
        for (i = 0; i < pred->probability_top_k_count; ++i) {
            if (pred->probability_top_k_keys[i] == engine->pending_rank_top_k) {
                status = copy_real_vector_owned(engine->allocator, pred->probability_top_k_values[i], n, &engine->pending_probability_top_k);
                if (status != BOLR_OK) return status;
                break;
            }
        }
    }
    return BOLR_OK;
}

static bolr_status capture_region_pending(struct bolr_replay_engine *engine, bolr_region_set *regions) {
    struct bolr_region_set *rs = (struct bolr_region_set *) regions;
    bolr_index i;
    bolr_allocator_free(engine->allocator, engine->pending_consensus_indices);
    bolr_allocator_free(engine->allocator, engine->pending_region_summaries);
    engine->pending_consensus_indices = NULL;
    engine->pending_region_summaries = NULL;
    engine->pending_region_count = 0;
    engine->pending_consensus_count = 0;
    if (rs == NULL) return BOLR_OK;
    engine->pending_region_count = rs->region_count;
    engine->pending_consensus_count = rs->consensus_count;
    if (rs->consensus_count > 0) {
        bolr_status status = copy_index_array(engine->allocator, rs->consensus_indices, rs->consensus_count, &engine->pending_consensus_indices);
        if (status != BOLR_OK) return status;
    }
    if (rs->region_count > 0) {
        engine->pending_region_summaries = (bolr_region_summary *) bolr_allocator_malloc(engine->allocator, (size_t) rs->region_count * sizeof(bolr_region_summary));
        if (engine->pending_region_summaries == NULL) return BOLR_ALLOCATION_FAILED;
        for (i = 0; i < rs->region_count; ++i) engine->pending_region_summaries[i] = rs->summaries[i];
    }
    return BOLR_OK;
}

static bolr_status copy_engine_pending_to_checkpoint(const struct bolr_replay_engine *engine, struct bolr_replay_checkpoint *checkpoint) {
    bolr_status status;
    destroy_checkpoint_pending_arrays(checkpoint);
    checkpoint->completed_day_index = engine->completed_day_index;
    checkpoint->pending_context_length = engine->pending_context_length;
    checkpoint->pending_ranking = engine->pending_ranking;
    checkpoint->pending_top_k_count = engine->pending_top_k_count;
    checkpoint->pending_decision_config = engine->pending_decision_config;
    checkpoint->pending_decision_id = engine->pending_decision_id;
    checkpoint->pending_candidate_count = engine->pending_candidate_count;
    checkpoint->pending_rank_top_k = engine->pending_rank_top_k;
    checkpoint->pending_rank_diagnostics = engine->pending_rank_diagnostics;
    checkpoint->pending_region_count = engine->pending_region_count;
    checkpoint->pending_consensus_count = engine->pending_consensus_count;
    checkpoint->graph_hash = engine->graph_hash;
    status = copy_real_vector_owned(checkpoint->allocator, engine->pending_context, engine->pending_context_length, &checkpoint->pending_context);
    if (status != BOLR_OK) return status;
    status = copy_index_array(checkpoint->allocator, engine->pending_top_k, engine->pending_top_k_count, &checkpoint->pending_top_k);
    if (status != BOLR_OK) return status;
    status = copy_real_vector_owned(checkpoint->allocator, engine->pending_score_mean, engine->pending_candidate_count, &checkpoint->pending_score_mean);
    if (status != BOLR_OK) return status;
    status = copy_real_vector_owned(checkpoint->allocator, engine->pending_score_variance, engine->pending_candidate_count, &checkpoint->pending_score_variance);
    if (status != BOLR_OK) return status;
    /* Rank/top-k arrays are optional: capture_prediction_pending only allocates them when present. */
    if (engine->pending_probability_best != NULL) {
        status = copy_real_vector_owned(checkpoint->allocator, engine->pending_probability_best, engine->pending_candidate_count, &checkpoint->pending_probability_best);
        if (status != BOLR_OK) return status;
    }
    if (engine->pending_expected_rank != NULL) {
        status = copy_real_vector_owned(checkpoint->allocator, engine->pending_expected_rank, engine->pending_candidate_count, &checkpoint->pending_expected_rank);
        if (status != BOLR_OK) return status;
    }
    if (engine->pending_rank_stddev != NULL) {
        status = copy_real_vector_owned(checkpoint->allocator, engine->pending_rank_stddev, engine->pending_candidate_count, &checkpoint->pending_rank_stddev);
        if (status != BOLR_OK) return status;
    }
    if (engine->pending_probability_top_k != NULL) {
        status = copy_real_vector_owned(checkpoint->allocator, engine->pending_probability_top_k, engine->pending_candidate_count, &checkpoint->pending_probability_top_k);
        if (status != BOLR_OK) return status;
    }
    status = copy_index_array(checkpoint->allocator, engine->pending_consensus_indices, engine->pending_consensus_count, &checkpoint->pending_consensus_indices);
    if (status != BOLR_OK) return status;
    if (engine->pending_region_count > 0) {
        size_t bytes;
        if (bolr_checked_size_mul((size_t) engine->pending_region_count, sizeof(bolr_region_summary), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
        checkpoint->pending_region_summaries = (bolr_region_summary *) bolr_allocator_malloc(checkpoint->allocator, bytes);
        if (checkpoint->pending_region_summaries == NULL) return BOLR_ALLOCATION_FAILED;
        memcpy(checkpoint->pending_region_summaries, engine->pending_region_summaries, bytes);
    }
    return BOLR_OK;
}

static bolr_status copy_checkpoint_pending_to_engine(const struct bolr_replay_checkpoint *checkpoint, struct bolr_replay_engine *engine) {
    bolr_status status;
    destroy_engine_pending_arrays(engine);
    engine->completed_day_index = checkpoint->completed_day_index;
    engine->pending_context_length = checkpoint->pending_context_length;
    engine->pending_ranking = checkpoint->pending_ranking;
    engine->pending_top_k_count = checkpoint->pending_top_k_count;
    engine->pending_decision_config = checkpoint->pending_decision_config;
    engine->pending_decision_id = checkpoint->pending_decision_id;
    engine->pending_candidate_count = checkpoint->pending_candidate_count;
    engine->pending_rank_top_k = checkpoint->pending_rank_top_k;
    engine->pending_rank_diagnostics = checkpoint->pending_rank_diagnostics;
    engine->pending_region_count = checkpoint->pending_region_count;
    engine->pending_consensus_count = checkpoint->pending_consensus_count;
    engine->graph_hash = checkpoint->graph_hash;
    status = copy_real_vector_owned(engine->allocator, checkpoint->pending_context, checkpoint->pending_context_length, &engine->pending_context);
    if (status != BOLR_OK) return status;
    status = copy_index_array(engine->allocator, checkpoint->pending_top_k, checkpoint->pending_top_k_count, &engine->pending_top_k);
    if (status != BOLR_OK) return status;
    status = copy_real_vector_owned(engine->allocator, checkpoint->pending_score_mean, checkpoint->pending_candidate_count, &engine->pending_score_mean);
    if (status != BOLR_OK) return status;
    status = copy_real_vector_owned(engine->allocator, checkpoint->pending_score_variance, checkpoint->pending_candidate_count, &engine->pending_score_variance);
    if (status != BOLR_OK) return status;
    if (checkpoint->pending_probability_best != NULL) {
        status = copy_real_vector_owned(engine->allocator, checkpoint->pending_probability_best, checkpoint->pending_candidate_count, &engine->pending_probability_best);
        if (status != BOLR_OK) return status;
    }
    if (checkpoint->pending_expected_rank != NULL) {
        status = copy_real_vector_owned(engine->allocator, checkpoint->pending_expected_rank, checkpoint->pending_candidate_count, &engine->pending_expected_rank);
        if (status != BOLR_OK) return status;
    }
    if (checkpoint->pending_rank_stddev != NULL) {
        status = copy_real_vector_owned(engine->allocator, checkpoint->pending_rank_stddev, checkpoint->pending_candidate_count, &engine->pending_rank_stddev);
        if (status != BOLR_OK) return status;
    }
    if (checkpoint->pending_probability_top_k != NULL) {
        status = copy_real_vector_owned(engine->allocator, checkpoint->pending_probability_top_k, checkpoint->pending_candidate_count, &engine->pending_probability_top_k);
        if (status != BOLR_OK) return status;
    }
    status = copy_index_array(engine->allocator, checkpoint->pending_consensus_indices, checkpoint->pending_consensus_count, &engine->pending_consensus_indices);
    if (status != BOLR_OK) return status;
    if (checkpoint->pending_region_count > 0) {
        size_t bytes;
        if (bolr_checked_size_mul((size_t) checkpoint->pending_region_count, sizeof(bolr_region_summary), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
        engine->pending_region_summaries = (bolr_region_summary *) bolr_allocator_malloc(engine->allocator, bytes);
        if (engine->pending_region_summaries == NULL) return BOLR_ALLOCATION_FAILED;
        memcpy(engine->pending_region_summaries, checkpoint->pending_region_summaries, bytes);
    }
    return BOLR_OK;
}

static void destroy_transition_storage(struct bolr_replay_engine *engine) {
    if (engine == NULL) return;
    bolr_allocator_free(engine->allocator, engine->transition_process_noise);
    bolr_allocator_free(engine->allocator, engine->transition_block_discount_scales);
    engine->transition_process_noise = NULL;
    engine->transition_block_discount_scales = NULL;
    engine->transition.process_noise = (bolr_const_matrix_view){NULL, 0, 0, 0, 0};
    engine->transition.block_discount_scales = (bolr_const_vector_view){NULL, 0, 1};
}

static void destroy_checkpoint_transition_storage(struct bolr_replay_checkpoint *checkpoint) {
    if (checkpoint == NULL) return;
    bolr_allocator_free(checkpoint->allocator, checkpoint->transition_process_noise);
    bolr_allocator_free(checkpoint->allocator, checkpoint->transition_block_discount_scales);
    checkpoint->transition_process_noise = NULL;
    checkpoint->transition_block_discount_scales = NULL;
    checkpoint->transition.process_noise = (bolr_const_matrix_view){NULL, 0, 0, 0, 0};
    checkpoint->transition.block_discount_scales = (bolr_const_vector_view){NULL, 0, 1};
}

static bolr_status engine_copy_transition(
    struct bolr_replay_engine *engine,
    const bolr_transition_config *transition,
    bolr_index dimension
) {
    if ((engine == NULL) || (transition == NULL)) return BOLR_INVALID_ARGUMENT;
    destroy_transition_storage(engine);
    engine->transition = *transition;
    if ((transition->process_noise.rows > 0) && (transition->process_noise.cols > 0)) {
        bolr_status status = copy_real_matrix_row_major(engine->allocator, transition->process_noise, &engine->transition_process_noise);
        if (status != BOLR_OK) return status;
        engine->transition.process_noise = (bolr_const_matrix_view){engine->transition_process_noise, dimension, dimension, dimension, 1};
    }
    if (transition->block_discount_scales.length > 0) {
        bolr_status status = copy_real_array(engine->allocator, transition->block_discount_scales, &engine->transition_block_discount_scales);
        if (status != BOLR_OK) return status;
        engine->transition.block_discount_scales = (bolr_const_vector_view){engine->transition_block_discount_scales, transition->block_discount_scales.length, 1};
    }
    return BOLR_OK;
}

static bolr_status checkpoint_copy_transition(
    struct bolr_replay_checkpoint *checkpoint,
    const bolr_transition_config *transition,
    bolr_index dimension
) {
    if ((checkpoint == NULL) || (transition == NULL)) return BOLR_INVALID_ARGUMENT;
    destroy_checkpoint_transition_storage(checkpoint);
    checkpoint->transition = *transition;
    if ((transition->process_noise.rows > 0) && (transition->process_noise.cols > 0)) {
        bolr_status status = copy_real_matrix_row_major(checkpoint->allocator, transition->process_noise, &checkpoint->transition_process_noise);
        if (status != BOLR_OK) return status;
        checkpoint->transition.process_noise = (bolr_const_matrix_view){checkpoint->transition_process_noise, dimension, dimension, dimension, 1};
    }
    if (transition->block_discount_scales.length > 0) {
        bolr_status status = copy_real_array(checkpoint->allocator, transition->block_discount_scales, &checkpoint->transition_block_discount_scales);
        if (status != BOLR_OK) return status;
        checkpoint->transition.block_discount_scales = (bolr_const_vector_view){checkpoint->transition_block_discount_scales, transition->block_discount_scales.length, 1};
    }
    return BOLR_OK;
}

static bolr_status clone_rng(const bolr_rng *source, const bolr_allocator *allocator, bolr_rng **out_rng) {
    bolr_rng_checkpoint *checkpoint = NULL;
    bolr_status status;
    *out_rng = NULL;
    status = bolr_rng_export(source, allocator, &checkpoint);
    if (status != BOLR_OK) return status;
    status = bolr_rng_import(checkpoint, allocator, out_rng);
    bolr_rng_checkpoint_destroy(checkpoint);
    return status;
}

static bolr_status clone_state(const bolr_gaussian_state *source, const bolr_allocator *allocator, bolr_gaussian_state **out_state) {
    return bolr_gaussian_state_clone(source, allocator, out_state);
}

static bolr_status clone_adaptive_state(
    const bolr_adaptive_policy *policy,
    const bolr_adaptive_state *source,
    const bolr_allocator *allocator,
    bolr_adaptive_state **out_state
) {
    void *buffer = NULL;
    size_t size = 0U;
    size_t written = 0U;
    bolr_status status;
    *out_state = NULL;
    status = bolr_adaptive_state_encoded_size(policy, source, &size);
    if (status != BOLR_OK) return status;
    buffer = bolr_allocator_malloc(allocator, size);
    if (buffer == NULL) return BOLR_ALLOCATION_FAILED;
    status = bolr_adaptive_state_encode(policy, source, buffer, size, &written);
    if (status != BOLR_OK) {
        bolr_allocator_free(allocator, buffer);
        return status;
    }
    status = bolr_adaptive_state_decode(policy, buffer, written, allocator, out_state);
    bolr_allocator_free(allocator, buffer);
    return status;
}

static void zero_decision(bolr_decision *decision) {
    if (decision != NULL) memset(decision, 0, sizeof(*decision));
}

static bolr_status build_predictive(
    struct bolr_replay_engine *engine,
    bolr_workspace *workspace,
    bolr_gaussian_state **out_predictive,
    bolr_adaptation_diagnostics *adaptive_diagnostics
) {
    if (engine->adaptive_enabled) {
        return bolr_adaptive_policy_predict(engine->adaptive_policy, engine->adaptive_state, engine->posterior, workspace, out_predictive, adaptive_diagnostics);
    }
    return bolr_gaussian_predict(engine->posterior, &engine->transition, workspace, out_predictive, NULL);
}

bolr_status bolr_replay_engine_create_fixed(
    const bolr_gaussian_state *posterior,
    const bolr_transition_config *transition,
    const bolr_rng *rng,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
) {
    struct bolr_replay_engine *engine;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_status status;
    if ((posterior == NULL) || (transition == NULL) || (rng == NULL) || (out_engine == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_engine = NULL;
    engine = (struct bolr_replay_engine *) bolr_allocator_calloc(active, 1U, sizeof(*engine));
    if (engine == NULL) return BOLR_ALLOCATION_FAILED;
    engine->allocator = active;
    engine->phase = BOLR_REPLAY_PHASE_READY;
    zero_decision(&engine->pending_decision);
    status = clone_state(posterior, active, &engine->posterior);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    status = clone_rng(rng, active, &engine->rng);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    status = engine_copy_transition(engine, transition, bolr_gaussian_state_dimension(posterior));
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    *out_engine = engine;
    return BOLR_OK;
}

bolr_status bolr_replay_engine_create_adaptive(
    const bolr_gaussian_state *posterior,
    const bolr_adaptive_policy *policy,
    const bolr_adaptive_state *adaptive_state,
    const bolr_rng *rng,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
) {
    struct bolr_replay_engine *engine;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_status status;
    if ((posterior == NULL) || (policy == NULL) || (adaptive_state == NULL) || (rng == NULL) || (out_engine == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_engine = NULL;
    engine = (struct bolr_replay_engine *) bolr_allocator_calloc(active, 1U, sizeof(*engine));
    if (engine == NULL) return BOLR_ALLOCATION_FAILED;
    engine->allocator = active;
    engine->adaptive_enabled = 1;
    engine->adaptive_policy = policy;
    engine->phase = BOLR_REPLAY_PHASE_READY;
    zero_decision(&engine->pending_decision);
    status = clone_state(posterior, active, &engine->posterior);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    status = clone_rng(rng, active, &engine->rng);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    status = clone_adaptive_state(policy, adaptive_state, active, &engine->adaptive_state);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    *out_engine = engine;
    return BOLR_OK;
}

void bolr_replay_engine_destroy(bolr_replay_engine *opaque) {
    struct bolr_replay_engine *engine = opaque;
    if (engine == NULL) return;
    destroy_transition_storage(engine);
    destroy_engine_pending_arrays(engine);
    bolr_adaptive_state_destroy(engine->adaptive_state);
    bolr_gaussian_state_destroy(engine->posterior);
    bolr_gaussian_state_destroy(engine->pending_predictive);
    bolr_rng_destroy(engine->rng);
    bolr_allocator_free(engine->allocator, engine);
}

bolr_replay_phase bolr_replay_engine_phase(const bolr_replay_engine *opaque) {
    const struct bolr_replay_engine *engine = opaque;
    return (engine == NULL) ? 0 : engine->phase;
}

bolr_status bolr_replay_engine_copy_posterior_mean(const bolr_replay_engine *opaque, bolr_vector_view output) {
    const struct bolr_replay_engine *engine = opaque;
    return (engine == NULL) ? BOLR_INVALID_ARGUMENT : bolr_gaussian_state_copy_mean(engine->posterior, output);
}

bolr_status bolr_replay_engine_copy_posterior_covariance(const bolr_replay_engine *opaque, bolr_matrix_view output) {
    const struct bolr_replay_engine *engine = opaque;
    return (engine == NULL) ? BOLR_INVALID_ARGUMENT : bolr_gaussian_state_copy_covariance(engine->posterior, output);
}

bolr_index bolr_replay_engine_pending_selected_index(const bolr_replay_engine *opaque) {
    const struct bolr_replay_engine *engine = opaque;
    if ((engine == NULL) || (engine->phase != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) || (!engine->pending_decision.selected)) return -1;
    return engine->pending_decision.selected_index;
}

bolr_status bolr_replay_engine_begin_day(
    bolr_replay_engine *opaque,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_replay_ranking_config *ranking,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    const bolr_decision_policy *decision_policy,
    const bolr_grid_graph *graph,
    const bolr_region_config *region_config,
    bolr_workspace *workspace,
    bolr_decision *out_decision,
    bolr_replay_begin_diagnostics *out_diagnostics
) {
    struct bolr_replay_engine *engine = opaque;
    bolr_gaussian_state *predictive = NULL;
    bolr_gaussian_state *pending_predictive_old;
    bolr_posterior_prediction *prediction = NULL;
    bolr_region_set *regions = NULL;
    bolr_monte_carlo_ranking_diagnostics rank_diag;
    bolr_adaptive_state *adaptive_clone = NULL;
    bolr_rng *rng_clone = NULL;
    bolr_rng *old_rng;
    bolr_adaptation_diagnostics adaptive_diag;
    struct bolr_replay_engine scratch;
    bolr_decision decision;
    bolr_decision_diagnostics decision_diag;
    bolr_status status;
    if ((engine == NULL) || (model == NULL) || (ranking == NULL) || (decision_policy == NULL) || (out_decision == NULL)) return BOLR_INVALID_ARGUMENT;
    if (engine->phase != BOLR_REPLAY_PHASE_READY) return BOLR_INVALID_ARGUMENT;
    memset(&decision, 0, sizeof(decision));
    memset(&decision_diag, 0, sizeof(decision_diag));
    memset(&rank_diag, 0, sizeof(rank_diag));
    memset(&adaptive_diag, 0, sizeof(adaptive_diag));
    scratch = *engine;
    status = clone_rng(engine->rng, engine->allocator, &rng_clone);
    if (status != BOLR_OK) return status;
    scratch.rng = rng_clone;
    if (engine->adaptive_enabled) {
        status = clone_adaptive_state(engine->adaptive_policy, engine->adaptive_state, engine->allocator, &adaptive_clone);
        if (status != BOLR_OK) { bolr_rng_destroy(rng_clone); return status; }
        scratch.adaptive_state = adaptive_clone;
    }
    status = build_predictive(&scratch, workspace, &predictive, &adaptive_diag);
    if (status != BOLR_OK) {
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    status = bolr_posterior_prediction_create(predictive, model, context, workspace, engine->allocator, &prediction, NULL);
    if (status != BOLR_OK) {
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    status = bolr_posterior_prediction_monte_carlo_rank_streaming(
        prediction,
        model,
        context,
        rng_clone,
        ranking->sample_count,
        (ranking->chunk_size > 0) ? ranking->chunk_size : ranking->sample_count,
        ranking->antithetic,
        top_k_values,
        top_k_count,
        ranking->retention,
        workspace,
        &rank_diag
    );
    if (status != BOLR_OK) {
        bolr_posterior_prediction_destroy(prediction);
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        return status;
    }
    if (region_config != NULL) {
        if (graph == NULL) {
            bolr_posterior_prediction_destroy(prediction);
            bolr_gaussian_state_destroy(predictive);
            bolr_adaptive_state_destroy(adaptive_clone);
            bolr_rng_destroy(rng_clone);
            return BOLR_INVALID_ARGUMENT;
        }
        status = bolr_region_set_build(prediction, graph, region_config, engine->allocator, &regions);
        if (status != BOLR_OK) {
            bolr_posterior_prediction_destroy(prediction);
            bolr_gaussian_state_destroy(predictive);
            bolr_adaptive_state_destroy(adaptive_clone);
            bolr_rng_destroy(rng_clone);
            return status;
        }
    }
    status = bolr_decision_policy_apply(decision_policy, prediction, regions, graph, &decision, &decision_diag);
    if (status != BOLR_OK) {
        bolr_region_set_destroy(regions);
        bolr_posterior_prediction_destroy(prediction);
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    status = capture_prediction_pending(engine, prediction, top_k_values, top_k_count, &rank_diag);
    if (status != BOLR_OK) {
        bolr_region_set_destroy(regions);
        bolr_posterior_prediction_destroy(prediction);
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    status = capture_region_pending(engine, regions);
    bolr_region_set_destroy(regions);
    bolr_posterior_prediction_destroy(prediction);
    if (status != BOLR_OK) {
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    engine->pending_ranking = *ranking;
    status = copy_index_array(engine->allocator, top_k_values, top_k_count, &engine->pending_top_k);
    if (status != BOLR_OK) {
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    engine->pending_top_k_count = top_k_count;
    engine->pending_decision_config = ((const struct bolr_decision_policy *) decision_policy)->config;
    engine->graph_hash = (graph != NULL) ? bolr_grid_graph_hash(graph) : 0ULL;
    status = copy_real_array(engine->allocator, context, &engine->pending_context);
    if (status != BOLR_OK) {
        bolr_gaussian_state_destroy(predictive);
        bolr_adaptive_state_destroy(adaptive_clone);
        bolr_rng_destroy(rng_clone);
        return status;
    }
    engine->pending_context_length = context.length;
    pending_predictive_old = engine->pending_predictive;
    old_rng = engine->rng;
    engine->pending_predictive = predictive;
    engine->rng = rng_clone;
    engine->phase = BOLR_REPLAY_PHASE_AWAITING_OUTCOME;
    engine->pending_decision = decision;
    engine->pending_decision_id = hash_pending_decision_id(&decision, &engine->pending_decision_config, engine->graph_hash);
    if (engine->adaptive_enabled) {
        bolr_adaptive_state_destroy(engine->adaptive_state);
        engine->adaptive_state = adaptive_clone;
    }
    bolr_gaussian_state_destroy(pending_predictive_old);
    bolr_rng_destroy(old_rng);
    *out_decision = decision;
    if (out_diagnostics != NULL) {
        memset(out_diagnostics, 0, sizeof(*out_diagnostics));
        out_diagnostics->phase = engine->phase;
        out_diagnostics->selected_index = decision.selected_index;
        out_diagnostics->selected_score_mean = decision.selected_score_mean;
        out_diagnostics->selected_score_variance = decision.selected_score_variance;
        out_diagnostics->selected_probability_best = decision.selected_probability_best;
        out_diagnostics->selected_expected_rank = decision.selected_expected_rank;
        out_diagnostics->retained_score_sample_count = rank_diag.retained_score_sample_count;
        out_diagnostics->region_count = (region_config != NULL) ? 1 : 0;
    }
    return BOLR_OK;
}

bolr_status bolr_replay_engine_finish_day(
    bolr_replay_engine *opaque,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_observation_operator *observation,
    const bolr_newton_config *newton_config,
    bolr_real effective_strength,
    bolr_real information_size,
    int informative,
    bolr_inference_workspace *workspace,
    bolr_laplace_diagnostics *out_laplace,
    bolr_adaptation_diagnostics *out_adaptation,
    bolr_replay_finish_diagnostics *out_diagnostics
) {
    struct bolr_replay_engine *engine = opaque;
    bolr_newton_config default_config;
    bolr_gaussian_state *posterior_new = NULL;
    bolr_gaussian_state *old_posterior;
    bolr_gaussian_state *old_pending;
    bolr_real gaussian_kl = 0.0;
    bolr_adaptive_state *adaptive_clone = NULL;
    bolr_status status;
    bolr_laplace_diagnostics laplace;
    bolr_adaptation_diagnostics adaptation;
    if ((engine == NULL) || (model == NULL) || (observation == NULL) || (workspace == NULL)) return BOLR_INVALID_ARGUMENT;
    if (engine->phase != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) return BOLR_INVALID_ARGUMENT;
    memset(&laplace, 0, sizeof(laplace));
    memset(&adaptation, 0, sizeof(adaptation));
    if (newton_config == NULL) {
        memset(&default_config, 0, sizeof(default_config));
        default_config.maximum_iterations = 30;
        default_config.gradient_tolerance = 1e-8;
        default_config.step_tolerance = 1e-10;
        default_config.objective_tolerance = 1e-12;
        default_config.initial_damping = 1e-6;
        default_config.damping_multiplier = 10.0;
        default_config.maximum_damping = 1e12;
        default_config.armijo_constant = 1e-4;
        default_config.line_search_reduction = 0.5;
        default_config.maximum_line_search_steps = 20;
        default_config.cholesky_initial_jitter = 1e-9;
        default_config.cholesky_jitter_multiplier = 10.0;
        default_config.maximum_cholesky_attempts = 6;
        newton_config = &default_config;
    }
    status = bolr_laplace_update(engine->pending_predictive, model, context, observation, newton_config, workspace, &posterior_new, &laplace);
    if (status != BOLR_OK) return status;
    if (engine->adaptive_enabled) {
        status = clone_adaptive_state(engine->adaptive_policy, engine->adaptive_state, engine->allocator, &adaptive_clone);
        if (status != BOLR_OK) {
            bolr_gaussian_state_destroy(posterior_new);
            return status;
        }
        status = bolr_gaussian_kl(posterior_new, engine->pending_predictive, &gaussian_kl);
        if (status != BOLR_OK) {
            bolr_gaussian_state_destroy(posterior_new);
            bolr_adaptive_state_destroy(adaptive_clone);
            return status;
        }
        status = bolr_adaptive_policy_observe(
            engine->adaptive_policy,
            adaptive_clone,
            engine->pending_predictive,
            posterior_new,
            &(bolr_surprise_input){
                informative,
                laplace.log_factor_at_predictive_mean,
                laplace.log_factor_at_posterior_mode,
                effective_strength,
                information_size,
                laplace.mahalanobis_update_norm,
                gaussian_kl,
                laplace.objective_improvement
            },
            &adaptation
        );
        if (status != BOLR_OK) {
            bolr_gaussian_state_destroy(posterior_new);
            bolr_adaptive_state_destroy(adaptive_clone);
            return status;
        }
    }
    old_posterior = engine->posterior;
    old_pending = engine->pending_predictive;
    engine->posterior = posterior_new;
    engine->pending_predictive = NULL;
    engine->phase = BOLR_REPLAY_PHASE_READY;
    if (engine->adaptive_enabled) {
        bolr_adaptive_state_destroy(engine->adaptive_state);
        engine->adaptive_state = adaptive_clone;
    }
    bolr_gaussian_state_destroy(old_posterior);
    bolr_gaussian_state_destroy(old_pending);
    if (out_laplace != NULL) *out_laplace = laplace;
    if ((out_adaptation != NULL) && engine->adaptive_enabled) *out_adaptation = adaptation;
    if (out_diagnostics != NULL) {
        memset(out_diagnostics, 0, sizeof(*out_diagnostics));
        out_diagnostics->phase_after = engine->phase;
        out_diagnostics->selected_index = engine->pending_decision.selected_index;
        out_diagnostics->objective_improvement = laplace.objective_improvement;
        out_diagnostics->posterior_trace = laplace.posterior_covariance_trace;
        out_diagnostics->adaptive_applied = engine->adaptive_enabled ? 1 : 0;
    }
    zero_decision(&engine->pending_decision);
    destroy_engine_pending_arrays(engine);
    engine->completed_day_index += 1;
    return BOLR_OK;
}

bolr_status bolr_replay_engine_export_checkpoint(
    const bolr_replay_engine *opaque,
    const bolr_allocator *allocator,
    bolr_replay_checkpoint **out_checkpoint
) {
    const struct bolr_replay_engine *engine = opaque;
    struct bolr_replay_checkpoint *checkpoint;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_status status;
    if ((engine == NULL) || (out_checkpoint == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_checkpoint = NULL;
    checkpoint = (struct bolr_replay_checkpoint *) bolr_allocator_calloc(active, 1U, sizeof(*checkpoint));
    if (checkpoint == NULL) return BOLR_ALLOCATION_FAILED;
    checkpoint->allocator = active;
    checkpoint->adaptive_enabled = engine->adaptive_enabled;
    checkpoint->phase = engine->phase;
    checkpoint->pending_decision = engine->pending_decision;
    status = bolr_gaussian_state_export(engine->posterior, active, &checkpoint->posterior_checkpoint);
    if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
    if (engine->pending_predictive != NULL) {
        status = bolr_gaussian_state_export(engine->pending_predictive, active, &checkpoint->pending_predictive_checkpoint);
        if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
    }
    status = bolr_rng_export(engine->rng, active, &checkpoint->rng_checkpoint);
    if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
    if (engine->adaptive_enabled) {
        checkpoint->adaptive_policy_hash = bolr_adaptive_policy_configuration_hash(engine->adaptive_policy);
        status = bolr_adaptive_state_encoded_size(engine->adaptive_policy, engine->adaptive_state, &checkpoint->adaptive_state_size);
        if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
        checkpoint->adaptive_state_bytes = bolr_allocator_malloc(active, checkpoint->adaptive_state_size);
        if (checkpoint->adaptive_state_bytes == NULL) { bolr_replay_checkpoint_destroy(checkpoint); return BOLR_ALLOCATION_FAILED; }
        status = bolr_adaptive_state_encode(engine->adaptive_policy, engine->adaptive_state, checkpoint->adaptive_state_bytes, checkpoint->adaptive_state_size, &checkpoint->adaptive_state_size);
        if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
    } else {
        status = checkpoint_copy_transition(checkpoint, &engine->transition, bolr_gaussian_state_dimension(engine->posterior));
        if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
    }
    status = copy_engine_pending_to_checkpoint(engine, checkpoint);
    if (status != BOLR_OK) { bolr_replay_checkpoint_destroy(checkpoint); return status; }
    *out_checkpoint = checkpoint;
    return BOLR_OK;
}

static bolr_status import_common_checkpoint(
    const struct bolr_replay_checkpoint *checkpoint,
    const bolr_allocator *allocator,
    struct bolr_replay_engine **out_engine
) {
    struct bolr_replay_engine *engine;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_status status;
    *out_engine = NULL;
    engine = (struct bolr_replay_engine *) bolr_allocator_calloc(active, 1U, sizeof(*engine));
    if (engine == NULL) return BOLR_ALLOCATION_FAILED;
    engine->allocator = active;
    engine->phase = checkpoint->phase;
    engine->pending_decision = checkpoint->pending_decision;
    status = bolr_gaussian_state_import(checkpoint->posterior_checkpoint, active, &engine->posterior);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    if (checkpoint->pending_predictive_checkpoint != NULL) {
        status = bolr_gaussian_state_import(checkpoint->pending_predictive_checkpoint, active, &engine->pending_predictive);
        if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    }
    status = bolr_rng_import(checkpoint->rng_checkpoint, active, &engine->rng);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    status = copy_checkpoint_pending_to_engine(checkpoint, engine);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    *out_engine = engine;
    return BOLR_OK;
}

bolr_status bolr_replay_engine_import_fixed(
    const bolr_replay_checkpoint *opaque,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
) {
    const struct bolr_replay_checkpoint *checkpoint = opaque;
    struct bolr_replay_engine *engine;
    bolr_status status;
    if ((checkpoint == NULL) || (out_engine == NULL) || checkpoint->adaptive_enabled) return BOLR_INVALID_ARGUMENT;
    status = import_common_checkpoint(checkpoint, allocator, &engine);
    if (status != BOLR_OK) return status;
    status = engine_copy_transition(engine, &checkpoint->transition, bolr_gaussian_state_dimension(engine->posterior));
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    *out_engine = engine;
    return BOLR_OK;
}

bolr_status bolr_replay_engine_import_adaptive(
    const bolr_replay_checkpoint *opaque,
    const bolr_adaptive_policy *policy,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
) {
    const struct bolr_replay_checkpoint *checkpoint = opaque;
    struct bolr_replay_engine *engine;
    bolr_status status;
    if ((checkpoint == NULL) || (policy == NULL) || (out_engine == NULL) || (!checkpoint->adaptive_enabled)) return BOLR_INVALID_ARGUMENT;
    if (checkpoint->adaptive_policy_hash != bolr_adaptive_policy_configuration_hash(policy)) return BOLR_SCHEMA_MISMATCH;
    status = import_common_checkpoint(checkpoint, allocator, &engine);
    if (status != BOLR_OK) return status;
    engine->adaptive_enabled = 1;
    engine->adaptive_policy = policy;
    status = bolr_adaptive_state_decode(policy, checkpoint->adaptive_state_bytes, checkpoint->adaptive_state_size, engine->allocator, &engine->adaptive_state);
    if (status != BOLR_OK) { bolr_replay_engine_destroy(engine); return status; }
    *out_engine = engine;
    return BOLR_OK;
}

void bolr_replay_checkpoint_destroy(bolr_replay_checkpoint *opaque) {
    struct bolr_replay_checkpoint *checkpoint = opaque;
    if (checkpoint == NULL) return;
    destroy_checkpoint_transition_storage(checkpoint);
    destroy_checkpoint_pending_arrays(checkpoint);
    bolr_checkpoint_state_destroy(checkpoint->posterior_checkpoint);
    bolr_checkpoint_state_destroy(checkpoint->pending_predictive_checkpoint);
    bolr_rng_checkpoint_destroy(checkpoint->rng_checkpoint);
    bolr_allocator_free(checkpoint->allocator, checkpoint->adaptive_state_bytes);
    bolr_allocator_free(checkpoint->allocator, checkpoint);
}

void bolr_replay_checkpoint_destroy_pending(bolr_replay_checkpoint *opaque) {
    struct bolr_replay_checkpoint *checkpoint = opaque;
    if (checkpoint == NULL) return;
    destroy_checkpoint_transition_storage(checkpoint);
    destroy_checkpoint_pending_arrays(checkpoint);
    bolr_checkpoint_state_destroy(checkpoint->posterior_checkpoint);
    bolr_checkpoint_state_destroy(checkpoint->pending_predictive_checkpoint);
    bolr_rng_checkpoint_destroy(checkpoint->rng_checkpoint);
    bolr_allocator_free(checkpoint->allocator, checkpoint->adaptive_state_bytes);
    checkpoint->posterior_checkpoint = NULL;
    checkpoint->pending_predictive_checkpoint = NULL;
    checkpoint->rng_checkpoint = NULL;
    checkpoint->adaptive_state_bytes = NULL;
    checkpoint->adaptive_state_size = 0U;
}

bolr_replay_phase bolr_replay_checkpoint_phase(const bolr_replay_checkpoint *opaque) {
    const struct bolr_replay_checkpoint *checkpoint = opaque;
    return (checkpoint == NULL) ? 0 : checkpoint->phase;
}

bolr_index bolr_replay_checkpoint_pending_selected_index(const bolr_replay_checkpoint *opaque) {
    const struct bolr_replay_checkpoint *checkpoint = opaque;
    if ((checkpoint == NULL) || (checkpoint->phase != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) || (!checkpoint->pending_decision.selected)) return -1;
    return checkpoint->pending_decision.selected_index;
}
