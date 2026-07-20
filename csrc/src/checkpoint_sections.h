#ifndef BOLR_CHECKPOINT_SECTIONS_IMPL_H
#define BOLR_CHECKPOINT_SECTIONS_IMPL_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/adaptation.h"
#include "bolr/allocator.h"
#include "bolr/decision.h"
#include "bolr/gaussian.h"
#include "bolr/replay.h"
#include "bolr/rng.h"
#include "bolr/region.h"
#include "internal.h"

uint64_t bolr_checkpoint_hash_decision_config(const bolr_decision_policy_config *config);
uint64_t bolr_checkpoint_hash_monte_carlo_config(const bolr_replay_ranking_config *ranking, const bolr_index *top_k, bolr_index top_k_count);
uint64_t bolr_checkpoint_hash_transition_config(const bolr_transition_config *transition, bolr_index dimension);
uint64_t bolr_checkpoint_hash_pending_decision_id(const bolr_decision *decision, const bolr_decision_policy_config *config, uint64_t graph_hash);

bolr_status bolr_checkpoint_section_size_gaussian(const struct bolr_gaussian_state *state, size_t *out_size);
bolr_status bolr_checkpoint_section_encode_gaussian(void *buf, size_t cap, size_t *cursor, const struct bolr_gaussian_state *state);
bolr_status bolr_checkpoint_section_decode_gaussian(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, struct bolr_gaussian_state **out_state);

bolr_status bolr_checkpoint_section_size_transition(const bolr_transition_config *transition, bolr_index dimension, size_t *out_size);
bolr_status bolr_checkpoint_section_encode_transition(void *buf, size_t cap, size_t *cursor, const bolr_transition_config *transition, bolr_index dimension);
bolr_status bolr_checkpoint_section_decode_transition(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_transition_config *out_transition, bolr_real **out_process_noise, bolr_real **out_block_scales, bolr_index *out_dimension);

bolr_status bolr_checkpoint_section_size_rng(const struct bolr_rng *rng, size_t *out_size);
bolr_status bolr_checkpoint_section_encode_rng(void *buf, size_t cap, size_t *cursor, const struct bolr_rng *rng);
bolr_status bolr_checkpoint_section_decode_rng(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_rng **out_rng);

bolr_status bolr_checkpoint_section_size_adaptive(const bolr_adaptive_policy *policy, const bolr_adaptive_state *state, size_t *out_size);
bolr_status bolr_checkpoint_section_encode_adaptive(void *buf, size_t cap, size_t *cursor, const bolr_adaptive_policy *policy, const bolr_adaptive_state *state);
bolr_status bolr_checkpoint_section_decode_adaptive(const void *buf, size_t cap, size_t *cursor, const bolr_adaptive_policy *policy, const bolr_allocator *allocator, bolr_adaptive_state **out_state, void **out_bytes, size_t *out_size);

bolr_status bolr_checkpoint_section_encode_decision_config(void *buf, size_t cap, size_t *cursor, const bolr_decision_policy_config *config);
bolr_status bolr_checkpoint_section_decode_decision_config(const void *buf, size_t cap, size_t *cursor, bolr_decision_policy_config *out_config);

bolr_status bolr_checkpoint_section_encode_monte_carlo(void *buf, size_t cap, size_t *cursor, const bolr_replay_ranking_config *ranking, const bolr_index *top_k, bolr_index top_k_count);
bolr_status bolr_checkpoint_section_decode_monte_carlo(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_replay_ranking_config *out_ranking, bolr_index **out_top_k, bolr_index *out_top_k_count);

bolr_status bolr_checkpoint_section_encode_pending_decision(void *buf, size_t cap, size_t *cursor, const bolr_decision *decision);
bolr_status bolr_checkpoint_section_decode_pending_decision(const void *buf, size_t cap, size_t *cursor, bolr_decision *out_decision);

bolr_status bolr_checkpoint_section_encode_provenance(void *buf, size_t cap, size_t *cursor);
bolr_status bolr_checkpoint_section_decode_provenance(const void *buf, size_t cap, size_t *cursor);

bolr_status bolr_checkpoint_section_encode_real_array(void *buf, size_t cap, size_t *cursor, const bolr_real *values, bolr_index count);
bolr_status bolr_checkpoint_section_decode_real_array(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_real **out_values, bolr_index *out_count);

bolr_status bolr_checkpoint_section_encode_index_array(void *buf, size_t cap, size_t *cursor, const bolr_index *values, bolr_index count);
bolr_status bolr_checkpoint_section_decode_index_array(const void *buf, size_t cap, size_t *cursor, const bolr_allocator *allocator, bolr_index **out_values, bolr_index *out_count);

#endif
