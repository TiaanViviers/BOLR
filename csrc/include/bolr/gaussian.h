#ifndef BOLR_GAUSSIAN_H
#define BOLR_GAUSSIAN_H

#include <stdint.h>

#include "bolr/allocator.h"
#include "bolr/array.h"
#include "bolr/checkpoint.h"
#include "bolr/workspace.h"

typedef struct bolr_gaussian_state bolr_gaussian_state;

typedef enum {
    BOLR_TRANSITION_ADDITIVE_Q = 1,
    BOLR_TRANSITION_GLOBAL_DISCOUNT = 2,
    BOLR_TRANSITION_BLOCK_DISCOUNT = 3
} bolr_transition_family;

typedef struct {
    bolr_transition_family family;
    bolr_const_matrix_view process_noise;
    bolr_real global_discount;
    bolr_const_vector_view block_discount_scales;
} bolr_transition_config;

typedef struct {
    bolr_real process_noise_trace;
    bolr_real predictive_covariance_trace;
    bolr_real minimum_cholesky_diagonal;
    bolr_real jitter_used;
} bolr_prediction_diagnostics;

bolr_status bolr_gaussian_state_create(
    bolr_const_vector_view mean,
    bolr_const_matrix_view covariance,
    uint64_t state_layout_hash,
    uint64_t model_schema_hash,
    const bolr_allocator *allocator,
    bolr_gaussian_state **out_state
);
void bolr_gaussian_state_destroy(bolr_gaussian_state *state);
bolr_status bolr_gaussian_state_clone(const bolr_gaussian_state *source, const bolr_allocator *allocator, bolr_gaussian_state **out_clone);
bolr_status bolr_gaussian_state_set(bolr_gaussian_state *state, bolr_const_vector_view mean, bolr_const_matrix_view covariance, uint64_t step_index);
bolr_index bolr_gaussian_state_dimension(const bolr_gaussian_state *state);
uint64_t bolr_gaussian_state_step_index(const bolr_gaussian_state *state);
uint64_t bolr_gaussian_state_state_layout_hash(const bolr_gaussian_state *state);
uint64_t bolr_gaussian_state_model_schema_hash(const bolr_gaussian_state *state);
uint32_t bolr_gaussian_state_schema_version(const bolr_gaussian_state *state);
bolr_status bolr_gaussian_state_copy_mean(const bolr_gaussian_state *state, bolr_vector_view output);
bolr_status bolr_gaussian_state_copy_covariance(const bolr_gaussian_state *state, bolr_matrix_view output);
bolr_status bolr_gaussian_predict(
    const bolr_gaussian_state *posterior,
    const bolr_transition_config *transition,
    bolr_workspace *workspace,
    bolr_gaussian_state **out_predictive,
    bolr_prediction_diagnostics *diagnostics
);
bolr_status bolr_gaussian_kl(const bolr_gaussian_state *posterior, const bolr_gaussian_state *predictive, bolr_real *out_kl);
bolr_status bolr_gaussian_state_export(const bolr_gaussian_state *state, const bolr_allocator *allocator, bolr_checkpoint_state **out_checkpoint);
bolr_status bolr_gaussian_state_import(const bolr_checkpoint_state *checkpoint, const bolr_allocator *allocator, bolr_gaussian_state **out_state);

#endif
