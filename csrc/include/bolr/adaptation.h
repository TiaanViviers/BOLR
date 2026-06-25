#ifndef BOLR_ADAPTATION_H
#define BOLR_ADAPTATION_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/allocator.h"
#include "bolr/bocpd.h"
#include "bolr/gaussian.h"
#include "bolr/standardizer.h"
#include "bolr/state_layout.h"
#include "bolr/status.h"
#include "bolr/surprise.h"

typedef struct bolr_adaptive_policy bolr_adaptive_policy;
typedef struct bolr_adaptive_state bolr_adaptive_state;

typedef enum {
    BOLR_ADAPTIVE_BLOCK_FIXED = 0,
    BOLR_ADAPTIVE_BLOCK_ADDITIVE = 1,
    BOLR_ADAPTIVE_BLOCK_DISCOUNT = 2,
    BOLR_ADAPTIVE_BLOCK_ZERO_NOISE = 3,
    BOLR_ADAPTIVE_BLOCK_FROZEN = 4
} bolr_adaptive_block_family;

typedef struct {
    const char *block_name;
    int transition_family;
    bolr_real maximum_multiplier;
    bolr_real minimum_multiplier;
    bolr_real decay;
    bolr_real attribution_floor;
    bolr_real minimum_discount;
    int minimum_discount_present;
    int reset_enabled;
    bolr_real reset_threshold;
    int reset_threshold_present;
    bolr_real reset_strength;
    int reset_strength_present;
    int reset_cooldown;
    bolr_real amplitude;
    int adaptive_enabled;
} bolr_adaptive_block_config;

typedef struct {
    bolr_surprise_mode surprise_mode;
    bolr_standardizer_config standardizer;
    bolr_bocpd_config detector;
    bolr_real activation_beta;
    bolr_real activation_z0;
    bolr_real attribution_epsilon;
} bolr_adaptive_policy_config;

typedef struct {
    bolr_real raw_surprise;
    bolr_real normalized_surprise;
    bolr_real information_normalized_surprise;
    bolr_real standardised_surprise;
    bolr_real standardizer_mean_before;
    bolr_real standardizer_scale_before;
    bolr_real change_probability;
    bolr_real map_run_length;
    bolr_real expected_run_length;
    bolr_real run_length_entropy;
    bolr_real predictive_log_density;
    bolr_real truncation_mass;
    bolr_real activation_value;
    int informative;
    int predictive_log_density_present;
    bolr_index block_count;
    bolr_real *euclidean_update_energy;
    bolr_real *mahalanobis_update_energy;
    bolr_real *attribution_weight;
    bolr_real *process_noise_multiplier;
    bolr_real *target_multiplier;
    bolr_real *active_discount;
    bolr_real *reset_strength;
    int32_t *reset_scheduled;
    int32_t *reset_applied;
    int64_t *days_since_reset;
} bolr_adaptation_diagnostics;

bolr_status bolr_adaptive_policy_create(
    const bolr_state_layout *layout,
    bolr_const_matrix_view base_process_noise,
    const bolr_adaptive_policy_config *config,
    const bolr_adaptive_block_config *block_configs,
    bolr_index block_config_count,
    const bolr_allocator *allocator,
    bolr_adaptive_policy **out_policy
);
void bolr_adaptive_policy_destroy(bolr_adaptive_policy *policy);
uint64_t bolr_adaptive_policy_configuration_hash(const bolr_adaptive_policy *policy);
bolr_index bolr_adaptive_policy_block_count(const bolr_adaptive_policy *policy);

bolr_status bolr_adaptive_state_create(const bolr_adaptive_policy *policy, const bolr_allocator *allocator, bolr_adaptive_state **out_state);
void bolr_adaptive_state_destroy(bolr_adaptive_state *state);
uint64_t bolr_adaptive_state_step_index(const bolr_adaptive_state *state);
bolr_status bolr_adaptive_state_copy_block_multipliers(const bolr_adaptive_state *state, bolr_vector_view output);
bolr_status bolr_adaptive_state_copy_block_discounts(const bolr_adaptive_state *state, bolr_vector_view output);
bolr_status bolr_adaptive_state_copy_run_length_posterior(const bolr_adaptive_state *state, bolr_vector_view output);

bolr_status bolr_adaptive_policy_predict(
    const bolr_adaptive_policy *policy,
    bolr_adaptive_state *state,
    const bolr_gaussian_state *posterior,
    bolr_workspace *workspace,
    bolr_gaussian_state **out_predictive,
    bolr_adaptation_diagnostics *diagnostics
);

bolr_status bolr_adaptive_policy_observe(
    const bolr_adaptive_policy *policy,
    bolr_adaptive_state *state,
    const bolr_gaussian_state *predictive,
    const bolr_gaussian_state *posterior,
    const bolr_surprise_input *surprise_input,
    bolr_adaptation_diagnostics *diagnostics
);

bolr_status bolr_adaptive_state_encoded_size(const bolr_adaptive_policy *policy, const bolr_adaptive_state *state, size_t *out_size);
bolr_status bolr_adaptive_state_encode(const bolr_adaptive_policy *policy, const bolr_adaptive_state *state, void *output, size_t output_size, size_t *out_written);
bolr_status bolr_adaptive_state_decode(const bolr_adaptive_policy *policy, const void *data, size_t data_size, const bolr_allocator *allocator, bolr_adaptive_state **out_state);

#endif
