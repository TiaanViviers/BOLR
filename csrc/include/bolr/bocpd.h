#ifndef BOLR_BOCPD_H
#define BOLR_BOCPD_H

#include <stddef.h>

#include "bolr/allocator.h"
#include "bolr/array.h"
#include "bolr/status.h"
#include "bolr/types.h"

typedef struct bolr_bocpd_state bolr_bocpd_state;

typedef struct {
    bolr_real hazard;
    bolr_index max_run_length;
    bolr_real prior_mean;
    bolr_real prior_kappa;
    bolr_real prior_alpha;
    bolr_real prior_beta;
    int missing_policy;
} bolr_bocpd_config;

typedef struct {
    bolr_real change_probability;
    bolr_real map_run_length;
    bolr_real expected_run_length;
    bolr_real run_length_entropy;
    bolr_real predictive_log_density;
    bolr_real truncation_mass;
    bolr_real hazard;
    int informative;
    int predictive_log_density_present;
    int missing_policy;
} bolr_bocpd_diagnostics;

enum {
    BOLR_BOCPD_MISSING_HOLD = 1,
    BOLR_BOCPD_MISSING_HAZARD_ONLY = 2
};

bolr_status bolr_bocpd_state_create(const bolr_bocpd_config *config, const bolr_allocator *allocator, bolr_bocpd_state **out_state);
void bolr_bocpd_state_destroy(bolr_bocpd_state *state);
bolr_status bolr_bocpd_step(bolr_bocpd_state *state, bolr_real value, int value_present, bolr_bocpd_diagnostics *diagnostics);
bolr_status bolr_bocpd_copy_run_length_posterior(const bolr_bocpd_state *state, bolr_vector_view output);
bolr_index bolr_bocpd_max_run_length(const bolr_bocpd_state *state);
uint64_t bolr_bocpd_step_index(const bolr_bocpd_state *state);
bolr_status bolr_bocpd_encoded_size(const bolr_bocpd_state *state, size_t *out_size);
bolr_status bolr_bocpd_encode(const bolr_bocpd_state *state, void *output, size_t output_size, size_t *out_written);
bolr_status bolr_bocpd_decode(const void *data, size_t data_size, const bolr_allocator *allocator, bolr_bocpd_state **out_state);

#endif
