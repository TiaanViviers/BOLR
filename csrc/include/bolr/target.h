#ifndef BOLR_TARGET_H
#define BOLR_TARGET_H

#include "bolr/array.h"

typedef struct {
    bolr_real kappa;
    bolr_real eta;
    bolr_real clip;
    bolr_real absolute_tolerance;
    bolr_real relative_tolerance;
    bolr_real min_scale;
    int no_update_if_degenerate;
} bolr_candidate_a_target_config;

typedef struct {
    bolr_index candidate_count;
    int informative;
    bolr_real target_sum;
    bolr_real target_minimum;
    bolr_real target_maximum;
    bolr_real target_entropy;
    bolr_index positive_candidate_count;
    bolr_index highest_group_count;
    bolr_real effective_temperature;
    int fallback_used;
    bolr_index tolerance_group_count;
    bolr_real utility_scale;
    bolr_real clipping_fraction;
    int all_irrelevant;
} bolr_candidate_a_target_diagnostics;

bolr_status bolr_candidate_a_target_build(
    const bolr_candidate_a_target_config *config,
    bolr_const_vector_view realised_values,
    bolr_vector_view output_target,
    bolr_real *out_update_weight,
    bolr_candidate_a_target_diagnostics *diagnostics
);

#endif
