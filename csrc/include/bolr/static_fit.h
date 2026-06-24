#ifndef BOLR_STATIC_FIT_H
#define BOLR_STATIC_FIT_H

#include "bolr/allocator.h"
#include "bolr/array.h"
#include "bolr/inference.h"

typedef struct bolr_candidate_a_static_dataset bolr_candidate_a_static_dataset;

typedef struct {
    bolr_real final_objective;
    bolr_real final_gradient_norm;
    bolr_real penalty_value;
    bolr_real observation_value;
    bolr_index iterations;
    int converged;
} bolr_static_fit_diagnostics;

bolr_status bolr_candidate_a_static_dataset_create(
    bolr_const_matrix_view design,
    const bolr_const_vector_view *targets,
    const bolr_real *effective_weights,
    bolr_index day_count,
    const bolr_allocator *allocator,
    bolr_candidate_a_static_dataset **out_dataset
);
void bolr_candidate_a_static_dataset_destroy(bolr_candidate_a_static_dataset *dataset);
bolr_status bolr_candidate_a_static_fit(
    const bolr_candidate_a_static_dataset *dataset,
    bolr_const_vector_view prior_mean,
    bolr_const_matrix_view prior_precision,
    const bolr_newton_config *config,
    bolr_inference_workspace *workspace,
    bolr_vector_view output_coefficients,
    bolr_vector_view output_scores,
    bolr_static_fit_diagnostics *diagnostics
);

#endif
