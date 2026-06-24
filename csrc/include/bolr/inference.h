#ifndef BOLR_INFERENCE_H
#define BOLR_INFERENCE_H

#include "bolr/gaussian.h"
#include "bolr/observation.h"
#include "bolr/optimizer.h"
#include "bolr/score.h"

typedef struct bolr_inference_workspace bolr_inference_workspace;

typedef struct {
    bolr_newton_diagnostics newton;
    bolr_real prior_covariance_trace;
    bolr_real posterior_covariance_trace;
    bolr_real posterior_covariance_log_determinant;
    bolr_real posterior_condition_estimate;
    bolr_real mean_update_norm;
    bolr_real mahalanobis_update_norm;
    bolr_real log_factor_at_predictive_mean;
    bolr_real log_factor_at_posterior_mode;
    bolr_real objective_improvement;
    bolr_real score_mean_min;
    bolr_real score_mean_max;
    bolr_real gradient_sum_diagnostic;
    bolr_real curvature_null_direction_diagnostic;
} bolr_laplace_diagnostics;

bolr_status bolr_inference_workspace_create(
    bolr_index state_dimension,
    bolr_index candidate_count,
    const bolr_allocator *allocator,
    bolr_inference_workspace **out_workspace
);
void bolr_inference_workspace_destroy(bolr_inference_workspace *workspace);
bolr_index bolr_inference_workspace_state_dimension(const bolr_inference_workspace *workspace);
bolr_index bolr_inference_workspace_candidate_count(const bolr_inference_workspace *workspace);

bolr_status bolr_laplace_update(
    const bolr_gaussian_state *predictive,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_observation_operator *observation,
    const bolr_newton_config *config,
    bolr_inference_workspace *workspace,
    bolr_gaussian_state **out_posterior,
    bolr_laplace_diagnostics *diagnostics
);

#endif
