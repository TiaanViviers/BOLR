#include "test_suite.h"

#include "bolr/gaussian.h"
#include "bolr/inference.h"
#include "bolr/observation.h"
#include "bolr/optimizer.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"
#include "bolr/status.h"

#include <math.h>

int test_laplace(void) {
    bolr_state_layout *layout = NULL;
    bolr_model *model = NULL;
    bolr_gaussian_state *predictive = NULL;
    bolr_gaussian_state *posterior = NULL;
    bolr_inference_workspace *workspace = NULL;
    bolr_candidate_a_observation *candidate_a = NULL;
    bolr_observation_operator observation;
    bolr_laplace_diagnostics diagnostics;
    bolr_newton_config config = {12, 1e-6, 1e-9, 1e-12, 1e-3, 10.0, 1e6, 1e-4, 0.5, 12, 1e-10, 10.0, 8};
    bolr_state_block_spec spec = {"dense", 0, 2, 2, 2, 1, 'C'};
    bolr_real static_scores[] = {0.0, 0.0};
    bolr_real design[] = {1.0, 0.0, 0.0, 1.0};
    bolr_real mean[] = {0.0, 0.0};
    bolr_real covariance[] = {1.0, 0.0, 0.0, 1.0};
    bolr_real context_data[] = {0.0};
    bolr_real target[] = {1.0, 0.0};
    bolr_real posterior_mean[] = {0.0, 0.0};
    if (bolr_state_layout_create(&spec, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 2, 1}, NULL, &model) != BOLR_OK) return 1;
    if (bolr_model_add_dense_block_copy(model, "dense", (bolr_const_matrix_view){design, 2, 2, 2, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, bolr_model_state_layout_hash(model), bolr_model_schema_hash(model), NULL, &predictive) != BOLR_OK) return 1;
    if (bolr_candidate_a_observation_create((bolr_const_vector_view){target, 2, 1}, 1.0, 1.0, NULL, &candidate_a) != BOLR_OK) return 1;
    if (bolr_candidate_a_observation_operator(candidate_a, &observation) != BOLR_OK) return 1;
    if (bolr_inference_workspace_create(2, 2, NULL, &workspace) != BOLR_OK) return 1;
    if (bolr_laplace_update(predictive, model, (bolr_const_vector_view){context_data, 0, 1}, &observation, &config, workspace, &posterior, &diagnostics) != BOLR_OK) return 1;
    if (bolr_gaussian_state_copy_mean(posterior, (bolr_vector_view){posterior_mean, 2, 1}) != BOLR_OK) return 1;
    if ((!isfinite(posterior_mean[0])) || (!isfinite(diagnostics.posterior_covariance_trace))) return 1;
    bolr_inference_workspace_destroy(workspace);
    bolr_candidate_a_observation_destroy(candidate_a);
    bolr_gaussian_state_destroy(posterior);
    bolr_gaussian_state_destroy(predictive);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    return 0;
}
