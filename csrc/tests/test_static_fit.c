#include "test_suite.h"

#include "bolr/inference.h"
#include "bolr/optimizer.h"
#include "bolr/static_fit.h"
#include "bolr/status.h"

#include <math.h>

int test_static_fit(void) {
    bolr_real design[] = {1.0, 0.0, 0.0, 1.0};
    bolr_real target_day_1[] = {1.0, 0.0};
    bolr_real target_day_2[] = {0.0, 1.0};
    bolr_const_vector_view targets[] = {{target_day_1, 2, 1}, {target_day_2, 2, 1}};
    bolr_real weights[] = {1.0, 1.0};
    bolr_real prior_mean[] = {0.0, 0.0};
    bolr_real prior_precision[] = {1.0, 0.0, 0.0, 1.0};
    bolr_real coefficients[] = {0.0, 0.0};
    bolr_real scores[] = {0.0, 0.0};
    bolr_candidate_a_static_dataset *dataset = NULL;
    bolr_inference_workspace *workspace = NULL;
    bolr_static_fit_diagnostics diagnostics;
    bolr_newton_config config = {6, 1e-6, 1e-9, 1e-12, 1e-3, 10.0, 1e6, 1e-4, 0.5, 6, 1e-10, 10.0, 8};
    if (bolr_candidate_a_static_dataset_create((bolr_const_matrix_view){design, 2, 2, 2, 1}, targets, weights, 2, NULL, &dataset) != BOLR_OK) return 1;
    if (bolr_inference_workspace_create(2, 2, NULL, &workspace) != BOLR_OK) return 1;
    if (bolr_candidate_a_static_fit(dataset, (bolr_const_vector_view){prior_mean, 2, 1}, (bolr_const_matrix_view){prior_precision, 2, 2, 2, 1}, &config, workspace, (bolr_vector_view){coefficients, 2, 1}, (bolr_vector_view){scores, 2, 1}, &diagnostics) != BOLR_OK) return 1;
    if ((!isfinite(coefficients[0])) || (!diagnostics.converged)) return 1;
    bolr_inference_workspace_destroy(workspace);
    bolr_candidate_a_static_dataset_destroy(dataset);
    return 0;
}
