#include "test_suite.h"

#include "bolr/gaussian.h"
#include "bolr/prediction.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"

#include <math.h>

int test_score_uncertainty(void) {
    bolr_state_layout *layout = NULL;
    bolr_model *model = NULL;
    bolr_gaussian_state *state = NULL;
    bolr_posterior_prediction *prediction = NULL;
    bolr_posterior_prediction_diagnostics diagnostics;
    bolr_state_block_spec spec = {"dense", 0, 2, 2, 2, 1, 'C'};
    bolr_real static_scores[] = {0.0, 0.0};
    bolr_real design[] = {1.0, 0.0, 0.0, 1.0};
    bolr_real mean[] = {1.5, -0.5};
    bolr_real covariance[] = {2.0, 0.3, 0.3, 1.5};
    bolr_real score_mean[] = {0.0, 0.0};
    bolr_real score_variance[] = {0.0, 0.0};
    bolr_real selected[] = {0.0, 0.0, 0.0, 0.0};
    bolr_index indices[] = {0, 1};
    if (bolr_state_layout_create(&spec, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 2, 1}, NULL, &model) != BOLR_OK) return 1;
    if (bolr_model_add_dense_block_copy(model, "dense", (bolr_const_matrix_view){design, 2, 2, 2, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, bolr_model_state_layout_hash(model), bolr_model_schema_hash(model), NULL, &state) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_create(state, model, (bolr_const_vector_view){NULL, 0, 1}, NULL, NULL, &prediction, &diagnostics) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_copy_score_mean(prediction, (bolr_vector_view){score_mean, 2, 1}) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_copy_score_variance(prediction, (bolr_vector_view){score_variance, 2, 1}) != BOLR_OK) return 1;
    if (bolr_selected_score_covariance(prediction, indices, 2, (bolr_matrix_view){selected, 2, 2, 2, 1}) != BOLR_OK) return 1;
    bolr_posterior_prediction_destroy(prediction);
    bolr_gaussian_state_destroy(state);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    if ((fabs(score_mean[0] - 1.5) > 1e-12) || (fabs(score_mean[1] + 0.5) > 1e-12)) return 1;
    if ((fabs(score_variance[0] - 2.0) > 1e-12) || (fabs(score_variance[1] - 1.5) > 1e-12)) return 1;
    if ((fabs(selected[0] - 2.0) > 1e-12) || (fabs(selected[1] - 0.3) > 1e-12) || (fabs(selected[2] - 0.3) > 1e-12) || (fabs(selected[3] - 1.5) > 1e-12)) return 1;
    return (diagnostics.explicit_design_frobenius_norm <= 0.0) ? 1 : 0;
}
