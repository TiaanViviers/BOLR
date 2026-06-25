#include "test_suite.h"

#include "bolr/gaussian.h"
#include "bolr/prediction.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"

#include <math.h>

int test_pairwise_probability(void) {
    bolr_state_layout *layout = NULL;
    bolr_model *model = NULL;
    bolr_gaussian_state *state = NULL;
    bolr_posterior_prediction *prediction = NULL;
    bolr_state_block_spec spec = {"dense", 0, 2, 2, 3, 1, 'C'};
    bolr_real static_scores[] = {0.0, 0.0, 0.0};
    bolr_real design[] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0};
    bolr_real mean[] = {2.0, 1.0};
    bolr_real covariance[] = {0.5, 0.0, 0.0, 0.5};
    bolr_index left[] = {0, 0};
    bolr_index right[] = {1, 2};
    bolr_pairwise_probability_result output[2];
    if (bolr_state_layout_create(&spec, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 3, 1}, NULL, &model) != BOLR_OK) return 1;
    if (bolr_model_add_dense_block_copy(model, "dense", (bolr_const_matrix_view){design, 3, 2, 2, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, bolr_model_state_layout_hash(model), bolr_model_schema_hash(model), NULL, &state) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_create(state, model, (bolr_const_vector_view){NULL, 0, 1}, NULL, NULL, &prediction, NULL) != BOLR_OK) return 1;
    if (bolr_pairwise_probability(prediction, left, right, 2, output) != BOLR_OK) return 1;
    bolr_posterior_prediction_destroy(prediction);
    bolr_gaussian_state_destroy(state);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    if (output[0].left_probability <= 0.5) return 1;
    if (fabs(output[1].variance_difference) > 1e-12) return 1;
    return (fabs(output[1].left_probability - 1.0) > 1e-12) ? 1 : 0;
}
