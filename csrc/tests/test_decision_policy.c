#include "test_suite.h"

#include "bolr/decision.h"
#include "bolr/gaussian.h"
#include "bolr/prediction.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"

int test_decision_policy(void) {
    bolr_state_layout *layout = NULL;
    bolr_model *model = NULL;
    bolr_gaussian_state *state = NULL;
    bolr_posterior_prediction *prediction = NULL;
    bolr_decision_policy *policy = NULL;
    bolr_decision_policy_config config = {BOLR_DECISION_POSTERIOR_MEAN, 0, 0, 0};
    bolr_decision decision;
    bolr_decision_diagnostics diagnostics;
    bolr_state_block_spec spec = {"dense", 0, 2, 2, 3, 1, 'C'};
    bolr_real static_scores[] = {1.0, 1.0, 0.0};
    bolr_real design[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    bolr_real mean[] = {0.0, 0.0};
    bolr_real covariance[] = {0.2, 0.0, 0.0, 0.1};
    bolr_real probability_best[] = {0.4, 0.5, 0.1};
    bolr_real expected_rank[] = {1.8, 1.2, 3.0};
    if (bolr_state_layout_create(&spec, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 3, 1}, NULL, &model) != BOLR_OK) return 1;
    if (bolr_model_add_dense_block_copy(model, "dense", (bolr_const_matrix_view){design, 3, 2, 2, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, bolr_model_state_layout_hash(model), bolr_model_schema_hash(model), NULL, &state) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_create(state, model, (bolr_const_vector_view){NULL, 0, 1}, NULL, NULL, &prediction, NULL) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_set_probability_best(prediction, (bolr_const_vector_view){probability_best, 3, 1}) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_set_expected_rank(prediction, (bolr_const_vector_view){expected_rank, 3, 1}) != BOLR_OK) return 1;
    if (bolr_decision_policy_create(&config, NULL, &policy) != BOLR_OK) return 1;
    if (bolr_decision_policy_apply(policy, prediction, NULL, NULL, &decision, &diagnostics) != BOLR_OK) return 1;
    if ((!decision.selected) || (decision.selected_index != 1) || (!diagnostics.tie_occurred)) return 1;
    bolr_decision_policy_destroy(policy);
    config.family = BOLR_DECISION_EXPECTED_RANK;
    if (bolr_decision_policy_create(&config, NULL, &policy) != BOLR_OK) return 1;
    if (bolr_decision_policy_apply(policy, prediction, NULL, NULL, &decision, NULL) != BOLR_OK) return 1;
    bolr_decision_policy_destroy(policy);
    bolr_posterior_prediction_destroy(prediction);
    bolr_gaussian_state_destroy(state);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    return (decision.selected_index != 1) ? 1 : 0;
}
