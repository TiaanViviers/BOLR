#include "test_suite.h"

#include "bolr/gaussian.h"
#include "bolr/status.h"

#include <math.h>

int test_gaussian_prediction(void) {
    bolr_real mean[] = {0.0, 0.0};
    bolr_real covariance[] = {1.0, 0.0, 0.0, 1.0};
    bolr_real process_noise[] = {0.2, 0.0, 0.0, 0.3};
    bolr_real predictive_covariance[] = {0.0, 0.0, 0.0, 0.0};
    bolr_gaussian_state *posterior = NULL;
    bolr_gaussian_state *predictive = NULL;
    bolr_prediction_diagnostics diagnostics;
    bolr_transition_config transition;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, 5ULL, 9ULL, NULL, &posterior) != BOLR_OK) return 1;
    transition.family = BOLR_TRANSITION_ADDITIVE_Q;
    transition.process_noise = (bolr_const_matrix_view){process_noise, 2, 2, 2, 1};
    transition.global_discount = 0.0;
    transition.block_discount_scales = (bolr_const_vector_view){NULL, 0, 1};
    if (bolr_gaussian_predict(posterior, &transition, NULL, &predictive, &diagnostics) != BOLR_OK) return 1;
    if (bolr_gaussian_state_copy_covariance(predictive, (bolr_matrix_view){predictive_covariance, 2, 2, 2, 1}) != BOLR_OK) return 1;
    if ((fabs(predictive_covariance[0] - 1.2) > 1e-12) || (fabs(predictive_covariance[3] - 1.3) > 1e-12)) return 1;
    if ((bolr_gaussian_state_step_index(predictive) != 1ULL) || (diagnostics.process_noise_trace <= 0.0)) return 1;
    bolr_gaussian_state_destroy(predictive);
    bolr_gaussian_state_destroy(posterior);
    return 0;
}
