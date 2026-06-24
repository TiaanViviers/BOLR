#include "test_suite.h"

#include "bolr/observation.h"
#include "bolr/status.h"

#include <math.h>

int test_posterior_objective(void) {
    bolr_real scores[] = {0.4, -0.1};
    bolr_real target[] = {0.8, 0.2};
    bolr_real vector[] = {1.0, -1.0};
    bolr_real gradient[] = {0.0, 0.0};
    bolr_real hvp[] = {0.0, 0.0};
    bolr_real value = 0.0;
    if (bolr_candidate_a_log_factor((bolr_const_vector_view){scores, 2, 1}, (bolr_const_vector_view){target, 2, 1}, 1.0, &value) != BOLR_OK) return 1;
    if (bolr_candidate_a_score_gradient((bolr_const_vector_view){scores, 2, 1}, (bolr_const_vector_view){target, 2, 1}, 1.0, (bolr_vector_view){gradient, 2, 1}) != BOLR_OK) return 1;
    if (bolr_candidate_a_score_hvp((bolr_const_vector_view){scores, 2, 1}, (bolr_const_vector_view){vector, 2, 1}, 1.0, (bolr_vector_view){hvp, 2, 1}) != BOLR_OK) return 1;
    if ((!isfinite(value)) || (!isfinite(gradient[0])) || (!isfinite(hvp[0]))) return 1;
    return 0;
}
