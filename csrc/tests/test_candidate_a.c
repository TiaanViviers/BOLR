#include "test_suite.h"

#include "bolr/observation.h"
#include "bolr/status.h"

#include <math.h>

int test_candidate_a(void) {
    bolr_real scores[] = {1.0, 0.0};
    bolr_real target[] = {1.0, 0.0};
    bolr_real grad[] = {0.0, 0.0};
    bolr_real value;
    if (bolr_candidate_a_log_factor((bolr_const_vector_view){scores, 2, 1}, (bolr_const_vector_view){target, 2, 1}, 1.0, &value) != BOLR_OK) return 1;
    if (bolr_candidate_a_score_gradient((bolr_const_vector_view){scores, 2, 1}, (bolr_const_vector_view){target, 2, 1}, 1.0, (bolr_vector_view){grad, 2, 1}) != BOLR_OK) return 1;
    return (fabs(grad[0] + grad[1]) > 1e-12) ? 1 : 0;
}
