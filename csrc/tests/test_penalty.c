#include "test_suite.h"

#include "bolr/penalty.h"
#include "bolr/status.h"

#include <math.h>

int test_penalty(void) {
    bolr_real precision[] = {2.0, 0.0, 0.0, 3.0};
    bolr_real state[] = {1.0, 2.0};
    bolr_real grad[] = {0.0, 0.0};
    if (bolr_quadratic_penalty_gradient((bolr_const_matrix_view){precision, 2, 2, 2, 1}, (bolr_const_vector_view){state, 2, 1}, (bolr_vector_view){grad, 2, 1}) != BOLR_OK) return 1;
    return (fabs(grad[1] - 6.0) > 1e-12) ? 1 : 0;
}
