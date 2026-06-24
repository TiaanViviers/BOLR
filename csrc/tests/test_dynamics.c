#include "test_suite.h"

#include "bolr/dynamics.h"
#include "bolr/status.h"

#include <math.h>

int test_dynamics(void) {
    bolr_real posterior[] = {1.0, 0.2, 0.2, 2.0};
    bolr_real noise[] = {0.1, 0.0, 0.0, 0.1};
    bolr_real out[] = {0.0, 0.0, 0.0, 0.0};
    bolr_real scale[] = {2.0, 0.5};
    if (bolr_additive_transition_covariance((bolr_const_matrix_view){posterior, 2, 2, 2, 1}, (bolr_const_matrix_view){noise, 2, 2, 2, 1}, (bolr_matrix_view){out, 2, 2, 2, 1}) != BOLR_OK) return 1;
    if (fabs(out[0] - 1.1) > 1e-12) return 1;
    if (bolr_heterogeneous_discount_covariance((bolr_const_matrix_view){posterior, 2, 2, 2, 1}, (bolr_const_vector_view){scale, 2, 1}, (bolr_matrix_view){out, 2, 2, 2, 1}) != BOLR_OK) return 1;
    return (fabs(out[1] - 0.2) > 1e-12) ? 1 : 0;
}
