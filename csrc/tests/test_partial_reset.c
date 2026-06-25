#include "test_suite.h"

#include "bolr/reset.h"

#include <math.h>

int test_partial_reset(void) {
    bolr_real mean[2] = {2.0, 0.5};
    bolr_real covariance[4] = {2.0, 0.4, 0.4, 1.0};
    bolr_real out_mean[2];
    bolr_real out_cov[4];
    bolr_real anchor_mean[1] = {0.0};
    bolr_real anchor_cov[1] = {1.0};
    if (bolr_apply_partial_reset(
        (bolr_const_vector_view){mean, 2, 1},
        (bolr_const_matrix_view){covariance, 2, 2, 2, 1},
        0, 1, 0.5,
        (bolr_const_vector_view){anchor_mean, 1, 1},
        (bolr_const_matrix_view){anchor_cov, 1, 1, 1, 1},
        (bolr_vector_view){out_mean, 2, 1},
        (bolr_matrix_view){out_cov, 2, 2, 2, 1}
    ) != BOLR_OK) return 1;
    return (fabs(out_mean[0] - 1.0) < 1e-12 && fabs(out_cov[1] - 0.2) < 1e-12) ? 0 : 1;
}
