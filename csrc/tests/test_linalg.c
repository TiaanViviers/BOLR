#include "test_suite.h"

#include "bolr/linalg.h"
#include "bolr/status.h"

#include <math.h>

int test_linalg(void) {
    bolr_real matrix[] = {4.0, 1.0, 1.0, 3.0};
    bolr_real rhs[] = {1.0, 2.0};
    bolr_real out[] = {0.0, 0.0};
    bolr_cholesky_diagnostics diag;
    if (bolr_cholesky_factor((bolr_matrix_view){matrix, 2, 2, 2, 1}, 1e-9, 10.0, 4, &diag) != BOLR_OK) return 1;
    if (bolr_cholesky_solve((bolr_const_matrix_view){matrix, 2, 2, 2, 1}, (bolr_const_vector_view){rhs, 2, 1}, (bolr_vector_view){out, 2, 1}) != BOLR_OK) return 1;
    if (fabs(out[0] - 0.090909090909) > 1e-9) return 1;
    return 0;
}
