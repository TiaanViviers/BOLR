#include "test_suite.h"

#include "bolr/math.h"
#include "bolr/status.h"

#include <math.h>

static int close_to(bolr_real a, bolr_real b, bolr_real tol) { return fabs(a - b) <= tol; }
int test_math(void) {
    bolr_real x[] = {-1000.0, 0.0, 1000.0};
    bolr_real out[3];
    bolr_real value;
    if (bolr_logsumexp((bolr_const_vector_view){x, 3, 1}, &value) != BOLR_OK) return 1;
    if (!close_to(value, 1000.0, 1e-12)) return 1;
    if (bolr_softmax((bolr_const_vector_view){x, 3, 1}, (bolr_vector_view){out, 3, 1}) != BOLR_OK) return 1;
    if (!(out[2] > 0.999999999999)) return 1;
    return 0;
}
