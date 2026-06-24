#include "bolr/penalty.h"
#include "bolr/linalg.h"

#include <stdlib.h>

bolr_status bolr_quadratic_penalty_value(bolr_const_matrix_view precision, bolr_const_vector_view state, bolr_real *out_value) {
    bolr_real *buffer;
    bolr_real dot;
    bolr_status status;
    bolr_index i;
    if (out_value == NULL) return BOLR_INVALID_ARGUMENT;
    buffer = (bolr_real *) malloc((size_t) state.length * sizeof(bolr_real));
    if (buffer == NULL) return BOLR_ALLOCATION_FAILED;
    status = bolr_quadratic_penalty_gradient(precision, state, (bolr_vector_view){buffer, state.length, 1});
    if (status == BOLR_OK) {
        dot = 0.0;
        for (i = 0; i < state.length; ++i) dot += state.data[i * state.stride] * buffer[i];
        *out_value = 0.5 * dot;
    }
    free(buffer);
    return status;
}
bolr_status bolr_quadratic_penalty_gradient(bolr_const_matrix_view precision, bolr_const_vector_view state, bolr_vector_view output_gradient) { return bolr_matvec(precision, state, output_gradient); }
bolr_status bolr_quadratic_penalty_hvp(bolr_const_matrix_view precision, bolr_const_vector_view vector, bolr_vector_view output_hvp) { return bolr_matvec(precision, vector, output_hvp); }
