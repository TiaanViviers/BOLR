#ifndef BOLR_PENALTY_H
#define BOLR_PENALTY_H

#include "bolr/array.h"

bolr_status bolr_quadratic_penalty_value(bolr_const_matrix_view precision, bolr_const_vector_view state, bolr_real *out_value);
bolr_status bolr_quadratic_penalty_gradient(bolr_const_matrix_view precision, bolr_const_vector_view state, bolr_vector_view output_gradient);
bolr_status bolr_quadratic_penalty_hvp(bolr_const_matrix_view precision, bolr_const_vector_view vector, bolr_vector_view output_hvp);

#endif
