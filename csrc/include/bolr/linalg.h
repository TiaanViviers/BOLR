#ifndef BOLR_LINALG_H
#define BOLR_LINALG_H

#include "bolr/array.h"
#include "bolr/status.h"

typedef struct {
    bolr_real jitter_used;
    bolr_index attempts;
    bolr_real minimum_diagonal;
} bolr_cholesky_diagnostics;

bolr_status bolr_copy(bolr_const_vector_view src, bolr_vector_view dst);
bolr_status bolr_fill(bolr_vector_view dst, bolr_real value);
bolr_status bolr_scale(bolr_vector_view dst, bolr_real value);
bolr_status bolr_axpy(bolr_real alpha, bolr_const_vector_view x, bolr_vector_view y);
bolr_status bolr_dot(bolr_const_vector_view x, bolr_const_vector_view y, bolr_real *out);
bolr_status bolr_norm2(bolr_const_vector_view x, bolr_real *out);
bolr_status bolr_sum(bolr_const_vector_view x, bolr_real *out);
bolr_status bolr_maximum(bolr_const_vector_view x, bolr_real *out);
bolr_status bolr_argmax(bolr_const_vector_view x, bolr_index *out);
bolr_status bolr_matvec(bolr_const_matrix_view matrix, bolr_const_vector_view vector, bolr_vector_view output);
bolr_status bolr_matvec_transpose(bolr_const_matrix_view matrix, bolr_const_vector_view vector, bolr_vector_view output);
bolr_status bolr_matmul(bolr_const_matrix_view left, bolr_const_matrix_view right, bolr_matrix_view output);
bolr_status bolr_sym_matvec(bolr_const_matrix_view matrix, bolr_const_vector_view vector, bolr_vector_view output);
bolr_status bolr_rank_one_update(bolr_matrix_view matrix, bolr_real alpha, bolr_const_vector_view x, bolr_const_vector_view y);
bolr_status bolr_symmetrize(bolr_matrix_view matrix);
bolr_status bolr_cholesky_factor(bolr_matrix_view matrix, bolr_real initial_jitter, bolr_real jitter_multiplier, bolr_index max_attempts, bolr_cholesky_diagnostics *diagnostics);
bolr_status bolr_cholesky_solve(bolr_const_matrix_view factor, bolr_const_vector_view rhs, bolr_vector_view output);
bolr_status bolr_triangular_solve_lower(bolr_const_matrix_view factor, bolr_const_vector_view rhs, bolr_vector_view output);
bolr_status bolr_logdet_from_cholesky(bolr_const_matrix_view factor, bolr_real *out);

#endif
