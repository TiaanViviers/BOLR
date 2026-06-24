#include "bolr/linalg.h"

#include <math.h>
#include <stddef.h>

static bolr_real vec_get(bolr_const_vector_view view, bolr_index i) { return view.data[i * view.stride]; }
static void vec_set(bolr_vector_view view, bolr_index i, bolr_real value) { view.data[i * view.stride] = value; }
static bolr_real mat_get(bolr_const_matrix_view view, bolr_index r, bolr_index c) { return view.data[r * view.row_stride + c * view.col_stride]; }
static void mat_set(bolr_matrix_view view, bolr_index r, bolr_index c, bolr_real value) { view.data[r * view.row_stride + c * view.col_stride] = value; }

bolr_status bolr_copy(bolr_const_vector_view src, bolr_vector_view dst) {
    bolr_index i;
    if ((bolr_vector_view_validate(src) != BOLR_OK) || (bolr_mutable_vector_view_validate(dst) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if (src.length != dst.length) return BOLR_INVALID_SHAPE;
    for (i = 0; i < src.length; ++i) vec_set(dst, i, vec_get(src, i));
    return BOLR_OK;
}
bolr_status bolr_fill(bolr_vector_view dst, bolr_real value) { bolr_index i; if (bolr_mutable_vector_view_validate(dst) != BOLR_OK) return BOLR_INVALID_ARGUMENT; for (i = 0; i < dst.length; ++i) vec_set(dst, i, value); return BOLR_OK; }
bolr_status bolr_scale(bolr_vector_view dst, bolr_real value) { bolr_index i; if (bolr_mutable_vector_view_validate(dst) != BOLR_OK) return BOLR_INVALID_ARGUMENT; for (i = 0; i < dst.length; ++i) vec_set(dst, i, vec_get((bolr_const_vector_view){dst.data, dst.length, dst.stride}, i) * value); return BOLR_OK; }
bolr_status bolr_axpy(bolr_real alpha, bolr_const_vector_view x, bolr_vector_view y) { bolr_index i; if ((bolr_vector_view_validate(x) != BOLR_OK) || (bolr_mutable_vector_view_validate(y) != BOLR_OK)) return BOLR_INVALID_ARGUMENT; if (x.length != y.length) return BOLR_INVALID_SHAPE; for (i = 0; i < x.length; ++i) vec_set(y, i, vec_get((bolr_const_vector_view){y.data, y.length, y.stride}, i) + alpha * vec_get(x, i)); return BOLR_OK; }
bolr_status bolr_dot(bolr_const_vector_view x, bolr_const_vector_view y, bolr_real *out) { bolr_index i; bolr_real sum = 0.0; if ((out == NULL) || (bolr_vector_view_validate(x) != BOLR_OK) || (bolr_vector_view_validate(y) != BOLR_OK)) return BOLR_INVALID_ARGUMENT; if (x.length != y.length) return BOLR_INVALID_SHAPE; for (i = 0; i < x.length; ++i) sum += vec_get(x, i) * vec_get(y, i); *out = sum; return BOLR_OK; }
bolr_status bolr_norm2(bolr_const_vector_view x, bolr_real *out) { bolr_real dot; bolr_status status = bolr_dot(x, x, &dot); if (status != BOLR_OK) return status; *out = sqrt(dot); return BOLR_OK; }
bolr_status bolr_sum(bolr_const_vector_view x, bolr_real *out) { bolr_index i; bolr_real sum = 0.0; if ((out == NULL) || (bolr_vector_view_validate(x) != BOLR_OK)) return BOLR_INVALID_ARGUMENT; for (i = 0; i < x.length; ++i) sum += vec_get(x, i); *out = sum; return BOLR_OK; }
bolr_status bolr_maximum(bolr_const_vector_view x, bolr_real *out) { bolr_index i; bolr_real maxv; if ((out == NULL) || (bolr_vector_view_validate(x) != BOLR_OK) || (x.length <= 0)) return BOLR_INVALID_ARGUMENT; maxv = vec_get(x, 0); for (i = 1; i < x.length; ++i) { bolr_real value = vec_get(x, i); if (value > maxv) maxv = value; } *out = maxv; return BOLR_OK; }
bolr_status bolr_argmax(bolr_const_vector_view x, bolr_index *out) { bolr_index i, arg = 0; bolr_real maxv; if ((out == NULL) || (bolr_vector_view_validate(x) != BOLR_OK) || (x.length <= 0)) return BOLR_INVALID_ARGUMENT; maxv = vec_get(x, 0); for (i = 1; i < x.length; ++i) { bolr_real value = vec_get(x, i); if (value > maxv) { maxv = value; arg = i; } } *out = arg; return BOLR_OK; }

bolr_status bolr_matvec(bolr_const_matrix_view matrix, bolr_const_vector_view vector, bolr_vector_view output) {
    bolr_index r, c;
    if ((bolr_matrix_view_validate(matrix) != BOLR_OK) || (bolr_vector_view_validate(vector) != BOLR_OK) || (bolr_mutable_vector_view_validate(output) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if ((matrix.cols != vector.length) || (matrix.rows != output.length)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < matrix.rows; ++r) {
        bolr_real sum = 0.0;
        for (c = 0; c < matrix.cols; ++c) sum += mat_get(matrix, r, c) * vec_get(vector, c);
        vec_set(output, r, sum);
    }
    return BOLR_OK;
}

bolr_status bolr_matvec_transpose(bolr_const_matrix_view matrix, bolr_const_vector_view vector, bolr_vector_view output) {
    bolr_index r, c;
    if ((bolr_matrix_view_validate(matrix) != BOLR_OK) || (bolr_vector_view_validate(vector) != BOLR_OK) || (bolr_mutable_vector_view_validate(output) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if ((matrix.rows != vector.length) || (matrix.cols != output.length)) return BOLR_INVALID_SHAPE;
    for (c = 0; c < matrix.cols; ++c) {
        bolr_real sum = 0.0;
        for (r = 0; r < matrix.rows; ++r) sum += mat_get(matrix, r, c) * vec_get(vector, r);
        vec_set(output, c, sum);
    }
    return BOLR_OK;
}

bolr_status bolr_matmul(bolr_const_matrix_view left, bolr_const_matrix_view right, bolr_matrix_view output) {
    bolr_index r, c, k;
    if ((bolr_matrix_view_validate(left) != BOLR_OK) || (bolr_matrix_view_validate(right) != BOLR_OK) || (bolr_mutable_matrix_view_validate(output) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if ((left.cols != right.rows) || (output.rows != left.rows) || (output.cols != right.cols)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < output.rows; ++r) {
        for (c = 0; c < output.cols; ++c) {
            bolr_real sum = 0.0;
            for (k = 0; k < left.cols; ++k) sum += mat_get(left, r, k) * mat_get(right, k, c);
            mat_set(output, r, c, sum);
        }
    }
    return BOLR_OK;
}

bolr_status bolr_sym_matvec(bolr_const_matrix_view matrix, bolr_const_vector_view vector, bolr_vector_view output) { return bolr_matvec(matrix, vector, output); }
bolr_status bolr_rank_one_update(bolr_matrix_view matrix, bolr_real alpha, bolr_const_vector_view x, bolr_const_vector_view y) {
    bolr_index r, c;
    if ((bolr_mutable_matrix_view_validate(matrix) != BOLR_OK) || (bolr_vector_view_validate(x) != BOLR_OK) || (bolr_vector_view_validate(y) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if ((matrix.rows != x.length) || (matrix.cols != y.length)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < matrix.rows; ++r) for (c = 0; c < matrix.cols; ++c) mat_set(matrix, r, c, mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, r, c) + alpha * vec_get(x, r) * vec_get(y, c));
    return BOLR_OK;
}
bolr_status bolr_symmetrize(bolr_matrix_view matrix) {
    bolr_index r, c;
    if ((bolr_mutable_matrix_view_validate(matrix) != BOLR_OK) || (matrix.rows != matrix.cols)) return BOLR_INVALID_ARGUMENT;
    for (r = 0; r < matrix.rows; ++r) for (c = r + 1; c < matrix.cols; ++c) { bolr_real value = 0.5 * (mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, r, c) + mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, c, r)); mat_set(matrix, r, c, value); mat_set(matrix, c, r, value); }
    return BOLR_OK;
}

bolr_status bolr_cholesky_factor(bolr_matrix_view matrix, bolr_real initial_jitter, bolr_real jitter_multiplier, bolr_index max_attempts, bolr_cholesky_diagnostics *diagnostics) {
    bolr_index attempt;
    if ((bolr_mutable_matrix_view_validate(matrix) != BOLR_OK) || (matrix.rows != matrix.cols)) return BOLR_INVALID_ARGUMENT;
    for (attempt = 0; attempt < max_attempts; ++attempt) {
        bolr_index i, j, k;
        bolr_real jitter = (attempt == 0) ? 0.0 : initial_jitter * pow(jitter_multiplier, (double) (attempt - 1));
        for (i = 0; i < matrix.rows; ++i) {
            for (j = 0; j <= i; ++j) {
                bolr_real sum = mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, i, j);
                if (i == j) sum += jitter;
                for (k = 0; k < j; ++k) sum -= mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, i, k) * mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, j, k);
                if (i == j) {
                    if (sum <= 0.0) break;
                    mat_set(matrix, i, j, sqrt(sum));
                } else {
                    mat_set(matrix, i, j, sum / mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, j, j));
                }
            }
            if (j <= i && i == j && mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, i, j) <= 0.0) break;
            for (j = i + 1; j < matrix.cols; ++j) mat_set(matrix, i, j, 0.0);
        }
        if (i == matrix.rows) {
            if (diagnostics != NULL) {
                diagnostics->jitter_used = jitter;
                diagnostics->attempts = attempt + 1;
                diagnostics->minimum_diagonal = mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, 0, 0);
                for (j = 1; j < matrix.rows; ++j) {
                    bolr_real diag = mat_get((bolr_const_matrix_view){matrix.data, matrix.rows, matrix.cols, matrix.row_stride, matrix.col_stride}, j, j);
                    if (diag < diagnostics->minimum_diagonal) diagnostics->minimum_diagonal = diag;
                }
            }
            return BOLR_OK;
        }
    }
    return BOLR_NOT_POSITIVE_DEFINITE;
}

bolr_status bolr_triangular_solve_lower(bolr_const_matrix_view factor, bolr_const_vector_view rhs, bolr_vector_view output) {
    bolr_index i, k;
    if ((bolr_matrix_view_validate(factor) != BOLR_OK) || (bolr_vector_view_validate(rhs) != BOLR_OK) || (bolr_mutable_vector_view_validate(output) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    if ((factor.rows != factor.cols) || (factor.rows != rhs.length) || (rhs.length != output.length)) return BOLR_INVALID_SHAPE;
    for (i = 0; i < factor.rows; ++i) {
        bolr_real sum = vec_get(rhs, i);
        for (k = 0; k < i; ++k) sum -= mat_get(factor, i, k) * vec_get((bolr_const_vector_view){output.data, output.length, output.stride}, k);
        vec_set(output, i, sum / mat_get(factor, i, i));
    }
    return BOLR_OK;
}

bolr_status bolr_cholesky_solve(bolr_const_matrix_view factor, bolr_const_vector_view rhs, bolr_vector_view output) {
    bolr_index i, k;
    bolr_status status = bolr_triangular_solve_lower(factor, rhs, output);
    if (status != BOLR_OK) return status;
    for (i = factor.rows - 1; i >= 0; --i) {
        bolr_real sum = vec_get((bolr_const_vector_view){output.data, output.length, output.stride}, i);
        for (k = i + 1; k < factor.cols; ++k) sum -= mat_get(factor, k, i) * vec_get((bolr_const_vector_view){output.data, output.length, output.stride}, k);
        vec_set(output, i, sum / mat_get(factor, i, i));
        if (i == 0) break;
    }
    return BOLR_OK;
}

bolr_status bolr_logdet_from_cholesky(bolr_const_matrix_view factor, bolr_real *out) {
    bolr_index i;
    bolr_real sum = 0.0;
    if ((out == NULL) || (bolr_matrix_view_validate(factor) != BOLR_OK) || (factor.rows != factor.cols)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < factor.rows; ++i) sum += log(mat_get(factor, i, i));
    *out = 2.0 * sum;
    return BOLR_OK;
}
