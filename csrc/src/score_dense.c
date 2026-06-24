#include "bolr/score.h"
#include "bolr/linalg.h"

#include <stddef.h>

struct bolr_dense_operator {
    const bolr_allocator *allocator;
    bolr_real *data;
    bolr_index rows;
    bolr_index cols;
};

bolr_status bolr_dense_operator_create_copy(bolr_const_matrix_view design, const bolr_allocator *allocator, bolr_dense_operator **out_operator) {
    size_t bytes;
    bolr_index r, c;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_dense_operator *op;
    if ((out_operator == NULL) || (bolr_matrix_view_validate(design) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    *out_operator = NULL;
    op = (bolr_dense_operator *) bolr_allocator_calloc(active, 1U, sizeof(*op));
    if (op == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_checked_size_mul((size_t) (design.rows * design.cols), sizeof(bolr_real), &bytes) != BOLR_OK) { bolr_dense_operator_destroy(op); return BOLR_DIMENSION_OVERFLOW; }
    op->data = (bolr_real *) bolr_allocator_malloc(active, bytes);
    if (op->data == NULL) { bolr_dense_operator_destroy(op); return BOLR_ALLOCATION_FAILED; }
    op->allocator = active; op->rows = design.rows; op->cols = design.cols;
    for (r = 0; r < design.rows; ++r) for (c = 0; c < design.cols; ++c) op->data[r * design.cols + c] = design.data[r * design.row_stride + c * design.col_stride];
    *out_operator = op;
    return BOLR_OK;
}

void bolr_dense_operator_destroy(bolr_dense_operator *op) { if (op == NULL) return; bolr_allocator_free(op->allocator, op->data); bolr_allocator_free(op->allocator, op); }
bolr_index bolr_dense_operator_rows(const bolr_dense_operator *op) { return (op == NULL) ? -1 : op->rows; }
bolr_index bolr_dense_operator_cols(const bolr_dense_operator *op) { return (op == NULL) ? -1 : op->cols; }

bolr_status bolr_dense_operator_forward(const bolr_dense_operator *op, bolr_const_vector_view state, bolr_vector_view scores, bolr_workspace *workspace) {
    (void) workspace;
    if (op == NULL) return BOLR_INVALID_ARGUMENT;
    return bolr_matvec((bolr_const_matrix_view){op->data, op->rows, op->cols, op->cols, 1}, state, scores);
}

bolr_status bolr_dense_operator_transpose(const bolr_dense_operator *op, bolr_const_vector_view score_vector, bolr_vector_view state_vector, bolr_workspace *workspace) {
    (void) workspace;
    if (op == NULL) return BOLR_INVALID_ARGUMENT;
    return bolr_matvec_transpose((bolr_const_matrix_view){op->data, op->rows, op->cols, op->cols, 1}, score_vector, state_vector);
}
