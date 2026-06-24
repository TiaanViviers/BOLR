#include "bolr/score.h"
#include "bolr/linalg.h"

#include <stddef.h>

struct bolr_context_operator {
    const bolr_allocator *allocator;
    bolr_real *basis;
    bolr_index rows;
    bolr_index candidate_dim;
    bolr_index context_dim;
};

bolr_status bolr_context_operator_create_copy(bolr_const_matrix_view candidate_basis, bolr_index context_dim, const bolr_allocator *allocator, bolr_context_operator **out_operator) {
    size_t bytes;
    bolr_index r, c;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_context_operator *op;
    if ((out_operator == NULL) || (context_dim <= 0) || (bolr_matrix_view_validate(candidate_basis) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    *out_operator = NULL;
    op = (bolr_context_operator *) bolr_allocator_calloc(active, 1U, sizeof(*op));
    if (op == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_checked_size_mul((size_t) (candidate_basis.rows * candidate_basis.cols), sizeof(bolr_real), &bytes) != BOLR_OK) { bolr_context_operator_destroy(op); return BOLR_DIMENSION_OVERFLOW; }
    op->basis = (bolr_real *) bolr_allocator_malloc(active, bytes);
    if (op->basis == NULL) { bolr_context_operator_destroy(op); return BOLR_ALLOCATION_FAILED; }
    op->allocator = active; op->rows = candidate_basis.rows; op->candidate_dim = candidate_basis.cols; op->context_dim = context_dim;
    for (r = 0; r < candidate_basis.rows; ++r) for (c = 0; c < candidate_basis.cols; ++c) op->basis[r * candidate_basis.cols + c] = candidate_basis.data[r * candidate_basis.row_stride + c * candidate_basis.col_stride];
    *out_operator = op;
    return BOLR_OK;
}

void bolr_context_operator_destroy(bolr_context_operator *op) { if (op == NULL) return; bolr_allocator_free(op->allocator, op->basis); bolr_allocator_free(op->allocator, op); }
bolr_index bolr_context_operator_candidate_dim(const bolr_context_operator *op) { return (op == NULL) ? -1 : op->candidate_dim; }
bolr_index bolr_context_operator_context_dim(const bolr_context_operator *op) { return (op == NULL) ? -1 : op->context_dim; }

bolr_status bolr_context_operator_forward(const bolr_context_operator *op, bolr_const_vector_view state, bolr_const_vector_view context, bolr_vector_view scores, bolr_workspace *workspace) {
    bolr_index row, cand, ctx;
    bolr_vector_view tmp;
    bolr_status status;
    if ((op == NULL) || (context.length != op->context_dim) || (state.length != op->candidate_dim * op->context_dim) || (scores.length != op->rows)) return BOLR_INVALID_SHAPE;
    if (workspace == NULL) return BOLR_INVALID_ARGUMENT;
    status = bolr_workspace_context_buffer(workspace, op->candidate_dim, &tmp);
    if (status != BOLR_OK) return status;
    for (cand = 0; cand < op->candidate_dim; ++cand) {
        bolr_real sum = 0.0;
        for (ctx = 0; ctx < op->context_dim; ++ctx) sum += state.data[(cand + ctx * op->candidate_dim) * state.stride] * context.data[ctx * context.stride];
        tmp.data[cand] = sum;
    }
    for (row = 0; row < op->rows; ++row) {
        bolr_real sum = 0.0;
        for (cand = 0; cand < op->candidate_dim; ++cand) sum += op->basis[row * op->candidate_dim + cand] * tmp.data[cand];
        scores.data[row * scores.stride] = sum;
    }
    return BOLR_OK;
}

bolr_status bolr_context_operator_transpose(const bolr_context_operator *op, bolr_const_vector_view score_vector, bolr_const_vector_view context, bolr_vector_view state_vector, bolr_workspace *workspace) {
    bolr_index cand, ctx, row;
    bolr_vector_view tmp;
    bolr_status status;
    if ((op == NULL) || (context.length != op->context_dim) || (score_vector.length != op->rows) || (state_vector.length != op->candidate_dim * op->context_dim)) return BOLR_INVALID_SHAPE;
    if (workspace == NULL) return BOLR_INVALID_ARGUMENT;
    status = bolr_workspace_context_buffer(workspace, op->candidate_dim, &tmp);
    if (status != BOLR_OK) return status;
    for (cand = 0; cand < op->candidate_dim; ++cand) {
        bolr_real sum = 0.0;
        for (row = 0; row < op->rows; ++row) sum += op->basis[row * op->candidate_dim + cand] * score_vector.data[row * score_vector.stride];
        tmp.data[cand] = sum;
    }
    for (ctx = 0; ctx < op->context_dim; ++ctx) for (cand = 0; cand < op->candidate_dim; ++cand) state_vector.data[(cand + ctx * op->candidate_dim) * state_vector.stride] = tmp.data[cand] * context.data[ctx * context.stride];
    return BOLR_OK;
}
