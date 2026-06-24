#include "bolr/score.h"

#include <stddef.h>

struct bolr_graph_operator { const bolr_allocator *allocator; bolr_dense_operator *dense; };

bolr_status bolr_graph_operator_create_copy(bolr_const_matrix_view basis, const bolr_allocator *allocator, bolr_graph_operator **out_operator) {
    bolr_graph_operator *op;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if (out_operator == NULL) return BOLR_INVALID_ARGUMENT;
    *out_operator = NULL;
    op = (bolr_graph_operator *) bolr_allocator_calloc(active, 1U, sizeof(*op));
    if (op == NULL) return BOLR_ALLOCATION_FAILED;
    op->allocator = active;
    if (bolr_dense_operator_create_copy(basis, active, &op->dense) != BOLR_OK) { bolr_graph_operator_destroy(op); return BOLR_ALLOCATION_FAILED; }
    *out_operator = op;
    return BOLR_OK;
}

void bolr_graph_operator_destroy(bolr_graph_operator *op) { if (op == NULL) return; bolr_dense_operator_destroy(op->dense); bolr_allocator_free(op->allocator, op); }
bolr_status bolr_graph_operator_forward(const bolr_graph_operator *op, bolr_const_vector_view state, bolr_vector_view scores, bolr_workspace *workspace) { if (op == NULL) return BOLR_INVALID_ARGUMENT; return bolr_dense_operator_forward(op->dense, state, scores, workspace); }
bolr_status bolr_graph_operator_transpose(const bolr_graph_operator *op, bolr_const_vector_view score_vector, bolr_vector_view state_vector, bolr_workspace *workspace) { if (op == NULL) return BOLR_INVALID_ARGUMENT; return bolr_dense_operator_transpose(op->dense, score_vector, state_vector, workspace); }
