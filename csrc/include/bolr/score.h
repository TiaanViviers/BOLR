#ifndef BOLR_SCORE_H
#define BOLR_SCORE_H

#include "bolr/allocator.h"
#include "bolr/array.h"
#include "bolr/state_layout.h"
#include "bolr/workspace.h"

typedef struct bolr_dense_operator bolr_dense_operator;
typedef struct bolr_context_operator bolr_context_operator;
typedef struct bolr_graph_operator bolr_graph_operator;
typedef struct bolr_model bolr_model;

bolr_status bolr_dense_operator_create_copy(bolr_const_matrix_view design, const bolr_allocator *allocator, bolr_dense_operator **out_operator);
void bolr_dense_operator_destroy(bolr_dense_operator *op);
bolr_status bolr_dense_operator_forward(const bolr_dense_operator *op, bolr_const_vector_view state, bolr_vector_view scores, bolr_workspace *workspace);
bolr_status bolr_dense_operator_transpose(const bolr_dense_operator *op, bolr_const_vector_view score_vector, bolr_vector_view state_vector, bolr_workspace *workspace);
bolr_index bolr_dense_operator_rows(const bolr_dense_operator *op);
bolr_index bolr_dense_operator_cols(const bolr_dense_operator *op);

bolr_status bolr_context_operator_create_copy(bolr_const_matrix_view candidate_basis, bolr_index context_dim, const bolr_allocator *allocator, bolr_context_operator **out_operator);
void bolr_context_operator_destroy(bolr_context_operator *op);
bolr_status bolr_context_operator_forward(const bolr_context_operator *op, bolr_const_vector_view state, bolr_const_vector_view context, bolr_vector_view scores, bolr_workspace *workspace);
bolr_status bolr_context_operator_transpose(const bolr_context_operator *op, bolr_const_vector_view score_vector, bolr_const_vector_view context, bolr_vector_view state_vector, bolr_workspace *workspace);
bolr_index bolr_context_operator_candidate_dim(const bolr_context_operator *op);
bolr_index bolr_context_operator_context_dim(const bolr_context_operator *op);

bolr_status bolr_graph_operator_create_copy(bolr_const_matrix_view basis, const bolr_allocator *allocator, bolr_graph_operator **out_operator);
void bolr_graph_operator_destroy(bolr_graph_operator *op);
bolr_status bolr_graph_operator_forward(const bolr_graph_operator *op, bolr_const_vector_view state, bolr_vector_view scores, bolr_workspace *workspace);
bolr_status bolr_graph_operator_transpose(const bolr_graph_operator *op, bolr_const_vector_view score_vector, bolr_vector_view state_vector, bolr_workspace *workspace);

bolr_status bolr_model_create(const bolr_state_layout *layout, bolr_const_vector_view static_scores, const bolr_allocator *allocator, bolr_model **out_model);
void bolr_model_destroy(bolr_model *model);
bolr_status bolr_model_add_dense_block_copy(bolr_model *model, const char *name, bolr_const_matrix_view design);
bolr_status bolr_model_add_context_block_copy(bolr_model *model, const char *name, bolr_const_matrix_view candidate_basis, bolr_index context_dim);
bolr_status bolr_model_add_graph_block_copy(bolr_model *model, const char *name, bolr_const_matrix_view basis);
bolr_status bolr_model_forward(const bolr_model *model, bolr_const_vector_view state, bolr_const_vector_view context, bolr_vector_view scores, bolr_workspace *workspace);
bolr_status bolr_model_dynamic_forward(const bolr_model *model, bolr_const_vector_view state, bolr_const_vector_view context, bolr_vector_view scores, bolr_workspace *workspace);
bolr_status bolr_model_transpose(const bolr_model *model, bolr_const_vector_view score_vector, bolr_const_vector_view context, bolr_vector_view state_vector, bolr_workspace *workspace);
bolr_index bolr_model_score_count(const bolr_model *model);
bolr_index bolr_model_state_dim(const bolr_model *model);
uint64_t bolr_model_schema_hash(const bolr_model *model);
uint64_t bolr_model_state_layout_hash(const bolr_model *model);
bolr_status bolr_model_copy_static_scores(const bolr_model *model, bolr_vector_view output);

#endif
