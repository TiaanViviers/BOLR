#include "bolr/score.h"
#include "bolr/linalg.h"

#include <stddef.h>
#include <string.h>

enum block_kind { BLOCK_NONE = 0, BLOCK_DENSE = 1, BLOCK_CONTEXT = 2, BLOCK_GRAPH = 3 };

struct model_block {
    enum block_kind kind;
    bolr_state_block_spec spec;
    bolr_dense_operator *dense;
    bolr_context_operator *context;
    bolr_graph_operator *graph;
};

struct bolr_model {
    const bolr_allocator *allocator;
    bolr_state_layout *layout_copy;
    bolr_real *static_scores;
    bolr_index score_count;
    bolr_index state_dim;
    bolr_index block_count;
    uint64_t schema_hash;
    uint64_t state_layout_hash;
    struct model_block *blocks;
};

static bolr_status copy_scores(bolr_const_vector_view static_scores, const bolr_allocator *allocator, bolr_real **out) {
    size_t bytes;
    bolr_index i;
    bolr_real *data;
    if (bolr_checked_size_mul((size_t) static_scores.length, sizeof(bolr_real), &bytes) != BOLR_OK) return BOLR_DIMENSION_OVERFLOW;
    data = (bolr_real *) bolr_allocator_malloc(allocator, bytes);
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < static_scores.length; ++i) data[i] = static_scores.data[i * static_scores.stride];
    *out = data;
    return BOLR_OK;
}

static struct model_block *find_block(bolr_model *model, const char *name) {
    bolr_index i;
    for (i = 0; i < model->block_count; ++i) if (strcmp(model->blocks[i].spec.name, name) == 0) return &model->blocks[i];
    return NULL;
}

bolr_status bolr_model_create(const bolr_state_layout *layout, bolr_const_vector_view static_scores, const bolr_allocator *allocator, bolr_model **out_model) {
    bolr_model *model;
    bolr_index i;
    bolr_status status;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((layout == NULL) || (out_model == NULL) || (bolr_vector_view_validate(static_scores) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    *out_model = NULL;
    model = (bolr_model *) bolr_allocator_calloc(active, 1U, sizeof(*model));
    if (model == NULL) return BOLR_ALLOCATION_FAILED;
    model->allocator = active;
    model->block_count = bolr_state_layout_block_count(layout);
    model->score_count = static_scores.length;
    model->state_dim = bolr_state_layout_total_dimension(layout);
    model->state_layout_hash = bolr_state_layout_schema_hash(layout);
    model->schema_hash = model->state_layout_hash;
    model->blocks = (struct model_block *) bolr_allocator_calloc(active, (size_t) model->block_count, sizeof(*model->blocks));
    if (model->blocks == NULL) { bolr_model_destroy(model); return BOLR_ALLOCATION_FAILED; }
    status = copy_scores(static_scores, active, &model->static_scores); if (status != BOLR_OK) { bolr_model_destroy(model); return status; }
    for (i = 0; i < model->block_count; ++i) {
        bolr_state_block_spec spec;
        (void) bolr_state_layout_block_spec(layout, i, &spec);
        model->blocks[i].spec = spec;
    }
    model->layout_copy = NULL;
    *out_model = model;
    return BOLR_OK;
}

void bolr_model_destroy(bolr_model *model) {
    bolr_index i;
    if (model == NULL) return;
    if (model->blocks != NULL) {
        for (i = 0; i < model->block_count; ++i) {
            bolr_dense_operator_destroy(model->blocks[i].dense);
            bolr_context_operator_destroy(model->blocks[i].context);
            bolr_graph_operator_destroy(model->blocks[i].graph);
        }
    }
    bolr_allocator_free(model->allocator, model->blocks);
    bolr_allocator_free(model->allocator, model->static_scores);
    bolr_allocator_free(model->allocator, model);
}

bolr_status bolr_model_add_dense_block_copy(bolr_model *model, const char *name, bolr_const_matrix_view design) {
    struct model_block *block = find_block(model, name);
    if ((model == NULL) || (name == NULL) || (block == NULL)) return BOLR_INVALID_ARGUMENT;
    block->kind = BLOCK_DENSE;
    return bolr_dense_operator_create_copy(design, model->allocator, &block->dense);
}
bolr_status bolr_model_add_context_block_copy(bolr_model *model, const char *name, bolr_const_matrix_view candidate_basis, bolr_index context_dim) {
    struct model_block *block = find_block(model, name);
    if ((model == NULL) || (name == NULL) || (block == NULL)) return BOLR_INVALID_ARGUMENT;
    block->kind = BLOCK_CONTEXT;
    return bolr_context_operator_create_copy(candidate_basis, context_dim, model->allocator, &block->context);
}
bolr_status bolr_model_add_graph_block_copy(bolr_model *model, const char *name, bolr_const_matrix_view basis) {
    struct model_block *block = find_block(model, name);
    if ((model == NULL) || (name == NULL) || (block == NULL)) return BOLR_INVALID_ARGUMENT;
    block->kind = BLOCK_GRAPH;
    return bolr_graph_operator_create_copy(basis, model->allocator, &block->graph);
}

bolr_status bolr_model_forward(const bolr_model *model, bolr_const_vector_view state, bolr_const_vector_view context, bolr_vector_view scores, bolr_workspace *workspace) {
    bolr_index i, j;
    bolr_vector_view scratch;
    bolr_status status;
    if ((model == NULL) || (scores.length != model->score_count) || (state.length != model->state_dim) || (workspace == NULL)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < model->score_count; ++i) scores.data[i * scores.stride] = model->static_scores[i];
    status = bolr_workspace_score_buffer(workspace, model->score_count, &scratch); if (status != BOLR_OK) return status;
    for (i = 0; i < model->block_count; ++i) {
        bolr_const_vector_view slice = {state.data + model->blocks[i].spec.start * state.stride, model->blocks[i].spec.stop - model->blocks[i].spec.start, state.stride};
        if (model->blocks[i].kind == BLOCK_DENSE) status = bolr_dense_operator_forward(model->blocks[i].dense, slice, scratch, workspace);
        else if (model->blocks[i].kind == BLOCK_CONTEXT) status = bolr_context_operator_forward(model->blocks[i].context, slice, context, scratch, workspace);
        else if (model->blocks[i].kind == BLOCK_GRAPH) status = bolr_graph_operator_forward(model->blocks[i].graph, slice, scratch, workspace);
        else continue;
        if (status != BOLR_OK) return status;
        for (j = 0; j < model->score_count; ++j) scores.data[j * scores.stride] += scratch.data[j];
    }
    return BOLR_OK;
}

bolr_status bolr_model_dynamic_forward(const bolr_model *model, bolr_const_vector_view state, bolr_const_vector_view context, bolr_vector_view scores, bolr_workspace *workspace) {
    bolr_index i;
    if (model == NULL) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < model->score_count; ++i) scores.data[i * scores.stride] = 0.0;
    return bolr_model_forward(model, state, context, scores, workspace) == BOLR_OK ? bolr_axpy(-1.0, (bolr_const_vector_view){model->static_scores, model->score_count, 1}, scores) : BOLR_NUMERICAL_FAILURE;
}

bolr_status bolr_model_transpose(const bolr_model *model, bolr_const_vector_view score_vector, bolr_const_vector_view context, bolr_vector_view state_vector, bolr_workspace *workspace) {
    bolr_index i, j;
    bolr_vector_view scratch;
    bolr_status status;
    if ((model == NULL) || (score_vector.length != model->score_count) || (state_vector.length != model->state_dim) || (workspace == NULL)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < model->state_dim; ++i) state_vector.data[i * state_vector.stride] = 0.0;
    status = bolr_workspace_state_buffer(workspace, model->state_dim, &scratch); if (status != BOLR_OK) return status;
    for (i = 0; i < model->block_count; ++i) {
        bolr_vector_view slice = {state_vector.data + model->blocks[i].spec.start * state_vector.stride, model->blocks[i].spec.stop - model->blocks[i].spec.start, state_vector.stride};
        if (model->blocks[i].kind == BLOCK_DENSE) status = bolr_dense_operator_transpose(model->blocks[i].dense, score_vector, slice, workspace);
        else if (model->blocks[i].kind == BLOCK_CONTEXT) status = bolr_context_operator_transpose(model->blocks[i].context, score_vector, context, slice, workspace);
        else if (model->blocks[i].kind == BLOCK_GRAPH) status = bolr_graph_operator_transpose(model->blocks[i].graph, score_vector, slice, workspace);
        else continue;
        if (status != BOLR_OK) return status;
        for (j = 0; j < slice.length; ++j) scratch.data[model->blocks[i].spec.start + j] = slice.data[j * slice.stride];
    }
    return BOLR_OK;
}

bolr_index bolr_model_score_count(const bolr_model *model) { return (model == NULL) ? -1 : model->score_count; }
bolr_index bolr_model_state_dim(const bolr_model *model) { return (model == NULL) ? -1 : model->state_dim; }
uint64_t bolr_model_schema_hash(const bolr_model *model) { return (model == NULL) ? 0ULL : model->schema_hash; }
uint64_t bolr_model_state_layout_hash(const bolr_model *model) { return (model == NULL) ? 0ULL : model->state_layout_hash; }
bolr_status bolr_model_copy_static_scores(const bolr_model *model, bolr_vector_view output) {
    return (model == NULL) ? BOLR_INVALID_ARGUMENT : bolr_copy((bolr_const_vector_view){model->static_scores, model->score_count, 1}, output);
}
