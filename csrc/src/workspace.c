#include "bolr/workspace.h"

#include <stddef.h>
#include <string.h>

struct bolr_workspace {
    const bolr_allocator *allocator;
    bolr_real *score_buffer;
    bolr_real *state_buffer;
    bolr_real *context_buffer;
    bolr_index score_capacity;
    bolr_index state_capacity;
    bolr_index context_capacity;
};

static bolr_status alloc_real_buffer(const bolr_allocator *allocator, bolr_index count, bolr_real **out) {
    size_t bytes;
    bolr_status status;
    if (count <= 0) { *out = NULL; return BOLR_OK; }
    status = bolr_checked_size_mul((size_t) count, sizeof(bolr_real), &bytes);
    if (status != BOLR_OK) return status;
    *out = (bolr_real *) bolr_allocator_calloc(allocator, 1U, bytes);
    return (*out == NULL) ? BOLR_ALLOCATION_FAILED : BOLR_OK;
}

bolr_status bolr_workspace_create(const bolr_workspace_config *config, const bolr_allocator *allocator, bolr_workspace **out_workspace) {
    bolr_workspace *workspace;
    bolr_status status;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((config == NULL) || (out_workspace == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_workspace = NULL;
    workspace = (bolr_workspace *) bolr_allocator_calloc(active, 1U, sizeof(*workspace));
    if (workspace == NULL) return BOLR_ALLOCATION_FAILED;
    workspace->allocator = active;
    workspace->score_capacity = config->score_capacity;
    workspace->state_capacity = config->state_capacity;
    workspace->context_capacity = config->context_capacity;
    status = alloc_real_buffer(active, config->score_capacity, &workspace->score_buffer); if (status != BOLR_OK) goto fail;
    status = alloc_real_buffer(active, config->state_capacity, &workspace->state_buffer); if (status != BOLR_OK) goto fail;
    status = alloc_real_buffer(active, config->context_capacity, &workspace->context_buffer); if (status != BOLR_OK) goto fail;
    *out_workspace = workspace;
    return BOLR_OK;
fail:
    bolr_workspace_destroy(workspace);
    return status;
}

void bolr_workspace_destroy(bolr_workspace *workspace) {
    if (workspace == NULL) return;
    bolr_allocator_free(workspace->allocator, workspace->score_buffer);
    bolr_allocator_free(workspace->allocator, workspace->state_buffer);
    bolr_allocator_free(workspace->allocator, workspace->context_buffer);
    bolr_allocator_free(workspace->allocator, workspace);
}

static bolr_status get_buffer(bolr_real *data, bolr_index capacity, bolr_index required, bolr_vector_view *out) {
    if ((required < 0) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    if (required > capacity) return BOLR_INVALID_SHAPE;
    *out = (bolr_vector_view){data, required, 1};
    return BOLR_OK;
}

bolr_status bolr_workspace_score_buffer(bolr_workspace *workspace, bolr_index required, bolr_vector_view *out) { if (workspace == NULL) return BOLR_INVALID_ARGUMENT; return get_buffer(workspace->score_buffer, workspace->score_capacity, required, out); }
bolr_status bolr_workspace_state_buffer(bolr_workspace *workspace, bolr_index required, bolr_vector_view *out) { if (workspace == NULL) return BOLR_INVALID_ARGUMENT; return get_buffer(workspace->state_buffer, workspace->state_capacity, required, out); }
bolr_status bolr_workspace_context_buffer(bolr_workspace *workspace, bolr_index required, bolr_vector_view *out) { if (workspace == NULL) return BOLR_INVALID_ARGUMENT; return get_buffer(workspace->context_buffer, workspace->context_capacity, required, out); }
