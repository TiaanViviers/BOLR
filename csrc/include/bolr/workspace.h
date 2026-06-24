#ifndef BOLR_WORKSPACE_H
#define BOLR_WORKSPACE_H

#include "bolr/allocator.h"
#include "bolr/array.h"
#include "bolr/types.h"

typedef struct bolr_workspace bolr_workspace;

typedef struct {
    bolr_index score_capacity;
    bolr_index state_capacity;
    bolr_index context_capacity;
} bolr_workspace_config;

bolr_status bolr_workspace_create(const bolr_workspace_config *config, const bolr_allocator *allocator, bolr_workspace **out_workspace);
void bolr_workspace_destroy(bolr_workspace *workspace);
bolr_status bolr_workspace_score_buffer(bolr_workspace *workspace, bolr_index required, bolr_vector_view *out);
bolr_status bolr_workspace_state_buffer(bolr_workspace *workspace, bolr_index required, bolr_vector_view *out);
bolr_status bolr_workspace_context_buffer(bolr_workspace *workspace, bolr_index required, bolr_vector_view *out);

#endif
