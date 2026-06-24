#ifndef BOLR_STATE_LAYOUT_H
#define BOLR_STATE_LAYOUT_H

#include <stdint.h>

#include "bolr/allocator.h"
#include "bolr/types.h"

typedef struct bolr_state_layout bolr_state_layout;

typedef struct {
    const char *name;
    bolr_index start;
    bolr_index stop;
    bolr_index rows;
    bolr_index cols;
    int dynamic;
    char vectorization_order;
} bolr_state_block_spec;

bolr_status bolr_state_layout_create(const bolr_state_block_spec *specs, bolr_index count, const bolr_allocator *allocator, bolr_state_layout **out_layout);
void bolr_state_layout_destroy(bolr_state_layout *layout);
bolr_index bolr_state_layout_block_count(const bolr_state_layout *layout);
bolr_index bolr_state_layout_total_dimension(const bolr_state_layout *layout);
uint64_t bolr_state_layout_schema_hash(const bolr_state_layout *layout);
bolr_status bolr_state_layout_block_index(const bolr_state_layout *layout, const char *name, bolr_index *out_index);
bolr_status bolr_state_layout_block_spec(const bolr_state_layout *layout, bolr_index index, bolr_state_block_spec *out_spec);

#endif
