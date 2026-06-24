#include "bolr/state_layout.h"

#include <stddef.h>
#include <string.h>

struct bolr_state_layout {
    const bolr_allocator *allocator;
    bolr_state_block_spec *specs;
    char **names;
    bolr_index count;
    bolr_index total_dimension;
    uint64_t schema_hash;
};

static uint64_t fnv1a_update(uint64_t state, const unsigned char *data, size_t size) {
    size_t i;
    for (i = 0; i < size; ++i) { state ^= (uint64_t) data[i]; state *= 1099511628211ULL; }
    return state;
}

static uint64_t hash_layout(const bolr_state_block_spec *specs, bolr_index count) {
    uint64_t h = 1469598103934665603ULL;
    bolr_index i;
    for (i = 0; i < count; ++i) {
        h = fnv1a_update(h, (const unsigned char *) specs[i].name, strlen(specs[i].name));
        h = fnv1a_update(h, (const unsigned char *) &specs[i].start, sizeof(specs[i].start));
        h = fnv1a_update(h, (const unsigned char *) &specs[i].stop, sizeof(specs[i].stop));
        h = fnv1a_update(h, (const unsigned char *) &specs[i].rows, sizeof(specs[i].rows));
        h = fnv1a_update(h, (const unsigned char *) &specs[i].cols, sizeof(specs[i].cols));
        h = fnv1a_update(h, (const unsigned char *) &specs[i].dynamic, sizeof(specs[i].dynamic));
        h = fnv1a_update(h, (const unsigned char *) &specs[i].vectorization_order, sizeof(specs[i].vectorization_order));
    }
    return h;
}

bolr_status bolr_state_layout_create(const bolr_state_block_spec *specs, bolr_index count, const bolr_allocator *allocator, bolr_state_layout **out_layout) {
    bolr_index i;
    bolr_state_layout *layout;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((specs == NULL) || (out_layout == NULL) || (count < 0)) return BOLR_INVALID_ARGUMENT;
    *out_layout = NULL;
    layout = (bolr_state_layout *) bolr_allocator_calloc(active, 1U, sizeof(*layout));
    if (layout == NULL) return BOLR_ALLOCATION_FAILED;
    layout->allocator = active;
    layout->count = count;
    layout->specs = (bolr_state_block_spec *) bolr_allocator_calloc(active, (size_t) count, sizeof(*layout->specs));
    layout->names = (char **) bolr_allocator_calloc(active, (size_t) count, sizeof(*layout->names));
    if ((layout->specs == NULL) || (layout->names == NULL)) { bolr_state_layout_destroy(layout); return BOLR_ALLOCATION_FAILED; }
    for (i = 0; i < count; ++i) {
        size_t bytes;
        if ((specs[i].name == NULL) || (specs[i].stop < specs[i].start)) { bolr_state_layout_destroy(layout); return BOLR_INVALID_ARGUMENT; }
        if ((i > 0) && (specs[i].start != specs[i - 1].stop)) { bolr_state_layout_destroy(layout); return BOLR_INVALID_SHAPE; }
        layout->specs[i] = specs[i];
        bytes = strlen(specs[i].name) + 1U;
        layout->names[i] = (char *) bolr_allocator_malloc(active, bytes);
        if (layout->names[i] == NULL) { bolr_state_layout_destroy(layout); return BOLR_ALLOCATION_FAILED; }
        memcpy(layout->names[i], specs[i].name, bytes);
        layout->specs[i].name = layout->names[i];
    }
    layout->total_dimension = (count == 0) ? 0 : specs[count - 1].stop;
    layout->schema_hash = hash_layout(layout->specs, count);
    *out_layout = layout;
    return BOLR_OK;
}

void bolr_state_layout_destroy(bolr_state_layout *layout) {
    bolr_index i;
    if (layout == NULL) return;
    if (layout->names != NULL) for (i = 0; i < layout->count; ++i) bolr_allocator_free(layout->allocator, layout->names[i]);
    bolr_allocator_free(layout->allocator, layout->names);
    bolr_allocator_free(layout->allocator, layout->specs);
    bolr_allocator_free(layout->allocator, layout);
}

bolr_index bolr_state_layout_block_count(const bolr_state_layout *layout) { return (layout == NULL) ? -1 : layout->count; }
bolr_index bolr_state_layout_total_dimension(const bolr_state_layout *layout) { return (layout == NULL) ? -1 : layout->total_dimension; }
uint64_t bolr_state_layout_schema_hash(const bolr_state_layout *layout) { return (layout == NULL) ? 0ULL : layout->schema_hash; }

bolr_status bolr_state_layout_block_index(const bolr_state_layout *layout, const char *name, bolr_index *out_index) {
    bolr_index i;
    if ((layout == NULL) || (name == NULL) || (out_index == NULL)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < layout->count; ++i) if (strcmp(layout->specs[i].name, name) == 0) { *out_index = i; return BOLR_OK; }
    return BOLR_INVALID_ARGUMENT;
}

bolr_status bolr_state_layout_block_spec(const bolr_state_layout *layout, bolr_index index, bolr_state_block_spec *out_spec) {
    if ((layout == NULL) || (out_spec == NULL) || (index < 0) || (index >= layout->count)) return BOLR_INVALID_ARGUMENT;
    *out_spec = layout->specs[index];
    return BOLR_OK;
}
