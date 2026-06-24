#include "bolr/allocator.h"

#include <stddef.h>
#include <stdlib.h>

static void *default_malloc(size_t size, void *context) { (void) context; return malloc(size); }
static void *default_calloc(size_t count, size_t size, void *context) { (void) context; return calloc(count, size); }
static void default_free(void *ptr, void *context) { (void) context; free(ptr); }

static const bolr_allocator DEFAULT_ALLOCATOR = {default_malloc, default_calloc, default_free, NULL};

const bolr_allocator *bolr_default_allocator(void) { return &DEFAULT_ALLOCATOR; }

bolr_status bolr_checked_size_mul(size_t a, size_t b, size_t *out) {
    if (out == NULL) return BOLR_INVALID_ARGUMENT;
    if ((a != 0U) && (b > SIZE_MAX / a)) return BOLR_DIMENSION_OVERFLOW;
    *out = a * b;
    return BOLR_OK;
}

bolr_status bolr_checked_size_add(size_t a, size_t b, size_t *out) {
    if (out == NULL) return BOLR_INVALID_ARGUMENT;
    if (a > SIZE_MAX - b) return BOLR_DIMENSION_OVERFLOW;
    *out = a + b;
    return BOLR_OK;
}

void *bolr_allocator_malloc(const bolr_allocator *allocator, size_t size) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if (active->malloc_fn == NULL) return NULL;
    return active->malloc_fn(size, active->context);
}

void *bolr_allocator_calloc(const bolr_allocator *allocator, size_t count, size_t size) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if (active->calloc_fn == NULL) return NULL;
    return active->calloc_fn(count, size, active->context);
}

void bolr_allocator_free(const bolr_allocator *allocator, void *ptr) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((ptr != NULL) && (active->free_fn != NULL)) active->free_fn(ptr, active->context);
}
