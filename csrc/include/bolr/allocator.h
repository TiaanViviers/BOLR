#ifndef BOLR_ALLOCATOR_H
#define BOLR_ALLOCATOR_H

#include <stddef.h>

#include "bolr/status.h"

typedef void *(*bolr_malloc_fn)(size_t size, void *context);
typedef void *(*bolr_calloc_fn)(size_t count, size_t size, void *context);
typedef void (*bolr_free_fn)(void *ptr, void *context);

typedef struct {
    bolr_malloc_fn malloc_fn;
    bolr_calloc_fn calloc_fn;
    bolr_free_fn free_fn;
    void *context;
} bolr_allocator;

const bolr_allocator *bolr_default_allocator(void);

bolr_status bolr_checked_size_mul(size_t a, size_t b, size_t *out);
bolr_status bolr_checked_size_add(size_t a, size_t b, size_t *out);

void *bolr_allocator_malloc(const bolr_allocator *allocator, size_t size);
void *bolr_allocator_calloc(const bolr_allocator *allocator, size_t count, size_t size);
void bolr_allocator_free(const bolr_allocator *allocator, void *ptr);

#endif
