#include "test_suite.h"

#include "bolr/allocator.h"
#include "bolr/gaussian.h"
#include "bolr/inference.h"
#include "bolr/status.h"

#include <stdlib.h>

struct alloc_state { int allocs; int frees; };
static void *count_malloc(size_t size, void *context) { struct alloc_state *s = (struct alloc_state *) context; s->allocs += 1; return malloc(size); }
static void *count_calloc(size_t count, size_t size, void *context) { struct alloc_state *s = (struct alloc_state *) context; s->allocs += 1; return calloc(count, size); }
static void count_free(void *ptr, void *context) { struct alloc_state *s = (struct alloc_state *) context; if (ptr != NULL) s->frees += 1; free(ptr); }

int test_inference_ownership(void) {
    struct alloc_state state = {0, 0};
    bolr_allocator allocator = {count_malloc, count_calloc, count_free, &state};
    bolr_real mean[] = {0.0, 0.0};
    bolr_real covariance[] = {1.0, 0.0, 0.0, 1.0};
    bolr_gaussian_state *gaussian = NULL;
    bolr_inference_workspace *workspace = NULL;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, 3ULL, 4ULL, &allocator, &gaussian) != BOLR_OK) return 1;
    if (bolr_inference_workspace_create(2, 2, &allocator, &workspace) != BOLR_OK) return 1;
    bolr_inference_workspace_destroy(workspace);
    bolr_gaussian_state_destroy(gaussian);
    return (state.frees == 0) ? 1 : 0;
}
