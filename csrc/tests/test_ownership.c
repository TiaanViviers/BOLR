#include "test_suite.h"

#include "bolr/allocator.h"
#include "bolr/score.h"
#include "bolr/status.h"
#include "bolr/workspace.h"

#include <stdlib.h>

struct alloc_state { int allocs; int frees; int fail_after; };
static void *count_malloc(size_t size, void *context) { struct alloc_state *s = (struct alloc_state *) context; if ((s->fail_after >= 0) && (s->allocs >= s->fail_after)) return NULL; s->allocs += 1; return malloc(size); }
static void *count_calloc(size_t count, size_t size, void *context) { struct alloc_state *s = (struct alloc_state *) context; if ((s->fail_after >= 0) && (s->allocs >= s->fail_after)) return NULL; s->allocs += 1; return calloc(count, size); }
static void count_free(void *ptr, void *context) { struct alloc_state *s = (struct alloc_state *) context; if (ptr != NULL) s->frees += 1; free(ptr); }

int test_ownership(void) {
    struct alloc_state state = {0, 0, -1};
    bolr_allocator allocator = {count_malloc, count_calloc, count_free, &state};
    bolr_workspace *workspace = NULL;
    bolr_workspace_config config = {4, 4, 4};
    if (bolr_workspace_create(&config, &allocator, &workspace) != BOLR_OK) return 1;
    bolr_workspace_destroy(workspace);
    return (state.frees == 0) ? 1 : 0;
}
