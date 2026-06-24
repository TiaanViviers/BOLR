#include "bolr/rng.h"

#include <stddef.h>

struct bolr_rng {
    const bolr_allocator *allocator;
    bolr_rng_seed seed;
};

bolr_status bolr_rng_create(bolr_rng_seed seed, const bolr_allocator *allocator, bolr_rng **out_rng) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_rng *rng;
    if (out_rng == NULL) return BOLR_INVALID_ARGUMENT;
    *out_rng = NULL;
    rng = (bolr_rng *) bolr_allocator_calloc(active, 1U, sizeof(*rng));
    if (rng == NULL) return BOLR_ALLOCATION_FAILED;
    rng->allocator = active;
    rng->seed = seed;
    *out_rng = rng;
    return BOLR_OK;
}
void bolr_rng_destroy(bolr_rng *rng) { if (rng == NULL) return; bolr_allocator_free(rng->allocator, rng); }
bolr_status bolr_rng_uniform_open01(bolr_rng *rng, bolr_real *out) { (void) rng; (void) out; return BOLR_UNSUPPORTED_OPERATION; }
bolr_status bolr_rng_standard_normal(bolr_rng *rng, bolr_real *out) { (void) rng; (void) out; return BOLR_UNSUPPORTED_OPERATION; }
bolr_status bolr_rng_fill_standard_normal(bolr_rng *rng, bolr_real *output, bolr_index count) { (void) rng; (void) output; (void) count; return BOLR_UNSUPPORTED_OPERATION; }
