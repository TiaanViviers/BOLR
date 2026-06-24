#ifndef BOLR_RNG_H
#define BOLR_RNG_H

#include <stdint.h>

#include "bolr/allocator.h"

typedef struct bolr_rng bolr_rng;

typedef struct {
    uint64_t seed;
    uint64_t stream;
} bolr_rng_seed;

bolr_status bolr_rng_create(bolr_rng_seed seed, const bolr_allocator *allocator, bolr_rng **out_rng);
void bolr_rng_destroy(bolr_rng *rng);
bolr_status bolr_rng_uniform_open01(bolr_rng *rng, bolr_real *out);
bolr_status bolr_rng_standard_normal(bolr_rng *rng, bolr_real *out);
bolr_status bolr_rng_fill_standard_normal(bolr_rng *rng, bolr_real *output, bolr_index count);

#endif
