#ifndef BOLR_RNG_H
#define BOLR_RNG_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/allocator.h"
#include "bolr/array.h"

typedef struct bolr_rng bolr_rng;
typedef struct bolr_rng_checkpoint bolr_rng_checkpoint;

typedef struct {
    uint64_t seed;
    uint64_t stream;
} bolr_rng_seed;

typedef struct {
    uint32_t schema_version;
    uint32_t algorithm_family;
    uint32_t algorithm_version;
    uint32_t pcg_variant;
    uint32_t ziggurat_layers;
    uint64_t table_hash;
    uint64_t seed;
    uint64_t stream;
    uint64_t u32_draw_count;
    uint64_t uniform_draw_count;
    uint64_t normal_draw_count;
} bolr_rng_metadata;

bolr_status bolr_rng_create(bolr_rng_seed seed, const bolr_allocator *allocator, bolr_rng **out_rng);
void bolr_rng_destroy(bolr_rng *rng);
bolr_status bolr_rng_clone(const bolr_rng *source, const bolr_allocator *allocator, bolr_rng **out_clone);
bolr_status bolr_rng_u32(bolr_rng *rng, uint32_t *out);
bolr_status bolr_rng_uniform_open01(bolr_rng *rng, bolr_real *out);
bolr_status bolr_rng_standard_normal(bolr_rng *rng, bolr_real *out);
bolr_status bolr_rng_fill_uniform_open01(bolr_rng *rng, bolr_vector_view output);
bolr_status bolr_rng_fill_standard_normal(bolr_rng *rng, bolr_vector_view output);
bolr_status bolr_rng_metadata_copy(const bolr_rng *rng, bolr_rng_metadata *out_metadata);
bolr_status bolr_rng_export(const bolr_rng *rng, const bolr_allocator *allocator, bolr_rng_checkpoint **out_checkpoint);
bolr_status bolr_rng_import(const bolr_rng_checkpoint *checkpoint, const bolr_allocator *allocator, bolr_rng **out_rng);
void bolr_rng_checkpoint_destroy(bolr_rng_checkpoint *checkpoint);
bolr_status bolr_rng_checkpoint_metadata_copy(const bolr_rng_checkpoint *checkpoint, bolr_rng_metadata *out_metadata);
bolr_status bolr_rng_checkpoint_encoded_size(const bolr_rng_checkpoint *checkpoint, size_t *out_size);
bolr_status bolr_rng_checkpoint_encode(const bolr_rng_checkpoint *checkpoint, void *output, size_t output_size, size_t *out_written);
bolr_status bolr_rng_checkpoint_decode(const void *data, size_t data_size, const bolr_allocator *allocator, bolr_rng_checkpoint **out_checkpoint);

#endif
