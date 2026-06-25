#ifndef BOLR_PARTITION_H
#define BOLR_PARTITION_H

#include "bolr/allocator.h"
#include "bolr/array.h"

enum {
    BOLR_ORDERED_PARTITION_SCALE_MAD = 1,
    BOLR_ORDERED_PARTITION_SCALE_IQR = 2,
    BOLR_ORDERED_PARTITION_SCALE_MAX = 3
};

enum {
    BOLR_ORDERED_PARTITION_ALWAYS_RELATIVE = 1,
    BOLR_ORDERED_PARTITION_NO_UPDATE = 2,
    BOLR_ORDERED_PARTITION_REDUCED_WEIGHT = 3
};

typedef struct {
    bolr_real absolute_tolerance;
    bolr_real relative_tolerance;
    bolr_real execution_tolerance;
    int robust_scale_mode;
    bolr_real scale_floor;
} bolr_ordered_partition_tolerance_config;

typedef struct {
    bolr_ordered_partition_tolerance_config tolerance;
    bolr_real positive_threshold;
    int all_irrelevant_policy;
    bolr_real reduced_weight;
} bolr_ordered_partition_config;

typedef struct bolr_ordered_partition bolr_ordered_partition;

typedef struct {
    bolr_index candidate_count;
    bolr_index group_count;
    bolr_real tolerance;
    bolr_real utility_maximum;
    bolr_real utility_median;
    bolr_real utility_scale;
    int all_irrelevant;
    bolr_real update_weight;
    bolr_index possible_pair_count;
    bolr_index largest_upper_partition;
    bolr_index partition_complexity_proxy;
    bolr_index high_group_size;
    bolr_index middle_group_size;
    bolr_index low_group_size;
} bolr_ordered_partition_diagnostics;

bolr_status bolr_ordered_partition_build(
    const bolr_ordered_partition_config *config,
    bolr_const_vector_view utilities,
    const bolr_allocator *allocator,
    bolr_ordered_partition **out_partition
);

bolr_status bolr_ordered_partition_create_copy(
    const bolr_index *group_offsets,
    const bolr_index *group_indices,
    bolr_index group_count,
    const bolr_index *candidate_to_group,
    bolr_index candidate_count,
    bolr_real tolerance,
    bolr_real utility_maximum,
    bolr_real utility_median,
    bolr_real utility_scale,
    int all_irrelevant,
    bolr_real update_weight,
    const bolr_allocator *allocator,
    bolr_ordered_partition **out_partition
);

void bolr_ordered_partition_destroy(bolr_ordered_partition *partition);

bolr_index bolr_ordered_partition_candidate_count(const bolr_ordered_partition *partition);
bolr_index bolr_ordered_partition_group_count(const bolr_ordered_partition *partition);
bolr_real bolr_ordered_partition_tolerance(const bolr_ordered_partition *partition);
bolr_real bolr_ordered_partition_update_weight(const bolr_ordered_partition *partition);
int bolr_ordered_partition_all_irrelevant(const bolr_ordered_partition *partition);
bolr_index bolr_ordered_partition_possible_pair_count(const bolr_ordered_partition *partition);

bolr_status bolr_ordered_partition_copy_candidate_to_group(const bolr_ordered_partition *partition, bolr_index *output, bolr_index length);
bolr_status bolr_ordered_partition_copy_group_offsets(const bolr_ordered_partition *partition, bolr_index *output, bolr_index length);
bolr_status bolr_ordered_partition_copy_group_indices(const bolr_ordered_partition *partition, bolr_index *output, bolr_index length);
bolr_status bolr_ordered_partition_get_diagnostics(const bolr_ordered_partition *partition, bolr_ordered_partition_diagnostics *out_diagnostics);

#endif
