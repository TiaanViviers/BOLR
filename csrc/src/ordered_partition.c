#include "bolr/partition.h"

#include <math.h>
#include <stdlib.h>

struct bolr_ordered_partition {
    const bolr_allocator *allocator;
    bolr_index candidate_count;
    bolr_index group_count;
    bolr_index *group_offsets;
    bolr_index *group_indices;
    bolr_index *candidate_to_group;
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
};

static int compare_real_ascending(const void *left, const void *right) {
    const bolr_real a = *(const bolr_real *) left;
    const bolr_real b = *(const bolr_real *) right;
    return (a > b) - (a < b);
}

static bolr_status copy_sorted_values(bolr_const_vector_view input, bolr_real *buffer) {
    bolr_index i;
    if ((buffer == NULL) || (bolr_vector_view_validate(input) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < input.length; ++i) {
        const bolr_real value = input.data[i * input.stride];
        if (!isfinite(value)) return BOLR_NONFINITE_INPUT;
        buffer[i] = value;
    }
    qsort(buffer, (size_t) input.length, sizeof(bolr_real), compare_real_ascending);
    return BOLR_OK;
}

static bolr_real median_from_sorted(const bolr_real *sorted, bolr_index length) {
    if ((length % 2) != 0) return sorted[length / 2];
    return 0.5 * (sorted[(length / 2) - 1] + sorted[length / 2]);
}

static bolr_real percentile_from_sorted(const bolr_real *sorted, bolr_index length, bolr_real q) {
    const bolr_real scaled = (q / 100.0) * (bolr_real) (length - 1);
    const bolr_index lower = (bolr_index) floor(scaled);
    const bolr_index upper = (bolr_index) ceil(scaled);
    const bolr_real fraction = scaled - (bolr_real) lower;
    if (upper <= lower) return sorted[lower];
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

static bolr_status compute_tolerance_and_scale(
    bolr_const_vector_view utilities,
    const bolr_ordered_partition_tolerance_config *config,
    bolr_real *out_tolerance,
    bolr_real *out_scale,
    bolr_real *out_median
) {
    bolr_real *sorted = NULL;
    bolr_real *absolute_centered = NULL;
    bolr_real median;
    bolr_real mad_scale;
    bolr_real q75;
    bolr_real q25;
    bolr_real iqr_scale;
    bolr_real scale;
    bolr_index i;
    bolr_status status;
    if ((config == NULL) || (out_tolerance == NULL) || (out_scale == NULL) || (out_median == NULL)) return BOLR_INVALID_ARGUMENT;
    sorted = (bolr_real *) malloc((size_t) utilities.length * sizeof(bolr_real));
    absolute_centered = (bolr_real *) malloc((size_t) utilities.length * sizeof(bolr_real));
    if ((sorted == NULL) || (absolute_centered == NULL)) {
        free(sorted);
        free(absolute_centered);
        return BOLR_ALLOCATION_FAILED;
    }
    status = copy_sorted_values(utilities, sorted);
    if (status != BOLR_OK) {
        free(sorted);
        free(absolute_centered);
        return status;
    }
    median = median_from_sorted(sorted, utilities.length);
    for (i = 0; i < utilities.length; ++i) absolute_centered[i] = fabs(utilities.data[i * utilities.stride] - median);
    qsort(absolute_centered, (size_t) utilities.length, sizeof(bolr_real), compare_real_ascending);
    mad_scale = 1.4826 * median_from_sorted(absolute_centered, utilities.length);
    q25 = percentile_from_sorted(sorted, utilities.length, 25.0);
    q75 = percentile_from_sorted(sorted, utilities.length, 75.0);
    iqr_scale = (q75 > q25) ? ((q75 - q25) / 1.349) : 0.0;
    if (config->robust_scale_mode == BOLR_ORDERED_PARTITION_SCALE_MAD) scale = fmax(mad_scale, config->scale_floor);
    else if (config->robust_scale_mode == BOLR_ORDERED_PARTITION_SCALE_IQR) scale = fmax(iqr_scale, config->scale_floor);
    else if (config->robust_scale_mode == BOLR_ORDERED_PARTITION_SCALE_MAX) scale = fmax(fmax(mad_scale, iqr_scale), config->scale_floor);
    else {
        free(sorted);
        free(absolute_centered);
        return BOLR_INVALID_ARGUMENT;
    }
    *out_tolerance = fmax(fmax(config->absolute_tolerance, config->relative_tolerance * scale), config->execution_tolerance);
    *out_scale = scale;
    *out_median = median;
    free(sorted);
    free(absolute_centered);
    return BOLR_OK;
}

static bolr_status allocate_partition_arrays(
    const bolr_allocator *allocator,
    bolr_index candidate_count,
    bolr_index group_count,
    bolr_ordered_partition *partition
) {
    partition->group_offsets = (bolr_index *) bolr_allocator_malloc(allocator, (size_t) (group_count + 1) * sizeof(bolr_index));
    partition->group_indices = (bolr_index *) bolr_allocator_malloc(allocator, (size_t) candidate_count * sizeof(bolr_index));
    partition->candidate_to_group = (bolr_index *) bolr_allocator_malloc(allocator, (size_t) candidate_count * sizeof(bolr_index));
    if ((partition->group_offsets == NULL) || (partition->group_indices == NULL) || (partition->candidate_to_group == NULL)) return BOLR_ALLOCATION_FAILED;
    return BOLR_OK;
}

static bolr_status finalize_partition_metadata(bolr_ordered_partition *partition) {
    bolr_index a;
    bolr_index largest_upper = 0;
    bolr_index possible_pair_count = 0;
    for (a = 0; a < partition->group_count; ++a) {
        const bolr_index group_size = partition->group_offsets[a + 1] - partition->group_offsets[a];
        bolr_index b;
        if ((a + 1) < partition->group_count && group_size > largest_upper) largest_upper = group_size;
        for (b = a + 1; b < partition->group_count; ++b) {
            const bolr_index lower_size = partition->group_offsets[b + 1] - partition->group_offsets[b];
            possible_pair_count += group_size * lower_size;
        }
    }
    partition->possible_pair_count = possible_pair_count;
    partition->largest_upper_partition = largest_upper;
    partition->partition_complexity_proxy = partition->candidate_count + (largest_upper * largest_upper * largest_upper);
    if (partition->group_count >= 1) partition->high_group_size = partition->group_offsets[1] - partition->group_offsets[0];
    else partition->high_group_size = 0;
    partition->middle_group_size = (partition->group_count == 3) ? (partition->group_offsets[2] - partition->group_offsets[1]) : 0;
    partition->low_group_size = (partition->group_count >= 2) ? (partition->group_offsets[partition->group_count] - partition->group_offsets[partition->group_count - 1]) : 0;
    return BOLR_OK;
}

bolr_status bolr_ordered_partition_build(
    const bolr_ordered_partition_config *config,
    bolr_const_vector_view utilities,
    const bolr_allocator *allocator,
    bolr_ordered_partition **out_partition
) {
    bolr_ordered_partition *partition = NULL;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_real tolerance;
    bolr_real scale;
    bolr_real utility_median;
    bolr_real utility_maximum;
    bolr_index i;
    bolr_index offset = 0;
    bolr_index current_group = 0;
    bolr_index group_count = 0;
    int has_high = 0;
    int has_middle = 0;
    int has_low = 0;
    bolr_index high_count = 0;
    bolr_index middle_count = 0;
    bolr_index low_count = 0;
    bolr_status status;
    if ((config == NULL) || (out_partition == NULL) || (utilities.length <= 0)) return BOLR_INVALID_ARGUMENT;
    *out_partition = NULL;
    if ((config->tolerance.absolute_tolerance < 0.0) || (config->tolerance.relative_tolerance < 0.0) || (config->tolerance.execution_tolerance < 0.0) || (config->tolerance.scale_floor <= 0.0)) return BOLR_INVALID_ARGUMENT;
    if ((config->all_irrelevant_policy != BOLR_ORDERED_PARTITION_ALWAYS_RELATIVE) && (config->all_irrelevant_policy != BOLR_ORDERED_PARTITION_NO_UPDATE) && (config->all_irrelevant_policy != BOLR_ORDERED_PARTITION_REDUCED_WEIGHT)) return BOLR_INVALID_ARGUMENT;
    if ((config->reduced_weight < 0.0) || (config->reduced_weight > 1.0)) return BOLR_INVALID_ARGUMENT;
    status = compute_tolerance_and_scale(utilities, &config->tolerance, &tolerance, &scale, &utility_median);
    if (status != BOLR_OK) return status;
    utility_maximum = utilities.data[0];
    for (i = 0; i < utilities.length; ++i) {
        const bolr_real value = utilities.data[i * utilities.stride];
        if (value > utility_maximum) utility_maximum = value;
        if (value >= utility_maximum - tolerance) high_count += 1;
    }
    {
        const bolr_real threshold = utility_maximum - tolerance;
        for (i = 0; i < utilities.length; ++i) {
            const bolr_real value = utilities.data[i * utilities.stride];
            if (value >= threshold) continue;
            if (value > config->positive_threshold) middle_count += 1;
            else low_count += 1;
        }
    }
    has_high = (high_count > 0);
    has_middle = (middle_count > 0);
    has_low = (low_count > 0);
    group_count = (has_high ? 1 : 0) + (has_middle ? 1 : 0) + (has_low ? 1 : 0);
    partition = (bolr_ordered_partition *) bolr_allocator_calloc(active, 1U, sizeof(*partition));
    if (partition == NULL) return BOLR_ALLOCATION_FAILED;
    partition->allocator = active;
    partition->candidate_count = utilities.length;
    partition->group_count = group_count;
    partition->tolerance = tolerance;
    partition->utility_maximum = utility_maximum;
    partition->utility_median = utility_median;
    partition->utility_scale = scale;
    status = allocate_partition_arrays(active, utilities.length, group_count, partition);
    if (status != BOLR_OK) {
        bolr_ordered_partition_destroy(partition);
        return status;
    }
    for (i = 0; i < utilities.length; ++i) partition->candidate_to_group[i] = -1;
    partition->group_offsets[0] = 0;
    if (has_high) {
        const bolr_real threshold = utility_maximum - tolerance;
        for (i = 0; i < utilities.length; ++i) {
            if (utilities.data[i * utilities.stride] >= threshold) {
                partition->group_indices[offset] = i;
                partition->candidate_to_group[i] = current_group;
                offset += 1;
            }
        }
        current_group += 1;
        partition->group_offsets[current_group] = offset;
    }
    if (has_middle) {
        const bolr_real threshold = utility_maximum - tolerance;
        for (i = 0; i < utilities.length; ++i) {
            const bolr_real value = utilities.data[i * utilities.stride];
            if ((value >= threshold) || !(value > config->positive_threshold)) continue;
            partition->group_indices[offset] = i;
            partition->candidate_to_group[i] = current_group;
            offset += 1;
        }
        current_group += 1;
        partition->group_offsets[current_group] = offset;
    }
    if (has_low) {
        const bolr_real threshold = utility_maximum - tolerance;
        for (i = 0; i < utilities.length; ++i) {
            const bolr_real value = utilities.data[i * utilities.stride];
            if ((value >= threshold) || (value > config->positive_threshold)) continue;
            partition->group_indices[offset] = i;
            partition->candidate_to_group[i] = current_group;
            offset += 1;
        }
        current_group += 1;
        partition->group_offsets[current_group] = offset;
    }
    for (i = 0; i < utilities.length; ++i) {
        if (partition->candidate_to_group[i] < 0) {
            bolr_ordered_partition_destroy(partition);
            return BOLR_NUMERICAL_FAILURE;
        }
    }
    partition->all_irrelevant = (group_count <= 1);
    if (!partition->all_irrelevant) {
        int any_positive = 0;
        for (i = 0; i < utilities.length; ++i) {
            if (utilities.data[i * utilities.stride] > config->positive_threshold) {
                any_positive = 1;
                break;
            }
        }
        if (!any_positive) partition->all_irrelevant = 1;
    }
    if (partition->all_irrelevant) {
        if (config->all_irrelevant_policy == BOLR_ORDERED_PARTITION_NO_UPDATE) partition->update_weight = 0.0;
        else if (config->all_irrelevant_policy == BOLR_ORDERED_PARTITION_REDUCED_WEIGHT) partition->update_weight = config->reduced_weight;
        else partition->update_weight = 1.0;
    } else partition->update_weight = 1.0;
    finalize_partition_metadata(partition);
    *out_partition = partition;
    return BOLR_OK;
}

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
) {
    bolr_ordered_partition *partition = NULL;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index i;
    if ((group_offsets == NULL) || (group_indices == NULL) || (candidate_to_group == NULL) || (out_partition == NULL) || (candidate_count <= 0) || (group_count <= 0)) return BOLR_INVALID_ARGUMENT;
    *out_partition = NULL;
    partition = (bolr_ordered_partition *) bolr_allocator_calloc(active, 1U, sizeof(*partition));
    if (partition == NULL) return BOLR_ALLOCATION_FAILED;
    partition->allocator = active;
    partition->candidate_count = candidate_count;
    partition->group_count = group_count;
    partition->tolerance = tolerance;
    partition->utility_maximum = utility_maximum;
    partition->utility_median = utility_median;
    partition->utility_scale = utility_scale;
    partition->all_irrelevant = all_irrelevant;
    partition->update_weight = update_weight;
    if (allocate_partition_arrays(active, candidate_count, group_count, partition) != BOLR_OK) {
        bolr_ordered_partition_destroy(partition);
        return BOLR_ALLOCATION_FAILED;
    }
    for (i = 0; i <= group_count; ++i) partition->group_offsets[i] = group_offsets[i];
    for (i = 0; i < candidate_count; ++i) {
        partition->group_indices[i] = group_indices[i];
        partition->candidate_to_group[i] = candidate_to_group[i];
    }
    finalize_partition_metadata(partition);
    *out_partition = partition;
    return BOLR_OK;
}

void bolr_ordered_partition_destroy(bolr_ordered_partition *partition) {
    if (partition == NULL) return;
    bolr_allocator_free(partition->allocator, partition->group_offsets);
    bolr_allocator_free(partition->allocator, partition->group_indices);
    bolr_allocator_free(partition->allocator, partition->candidate_to_group);
    bolr_allocator_free(partition->allocator, partition);
}

bolr_index bolr_ordered_partition_candidate_count(const bolr_ordered_partition *partition) {
    return (partition == NULL) ? 0 : partition->candidate_count;
}

bolr_index bolr_ordered_partition_group_count(const bolr_ordered_partition *partition) {
    return (partition == NULL) ? 0 : partition->group_count;
}

bolr_real bolr_ordered_partition_tolerance(const bolr_ordered_partition *partition) {
    return (partition == NULL) ? 0.0 : partition->tolerance;
}

bolr_real bolr_ordered_partition_update_weight(const bolr_ordered_partition *partition) {
    return (partition == NULL) ? 0.0 : partition->update_weight;
}

int bolr_ordered_partition_all_irrelevant(const bolr_ordered_partition *partition) {
    return (partition == NULL) ? 1 : partition->all_irrelevant;
}

bolr_index bolr_ordered_partition_possible_pair_count(const bolr_ordered_partition *partition) {
    return (partition == NULL) ? 0 : partition->possible_pair_count;
}

bolr_status bolr_ordered_partition_copy_candidate_to_group(const bolr_ordered_partition *partition, bolr_index *output, bolr_index length) {
    bolr_index i;
    if ((partition == NULL) || (output == NULL) || (length != partition->candidate_count)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < length; ++i) output[i] = partition->candidate_to_group[i];
    return BOLR_OK;
}

bolr_status bolr_ordered_partition_copy_group_offsets(const bolr_ordered_partition *partition, bolr_index *output, bolr_index length) {
    bolr_index i;
    if ((partition == NULL) || (output == NULL) || (length != (partition->group_count + 1))) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < length; ++i) output[i] = partition->group_offsets[i];
    return BOLR_OK;
}

bolr_status bolr_ordered_partition_copy_group_indices(const bolr_ordered_partition *partition, bolr_index *output, bolr_index length) {
    bolr_index i;
    if ((partition == NULL) || (output == NULL) || (length != partition->candidate_count)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < length; ++i) output[i] = partition->group_indices[i];
    return BOLR_OK;
}

bolr_status bolr_ordered_partition_get_diagnostics(const bolr_ordered_partition *partition, bolr_ordered_partition_diagnostics *out_diagnostics) {
    if ((partition == NULL) || (out_diagnostics == NULL)) return BOLR_INVALID_ARGUMENT;
    out_diagnostics->candidate_count = partition->candidate_count;
    out_diagnostics->group_count = partition->group_count;
    out_diagnostics->tolerance = partition->tolerance;
    out_diagnostics->utility_maximum = partition->utility_maximum;
    out_diagnostics->utility_median = partition->utility_median;
    out_diagnostics->utility_scale = partition->utility_scale;
    out_diagnostics->all_irrelevant = partition->all_irrelevant;
    out_diagnostics->update_weight = partition->update_weight;
    out_diagnostics->possible_pair_count = partition->possible_pair_count;
    out_diagnostics->largest_upper_partition = partition->largest_upper_partition;
    out_diagnostics->partition_complexity_proxy = partition->partition_complexity_proxy;
    out_diagnostics->high_group_size = partition->high_group_size;
    out_diagnostics->middle_group_size = partition->middle_group_size;
    out_diagnostics->low_group_size = partition->low_group_size;
    return BOLR_OK;
}
