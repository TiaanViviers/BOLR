#include "bolr/target.h"
#include "bolr/math.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    bolr_real value;
    bolr_index index;
} bolr_value_index_pair;

static int compare_real_ascending(const void *left, const void *right) {
    const bolr_real a = *(const bolr_real *) left;
    const bolr_real b = *(const bolr_real *) right;
    return (a > b) - (a < b);
}

static int compare_pair_descending(const void *left, const void *right) {
    const bolr_value_index_pair *a = (const bolr_value_index_pair *) left;
    const bolr_value_index_pair *b = (const bolr_value_index_pair *) right;
    if (a->value < b->value) return 1;
    if (a->value > b->value) return -1;
    return (a->index > b->index) - (a->index < b->index);
}

static bolr_status copy_sorted_values(bolr_const_vector_view input, bolr_real *buffer) {
    bolr_index i;
    if ((bolr_vector_view_validate(input) != BOLR_OK) || (buffer == NULL)) return BOLR_INVALID_ARGUMENT;
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

static bolr_status compute_soft_target_tolerance(
    bolr_const_vector_view utilities,
    const bolr_candidate_a_target_config *config,
    bolr_real *out_tolerance
) {
    bolr_real *sorted = NULL;
    bolr_real median;
    bolr_real mad;
    bolr_real robust_std;
    bolr_index i;
    bolr_status status;
    if ((config == NULL) || (out_tolerance == NULL)) return BOLR_INVALID_ARGUMENT;
    sorted = (bolr_real *) malloc((size_t) utilities.length * sizeof(bolr_real));
    if (sorted == NULL) return BOLR_ALLOCATION_FAILED;
    status = copy_sorted_values(utilities, sorted);
    if (status != BOLR_OK) {
        free(sorted);
        return status;
    }
    median = median_from_sorted(sorted, utilities.length);
    for (i = 0; i < utilities.length; ++i) sorted[i] = fabs(utilities.data[i * utilities.stride] - median);
    qsort(sorted, (size_t) utilities.length, sizeof(bolr_real), compare_real_ascending);
    mad = median_from_sorted(sorted, utilities.length);
    robust_std = mad * 1.4826;
    if (robust_std < config->min_scale) robust_std = config->min_scale;
    *out_tolerance = fmax(config->absolute_tolerance, config->relative_tolerance * robust_std);
    free(sorted);
    return BOLR_OK;
}

static bolr_status collapse_by_tolerance(
    bolr_const_vector_view utilities,
    bolr_real tolerance,
    bolr_real *collapsed,
    bolr_index *out_group_count,
    bolr_index *out_highest_group_count
) {
    bolr_value_index_pair *pairs = NULL;
    bolr_real current_sum;
    bolr_real anchor;
    bolr_index i;
    bolr_index group_start;
    bolr_index group_count = 0;
    bolr_index highest_group = 0;
    if ((collapsed == NULL) || (out_group_count == NULL) || (out_highest_group_count == NULL)) return BOLR_INVALID_ARGUMENT;
    pairs = (bolr_value_index_pair *) malloc((size_t) utilities.length * sizeof(bolr_value_index_pair));
    if (pairs == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < utilities.length; ++i) {
        pairs[i].value = utilities.data[i * utilities.stride];
        pairs[i].index = i;
    }
    qsort(pairs, (size_t) utilities.length, sizeof(bolr_value_index_pair), compare_pair_descending);
    group_start = 0;
    anchor = pairs[0].value;
    current_sum = pairs[0].value;
    for (i = 1; i < utilities.length; ++i) {
        if (fabs(pairs[i].value - anchor) <= tolerance) {
            current_sum += pairs[i].value;
            continue;
        }
        {
            const bolr_index group_size = i - group_start;
            const bolr_real mean_value = current_sum / (bolr_real) group_size;
            bolr_index j;
            for (j = group_start; j < i; ++j) collapsed[pairs[j].index] = mean_value;
            if (group_size > highest_group) highest_group = group_size;
            group_count += 1;
        }
        group_start = i;
        anchor = pairs[i].value;
        current_sum = pairs[i].value;
    }
    {
        const bolr_index group_size = utilities.length - group_start;
        const bolr_real mean_value = current_sum / (bolr_real) group_size;
        bolr_index j;
        for (j = group_start; j < utilities.length; ++j) collapsed[pairs[j].index] = mean_value;
        if (group_size > highest_group) highest_group = group_size;
        group_count += 1;
    }
    *out_group_count = group_count;
    *out_highest_group_count = highest_group;
    free(pairs);
    return BOLR_OK;
}

static bolr_status robust_transform(
    bolr_const_vector_view collapsed,
    const bolr_candidate_a_target_config *config,
    bolr_real *transformed,
    bolr_real *out_scale,
    bolr_real *out_clipping_fraction
) {
    bolr_real *sorted = NULL;
    bolr_real *absolute_centered = NULL;
    bolr_real median;
    bolr_real mad;
    bolr_real q75;
    bolr_real q25;
    bolr_real iqr;
    bolr_real scale;
    bolr_index i;
    bolr_index clipped = 0;
    bolr_status status;
    if ((config == NULL) || (transformed == NULL) || (out_scale == NULL) || (out_clipping_fraction == NULL)) return BOLR_INVALID_ARGUMENT;
    sorted = (bolr_real *) malloc((size_t) collapsed.length * sizeof(bolr_real));
    absolute_centered = (bolr_real *) malloc((size_t) collapsed.length * sizeof(bolr_real));
    if ((sorted == NULL) || (absolute_centered == NULL)) {
        free(sorted);
        free(absolute_centered);
        return BOLR_ALLOCATION_FAILED;
    }
    status = copy_sorted_values(collapsed, sorted);
    if (status != BOLR_OK) {
        free(sorted);
        free(absolute_centered);
        return status;
    }
    median = median_from_sorted(sorted, collapsed.length);
    for (i = 0; i < collapsed.length; ++i) absolute_centered[i] = fabs(collapsed.data[i * collapsed.stride] - median);
    qsort(absolute_centered, (size_t) collapsed.length, sizeof(bolr_real), compare_real_ascending);
    mad = median_from_sorted(absolute_centered, collapsed.length);
    q25 = percentile_from_sorted(sorted, collapsed.length, 25.0);
    q75 = percentile_from_sorted(sorted, collapsed.length, 75.0);
    iqr = q75 - q25;
    scale = fmax(mad, 0.7413 * iqr);
    if (scale < config->min_scale) scale = config->min_scale;
    for (i = 0; i < collapsed.length; ++i) {
        const bolr_real scaled = (collapsed.data[i * collapsed.stride] - median) / scale;
        if (fabs(scaled) > config->clip) clipped += 1;
        if (scaled > config->clip) transformed[i] = config->clip;
        else if (scaled < -config->clip) transformed[i] = -config->clip;
        else transformed[i] = scaled;
    }
    *out_scale = scale;
    *out_clipping_fraction = (bolr_real) clipped / (bolr_real) collapsed.length;
    free(sorted);
    free(absolute_centered);
    return BOLR_OK;
}

bolr_status bolr_candidate_a_target_build(
    const bolr_candidate_a_target_config *config,
    bolr_const_vector_view realised_values,
    bolr_vector_view output_target,
    bolr_real *out_update_weight,
    bolr_candidate_a_target_diagnostics *diagnostics
) {
    bolr_real *collapsed = NULL;
    bolr_real *transformed = NULL;
    bolr_real *scaled = NULL;
    bolr_real tolerance;
    bolr_real update_weight;
    bolr_real scale;
    bolr_real clipping_fraction;
    bolr_real target_sum = 0.0;
    bolr_real target_min = 0.0;
    bolr_real target_max = 0.0;
    bolr_real entropy = 0.0;
    bolr_index tolerance_group_count;
    bolr_index highest_group_count;
    bolr_index positive_count = 0;
    bolr_index i;
    bolr_status status;
    int all_irrelevant;
    if ((config == NULL) || (realised_values.length <= 0)) return BOLR_INVALID_ARGUMENT;
    status = bolr_vector_view_validate(realised_values);
    if (status != BOLR_OK) return status;
    status = bolr_mutable_vector_view_validate(output_target);
    if (status != BOLR_OK) return status;
    if (output_target.length != realised_values.length) return BOLR_INVALID_SHAPE;
    if ((config->kappa <= 0.0) || (config->eta < 0.0) || (config->clip <= 0.0) || (config->absolute_tolerance < 0.0) || (config->relative_tolerance < 0.0) || (config->min_scale <= 0.0)) return BOLR_INVALID_ARGUMENT;
    collapsed = (bolr_real *) malloc((size_t) realised_values.length * sizeof(bolr_real));
    transformed = (bolr_real *) malloc((size_t) realised_values.length * sizeof(bolr_real));
    scaled = (bolr_real *) malloc((size_t) realised_values.length * sizeof(bolr_real));
    if ((collapsed == NULL) || (transformed == NULL) || (scaled == NULL)) {
        free(collapsed);
        free(transformed);
        free(scaled);
        return BOLR_ALLOCATION_FAILED;
    }
    status = compute_soft_target_tolerance(realised_values, config, &tolerance);
    if (status != BOLR_OK) goto cleanup;
    status = collapse_by_tolerance(realised_values, tolerance, collapsed, &tolerance_group_count, &highest_group_count);
    if (status != BOLR_OK) goto cleanup;
    status = robust_transform((bolr_const_vector_view){collapsed, realised_values.length, 1}, config, transformed, &scale, &clipping_fraction);
    if (status != BOLR_OK) goto cleanup;
    for (i = 0; i < realised_values.length; ++i) {
        if (realised_values.data[i * realised_values.stride] > 0.0) positive_count += 1;
        scaled[i] = config->kappa * transformed[i];
    }
    status = bolr_softmax((bolr_const_vector_view){scaled, realised_values.length, 1}, output_target);
    if (status != BOLR_OK) goto cleanup;
    all_irrelevant = (tolerance_group_count <= 1);
    update_weight = (all_irrelevant && config->no_update_if_degenerate) ? 0.0 : config->eta;
    for (i = 0; i < output_target.length; ++i) {
        const bolr_real p = output_target.data[i * output_target.stride];
        target_sum += p;
        if (i == 0 || p < target_min) target_min = p;
        if (i == 0 || p > target_max) target_max = p;
        if (p > 0.0) entropy -= p * log(p);
    }
    if (out_update_weight != NULL) *out_update_weight = update_weight;
    if (diagnostics != NULL) {
        diagnostics->candidate_count = realised_values.length;
        diagnostics->informative = !all_irrelevant;
        diagnostics->target_sum = target_sum;
        diagnostics->target_minimum = target_min;
        diagnostics->target_maximum = target_max;
        diagnostics->target_entropy = entropy;
        diagnostics->positive_candidate_count = positive_count;
        diagnostics->highest_group_count = highest_group_count;
        diagnostics->effective_temperature = config->kappa;
        diagnostics->fallback_used = all_irrelevant && config->no_update_if_degenerate;
        diagnostics->tolerance_group_count = tolerance_group_count;
        diagnostics->utility_scale = scale;
        diagnostics->clipping_fraction = clipping_fraction;
        diagnostics->all_irrelevant = all_irrelevant;
    }
    status = BOLR_OK;

cleanup:
    free(collapsed);
    free(transformed);
    free(scaled);
    return status;
}
