#include "bolr/decision.h"

#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

enum {
    BOLR_TIE_STAGE_PRIMARY = 1,
    BOLR_TIE_STAGE_POSTERIOR_MEAN = 2,
    BOLR_TIE_STAGE_POSTERIOR_VARIANCE = 3
};

static bolr_status break_ties(
    bolr_const_vector_view primary,
    bolr_const_vector_view score_mean,
    bolr_const_vector_view score_variance,
    bolr_index *out_index,
    bolr_decision_diagnostics *diagnostics
) {
    bolr_real max_primary = -INFINITY;
    bolr_real max_mean = -INFINITY;
    bolr_real min_variance = INFINITY;
    bolr_index selected = -1;
    bolr_index i;
    bolr_index tie_count = 0;
    if ((out_index == NULL) || (primary.length <= 0) || (primary.length != score_mean.length) || (primary.length != score_variance.length)) {
        return BOLR_INVALID_ARGUMENT;
    }
    for (i = 0; i < primary.length; ++i) {
        bolr_real value = primary.data[i * primary.stride];
        if (value > max_primary) max_primary = value;
    }
    for (i = 0; i < primary.length; ++i) {
        if (primary.data[i * primary.stride] == max_primary) {
            tie_count += 1;
            if (score_mean.data[i * score_mean.stride] > max_mean) max_mean = score_mean.data[i * score_mean.stride];
        }
    }
    for (i = 0; i < primary.length; ++i) {
        if ((primary.data[i * primary.stride] == max_primary) &&
            (score_mean.data[i * score_mean.stride] == max_mean) &&
            (score_variance.data[i * score_variance.stride] < min_variance)) {
            min_variance = score_variance.data[i * score_variance.stride];
        }
    }
    for (i = 0; i < primary.length; ++i) {
        if ((primary.data[i * primary.stride] == max_primary) &&
            (score_mean.data[i * score_mean.stride] == max_mean) &&
            (score_variance.data[i * score_variance.stride] == min_variance)) {
            selected = i;
            break;
        }
    }
    if (selected < 0) return BOLR_NUMERICAL_FAILURE;
    if (diagnostics != NULL) {
        diagnostics->tie_occurred = (tie_count > 1) ? 1 : 0;
        diagnostics->tie_break_stage_count = (tie_count > 1) ? 3 : 1;
        diagnostics->tie_break_stages[0] = BOLR_TIE_STAGE_PRIMARY;
        diagnostics->tie_break_stages[1] = BOLR_TIE_STAGE_POSTERIOR_MEAN;
        diagnostics->tie_break_stages[2] = BOLR_TIE_STAGE_POSTERIOR_VARIANCE;
    }
    *out_index = selected;
    return BOLR_OK;
}

static bolr_status decision_from_index(
    bolr_index index,
    const struct bolr_posterior_prediction *prediction,
    bolr_index region_id,
    bolr_real region_mass,
    bolr_real region_best_mass,
    bolr_decision *out_decision
) {
    if ((prediction == NULL) || (out_decision == NULL) || (index < 0) || (index >= prediction->candidate_count)) return BOLR_INVALID_ARGUMENT;
    memset(out_decision, 0, sizeof(*out_decision));
    out_decision->selected = 1;
    out_decision->selected_index = index;
    out_decision->selected_score_mean = prediction->score_mean[index];
    out_decision->selected_score_variance = prediction->score_variance[index];
    out_decision->selected_probability_best = (prediction->probability_best == NULL) ? 0.0 : prediction->probability_best[index];
    out_decision->selected_expected_rank = (prediction->expected_rank == NULL) ? 0.0 : prediction->expected_rank[index];
    out_decision->selected_region_id = region_id;
    out_decision->selected_region_mass = region_mass;
    out_decision->selected_region_probability_best = region_best_mass;
    return BOLR_OK;
}

static bolr_index find_top_k_slot(const struct bolr_posterior_prediction *prediction, bolr_index top_k) {
    bolr_index i;
    for (i = 0; i < prediction->probability_top_k_count; ++i) {
        if (prediction->probability_top_k_keys[i] == top_k) return i;
    }
    return -1;
}

bolr_status bolr_decision_policy_create(const bolr_decision_policy_config *config, const bolr_allocator *allocator, bolr_decision_policy **out_policy) {
    struct bolr_decision_policy *policy;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((config == NULL) || (out_policy == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_policy = NULL;
    policy = (struct bolr_decision_policy *) bolr_allocator_calloc(active, 1U, sizeof(*policy));
    if (policy == NULL) return BOLR_ALLOCATION_FAILED;
    policy->allocator = active;
    policy->config = *config;
    *out_policy = policy;
    return BOLR_OK;
}

void bolr_decision_policy_destroy(bolr_decision_policy *opaque) {
    struct bolr_decision_policy *policy = opaque;
    if (policy == NULL) return;
    bolr_allocator_free(policy->allocator, policy);
}

bolr_status bolr_decision_policy_apply(
    const bolr_decision_policy *policy_opaque,
    const bolr_posterior_prediction *prediction_opaque,
    const bolr_region_set *regions_opaque,
    const bolr_grid_graph *graph_opaque,
    bolr_decision *out_decision,
    bolr_decision_diagnostics *diagnostics
) {
    const struct bolr_decision_policy *policy = policy_opaque;
    const struct bolr_posterior_prediction *prediction = prediction_opaque;
    const struct bolr_region_set *regions = regions_opaque;
    const struct bolr_grid_graph *graph = graph_opaque;
    bolr_status status;
    bolr_index index = -1;
    bolr_index i;
    if ((policy == NULL) || (prediction == NULL) || (out_decision == NULL)) return BOLR_INVALID_ARGUMENT;
    if (diagnostics != NULL) memset(diagnostics, 0, sizeof(*diagnostics));
    if (policy->config.family == BOLR_DECISION_REGION) {
        bolr_real *region_primary;
        bolr_real *region_mean;
        bolr_real *region_variance;
        bolr_index region_index = -1;
        bolr_region_summary summary;
        if ((regions == NULL) || (regions->region_count <= 0)) return BOLR_INVALID_ARGUMENT;
        region_primary = (bolr_real *) malloc((size_t) regions->region_count * sizeof(*region_primary));
        region_mean = (bolr_real *) malloc((size_t) regions->region_count * sizeof(*region_mean));
        region_variance = (bolr_real *) malloc((size_t) regions->region_count * sizeof(*region_variance));
        if ((region_primary == NULL) || (region_mean == NULL) || (region_variance == NULL)) {
            free(region_primary);
            free(region_mean);
            free(region_variance);
            return BOLR_ALLOCATION_FAILED;
        }
        for (i = 0; i < regions->region_count; ++i) {
            summary = regions->summaries[i];
            region_primary[i] = (policy->config.region_selection_statistic == BOLR_REGION_STATISTIC_PROBABILITY_BEST) ? summary.probability_best_mass : summary.inclusion_mass;
            region_mean[i] = summary.maximum_score_mean;
            region_variance[i] = -(bolr_real) summary.candidate_count;
        }
        status = break_ties(
            (bolr_const_vector_view){region_primary, regions->region_count, 1},
            (bolr_const_vector_view){region_mean, regions->region_count, 1},
            (bolr_const_vector_view){region_variance, regions->region_count, 1},
            &region_index,
            diagnostics
        );
        free(region_primary);
        free(region_mean);
        free(region_variance);
        if (status != BOLR_OK) return status;
        summary = regions->summaries[region_index];
        if (policy->config.representative_policy == BOLR_REGION_REP_WEIGHTED_MEDOID) {
            bolr_real *weights;
            bolr_real objective = 0.0;
            if (graph == NULL) return BOLR_INVALID_ARGUMENT;
            weights = (bolr_real *) malloc((size_t) summary.candidate_count * sizeof(*weights));
            if (weights == NULL) return BOLR_ALLOCATION_FAILED;
            for (i = 0; i < summary.candidate_count; ++i) {
                bolr_index candidate = regions->region_candidates[summary.candidate_offset + i];
                weights[i] = regions->inclusion_probability[candidate];
            }
            status = bolr_weighted_graph_medoid(
                graph,
                regions->region_candidates + summary.candidate_offset,
                (bolr_const_vector_view){weights, summary.candidate_count, 1},
                summary.candidate_count,
                &index,
                &objective
            );
            free(weights);
            if (status != BOLR_OK) return status;
            if (diagnostics != NULL) diagnostics->medoid_objective = objective;
        } else {
            bolr_real *candidate_primary = (bolr_real *) malloc((size_t) summary.candidate_count * sizeof(*candidate_primary));
            bolr_real *candidate_mean = (bolr_real *) malloc((size_t) summary.candidate_count * sizeof(*candidate_mean));
            bolr_real *candidate_variance = (bolr_real *) malloc((size_t) summary.candidate_count * sizeof(*candidate_variance));
            bolr_index local_index = -1;
            if ((candidate_primary == NULL) || (candidate_mean == NULL) || (candidate_variance == NULL)) {
                free(candidate_primary);
                free(candidate_mean);
                free(candidate_variance);
                return BOLR_ALLOCATION_FAILED;
            }
            for (i = 0; i < summary.candidate_count; ++i) {
                bolr_index candidate = regions->region_candidates[summary.candidate_offset + i];
                candidate_mean[i] = prediction->score_mean[candidate];
                candidate_variance[i] = prediction->score_variance[candidate];
                if (policy->config.representative_policy == BOLR_REGION_REP_POSTERIOR_MEAN) {
                    candidate_primary[i] = prediction->score_mean[candidate];
                } else if (policy->config.representative_policy == BOLR_REGION_REP_PROBABILITY_BEST) {
                    if (prediction->probability_best == NULL) {
                        free(candidate_primary);
                        free(candidate_mean);
                        free(candidate_variance);
                        return BOLR_INVALID_ARGUMENT;
                    }
                    candidate_primary[i] = prediction->probability_best[candidate];
                } else {
                    bolr_index top_k = (policy->config.top_k > 0) ? policy->config.top_k : summary.candidate_count;
                    bolr_index slot = find_top_k_slot(prediction, top_k);
                    if (slot < 0) slot = find_top_k_slot(prediction, regions->top_k);
                    if (slot < 0) {
                        free(candidate_primary);
                        free(candidate_mean);
                        free(candidate_variance);
                        return BOLR_INVALID_ARGUMENT;
                    }
                    candidate_primary[i] = prediction->probability_top_k_values[slot][candidate];
                }
            }
            status = break_ties(
                (bolr_const_vector_view){candidate_primary, summary.candidate_count, 1},
                (bolr_const_vector_view){candidate_mean, summary.candidate_count, 1},
                (bolr_const_vector_view){candidate_variance, summary.candidate_count, 1},
                &local_index,
                diagnostics
            );
            free(candidate_primary);
            free(candidate_mean);
            free(candidate_variance);
            if (status != BOLR_OK) return status;
            index = regions->region_candidates[summary.candidate_offset + local_index];
        }
        if (diagnostics != NULL) diagnostics->selected_region_candidate_count = summary.candidate_count;
        return decision_from_index(index, prediction, region_index, summary.inclusion_mass, summary.probability_best_mass, out_decision);
    }
    {
        bolr_real *primary = (bolr_real *) malloc((size_t) prediction->candidate_count * sizeof(*primary));
        if (primary == NULL) return BOLR_ALLOCATION_FAILED;
        for (i = 0; i < prediction->candidate_count; ++i) {
            if (policy->config.family == BOLR_DECISION_POSTERIOR_MEAN) {
                primary[i] = prediction->score_mean[i];
            } else if (policy->config.family == BOLR_DECISION_PROBABILITY_BEST) {
                if (prediction->probability_best == NULL) {
                    free(primary);
                    return BOLR_INVALID_ARGUMENT;
                }
                primary[i] = prediction->probability_best[i];
            } else if (policy->config.family == BOLR_DECISION_PROBABILITY_TOP_K) {
                bolr_index slot = find_top_k_slot(prediction, policy->config.top_k);
                if (slot < 0) {
                    free(primary);
                    return BOLR_INVALID_ARGUMENT;
                }
                primary[i] = prediction->probability_top_k_values[slot][i];
            } else if (policy->config.family == BOLR_DECISION_EXPECTED_RANK) {
                if (prediction->expected_rank == NULL) {
                    free(primary);
                    return BOLR_INVALID_ARGUMENT;
                }
                primary[i] = -prediction->expected_rank[i];
            } else {
                if ((prediction->score_samples == NULL) || (prediction->score_sample_count <= 0)) {
                    free(primary);
                    return BOLR_INVALID_ARGUMENT;
                }
                primary[i] = prediction->score_samples[i];
            }
        }
        status = break_ties(
            (bolr_const_vector_view){primary, prediction->candidate_count, 1},
            (bolr_const_vector_view){prediction->score_mean, prediction->candidate_count, 1},
            (bolr_const_vector_view){prediction->score_variance, prediction->candidate_count, 1},
            &index,
            diagnostics
        );
        free(primary);
        if (status != BOLR_OK) return status;
    }
    return decision_from_index(index, prediction, -1, 0.0, 0.0, out_decision);
}

bolr_status bolr_realized_best_distribution(bolr_const_vector_view utilities, bolr_real tolerance, bolr_vector_view output) {
    bolr_real best = -INFINITY;
    bolr_index count = 0;
    bolr_index i;
    if ((utilities.length != output.length) || (tolerance < 0.0)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < utilities.length; ++i) if (utilities.data[i * utilities.stride] > best) best = utilities.data[i * utilities.stride];
    for (i = 0; i < output.length; ++i) {
        output.data[i * output.stride] = (utilities.data[i * utilities.stride] >= (best - tolerance)) ? 1.0 : 0.0;
        if (output.data[i * output.stride] > 0.0) count += 1;
    }
    if (count <= 0) return BOLR_NUMERICAL_FAILURE;
    for (i = 0; i < output.length; ++i) output.data[i * output.stride] /= (bolr_real) count;
    return BOLR_OK;
}

bolr_status bolr_realized_top_k_indicator(bolr_const_vector_view utilities, bolr_index top_k, bolr_vector_view output) {
    bolr_index i;
    bolr_index chosen;
    if ((utilities.length != output.length) || (top_k <= 0) || (top_k > utilities.length)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < output.length; ++i) output.data[i * output.stride] = 0.0;
    for (chosen = 0; chosen < top_k; ++chosen) {
        bolr_index best = -1;
        for (i = 0; i < utilities.length; ++i) {
            if (output.data[i * output.stride] > 0.0) continue;
            if ((best < 0) || (utilities.data[i * utilities.stride] > utilities.data[best * utilities.stride])) best = i;
        }
        output.data[best * output.stride] = 1.0;
    }
    return BOLR_OK;
}

bolr_status bolr_probability_best_brier(bolr_const_vector_view probability_best, bolr_const_vector_view utilities, bolr_real tolerance, bolr_real *out_brier) {
    bolr_real *target;
    bolr_real brier = 0.0;
    bolr_index i;
    if ((out_brier == NULL) || (probability_best.length != utilities.length)) return BOLR_INVALID_ARGUMENT;
    target = (bolr_real *) malloc((size_t) utilities.length * sizeof(*target));
    if (target == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_realized_best_distribution(utilities, tolerance, (bolr_vector_view){target, utilities.length, 1}) != BOLR_OK) {
        free(target);
        return BOLR_INVALID_ARGUMENT;
    }
    for (i = 0; i < probability_best.length; ++i) {
        bolr_real diff = probability_best.data[i * probability_best.stride] - target[i];
        brier += diff * diff;
    }
    free(target);
    *out_brier = brier;
    return BOLR_OK;
}

bolr_status bolr_top_k_brier(bolr_const_vector_view probability_top_k, bolr_const_vector_view utilities, bolr_index top_k, bolr_real *out_brier) {
    bolr_real *target;
    bolr_real brier = 0.0;
    bolr_index i;
    if ((out_brier == NULL) || (probability_top_k.length != utilities.length)) return BOLR_INVALID_ARGUMENT;
    target = (bolr_real *) malloc((size_t) utilities.length * sizeof(*target));
    if (target == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_realized_top_k_indicator(utilities, top_k, (bolr_vector_view){target, utilities.length, 1}) != BOLR_OK) {
        free(target);
        return BOLR_INVALID_ARGUMENT;
    }
    for (i = 0; i < probability_top_k.length; ++i) {
        bolr_real diff = probability_top_k.data[i * probability_top_k.stride] - target[i];
        brier += diff * diff;
    }
    free(target);
    *out_brier = brier;
    return BOLR_OK;
}

bolr_status bolr_region_coverage(const bolr_index *region_indices, bolr_index region_count, bolr_const_vector_view utilities, bolr_real tolerance, int *out_covered) {
    bolr_real *target;
    bolr_index i;
    if ((region_indices == NULL) || (out_covered == NULL) || (region_count < 0)) return BOLR_INVALID_ARGUMENT;
    target = (bolr_real *) malloc((size_t) utilities.length * sizeof(*target));
    if (target == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_realized_best_distribution(utilities, tolerance, (bolr_vector_view){target, utilities.length, 1}) != BOLR_OK) {
        free(target);
        return BOLR_INVALID_ARGUMENT;
    }
    *out_covered = 0;
    for (i = 0; i < region_count; ++i) {
        if ((region_indices[i] < 0) || (region_indices[i] >= utilities.length)) {
            free(target);
            return BOLR_INVALID_ARGUMENT;
        }
        if (target[region_indices[i]] > 0.0) {
            *out_covered = 1;
            break;
        }
    }
    free(target);
    return BOLR_OK;
}
