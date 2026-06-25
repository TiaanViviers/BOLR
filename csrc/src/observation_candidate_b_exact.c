#include "bolr/observation_candidate_b.h"
#include "bolr/math.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    const bolr_allocator *allocator;
    bolr_index candidate_count;
    bolr_index pair_count;
    bolr_index possible_pair_count;
    bolr_index duplicate_sample_count;
    bolr_real update_weight;
    int normalize_pair_losses;
    bolr_index *winner_indices;
    bolr_index *loser_indices;
    bolr_real *pair_weights;
} bolr_candidate_b_pair_observation;

struct bolr_candidate_b_exact_observation {
    bolr_candidate_b_pair_observation core;
};

struct bolr_candidate_b_sampled_observation {
    bolr_candidate_b_pair_observation core;
};

static bolr_status evaluate_candidate_b(
    const bolr_candidate_b_pair_observation *observation,
    bolr_const_vector_view scores,
    bolr_real *out_value,
    bolr_vector_view output_gradient,
    bolr_const_vector_view vector,
    bolr_vector_view output_hvp
) {
    bolr_index k;
    if ((observation == NULL) || (scores.length != observation->candidate_count)) return BOLR_INVALID_ARGUMENT;
    if (out_value != NULL) *out_value = 0.0;
    if (output_gradient.data != NULL) {
        if (output_gradient.length != observation->candidate_count) return BOLR_INVALID_SHAPE;
        for (k = 0; k < output_gradient.length; ++k) output_gradient.data[k * output_gradient.stride] = 0.0;
    }
    if (output_hvp.data != NULL) {
        if ((vector.length != observation->candidate_count) || (output_hvp.length != observation->candidate_count)) return BOLR_INVALID_SHAPE;
        for (k = 0; k < output_hvp.length; ++k) output_hvp.data[k * output_hvp.stride] = 0.0;
    }
    for (k = 0; k < observation->pair_count; ++k) {
        const bolr_index winner = observation->winner_indices[k];
        const bolr_index loser = observation->loser_indices[k];
        const bolr_real weight = observation->update_weight * observation->pair_weights[k];
        const bolr_real difference = scores.data[loser * scores.stride] - scores.data[winner * scores.stride];
        bolr_real probability;
        bolr_real softplus_value;
        if ((winner < 0) || (winner >= observation->candidate_count) || (loser < 0) || (loser >= observation->candidate_count)) return BOLR_INVALID_ARGUMENT;
        if (bolr_sigmoid(difference, &probability) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
        if (bolr_softplus(difference, &softplus_value) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
        if (out_value != NULL) *out_value -= weight * softplus_value;
        if (output_gradient.data != NULL) {
            output_gradient.data[winner * output_gradient.stride] += weight * probability;
            output_gradient.data[loser * output_gradient.stride] += -weight * probability;
        }
        if (output_hvp.data != NULL) {
            const bolr_real curvature = weight * probability * (1.0 - probability);
            const bolr_real displacement = vector.data[winner * vector.stride] - vector.data[loser * vector.stride];
            output_hvp.data[winner * output_hvp.stride] += curvature * displacement;
            output_hvp.data[loser * output_hvp.stride] -= curvature * displacement;
        }
    }
    return BOLR_OK;
}

static bolr_status pair_observation_value_callback(const void *context, bolr_const_vector_view scores, bolr_real *out_log_factor, void *observation_workspace) {
    (void) observation_workspace;
    return evaluate_candidate_b((const bolr_candidate_b_pair_observation *) context, scores, out_log_factor, (bolr_vector_view){NULL, 0, 1}, (bolr_const_vector_view){NULL, 0, 1}, (bolr_vector_view){NULL, 0, 1});
}

static bolr_status pair_observation_gradient_callback(const void *context, bolr_const_vector_view scores, bolr_vector_view output_gradient, void *observation_workspace) {
    (void) observation_workspace;
    return evaluate_candidate_b((const bolr_candidate_b_pair_observation *) context, scores, NULL, output_gradient, (bolr_const_vector_view){NULL, 0, 1}, (bolr_vector_view){NULL, 0, 1});
}

static bolr_status pair_observation_curvature_hvp_callback(const void *context, bolr_const_vector_view scores, bolr_const_vector_view vector, bolr_vector_view output, void *observation_workspace) {
    (void) observation_workspace;
    return evaluate_candidate_b((const bolr_candidate_b_pair_observation *) context, scores, NULL, (bolr_vector_view){NULL, 0, 1}, vector, output);
}

static void destroy_pair_observation(bolr_candidate_b_pair_observation *observation) {
    if (observation == NULL) return;
    bolr_allocator_free(observation->allocator, observation->winner_indices);
    bolr_allocator_free(observation->allocator, observation->loser_indices);
    bolr_allocator_free(observation->allocator, observation->pair_weights);
}

static bolr_status initialize_pair_observation(
    bolr_candidate_b_pair_observation *observation,
    const bolr_allocator *allocator,
    bolr_index candidate_count,
    bolr_index pair_count
) {
    observation->allocator = allocator;
    observation->candidate_count = candidate_count;
    observation->pair_count = pair_count;
    if (pair_count == 0) return BOLR_OK;
    observation->winner_indices = (bolr_index *) bolr_allocator_malloc(allocator, (size_t) pair_count * sizeof(bolr_index));
    observation->loser_indices = (bolr_index *) bolr_allocator_malloc(allocator, (size_t) pair_count * sizeof(bolr_index));
    observation->pair_weights = (bolr_real *) bolr_allocator_malloc(allocator, (size_t) pair_count * sizeof(bolr_real));
    if ((observation->winner_indices == NULL) || (observation->loser_indices == NULL) || (observation->pair_weights == NULL)) return BOLR_ALLOCATION_FAILED;
    return BOLR_OK;
}

bolr_status bolr_candidate_b_exact_observation_create(
    const bolr_ordered_partition *partition,
    int normalize_pair_losses,
    const bolr_allocator *allocator,
    bolr_candidate_b_exact_observation **out_observation
) {
    bolr_candidate_b_exact_observation *observation = NULL;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index group_count;
    bolr_index candidate_count;
    bolr_index pair_count;
    bolr_index group_offsets_length;
    bolr_index *group_offsets = NULL;
    bolr_index *group_indices = NULL;
    bolr_index a;
    bolr_index write_idx = 0;
    bolr_real update_weight;
    if ((partition == NULL) || (out_observation == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_observation = NULL;
    group_count = bolr_ordered_partition_group_count(partition);
    candidate_count = bolr_ordered_partition_candidate_count(partition);
    pair_count = bolr_ordered_partition_possible_pair_count(partition);
    group_offsets_length = group_count + 1;
    update_weight = bolr_ordered_partition_update_weight(partition);
    group_offsets = (bolr_index *) malloc((size_t) group_offsets_length * sizeof(bolr_index));
    group_indices = (bolr_index *) malloc((size_t) candidate_count * sizeof(bolr_index));
    if ((group_offsets == NULL) || (group_indices == NULL)) {
        free(group_offsets);
        free(group_indices);
        return BOLR_ALLOCATION_FAILED;
    }
    if ((bolr_ordered_partition_copy_group_offsets(partition, group_offsets, group_offsets_length) != BOLR_OK) || (bolr_ordered_partition_copy_group_indices(partition, group_indices, candidate_count) != BOLR_OK)) {
        free(group_offsets);
        free(group_indices);
        return BOLR_INVALID_ARGUMENT;
    }
    observation = (bolr_candidate_b_exact_observation *) bolr_allocator_calloc(active, 1U, sizeof(*observation));
    if (observation == NULL) {
        free(group_offsets);
        free(group_indices);
        return BOLR_ALLOCATION_FAILED;
    }
    if (initialize_pair_observation(&observation->core, active, candidate_count, pair_count) != BOLR_OK) {
        free(group_offsets);
        free(group_indices);
        bolr_candidate_b_exact_observation_destroy(observation);
        return BOLR_ALLOCATION_FAILED;
    }
    observation->core.possible_pair_count = pair_count;
    observation->core.duplicate_sample_count = 0;
    observation->core.update_weight = update_weight;
    observation->core.normalize_pair_losses = normalize_pair_losses;
    for (a = 0; a < group_count; ++a) {
        const bolr_index start_a = group_offsets[a];
        const bolr_index stop_a = group_offsets[a + 1];
        bolr_index b;
        for (b = a + 1; b < group_count; ++b) {
            const bolr_index start_b = group_offsets[b];
            const bolr_index stop_b = group_offsets[b + 1];
            const bolr_index group_pair_count = (stop_a - start_a) * (stop_b - start_b);
            const bolr_real group_weight = 1.0 / (bolr_real) ((group_count * (group_count - 1)) / 2);
            const bolr_real coeff = normalize_pair_losses ? (group_weight / (bolr_real) group_pair_count) : group_weight;
            bolr_index ia;
            for (ia = start_a; ia < stop_a; ++ia) {
                bolr_index ib;
                for (ib = start_b; ib < stop_b; ++ib) {
                    observation->core.winner_indices[write_idx] = group_indices[ia];
                    observation->core.loser_indices[write_idx] = group_indices[ib];
                    observation->core.pair_weights[write_idx] = coeff;
                    write_idx += 1;
                }
            }
        }
    }
    free(group_offsets);
    free(group_indices);
    *out_observation = observation;
    return BOLR_OK;
}

void bolr_candidate_b_exact_observation_destroy(bolr_candidate_b_exact_observation *observation) {
    if (observation == NULL) return;
    destroy_pair_observation(&observation->core);
    bolr_allocator_free(observation->core.allocator, observation);
}

bolr_status bolr_candidate_b_exact_observation_operator(
    const bolr_candidate_b_exact_observation *observation,
    bolr_observation_operator *out_operator
) {
    if ((observation == NULL) || (out_operator == NULL)) return BOLR_INVALID_ARGUMENT;
    out_operator->value = pair_observation_value_callback;
    out_operator->gradient = pair_observation_gradient_callback;
    out_operator->curvature_hvp = pair_observation_curvature_hvp_callback;
    out_operator->context = &observation->core;
    return BOLR_OK;
}

bolr_status bolr_candidate_b_exact_observation_diagnostics(
    const bolr_candidate_b_exact_observation *observation,
    bolr_candidate_b_diagnostics *out_diagnostics
) {
    if ((observation == NULL) || (out_diagnostics == NULL)) return BOLR_INVALID_ARGUMENT;
    out_diagnostics->candidate_count = observation->core.candidate_count;
    out_diagnostics->possible_pair_count = observation->core.possible_pair_count;
    out_diagnostics->used_pair_count = observation->core.pair_count;
    out_diagnostics->duplicate_sample_count = 0;
    out_diagnostics->update_weight = observation->core.update_weight;
    out_diagnostics->normalize_pair_losses = observation->core.normalize_pair_losses;
    return BOLR_OK;
}

bolr_status bolr_candidate_b_sampled_observation_create(
    bolr_index candidate_count,
    const bolr_index *winner_indices,
    const bolr_index *loser_indices,
    const bolr_real *pair_weights,
    bolr_index pair_count,
    bolr_real update_weight,
    bolr_index possible_pair_count,
    bolr_index duplicate_sample_count,
    int normalize_pair_losses,
    const bolr_allocator *allocator,
    bolr_candidate_b_sampled_observation **out_observation
) {
    bolr_candidate_b_sampled_observation *observation = NULL;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index k;
    if ((out_observation == NULL) || (candidate_count <= 0) || (pair_count < 0)) return BOLR_INVALID_ARGUMENT;
    *out_observation = NULL;
    if (((pair_count > 0) && (winner_indices == NULL || loser_indices == NULL || pair_weights == NULL)) || !isfinite(update_weight)) return BOLR_INVALID_ARGUMENT;
    observation = (bolr_candidate_b_sampled_observation *) bolr_allocator_calloc(active, 1U, sizeof(*observation));
    if (observation == NULL) return BOLR_ALLOCATION_FAILED;
    if (initialize_pair_observation(&observation->core, active, candidate_count, pair_count) != BOLR_OK) {
        bolr_candidate_b_sampled_observation_destroy(observation);
        return BOLR_ALLOCATION_FAILED;
    }
    observation->core.possible_pair_count = possible_pair_count;
    observation->core.duplicate_sample_count = duplicate_sample_count;
    observation->core.update_weight = update_weight;
    observation->core.normalize_pair_losses = normalize_pair_losses;
    for (k = 0; k < pair_count; ++k) {
        if ((winner_indices[k] < 0) || (winner_indices[k] >= candidate_count) || (loser_indices[k] < 0) || (loser_indices[k] >= candidate_count) || !isfinite(pair_weights[k])) {
            bolr_candidate_b_sampled_observation_destroy(observation);
            return BOLR_INVALID_ARGUMENT;
        }
        observation->core.winner_indices[k] = winner_indices[k];
        observation->core.loser_indices[k] = loser_indices[k];
        observation->core.pair_weights[k] = pair_weights[k];
    }
    *out_observation = observation;
    return BOLR_OK;
}

void bolr_candidate_b_sampled_observation_destroy(bolr_candidate_b_sampled_observation *observation) {
    if (observation == NULL) return;
    destroy_pair_observation(&observation->core);
    bolr_allocator_free(observation->core.allocator, observation);
}

bolr_status bolr_candidate_b_sampled_observation_operator(
    const bolr_candidate_b_sampled_observation *observation,
    bolr_observation_operator *out_operator
) {
    if ((observation == NULL) || (out_operator == NULL)) return BOLR_INVALID_ARGUMENT;
    out_operator->value = pair_observation_value_callback;
    out_operator->gradient = pair_observation_gradient_callback;
    out_operator->curvature_hvp = pair_observation_curvature_hvp_callback;
    out_operator->context = &observation->core;
    return BOLR_OK;
}

bolr_status bolr_candidate_b_sampled_observation_diagnostics(
    const bolr_candidate_b_sampled_observation *observation,
    bolr_candidate_b_diagnostics *out_diagnostics
) {
    if ((observation == NULL) || (out_diagnostics == NULL)) return BOLR_INVALID_ARGUMENT;
    out_diagnostics->candidate_count = observation->core.candidate_count;
    out_diagnostics->possible_pair_count = observation->core.possible_pair_count;
    out_diagnostics->used_pair_count = observation->core.pair_count;
    out_diagnostics->duplicate_sample_count = observation->core.duplicate_sample_count;
    out_diagnostics->update_weight = observation->core.update_weight;
    out_diagnostics->normalize_pair_losses = observation->core.normalize_pair_losses;
    return BOLR_OK;
}
