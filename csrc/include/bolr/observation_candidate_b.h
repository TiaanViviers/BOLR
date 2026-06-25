#ifndef BOLR_OBSERVATION_CANDIDATE_B_H
#define BOLR_OBSERVATION_CANDIDATE_B_H

#include "bolr/observation.h"
#include "bolr/partition.h"

typedef struct bolr_candidate_b_exact_observation bolr_candidate_b_exact_observation;
typedef struct bolr_candidate_b_sampled_observation bolr_candidate_b_sampled_observation;

typedef struct {
    bolr_index candidate_count;
    bolr_index possible_pair_count;
    bolr_index used_pair_count;
    bolr_index duplicate_sample_count;
    bolr_real update_weight;
    int normalize_pair_losses;
} bolr_candidate_b_diagnostics;

bolr_status bolr_candidate_b_exact_observation_create(
    const bolr_ordered_partition *partition,
    int normalize_pair_losses,
    const bolr_allocator *allocator,
    bolr_candidate_b_exact_observation **out_observation
);
void bolr_candidate_b_exact_observation_destroy(bolr_candidate_b_exact_observation *observation);
bolr_status bolr_candidate_b_exact_observation_operator(
    const bolr_candidate_b_exact_observation *observation,
    bolr_observation_operator *out_operator
);
bolr_status bolr_candidate_b_exact_observation_diagnostics(
    const bolr_candidate_b_exact_observation *observation,
    bolr_candidate_b_diagnostics *out_diagnostics
);

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
);
void bolr_candidate_b_sampled_observation_destroy(bolr_candidate_b_sampled_observation *observation);
bolr_status bolr_candidate_b_sampled_observation_operator(
    const bolr_candidate_b_sampled_observation *observation,
    bolr_observation_operator *out_operator
);
bolr_status bolr_candidate_b_sampled_observation_diagnostics(
    const bolr_candidate_b_sampled_observation *observation,
    bolr_candidate_b_diagnostics *out_diagnostics
);

#endif
