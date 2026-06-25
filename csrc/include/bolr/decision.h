#ifndef BOLR_DECISION_H
#define BOLR_DECISION_H

#include "bolr/allocator.h"
#include "bolr/prediction.h"
#include "bolr/region.h"

typedef struct bolr_decision_policy bolr_decision_policy;

typedef enum {
    BOLR_DECISION_POSTERIOR_MEAN = 1,
    BOLR_DECISION_PROBABILITY_BEST = 2,
    BOLR_DECISION_PROBABILITY_TOP_K = 3,
    BOLR_DECISION_EXPECTED_RANK = 4,
    BOLR_DECISION_REGION = 5
} bolr_decision_family;

typedef enum {
    BOLR_REGION_STATISTIC_PROBABILITY_BEST = 1,
    BOLR_REGION_STATISTIC_INCLUSION_MASS = 2
} bolr_region_statistic;

typedef enum {
    BOLR_REGION_REP_POSTERIOR_MEAN = 1,
    BOLR_REGION_REP_PROBABILITY_BEST = 2,
    BOLR_REGION_REP_PROBABILITY_TOP_K = 3,
    BOLR_REGION_REP_WEIGHTED_MEDOID = 4
} bolr_region_representative;

typedef struct {
    bolr_decision_family family;
    bolr_index top_k;
    bolr_region_statistic region_selection_statistic;
    bolr_region_representative representative_policy;
} bolr_decision_policy_config;

typedef struct {
    bolr_index selected_index;
    int selected;
    int abstained;
    bolr_real selected_score_mean;
    bolr_real selected_score_variance;
    bolr_real selected_probability_best;
    bolr_real selected_expected_rank;
    bolr_index selected_region_id;
    bolr_real selected_region_mass;
    bolr_real selected_region_probability_best;
    uint32_t tie_flags;
    uint32_t reason_code;
} bolr_decision;

typedef struct {
    int tie_occurred;
    int tie_break_stage_count;
    int tie_break_stages[4];
    bolr_index selected_region_candidate_count;
    bolr_real medoid_objective;
} bolr_decision_diagnostics;

bolr_status bolr_decision_policy_create(const bolr_decision_policy_config *config, const bolr_allocator *allocator, bolr_decision_policy **out_policy);
void bolr_decision_policy_destroy(bolr_decision_policy *policy);
bolr_status bolr_decision_policy_apply(
    const bolr_decision_policy *policy,
    const bolr_posterior_prediction *prediction,
    const bolr_region_set *regions,
    const bolr_grid_graph *graph,
    bolr_decision *out_decision,
    bolr_decision_diagnostics *diagnostics
);

bolr_status bolr_realized_best_distribution(bolr_const_vector_view utilities, bolr_real tolerance, bolr_vector_view output);
bolr_status bolr_realized_top_k_indicator(bolr_const_vector_view utilities, bolr_index top_k, bolr_vector_view output);
bolr_status bolr_probability_best_brier(bolr_const_vector_view probability_best, bolr_const_vector_view utilities, bolr_real tolerance, bolr_real *out_brier);
bolr_status bolr_top_k_brier(bolr_const_vector_view probability_top_k, bolr_const_vector_view utilities, bolr_index top_k, bolr_real *out_brier);
bolr_status bolr_region_coverage(const bolr_index *region_indices, bolr_index region_count, bolr_const_vector_view utilities, bolr_real tolerance, int *out_covered);

#endif
