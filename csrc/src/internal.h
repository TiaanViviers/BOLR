#ifndef BOLR_INTERNAL_H
#define BOLR_INTERNAL_H

#include "bolr/checkpoint.h"
#include "bolr/gaussian.h"
#include "bolr/prediction.h"
#include "bolr/region.h"
#include "bolr/decision.h"
#include "bolr/score.h"
#include "bolr/workspace.h"

struct bolr_gaussian_state {
    const bolr_allocator *allocator;
    bolr_real *mean;
    bolr_real *covariance;
    bolr_index dimension;
    uint64_t step_index;
    uint64_t state_layout_hash;
    uint64_t model_schema_hash;
    uint32_t schema_version;
};

struct bolr_checkpoint_state {
    const bolr_allocator *allocator;
    bolr_checkpoint_header header;
    bolr_real *mean;
    bolr_real *covariance;
    bolr_index dimension;
    uint64_t step_index;
    uint64_t state_layout_hash;
    uint64_t model_schema_hash;
    uint32_t gaussian_state_schema_version;
};

struct bolr_inference_workspace {
    const bolr_allocator *allocator;
    bolr_index state_dimension;
    bolr_index candidate_count;
    bolr_workspace *score_workspace;
    bolr_real *state_displacement;
    bolr_real *prior_solve;
    bolr_real *score_vector;
    bolr_real *score_gradient;
    bolr_real *score_hvp;
    bolr_real *parameter_gradient;
    bolr_real *parameter_curvature;
    bolr_real *parameter_hvp;
    bolr_real *newton_step;
    bolr_real *current_state;
    bolr_real *trial_state;
    bolr_real *trial_scores;
    bolr_real *prior_cholesky;
    bolr_real *dense_hessian;
    bolr_real *damped_hessian;
    bolr_real *posterior_covariance;
    bolr_real *identity_rhs;
};

struct bolr_posterior_prediction {
    const bolr_allocator *allocator;
    bolr_real *score_mean;
    bolr_real *score_variance;
    bolr_real *state_mean;
    bolr_real *state_covariance;
    bolr_real *design_matrix;
    bolr_real *probability_best;
    bolr_real **probability_top_k_values;
    bolr_index *probability_top_k_keys;
    bolr_index probability_top_k_count;
    bolr_real *expected_rank;
    bolr_index candidate_count;
    bolr_index state_dim;
    uint64_t model_schema_hash;
    uint64_t state_layout_hash;
};

struct bolr_grid_graph {
    const bolr_allocator *allocator;
    bolr_index node_count;
    bolr_index edge_count;
    bolr_index *edge_index;
    bolr_index *entry_indices;
    bolr_index *stop_indices;
    uint64_t graph_hash;
};

struct bolr_region_set {
    const bolr_allocator *allocator;
    bolr_real *inclusion_probability;
    bolr_index candidate_count;
    bolr_index *consensus_indices;
    bolr_index consensus_count;
    bolr_index top_k;
    int empty_consensus;
    bolr_region_summary *summaries;
    bolr_index *region_candidates;
    bolr_index region_candidate_count;
    bolr_index region_count;
};

struct bolr_decision_policy {
    const bolr_allocator *allocator;
    bolr_decision_policy_config config;
};

#endif
