#ifndef BOLR_REPLAY_H
#define BOLR_REPLAY_H

#include "bolr/adaptation.h"
#include "bolr/decision.h"
#include "bolr/gaussian.h"
#include "bolr/inference.h"
#include "bolr/observation.h"
#include "bolr/prediction.h"
#include "bolr/region.h"
#include "bolr/rng.h"

typedef struct bolr_replay_engine bolr_replay_engine;
typedef struct bolr_replay_checkpoint bolr_replay_checkpoint;

typedef enum {
    BOLR_REPLAY_PHASE_READY = 1,
    BOLR_REPLAY_PHASE_AWAITING_OUTCOME = 2
} bolr_replay_phase;

typedef struct {
    bolr_index sample_count;
    bolr_index chunk_size;
    int antithetic;
    bolr_score_retention retention;
} bolr_replay_ranking_config;

typedef struct {
    bolr_replay_phase phase;
    bolr_index selected_index;
    bolr_real selected_score_mean;
    bolr_real selected_score_variance;
    bolr_real selected_probability_best;
    bolr_real selected_expected_rank;
    bolr_index retained_score_sample_count;
    bolr_index region_count;
} bolr_replay_begin_diagnostics;

typedef struct {
    bolr_replay_phase phase_after;
    bolr_index selected_index;
    bolr_real objective_improvement;
    bolr_real posterior_trace;
    int adaptive_applied;
} bolr_replay_finish_diagnostics;

bolr_status bolr_replay_engine_create_fixed(
    const bolr_gaussian_state *posterior,
    const bolr_transition_config *transition,
    const bolr_rng *rng,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
);
bolr_status bolr_replay_engine_create_adaptive(
    const bolr_gaussian_state *posterior,
    const bolr_adaptive_policy *policy,
    const bolr_adaptive_state *adaptive_state,
    const bolr_rng *rng,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
);
void bolr_replay_engine_destroy(bolr_replay_engine *engine);
bolr_replay_phase bolr_replay_engine_phase(const bolr_replay_engine *engine);
bolr_status bolr_replay_engine_copy_posterior_mean(const bolr_replay_engine *engine, bolr_vector_view output);
bolr_status bolr_replay_engine_copy_posterior_covariance(const bolr_replay_engine *engine, bolr_matrix_view output);
bolr_index bolr_replay_engine_pending_selected_index(const bolr_replay_engine *engine);

bolr_status bolr_replay_engine_begin_day(
    bolr_replay_engine *engine,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_replay_ranking_config *ranking,
    const bolr_index *top_k_values,
    bolr_index top_k_count,
    const bolr_decision_policy *decision_policy,
    const bolr_grid_graph *graph,
    const bolr_region_config *region_config,
    bolr_workspace *workspace,
    bolr_decision *out_decision,
    bolr_replay_begin_diagnostics *out_diagnostics
);

bolr_status bolr_replay_engine_finish_day(
    bolr_replay_engine *engine,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_observation_operator *observation,
    const bolr_newton_config *newton_config,
    bolr_real effective_strength,
    bolr_real information_size,
    int informative,
    bolr_inference_workspace *workspace,
    bolr_laplace_diagnostics *out_laplace,
    bolr_adaptation_diagnostics *out_adaptation,
    bolr_replay_finish_diagnostics *out_diagnostics
);

bolr_status bolr_replay_engine_export_checkpoint(
    const bolr_replay_engine *engine,
    const bolr_allocator *allocator,
    bolr_replay_checkpoint **out_checkpoint
);
bolr_status bolr_replay_engine_import_fixed(
    const bolr_replay_checkpoint *checkpoint,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
);
bolr_status bolr_replay_engine_import_adaptive(
    const bolr_replay_checkpoint *checkpoint,
    const bolr_adaptive_policy *policy,
    const bolr_allocator *allocator,
    bolr_replay_engine **out_engine
);

void bolr_replay_checkpoint_destroy(bolr_replay_checkpoint *checkpoint);
bolr_replay_phase bolr_replay_checkpoint_phase(const bolr_replay_checkpoint *checkpoint);
bolr_index bolr_replay_checkpoint_pending_selected_index(const bolr_replay_checkpoint *checkpoint);

#endif
