#include "test_suite.h"

#include "bolr/inference.h"
#include "bolr/observation.h"
#include "bolr/replay.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"

int test_replay(void) {
    bolr_state_layout *layout = NULL;
    bolr_model *model = NULL;
    bolr_gaussian_state *posterior = NULL;
    bolr_rng *rng = NULL;
    bolr_replay_engine *engine = NULL;
    bolr_replay_checkpoint *checkpoint = NULL;
    bolr_replay_engine *restored = NULL;
    bolr_workspace *workspace = NULL;
    bolr_inference_workspace *inference = NULL;
    bolr_candidate_a_observation *observation = NULL;
    bolr_decision_policy *policy = NULL;
    bolr_decision decision;
    bolr_replay_begin_diagnostics begin_diag;
    bolr_replay_finish_diagnostics finish_diag;
    bolr_laplace_diagnostics laplace_diag;
    bolr_replay_ranking_config ranking = {16, 5, 1, BOLR_SCORE_RETENTION_SAMPLE_ZERO};
    bolr_decision_policy_config policy_cfg = {BOLR_DECISION_THOMPSON, 0, 0, 0};
    bolr_state_block_spec spec = {"dense", 0, 2, 2, 3, 1, 'C'};
    bolr_real static_scores[] = {0.4, 0.2, -0.1};
    bolr_real design[] = {1.0, 0.0, 0.5, 0.5, 0.0, 1.0};
    bolr_real mean[] = {0.0, 0.0};
    bolr_real covariance[] = {0.2, 0.0, 0.0, 0.2};
    bolr_real process_noise[] = {0.01, 0.0, 0.0, 0.01};
    bolr_real target[] = {0.6, 0.3, 0.1};
    bolr_real posterior_mean_out[2];
    bolr_transition_config transition;
    if (bolr_state_layout_create(&spec, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 3, 1}, NULL, &model) != BOLR_OK) return 1;
    if (bolr_model_add_dense_block_copy(model, "dense", (bolr_const_matrix_view){design, 3, 2, 2, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, bolr_model_state_layout_hash(model), bolr_model_schema_hash(model), NULL, &posterior) != BOLR_OK) return 1;
    if (bolr_rng_create((bolr_rng_seed){7ULL, 2ULL}, NULL, &rng) != BOLR_OK) return 1;
    transition.family = BOLR_TRANSITION_ADDITIVE_Q;
    transition.process_noise = (bolr_const_matrix_view){process_noise, 2, 2, 2, 1};
    transition.global_discount = 0.0;
    transition.block_discount_scales = (bolr_const_vector_view){NULL, 0, 1};
    if (bolr_replay_engine_create_fixed(posterior, &transition, rng, NULL, &engine) != BOLR_OK) return 1;
    if (bolr_workspace_create(&(bolr_workspace_config){3, 2, 0}, NULL, &workspace) != BOLR_OK) return 1;
    if (bolr_inference_workspace_create(2, 3, NULL, &inference) != BOLR_OK) return 1;
    if (bolr_candidate_a_observation_create((bolr_const_vector_view){target, 3, 1}, 1.0, 1.0, NULL, &observation) != BOLR_OK) return 1;
    if (bolr_decision_policy_create(&policy_cfg, NULL, &policy) != BOLR_OK) return 1;
    if (bolr_replay_engine_begin_day(engine, model, (bolr_const_vector_view){NULL, 0, 1}, &ranking, (const bolr_index[]){1, 2}, 2, policy, NULL, NULL, workspace, &decision, &begin_diag) != BOLR_OK) return 1;
    if ((begin_diag.phase != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) || (decision.selected_index < 0)) return 1;
    if (bolr_replay_engine_export_checkpoint(engine, NULL, &checkpoint) != BOLR_OK) return 1;
    if (bolr_replay_engine_import_fixed(checkpoint, NULL, &restored) != BOLR_OK) return 1;
    if (bolr_replay_checkpoint_phase(checkpoint) != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) return 1;
    if (bolr_replay_engine_pending_selected_index(restored) != decision.selected_index) return 1;
    {
        bolr_observation_operator op;
        if (bolr_candidate_a_observation_operator(observation, &op) != BOLR_OK) return 1;
        if (bolr_replay_engine_finish_day(restored, model, (bolr_const_vector_view){NULL, 0, 1}, &op, NULL, 1.0, 1.0, 1, inference, &laplace_diag, NULL, &finish_diag) != BOLR_OK) return 1;
    }
    if (finish_diag.phase_after != BOLR_REPLAY_PHASE_READY) return 1;
    if (bolr_replay_engine_copy_posterior_mean(restored, (bolr_vector_view){posterior_mean_out, 2, 1}) != BOLR_OK) return 1;
    if ((posterior_mean_out[0] == 0.0) && (posterior_mean_out[1] == 0.0)) return 1;
    bolr_decision_policy_destroy(policy);
    bolr_candidate_a_observation_destroy(observation);
    bolr_inference_workspace_destroy(inference);
    bolr_workspace_destroy(workspace);
    bolr_replay_engine_destroy(restored);
    bolr_replay_checkpoint_destroy(checkpoint);
    bolr_replay_engine_destroy(engine);
    bolr_rng_destroy(rng);
    bolr_gaussian_state_destroy(posterior);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    return 0;
}
