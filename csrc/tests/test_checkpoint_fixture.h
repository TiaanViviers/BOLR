#ifndef BOLR_TEST_CHECKPOINT_FIXTURE_H
#define BOLR_TEST_CHECKPOINT_FIXTURE_H

#include "bolr/bolr.h"

#include <math.h>
#include <string.h>

#define BOLR_TEST_CHECKPOINT_HEADER_SIZE 180U
#define BOLR_TEST_CHECKPOINT_DIRECTORY_ENTRY_SIZE 44U

typedef struct {
    bolr_state_layout *layout;
    bolr_model *model;
    bolr_gaussian_state *posterior;
    bolr_rng *rng;
    bolr_replay_engine *engine;
    bolr_workspace *workspace;
    bolr_inference_workspace *inference;
    bolr_candidate_a_observation *observation;
    bolr_decision_policy *policy;
    bolr_replay_ranking_config ranking;
    bolr_decision_policy_config policy_cfg;
    bolr_transition_config transition;
    bolr_index top_k[2];
} bolr_test_checkpoint_fixture;

static inline bolr_replay_restore_context bolr_test_checkpoint_restore_context(const bolr_model *model) {
    bolr_replay_restore_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.limits = bolr_checkpoint_limits_default();
    ctx.expected_model_schema_hash = bolr_model_schema_hash(model);
    ctx.expected_state_layout_hash = bolr_model_state_layout_hash(model);
    return ctx;
}

static inline bolr_status bolr_test_checkpoint_fixture_create(bolr_test_checkpoint_fixture *fixture) {
    bolr_state_block_spec spec = {"dense", 0, 2, 2, 3, 1, 'C'};
    bolr_real static_scores[] = {0.4, 0.2, -0.1};
    bolr_real design[] = {1.0, 0.0, 0.5, 0.5, 0.0, 1.0};
    bolr_real mean[] = {0.0, 0.0};
    bolr_real covariance[] = {0.2, 0.0, 0.0, 0.2};
    bolr_real process_noise[] = {0.01, 0.0, 0.0, 0.01};
    bolr_real target[] = {0.6, 0.3, 0.1};
    if (fixture == NULL) return BOLR_INVALID_ARGUMENT;
    memset(fixture, 0, sizeof(*fixture));
    fixture->ranking = (bolr_replay_ranking_config){16, 5, 1, BOLR_SCORE_RETENTION_SAMPLE_ZERO};
    fixture->policy_cfg = (bolr_decision_policy_config){BOLR_DECISION_THOMPSON, 0, 0, 0};
    fixture->top_k[0] = 1;
    fixture->top_k[1] = 2;
    if (bolr_state_layout_create(&spec, 1, NULL, &fixture->layout) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_model_create(fixture->layout, (bolr_const_vector_view){static_scores, 3, 1}, NULL, &fixture->model) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_model_add_dense_block_copy(fixture->model, "dense", (bolr_const_matrix_view){design, 3, 2, 2, 1}) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, bolr_model_state_layout_hash(fixture->model), bolr_model_schema_hash(fixture->model), NULL, &fixture->posterior) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_rng_create((bolr_rng_seed){7ULL, 2ULL}, NULL, &fixture->rng) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    fixture->transition.family = BOLR_TRANSITION_ADDITIVE_Q;
    fixture->transition.process_noise = (bolr_const_matrix_view){process_noise, 2, 2, 2, 1};
    fixture->transition.global_discount = 0.0;
    fixture->transition.block_discount_scales = (bolr_const_vector_view){NULL, 0, 1};
    if (bolr_replay_engine_create_fixed(fixture->posterior, &fixture->transition, fixture->rng, NULL, &fixture->engine) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_workspace_create(&(bolr_workspace_config){3, 2, 0}, NULL, &fixture->workspace) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_inference_workspace_create(2, 3, NULL, &fixture->inference) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_candidate_a_observation_create((bolr_const_vector_view){target, 3, 1}, 1.0, 1.0, NULL, &fixture->observation) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (bolr_decision_policy_create(&fixture->policy_cfg, NULL, &fixture->policy) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    return BOLR_OK;
}

static inline void bolr_test_checkpoint_fixture_destroy(bolr_test_checkpoint_fixture *fixture) {
    if (fixture == NULL) return;
    bolr_decision_policy_destroy(fixture->policy);
    bolr_candidate_a_observation_destroy(fixture->observation);
    bolr_inference_workspace_destroy(fixture->inference);
    bolr_workspace_destroy(fixture->workspace);
    bolr_replay_engine_destroy(fixture->engine);
    bolr_rng_destroy(fixture->rng);
    bolr_gaussian_state_destroy(fixture->posterior);
    bolr_model_destroy(fixture->model);
    bolr_state_layout_destroy(fixture->layout);
    memset(fixture, 0, sizeof(*fixture));
}

static inline int bolr_test_checkpoint_vectors_close(const bolr_real *a, const bolr_real *b, bolr_index n, bolr_real tol) {
    bolr_index i;
    for (i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > tol) return 0;
    }
    return 1;
}

static inline bolr_status bolr_test_checkpoint_begin_day(bolr_test_checkpoint_fixture *fixture, bolr_decision *out_decision) {
    return bolr_replay_engine_begin_day(
        fixture->engine,
        fixture->model,
        (bolr_const_vector_view){NULL, 0, 1},
        &fixture->ranking,
        fixture->top_k,
        2,
        fixture->policy,
        NULL,
        NULL,
        fixture->workspace,
        out_decision,
        NULL
    );
}

static inline bolr_status bolr_test_checkpoint_finish_day(bolr_test_checkpoint_fixture *fixture) {
    bolr_observation_operator op;
    bolr_status status = bolr_candidate_a_observation_operator(fixture->observation, &op);
    if (status != BOLR_OK) return status;
    return bolr_replay_engine_finish_day(
        fixture->engine,
        fixture->model,
        (bolr_const_vector_view){NULL, 0, 1},
        &op,
        NULL,
        1.0,
        1.0,
        1,
        fixture->inference,
        NULL,
        NULL,
        NULL
    );
}

#endif
