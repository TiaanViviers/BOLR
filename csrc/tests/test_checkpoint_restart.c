#include "test_suite.h"
#include "test_checkpoint_fixture.h"

#include "bolr/checkpoint_codec.h"
#include "bolr/checkpoint_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int compare_posterior_means(bolr_replay_engine *left, bolr_replay_engine *right) {
    bolr_real left_mean[2];
    bolr_real right_mean[2];
    if (bolr_replay_engine_copy_posterior_mean(left, (bolr_vector_view){left_mean, 2, 1}) != BOLR_OK) return 0;
    if (bolr_replay_engine_copy_posterior_mean(right, (bolr_vector_view){right_mean, 2, 1}) != BOLR_OK) return 0;
    return bolr_test_checkpoint_vectors_close(left_mean, right_mean, 2, 1e-9);
}

int test_checkpoint_restart(void) {
    bolr_test_checkpoint_fixture fixture;
    bolr_test_checkpoint_fixture twin;
    bolr_replay_restore_context ctx;
    bolr_replay_engine *restored = NULL;
    bolr_decision decision_twin;
    bolr_decision decision_restored;
    bolr_decision decision_scratch;
    bolr_checkpoint_file_options options;
    char ready_path[512];
    char pending_path[512];
    int rc = 1;

    bolr_checkpoint_io_hooks_reset();
    (void) system("mkdir -p build/l4b23-debug-gcc");
    snprintf(ready_path, sizeof(ready_path), "build/l4b23-debug-gcc/bolr_checkpoint_ready_%d.cp", (int) getpid());
    snprintf(pending_path, sizeof(pending_path), "build/l4b23-debug-gcc/bolr_checkpoint_pending_%d.cp", (int) getpid());
    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_fixture_create(&twin) != BOLR_OK) goto cleanup;
    ctx = bolr_test_checkpoint_restore_context(fixture.model);
    options = bolr_checkpoint_file_options_default();
    options.fsync_directory = 0;
    options.replace_existing = 1;

    if (bolr_test_checkpoint_begin_day(&fixture, &decision_scratch) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_finish_day(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_write_atomic(fixture.engine, ready_path, &options) != BOLR_OK) goto cleanup;

    if (bolr_test_checkpoint_begin_day(&twin, &decision_scratch) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_finish_day(&twin) != BOLR_OK) goto cleanup;

    if (bolr_replay_checkpoint_read_file(ready_path, &ctx, NULL, &restored) != BOLR_OK) goto cleanup;
    if (bolr_replay_engine_phase(restored) != BOLR_REPLAY_PHASE_READY) goto cleanup;
    if (bolr_replay_engine_begin_day(restored, twin.model, (bolr_const_vector_view){NULL, 0, 1}, &twin.ranking, twin.top_k, 2, twin.policy, NULL, NULL, twin.workspace, &decision_restored, NULL) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_begin_day(&twin, &decision_twin) != BOLR_OK) goto cleanup;
    if (decision_restored.selected_index != decision_twin.selected_index) goto cleanup;
    {
        bolr_observation_operator op;
        if (bolr_candidate_a_observation_operator(twin.observation, &op) != BOLR_OK) goto cleanup;
        if (bolr_replay_engine_finish_day(restored, twin.model, (bolr_const_vector_view){NULL, 0, 1}, &op, NULL, 1.0, 1.0, 1, twin.inference, NULL, NULL, NULL) != BOLR_OK) goto cleanup;
        if (bolr_test_checkpoint_finish_day(&twin) != BOLR_OK) goto cleanup;
    }
    if (!compare_posterior_means(restored, twin.engine)) goto cleanup;
    bolr_replay_engine_destroy(restored);
    restored = NULL;

    bolr_test_checkpoint_fixture_destroy(&twin);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    memset(&fixture, 0, sizeof(fixture));
    memset(&twin, 0, sizeof(twin));
    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_fixture_create(&twin) != BOLR_OK) goto cleanup;
    ctx = bolr_test_checkpoint_restore_context(fixture.model);

    if (bolr_test_checkpoint_begin_day(&fixture, &decision_restored) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_begin_day(&twin, &decision_twin) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_write_atomic(fixture.engine, pending_path, &options) != BOLR_OK) goto cleanup;
    bolr_replay_engine_destroy(fixture.engine);
    fixture.engine = NULL;

    if (bolr_replay_checkpoint_read_file(pending_path, &ctx, NULL, &restored) != BOLR_OK) goto cleanup;
    if (bolr_replay_engine_phase(restored) != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) goto cleanup;
    if (bolr_replay_engine_pending_selected_index(restored) != decision_restored.selected_index) goto cleanup;
    {
        bolr_observation_operator op;
        if (bolr_candidate_a_observation_operator(fixture.observation, &op) != BOLR_OK) goto cleanup;
        if (bolr_replay_engine_finish_day(restored, twin.model, (bolr_const_vector_view){NULL, 0, 1}, &op, NULL, 1.0, 1.0, 1, fixture.inference, NULL, NULL, NULL) != BOLR_OK) goto cleanup;
        if (bolr_test_checkpoint_finish_day(&twin) != BOLR_OK) goto cleanup;
    }
    if (!compare_posterior_means(restored, twin.engine)) goto cleanup;

    rc = 0;

cleanup:
    bolr_replay_engine_destroy(restored);
    unlink(ready_path);
    unlink(pending_path);
    bolr_test_checkpoint_fixture_destroy(&twin);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    bolr_checkpoint_io_hooks_reset();
    return rc;
}
