#include "test_suite.h"
#include "test_checkpoint_fixture.h"

#include "bolr/checkpoint_codec.h"
#include "bolr/endian.h"

#include <stdlib.h>
#include <string.h>

static int compare_decisions(const bolr_decision *a, const bolr_decision *b) {
    if (a->selected_index != b->selected_index) return 0;
    if (a->selected_region_id != b->selected_region_id) return 0;
    return 1;
}

static int compare_posterior_means(bolr_replay_engine *left, bolr_replay_engine *right) {
    bolr_real left_mean[2];
    bolr_real right_mean[2];
    if (bolr_replay_engine_copy_posterior_mean(left, (bolr_vector_view){left_mean, 2, 1}) != BOLR_OK) return 0;
    if (bolr_replay_engine_copy_posterior_mean(right, (bolr_vector_view){right_mean, 2, 1}) != BOLR_OK) return 0;
    return bolr_test_checkpoint_vectors_close(left_mean, right_mean, 2, 1e-9);
}

int test_checkpoint_codec(void) {
    bolr_test_checkpoint_fixture fixture;
    bolr_test_checkpoint_fixture twin;
    bolr_replay_restore_context ctx;
    bolr_replay_engine *restored = NULL;
    void *encoded_once = NULL;
    void *encoded_twice = NULL;
    size_t encoded_size = 0U;
    bolr_decision decision_main;
    bolr_decision decision_twin;
    bolr_decision decision_restored;
    bolr_checkpoint_size_report report;
    int rc = 1;

    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_fixture_create(&twin) != BOLR_OK) goto cleanup;
    ctx = bolr_test_checkpoint_restore_context(fixture.model);

    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, &encoded_once, &encoded_size) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, &encoded_twice, &encoded_size) != BOLR_OK) goto cleanup;
    if (memcmp(encoded_once, encoded_twice, encoded_size) != 0) goto cleanup;

    if (bolr_replay_checkpoint_size_report(fixture.engine, &report) != BOLR_OK) goto cleanup;
    if (report.total_bytes != encoded_size) goto cleanup;
    if (report.header_bytes != BOLR_TEST_CHECKPOINT_HEADER_SIZE) goto cleanup;
    if (report.candidate_count != 0) goto cleanup;
    if (report.state_dimension != 2) goto cleanup;
    if ((report.header_bytes + report.directory_bytes + report.payload_bytes) != report.total_bytes) goto cleanup;

    if (bolr_replay_checkpoint_decode(encoded_once, encoded_size, &ctx, NULL, &restored) != BOLR_OK) goto cleanup;
    if (bolr_replay_engine_phase(restored) != BOLR_REPLAY_PHASE_READY) goto cleanup;
    bolr_replay_engine_destroy(fixture.engine);
    fixture.engine = NULL;
    if (bolr_replay_engine_begin_day(restored, twin.model, (bolr_const_vector_view){NULL, 0, 1}, &twin.ranking, twin.top_k, 2, twin.policy, NULL, NULL, twin.workspace, &decision_restored, NULL) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_begin_day(&twin, &decision_twin) != BOLR_OK) goto cleanup;
    if (!compare_decisions(&decision_restored, &decision_twin)) goto cleanup;
    bolr_replay_engine_destroy(restored);
    restored = NULL;
    bolr_test_checkpoint_fixture_destroy(&fixture);
    bolr_test_checkpoint_fixture_destroy(&twin);
    free(encoded_once);
    encoded_once = NULL;
    free(encoded_twice);
    encoded_twice = NULL;

    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_fixture_create(&twin) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_begin_day(&fixture, &decision_main) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_begin_day(&twin, &decision_twin) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, &encoded_once, &encoded_size) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_decode(encoded_once, encoded_size, &ctx, NULL, &restored) != BOLR_OK) goto cleanup;
    if (bolr_replay_engine_phase(restored) != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) goto cleanup;
    if (bolr_replay_engine_pending_selected_index(restored) != decision_main.selected_index) goto cleanup;
    bolr_replay_engine_destroy(fixture.engine);
    fixture.engine = NULL;
    if (bolr_test_checkpoint_finish_day(&twin) != BOLR_OK) goto cleanup;
    {
        bolr_observation_operator op;
        if (bolr_candidate_a_observation_operator(fixture.observation, &op) != BOLR_OK) goto cleanup;
        if (bolr_replay_engine_finish_day(restored, twin.model, (bolr_const_vector_view){NULL, 0, 1}, &op, NULL, 1.0, 1.0, 1, fixture.inference, NULL, NULL, NULL) != BOLR_OK) goto cleanup;
    }
    if (!compare_posterior_means(restored, twin.engine)) goto cleanup;
    bolr_replay_engine_destroy(restored);
    restored = NULL;
    bolr_test_checkpoint_fixture_destroy(&fixture);
    bolr_test_checkpoint_fixture_destroy(&twin);
    free(encoded_once);
    encoded_once = NULL;

    /* Regression: empty top_k leaves optional probability_top_k unallocated. */
    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_fixture_create(&twin) != BOLR_OK) goto cleanup;
    ctx = bolr_test_checkpoint_restore_context(fixture.model);
    if (bolr_replay_engine_begin_day(
            fixture.engine,
            fixture.model,
            (bolr_const_vector_view){NULL, 0, 1},
            &fixture.ranking,
            NULL,
            0,
            fixture.policy,
            NULL,
            NULL,
            fixture.workspace,
            &decision_main,
            NULL
        ) != BOLR_OK) goto cleanup;
    {
        bolr_replay_checkpoint *memory_checkpoint = NULL;
        if (bolr_replay_engine_export_checkpoint(fixture.engine, NULL, &memory_checkpoint) != BOLR_OK) goto cleanup;
        if (bolr_replay_checkpoint_phase(memory_checkpoint) != BOLR_REPLAY_PHASE_AWAITING_OUTCOME) {
            bolr_replay_checkpoint_destroy(memory_checkpoint);
            goto cleanup;
        }
        if (bolr_replay_checkpoint_pending_selected_index(memory_checkpoint) != decision_main.selected_index) {
            bolr_replay_checkpoint_destroy(memory_checkpoint);
            goto cleanup;
        }
        bolr_replay_checkpoint_destroy(memory_checkpoint);
    }
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, &encoded_once, &encoded_size) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_decode(encoded_once, encoded_size, &ctx, NULL, &restored) != BOLR_OK) goto cleanup;
    if (bolr_replay_engine_pending_selected_index(restored) != decision_main.selected_index) goto cleanup;
    bolr_replay_engine_destroy(fixture.engine);
    fixture.engine = NULL;
    {
        bolr_observation_operator op;
        if (bolr_candidate_a_observation_operator(fixture.observation, &op) != BOLR_OK) goto cleanup;
        if (bolr_replay_engine_finish_day(restored, fixture.model, (bolr_const_vector_view){NULL, 0, 1}, &op, NULL, 1.0, 1.0, 1, fixture.inference, NULL, NULL, NULL) != BOLR_OK) goto cleanup;
    }
    if (bolr_replay_engine_phase(restored) != BOLR_REPLAY_PHASE_READY) goto cleanup;

    /* Corrupt a payload byte under a valid header/directory: decode must fail and leave out_engine NULL. */
    bolr_replay_engine_destroy(restored);
    restored = NULL;
    free(encoded_once);
    encoded_once = NULL;
    if (bolr_replay_checkpoint_encode_buffer(twin.engine, NULL, &encoded_once, &encoded_size) != BOLR_OK) goto cleanup;
    if (encoded_size < BOLR_TEST_CHECKPOINT_HEADER_SIZE + 64U) goto cleanup;
    {
        unsigned char *bytes = (unsigned char *) encoded_once;
        uint64_t payload_offset = 0ULL;
        size_t cursor = 44U;
        if (bolr_decode_u64_le(bytes, encoded_size, &cursor, &payload_offset) != BOLR_OK) goto cleanup;
        if ((payload_offset >= encoded_size) || (payload_offset < BOLR_TEST_CHECKPOINT_HEADER_SIZE)) goto cleanup;
        bytes[payload_offset] ^= 0xFFu;
        if (bolr_replay_checkpoint_decode(encoded_once, encoded_size, &ctx, NULL, &restored) == BOLR_OK) goto cleanup;
        if (restored != NULL) goto cleanup;
    }
    rc = 0;

cleanup:
    bolr_replay_engine_destroy(restored);
    free(encoded_twice);
    free(encoded_once);
    bolr_test_checkpoint_fixture_destroy(&twin);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    return rc;
}
