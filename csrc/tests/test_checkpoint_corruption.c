#include "test_suite.h"
#include "test_checkpoint_fixture.h"

#include "bolr/checkpoint_codec.h"
#include "bolr/endian.h"

#include <stdlib.h>
#include <string.h>

static uint32_t corruption_lcg(uint32_t state) {
    return (1103515245U * state + 12345U);
}

static int decode_fails_safe(bolr_status status) {
    return (status != BOLR_OK);
}

int test_checkpoint_corruption(void) {
    bolr_test_checkpoint_fixture fixture;
    bolr_replay_restore_context ctx;
    bolr_replay_restore_context bad_ctx;
    bolr_replay_engine *restored = NULL;
    unsigned char *buffer = NULL;
    size_t size = 0U;
    size_t header_only = BOLR_TEST_CHECKPOINT_HEADER_SIZE;
    size_t partial_directory = BOLR_TEST_CHECKPOINT_HEADER_SIZE + 20U;
    uint32_t lcg = 1U;
    size_t i;
    int rc = 1;

    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    ctx = bolr_test_checkpoint_restore_context(fixture.model);
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, (void **) &buffer, &size) != BOLR_OK) goto cleanup;

    memcpy(buffer, "XOLRCP01", 8U);
    if (bolr_replay_checkpoint_decode(buffer, size, &ctx, NULL, &restored) != BOLR_CHECKPOINT_BAD_MAGIC) goto cleanup;

    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, (void **) &buffer, &size) != BOLR_OK) goto cleanup;
    {
        size_t cursor = 44U;
        uint64_t payload_offset = 0ULL;
        if (bolr_decode_u64_le(buffer, size, &cursor, &payload_offset) != BOLR_OK) goto cleanup;
        if (payload_offset + 1U >= size) goto cleanup;
        buffer[payload_offset + 1U] ^= 0x5AU;
    }
    if (bolr_replay_checkpoint_decode(buffer, size, &ctx, NULL, &restored) != BOLR_CHECKPOINT_CHECKSUM_MISMATCH) goto cleanup;

    if (!decode_fails_safe(bolr_replay_checkpoint_decode(buffer, 0U, &ctx, NULL, &restored))) goto cleanup;
    if (!decode_fails_safe(bolr_replay_checkpoint_decode(buffer, 4U, &ctx, NULL, &restored))) goto cleanup;
    if (bolr_replay_checkpoint_decode(buffer, header_only, &ctx, NULL, &restored) != BOLR_CHECKPOINT_TRUNCATED) goto cleanup;
    if (!decode_fails_safe(bolr_replay_checkpoint_decode(buffer, partial_directory, &ctx, NULL, &restored))) goto cleanup;

    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, (void **) &buffer, &size) != BOLR_OK) goto cleanup;
    buffer[72U] = (unsigned char) BOLR_REPLAY_PHASE_AWAITING_OUTCOME;
    {
        bolr_status phase_status = bolr_replay_checkpoint_decode(buffer, size, &ctx, NULL, &restored);
        if ((phase_status != BOLR_CHECKPOINT_INVALID_DIRECTORY) && (phase_status != BOLR_CHECKPOINT_CHECKSUM_MISMATCH)) goto cleanup;
    }

    bad_ctx = ctx;
    bad_ctx.expected_model_schema_hash ^= 1ULL;
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, (void **) &buffer, &size) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_decode(buffer, size, &bad_ctx, NULL, &restored) != BOLR_INCOMPATIBLE_CHECKPOINT) goto cleanup;

    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, (void **) &buffer, &size) != BOLR_OK) goto cleanup;
    {
        size_t cursor = 44U;
        uint64_t payload_offset = 0ULL;
        uint64_t payload_size = 0ULL;
        if (bolr_decode_u64_le(buffer, size, &cursor, &payload_offset) != BOLR_OK) goto cleanup;
        cursor = 52U;
        if (bolr_decode_u64_le(buffer, size, &cursor, &payload_size) != BOLR_OK) goto cleanup;
        if (payload_size == 0ULL) goto cleanup;
        for (i = 0U; i < 200U; ++i) {
            size_t flip_index;
            bolr_status status;
            lcg = corruption_lcg(lcg);
            flip_index = (size_t) payload_offset + (size_t) ((lcg % (uint32_t) payload_size));
            buffer[flip_index] ^= (unsigned char) ((lcg % 255U) + 1U);
            status = bolr_replay_checkpoint_decode(buffer, size, &ctx, NULL, &restored);
            if (status == BOLR_OK) goto cleanup;
            if (restored != NULL) {
                bolr_replay_engine_destroy(restored);
                restored = NULL;
            }
            if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, (void **) &buffer, &size) != BOLR_OK) goto cleanup;
        }
    }
    rc = 0;

cleanup:
    bolr_replay_engine_destroy(restored);
    free(buffer);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    return rc;
}
