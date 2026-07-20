#include "test_suite.h"
#include "test_checkpoint_fixture.h"

#include "bolr/checkpoint_codec.h"
#include "bolr/checkpoint_sections.h"
#include "bolr/endian.h"

#include <stdlib.h>
#include <string.h>

static int section_type_present(const uint32_t *types, size_t count, uint32_t target) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (types[i] == target) return 1;
    }
    return 0;
}

static int parse_section_types(const unsigned char *bytes, size_t size, uint32_t *out_types, size_t max_types, size_t *out_count) {
    uint32_t section_count = 0U;
    size_t cursor = 20U;
    size_t i;
    if ((bytes == NULL) || (out_types == NULL) || (out_count == NULL)) return 0;
    if (size < BOLR_TEST_CHECKPOINT_HEADER_SIZE) return 0;
    if (memcmp(bytes, BOLR_CHECKPOINT_MAGIC, 8U) != 0) return 0;
    if (bolr_decode_u32_le(bytes, size, &cursor, &section_count) != BOLR_OK) return 0;
    if (section_count > max_types) return 0;
    cursor = BOLR_TEST_CHECKPOINT_HEADER_SIZE;
    for (i = 0U; i < (size_t) section_count; ++i) {
        uint16_t schema_major = 0U;
        uint16_t schema_minor = 0U;
        uint32_t section_flags = 0U;
        uint64_t payload_offset = 0ULL;
        uint64_t payload_length = 0ULL;
        uint64_t element_count = 0ULL;
        uint32_t section_crc = 0U;
        uint32_t reserved = 0U;
        if (bolr_decode_u32_le(bytes, size, &cursor, &out_types[i]) != BOLR_OK) return 0;
        if (bolr_decode_u16_le(bytes, size, &cursor, &schema_major) != BOLR_OK) return 0;
        if (bolr_decode_u16_le(bytes, size, &cursor, &schema_minor) != BOLR_OK) return 0;
        if (bolr_decode_u32_le(bytes, size, &cursor, &section_flags) != BOLR_OK) return 0;
        if (bolr_decode_u64_le(bytes, size, &cursor, &payload_offset) != BOLR_OK) return 0;
        if (bolr_decode_u64_le(bytes, size, &cursor, &payload_length) != BOLR_OK) return 0;
        if (bolr_decode_u64_le(bytes, size, &cursor, &element_count) != BOLR_OK) return 0;
        if (bolr_decode_u32_le(bytes, size, &cursor, &section_crc) != BOLR_OK) return 0;
        if (bolr_decode_u32_le(bytes, size, &cursor, &reserved) != BOLR_OK) return 0;
        (void) schema_major;
        (void) schema_minor;
        (void) section_flags;
        (void) payload_offset;
        (void) payload_length;
        (void) element_count;
        (void) section_crc;
        (void) reserved;
    }
    *out_count = (size_t) section_count;
    return 1;
}

static int verify_ready_sections(const unsigned char *bytes, size_t size) {
    uint32_t types[32];
    size_t count = 0U;
    if (!parse_section_types(bytes, size, types, 32U, &count)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_REPLAY_METADATA)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_GAUSSIAN_POSTERIOR)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_RNG_STATE)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_TRANSITION_CONFIG)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_DECISION_CONFIG)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_MONTE_CARLO_CONFIG)) return 0;
    return 1;
}

static int verify_awaiting_sections(const unsigned char *bytes, size_t size) {
    uint32_t types[32];
    size_t count = 0U;
    if (!parse_section_types(bytes, size, types, 32U, &count)) return 0;
    if (!verify_ready_sections(bytes, size)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_PENDING_DAY_METADATA)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_PENDING_SCORE_CONTEXT)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_PENDING_PREDICTIVE_GAUSSIAN)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_PENDING_POSTERIOR_PREDICTION)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_PENDING_RANK_SUMMARY)) return 0;
    if (!section_type_present(types, count, BOLR_CHECKPOINT_SECTION_PENDING_DECISION)) return 0;
    return 1;
}

int test_checkpoint_sections(void) {
    bolr_test_checkpoint_fixture fixture;
    void *ready_buf = NULL;
    void *await_buf = NULL;
    size_t ready_size = 0U;
    size_t await_size = 0U;
    bolr_decision decision;
    int rc = 1;

    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, &ready_buf, &ready_size) != BOLR_OK) goto cleanup;
    if (!verify_ready_sections((const unsigned char *) ready_buf, ready_size)) goto cleanup;
    if (bolr_test_checkpoint_begin_day(&fixture, &decision) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_encode_buffer(fixture.engine, NULL, &await_buf, &await_size) != BOLR_OK) goto cleanup;
    if (!verify_awaiting_sections((const unsigned char *) await_buf, await_size)) goto cleanup;
    rc = 0;

cleanup:
    free(await_buf);
    free(ready_buf);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    return rc;
}
