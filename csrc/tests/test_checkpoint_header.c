#include "test_suite.h"
#include "test_checkpoint_fixture.h"

#include "bolr/checkpoint_codec.h"
#include "bolr/checkpoint_sections.h"
#include "bolr/endian.h"

#include <stdlib.h>
#include <string.h>

int test_checkpoint_header(void) {
    bolr_test_checkpoint_fixture fixture;
    void *buffer = NULL;
    size_t encoded_size = 0U;
    size_t written = 0U;
    uint16_t format_major = 0U;
    uint16_t format_minor = 0U;
    size_t cursor = 0U;
    int rc = 1;

    if (bolr_platform_validate() != BOLR_OK) goto cleanup;
    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_encoded_size(fixture.engine, &encoded_size) != BOLR_OK) goto cleanup;
    if (encoded_size < BOLR_TEST_CHECKPOINT_HEADER_SIZE) goto cleanup;
    buffer = malloc(encoded_size);
    if (buffer == NULL) goto cleanup;
    if (bolr_replay_checkpoint_encode(fixture.engine, buffer, encoded_size, &written) != BOLR_OK) goto cleanup;
    if (written != encoded_size) goto cleanup;
    if (memcmp(buffer, BOLR_CHECKPOINT_MAGIC, 8U) != 0) goto cleanup;
    cursor = 8U;
    if (bolr_decode_u16_le(buffer, encoded_size, &cursor, &format_major) != BOLR_OK) goto cleanup;
    if (bolr_decode_u16_le(buffer, encoded_size, &cursor, &format_minor) != BOLR_OK) goto cleanup;
    if ((format_major != BOLR_CHECKPOINT_FORMAT_MAJOR) || (format_minor != BOLR_CHECKPOINT_FORMAT_MINOR)) goto cleanup;
    if (bolr_replay_engine_phase(fixture.engine) != BOLR_REPLAY_PHASE_READY) goto cleanup;
    rc = 0;

cleanup:
    free(buffer);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    return rc;
}
