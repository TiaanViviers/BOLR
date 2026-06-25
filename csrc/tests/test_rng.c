#include "bolr/rng.h"
#include "test_suite.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int check_sequence(bolr_rng_seed seed, const uint32_t *expected, size_t count) {
    bolr_rng *rng = NULL;
    size_t i;
    if (bolr_rng_create(seed, NULL, &rng) != BOLR_OK) return 1;
    for (i = 0; i < count; ++i) {
        uint32_t value = 0U;
        if (bolr_rng_u32(rng, &value) != BOLR_OK) return 1;
        if (value != expected[i]) return 1;
    }
    bolr_rng_destroy(rng);
    return 0;
}

int test_rng(void) {
    static const uint32_t expected_seed0_stream0[10] = {
        3837872008u, 932996374u, 1548399547u, 1612522464u, 473443212u,
        3522865942u, 1734871597u, 2449558126u, 1653269385u, 2811495245u
    };
    static const uint32_t expected_seed1_stream0[10] = {
        3795398737u, 17903413u, 3545275701u, 194195274u, 2326030198u,
        2354257974u, 2697798104u, 3102124240u, 3710314595u, 2357030906u
    };
    bolr_rng *left = NULL;
    bolr_rng *right = NULL;
    bolr_rng *clone = NULL;
    bolr_rng_checkpoint *checkpoint = NULL;
    bolr_rng_checkpoint *decoded = NULL;
    bolr_rng *restored = NULL;
    bolr_rng_metadata metadata;
    bolr_real uniform = 0.0;
    bolr_real normals_left[8];
    bolr_real normals_right[8];
    unsigned char *payload = NULL;
    size_t payload_size = 0U;
    size_t written = 0U;
    int rc = 1;
    size_t i;
    uint32_t left_first = 0U;
    uint32_t right_first = 0U;

    if (check_sequence((bolr_rng_seed){0ULL, 0ULL}, expected_seed0_stream0, 10U) != 0) return 1;
    if (check_sequence((bolr_rng_seed){1ULL, 0ULL}, expected_seed1_stream0, 10U) != 0) return 1;

    if (bolr_rng_create((bolr_rng_seed){17ULL, 3ULL}, NULL, &left) != BOLR_OK) goto cleanup;
    if (bolr_rng_create((bolr_rng_seed){17ULL, 4ULL}, NULL, &right) != BOLR_OK) goto cleanup;
    if (bolr_rng_u32(left, &left_first) != BOLR_OK) goto cleanup;
    if (bolr_rng_u32(right, &right_first) != BOLR_OK) goto cleanup;
    if (left_first == right_first) goto cleanup;

    bolr_rng_destroy(left);
    left = NULL;
    if (bolr_rng_create((bolr_rng_seed){9ULL, 5ULL}, NULL, &left) != BOLR_OK) goto cleanup;
    if (bolr_rng_clone(left, NULL, &clone) != BOLR_OK) goto cleanup;
    for (i = 0; i < 8U; ++i) {
        if (bolr_rng_standard_normal(left, &normals_left[i]) != BOLR_OK) goto cleanup;
        if (bolr_rng_standard_normal(clone, &normals_right[i]) != BOLR_OK) goto cleanup;
        if (memcmp(&normals_left[i], &normals_right[i], sizeof(bolr_real)) != 0) goto cleanup;
    }
    if (bolr_rng_uniform_open01(left, &uniform) != BOLR_OK) goto cleanup;
    if (!(uniform > 0.0 && uniform < 1.0)) goto cleanup;
    if (bolr_rng_metadata_copy(left, &metadata) != BOLR_OK) goto cleanup;
    if ((metadata.stream != 5ULL) || (metadata.algorithm_family != 1U) || (metadata.ziggurat_layers != 128U)) goto cleanup;

    if (bolr_rng_export(left, NULL, &checkpoint) != BOLR_OK) goto cleanup;
    if (bolr_rng_checkpoint_encoded_size(checkpoint, &payload_size) != BOLR_OK) goto cleanup;
    payload = (unsigned char *) malloc(payload_size);
    if (payload == NULL) goto cleanup;
    if (bolr_rng_checkpoint_encode(checkpoint, payload, payload_size, &written) != BOLR_OK) goto cleanup;
    if (written != payload_size) goto cleanup;
    if (bolr_rng_checkpoint_decode(payload, payload_size, NULL, &decoded) != BOLR_OK) goto cleanup;
    if (bolr_rng_import(decoded, NULL, &restored) != BOLR_OK) goto cleanup;
    for (i = 0; i < 8U; ++i) {
        bolr_real a = 0.0;
        bolr_real b = 0.0;
        if (bolr_rng_standard_normal(left, &a) != BOLR_OK) goto cleanup;
        if (bolr_rng_standard_normal(restored, &b) != BOLR_OK) goto cleanup;
        if (memcmp(&a, &b, sizeof(bolr_real)) != 0) goto cleanup;
    }
    payload[16] &= 0xfeU;
    if (bolr_rng_checkpoint_decode(payload, payload_size, NULL, &decoded) != BOLR_INCOMPATIBLE_CHECKPOINT) goto cleanup;

    rc = 0;
cleanup:
    free(payload);
    bolr_rng_destroy(left);
    bolr_rng_destroy(right);
    bolr_rng_destroy(clone);
    bolr_rng_destroy(restored);
    bolr_rng_checkpoint_destroy(checkpoint);
    bolr_rng_checkpoint_destroy(decoded);
    return rc;
}
