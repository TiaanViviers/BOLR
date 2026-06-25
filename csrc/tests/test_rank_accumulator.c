#include "test_suite.h"

#include "bolr/prediction.h"

#include <math.h>
#include <string.h>

static int vector_close(const bolr_real *left, const bolr_real *right, bolr_index length, bolr_real tol) {
    bolr_index i;
    for (i = 0; i < length; ++i) {
        if (fabs(left[i] - right[i]) > tol) return 0;
    }
    return 1;
}

int test_rank_accumulator(void) {
    bolr_rank_accumulator *full = NULL;
    bolr_rank_accumulator *left = NULL;
    bolr_rank_accumulator *right = NULL;
    const bolr_index top_k[] = {1, 2};
    const bolr_real score_rows[] = {
        2.0, 1.0, 0.0,
        1.0, 3.0, 2.0,
        0.5, 0.5, 0.1,
        4.0, 2.0, 3.0
    };
    bolr_real full_best[3];
    bolr_real merged_best[3];
    bolr_real full_top2[3];
    bolr_real merged_top2[3];
    bolr_real full_rank[3];
    bolr_real merged_rank[3];
    bolr_real full_std[3];
    bolr_real merged_std[3];
    const bolr_const_matrix_view all_scores = {score_rows, 4, 3, 3, 1};
    const bolr_const_matrix_view first_half = {score_rows, 2, 3, 3, 1};
    const bolr_const_matrix_view second_half = {score_rows + 6, 2, 3, 3, 1};
    int rc = 1;

    if (bolr_rank_accumulator_create(3, top_k, 2, NULL, &full) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_create(3, top_k, 2, NULL, &left) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_create(3, top_k, 2, NULL, &right) != BOLR_OK) goto cleanup;

    if (bolr_rank_accumulator_accumulate_scores(full, all_scores) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_accumulate_scores(left, first_half) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_accumulate_scores(right, second_half) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_merge(left, right) != BOLR_OK) goto cleanup;

    if (bolr_rank_accumulator_sample_count(full) != bolr_rank_accumulator_sample_count(left)) goto cleanup;
    if (bolr_rank_accumulator_tie_count(full) != bolr_rank_accumulator_tie_count(left)) goto cleanup;

    if (bolr_rank_accumulator_copy_probability_best(full, (bolr_vector_view){full_best, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_probability_best(left, (bolr_vector_view){merged_best, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_probability_top_k(full, 2, (bolr_vector_view){full_top2, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_probability_top_k(left, 2, (bolr_vector_view){merged_top2, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_expected_rank(full, (bolr_vector_view){full_rank, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_expected_rank(left, (bolr_vector_view){merged_rank, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_rank_stddev(full, (bolr_vector_view){full_std, 3, 1}) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_copy_rank_stddev(left, (bolr_vector_view){merged_std, 3, 1}) != BOLR_OK) goto cleanup;

    if (!vector_close(full_best, merged_best, 3, 1e-12)) goto cleanup;
    if (!vector_close(full_top2, merged_top2, 3, 1e-12)) goto cleanup;
    if (!vector_close(full_rank, merged_rank, 3, 1e-12)) goto cleanup;
    if (!vector_close(full_std, merged_std, 3, 1e-12)) goto cleanup;

    if (bolr_rank_accumulator_reset(left) != BOLR_OK) goto cleanup;
    if (bolr_rank_accumulator_sample_count(left) != 0U) goto cleanup;
    if (bolr_rank_accumulator_tie_count(left) != 0) goto cleanup;
    if (bolr_rank_accumulator_accumulate_scores(left, all_scores) != BOLR_OK) goto cleanup;
    memset(merged_best, 0, sizeof(merged_best));
    if (bolr_rank_accumulator_copy_probability_best(left, (bolr_vector_view){merged_best, 3, 1}) != BOLR_OK) goto cleanup;
    if (!vector_close(full_best, merged_best, 3, 1e-12)) goto cleanup;

    rc = 0;
cleanup:
    bolr_rank_accumulator_destroy(full);
    bolr_rank_accumulator_destroy(left);
    bolr_rank_accumulator_destroy(right);
    return rc;
}
