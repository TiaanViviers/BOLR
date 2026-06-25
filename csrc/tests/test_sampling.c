#include "bolr/gaussian.h"
#include "bolr/rng.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"
#include "bolr/workspace.h"
#include "test_suite.h"

#include <math.h>
#include <stdlib.h>

int test_sampling(void) {
    static const bolr_real mean_data[2] = {0.5, -0.25};
    static const bolr_real covariance_data[4] = {1.0, 0.3, 0.3, 2.0};
    static const bolr_real static_scores[3] = {0.2, -0.1, 0.0};
    static const bolr_real design_data[6] = {1.0, 0.0, 0.0, 1.0, 1.0, -1.0};
    static const bolr_real context_empty[1] = {0.0};
    bolr_state_layout *layout = NULL;
    bolr_state_block_spec block = {"surface", 0, 2, 1, 2, 1, 'r'};
    bolr_model *model = NULL;
    bolr_gaussian_state *state = NULL;
    bolr_rng *rng = NULL;
    bolr_workspace *workspace = NULL;
    bolr_real state_samples[12];
    bolr_real score_samples[18];
    bolr_real posterior_scores[18];
    bolr_sampling_diagnostics diagnostics;
    bolr_score_sampling_diagnostics score_diag;
    bolr_index row;
    bolr_index col;
    int rc = 1;

    if (bolr_state_layout_create(&block, 1, NULL, &layout) != BOLR_OK) goto cleanup;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 3, 1}, NULL, &model) != BOLR_OK) goto cleanup;
    if (bolr_model_add_dense_block_copy(model, "surface", (bolr_const_matrix_view){design_data, 3, 2, 2, 1}) != BOLR_OK) goto cleanup;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean_data, 2, 1}, (bolr_const_matrix_view){covariance_data, 2, 2, 2, 1}, 11ULL, 13ULL, NULL, &state) != BOLR_OK) goto cleanup;
    if (bolr_rng_create((bolr_rng_seed){123ULL, 7ULL}, NULL, &rng) != BOLR_OK) goto cleanup;
    if (bolr_workspace_create(&(bolr_workspace_config){3, 2, 0}, NULL, &workspace) != BOLR_OK) goto cleanup;

    if (bolr_gaussian_state_sample(state, rng, 6, 1, (bolr_matrix_view){state_samples, 6, 2, 2, 1}, &diagnostics, workspace) != BOLR_OK) goto cleanup;
    if ((diagnostics.sample_count != 6) || (diagnostics.state_dimension != 2) || (diagnostics.antithetic != 1)) goto cleanup;
    if (fabs(state_samples[0] + state_samples[6] - (2.0 * mean_data[0])) > 1e-12) goto cleanup;
    if (fabs(state_samples[1] + state_samples[7] - (2.0 * mean_data[1])) > 1e-12) goto cleanup;
    if (fabs(state_samples[2] + state_samples[8] - (2.0 * mean_data[0])) > 1e-12) goto cleanup;
    if (fabs(state_samples[3] + state_samples[9] - (2.0 * mean_data[1])) > 1e-12) goto cleanup;

    if (bolr_composite_score_samples(
            model,
            (bolr_const_vector_view){context_empty, 0, 1},
            (bolr_const_matrix_view){state_samples, 6, 2, 2, 1},
            (bolr_matrix_view){score_samples, 6, 3, 3, 1},
            workspace,
            &score_diag
        ) != BOLR_OK) goto cleanup;
    if ((score_diag.sample_count != 6) || (score_diag.candidate_count != 3) || (score_diag.state_dimension != 2)) goto cleanup;
    for (row = 0; row < 6; ++row) {
        bolr_real theta0 = state_samples[row * 2];
        bolr_real theta1 = state_samples[row * 2 + 1];
        bolr_real expected[3] = {0.2 + theta0, -0.1 + theta1, theta0 - theta1};
        for (col = 0; col < 3; ++col) {
            if (fabs(score_samples[row * 3 + col] - expected[col]) > 1e-12) goto cleanup;
        }
    }

    bolr_rng_destroy(rng);
    rng = NULL;
    if (bolr_rng_create((bolr_rng_seed){123ULL, 7ULL}, NULL, &rng) != BOLR_OK) goto cleanup;
    if (bolr_posterior_score_sample(
            state,
            model,
            (bolr_const_vector_view){context_empty, 0, 1},
            rng,
            6,
            1,
            (bolr_matrix_view){posterior_scores, 6, 3, 3, 1},
            &diagnostics,
            workspace
        ) != BOLR_OK) goto cleanup;
    for (row = 0; row < 18; ++row) {
        if (fabs(score_samples[row] - posterior_scores[row]) > 1e-12) goto cleanup;
    }

    rc = 0;
cleanup:
    bolr_workspace_destroy(workspace);
    bolr_rng_destroy(rng);
    bolr_gaussian_state_destroy(state);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    return rc;
}
