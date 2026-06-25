#include "bolr/score.h"

#include <stddef.h>
#include <stdlib.h>

bolr_status bolr_composite_score_samples(
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_const_matrix_view state_samples,
    bolr_matrix_view output_scores,
    bolr_workspace *workspace,
    bolr_score_sampling_diagnostics *diagnostics
) {
    const bolr_allocator *active;
    bolr_workspace *local_workspace = NULL;
    bolr_index row;
    bolr_index sample_count;
    bolr_index candidate_count;
    bolr_index state_dim;
    bolr_status status;
    if ((model == NULL) || (bolr_matrix_view_validate(state_samples) != BOLR_OK) || (bolr_mutable_matrix_view_validate(output_scores) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    sample_count = state_samples.rows;
    candidate_count = bolr_model_score_count(model);
    state_dim = bolr_model_state_dim(model);
    if ((state_samples.cols != state_dim) || (output_scores.rows != sample_count) || (output_scores.cols != candidate_count)) return BOLR_INVALID_SHAPE;
    if (diagnostics != NULL) {
        diagnostics->sample_count = sample_count;
        diagnostics->candidate_count = candidate_count;
        diagnostics->state_dimension = state_dim;
    }
    if (sample_count == 0) return BOLR_OK;
    active = bolr_default_allocator();
    if (workspace == NULL) {
        bolr_workspace_config config = {candidate_count, state_dim, context.length};
        status = bolr_workspace_create(&config, active, &local_workspace);
        if (status != BOLR_OK) return status;
        workspace = local_workspace;
    }
    for (row = 0; row < sample_count; ++row) {
        status = bolr_model_forward(
            model,
            (bolr_const_vector_view){state_samples.data + row * state_samples.row_stride, state_dim, state_samples.col_stride},
            context,
            (bolr_vector_view){output_scores.data + row * output_scores.row_stride, candidate_count, output_scores.col_stride},
            workspace
        );
        if (status != BOLR_OK) {
            bolr_workspace_destroy(local_workspace);
            return status;
        }
    }
    bolr_workspace_destroy(local_workspace);
    return BOLR_OK;
}

bolr_status bolr_posterior_score_sample(
    const bolr_gaussian_state *state,
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_rng *rng,
    bolr_index sample_count,
    int antithetic,
    bolr_matrix_view output_scores,
    bolr_sampling_diagnostics *diagnostics,
    bolr_workspace *workspace
) {
    bolr_real *state_samples;
    bolr_status status;
    bolr_score_sampling_diagnostics ignored;
    bolr_index state_dim;
    if ((state == NULL) || (model == NULL) || (rng == NULL)) return BOLR_INVALID_ARGUMENT;
    state_dim = bolr_gaussian_state_dimension(state);
    if (output_scores.rows != sample_count) return BOLR_INVALID_SHAPE;
    if (sample_count == 0) {
        if (diagnostics != NULL) {
            diagnostics->sample_count = 0;
            diagnostics->state_dimension = state_dim;
            diagnostics->antithetic = antithetic ? 1 : 0;
            diagnostics->normal_draw_count = 0ULL;
            diagnostics->cholesky_jitter = 0.0;
            diagnostics->minimum_cholesky_diagonal = 0.0;
        }
        return BOLR_OK;
    }
    state_samples = (bolr_real *) malloc((size_t) (sample_count * state_dim) * sizeof(bolr_real));
    if (state_samples == NULL) return BOLR_ALLOCATION_FAILED;
    status = bolr_gaussian_state_sample(
        state,
        rng,
        sample_count,
        antithetic,
        (bolr_matrix_view){state_samples, sample_count, state_dim, state_dim, 1},
        diagnostics,
        workspace
    );
    if (status != BOLR_OK) {
        free(state_samples);
        return status;
    }
    status = bolr_composite_score_samples(
        model,
        context,
        (bolr_const_matrix_view){state_samples, sample_count, state_dim, state_dim, 1},
        output_scores,
        workspace,
        &ignored
    );
    free(state_samples);
    return status;
}
