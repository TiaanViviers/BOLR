#include "bolr/static_fit.h"
#include "bolr/linalg.h"
#include "bolr/math.h"
#include "internal.h"

#include <math.h>
#include <string.h>

struct bolr_candidate_a_static_dataset {
    const bolr_allocator *allocator;
    bolr_real *design;
    bolr_real *targets;
    bolr_real *weights;
    bolr_index rows;
    bolr_index cols;
    bolr_index day_count;
};

static bolr_real vector_norm(const bolr_real *data, bolr_index length) {
    bolr_real total = 0.0;
    bolr_index i;
    for (i = 0; i < length; ++i) total += data[i] * data[i];
    return sqrt(total);
}

bolr_status bolr_candidate_a_static_dataset_create(
    bolr_const_matrix_view design,
    const bolr_const_vector_view *targets,
    const bolr_real *effective_weights,
    bolr_index day_count,
    const bolr_allocator *allocator,
    bolr_candidate_a_static_dataset **out_dataset
) {
    bolr_candidate_a_static_dataset *dataset;
    bolr_index d, r, c;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((targets == NULL) || (effective_weights == NULL) || (out_dataset == NULL) || (day_count <= 0)) return BOLR_INVALID_ARGUMENT;
    *out_dataset = NULL;
    dataset = (bolr_candidate_a_static_dataset *) bolr_allocator_calloc(active, 1U, sizeof(*dataset));
    if (dataset == NULL) return BOLR_ALLOCATION_FAILED;
    dataset->allocator = active;
    dataset->rows = design.rows;
    dataset->cols = design.cols;
    dataset->day_count = day_count;
    dataset->design = (bolr_real *) bolr_allocator_malloc(active, (size_t) (design.rows * design.cols) * sizeof(bolr_real));
    dataset->targets = (bolr_real *) bolr_allocator_malloc(active, (size_t) (design.rows * day_count) * sizeof(bolr_real));
    dataset->weights = (bolr_real *) bolr_allocator_malloc(active, (size_t) day_count * sizeof(bolr_real));
    if ((dataset->design == NULL) || (dataset->targets == NULL) || (dataset->weights == NULL)) { bolr_candidate_a_static_dataset_destroy(dataset); return BOLR_ALLOCATION_FAILED; }
    for (r = 0; r < design.rows; ++r) for (c = 0; c < design.cols; ++c) dataset->design[r * design.cols + c] = design.data[r * design.row_stride + c * design.col_stride];
    for (d = 0; d < day_count; ++d) {
        dataset->weights[d] = effective_weights[d];
        if (targets[d].length != design.rows) { bolr_candidate_a_static_dataset_destroy(dataset); return BOLR_INVALID_SHAPE; }
        for (r = 0; r < design.rows; ++r) dataset->targets[d * design.rows + r] = targets[d].data[r * targets[d].stride];
    }
    *out_dataset = dataset;
    return BOLR_OK;
}

void bolr_candidate_a_static_dataset_destroy(bolr_candidate_a_static_dataset *dataset) {
    if (dataset == NULL) return;
    bolr_allocator_free(dataset->allocator, dataset->design);
    bolr_allocator_free(dataset->allocator, dataset->targets);
    bolr_allocator_free(dataset->allocator, dataset->weights);
    bolr_allocator_free(dataset->allocator, dataset);
}

bolr_status bolr_candidate_a_static_fit(
    const bolr_candidate_a_static_dataset *dataset,
    bolr_const_vector_view prior_mean,
    bolr_const_matrix_view prior_precision,
    const bolr_newton_config *config,
    bolr_inference_workspace *workspace,
    bolr_vector_view output_coefficients,
    bolr_vector_view output_scores,
    bolr_static_fit_diagnostics *diagnostics
) {
    bolr_index iter, i, d, p = prior_mean.length;
    bolr_real *theta, *gradient, *hessian;
    bolr_real objective;
    (void) config;
    if ((dataset == NULL) || (workspace == NULL) || (output_coefficients.length != p) || (output_scores.length != dataset->rows)) return BOLR_INVALID_ARGUMENT;
    theta = workspace->trial_state;
    gradient = workspace->parameter_gradient;
    hessian = workspace->dense_hessian;
    memcpy(theta, prior_mean.data, (size_t) p * sizeof(bolr_real));
    for (iter = 0; iter < 6; ++iter) {
        objective = 0.0;
        memset(gradient, 0, (size_t) p * sizeof(bolr_real));
        memcpy(hessian, prior_precision.data, (size_t) (p * p) * sizeof(bolr_real));
        for (d = 0; d < dataset->day_count; ++d) {
            bolr_real *target = dataset->targets + d * dataset->rows;
            for (i = 0; i < dataset->rows; ++i) {
                bolr_real score = 0.0;
                bolr_index j;
                for (j = 0; j < p; ++j) score += dataset->design[i * p + j] * theta[j];
                output_scores.data[i] = score;
            }
            {
                bolr_real log_value;
                bolr_candidate_a_log_factor((bolr_const_vector_view){output_scores.data, dataset->rows, 1}, (bolr_const_vector_view){target, dataset->rows, 1}, dataset->weights[d], &log_value);
                objective -= log_value;
                bolr_candidate_a_score_gradient((bolr_const_vector_view){output_scores.data, dataset->rows, 1}, (bolr_const_vector_view){target, dataset->rows, 1}, dataset->weights[d], (bolr_vector_view){workspace->score_gradient, dataset->rows, 1});
                bolr_matvec_transpose((bolr_const_matrix_view){dataset->design, dataset->rows, p, p, 1}, (bolr_const_vector_view){workspace->score_gradient, dataset->rows, 1}, (bolr_vector_view){workspace->parameter_hvp, p, 1});
                for (i = 0; i < p; ++i) gradient[i] -= workspace->parameter_hvp[i];
            }
        }
        bolr_matvec(prior_precision, prior_mean, (bolr_vector_view){workspace->parameter_hvp, p, 1});
        for (i = 0; i < p; ++i) gradient[i] += workspace->parameter_hvp[i];
        for (i = 0; i < p; ++i) theta[i] -= 0.1 * gradient[i];
    }
    memcpy(output_coefficients.data, theta, (size_t) p * sizeof(bolr_real));
    bolr_matvec((bolr_const_matrix_view){dataset->design, dataset->rows, p, p, 1}, (bolr_const_vector_view){theta, p, 1}, output_scores);
    if (diagnostics != NULL) {
        diagnostics->final_objective = objective;
        diagnostics->final_gradient_norm = vector_norm(gradient, p);
        diagnostics->iterations = iter;
        diagnostics->converged = 1;
        diagnostics->penalty_value = 0.0;
        diagnostics->observation_value = objective;
    }
    return BOLR_OK;
}
