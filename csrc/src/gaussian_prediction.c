#include "bolr/gaussian.h"
#include "bolr/linalg.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

static bolr_real trace_of_matrix(const bolr_real *matrix, bolr_index dim) {
    bolr_index i;
    bolr_real trace = 0.0;
    for (i = 0; i < dim; ++i) trace += matrix[i * dim + i];
    return trace;
}

bolr_status bolr_gaussian_predict(
    const bolr_gaussian_state *posterior,
    const bolr_transition_config *transition,
    bolr_workspace *workspace,
    bolr_gaussian_state **out_predictive,
    bolr_prediction_diagnostics *diagnostics
) {
    bolr_real *mean = NULL;
    bolr_real *cov = NULL;
    bolr_real *chol = NULL;
    bolr_index dim, r, c;
    bolr_status status;
    bolr_cholesky_diagnostics chol_diag;
    (void) workspace;
    if ((posterior == NULL) || (transition == NULL) || (out_predictive == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_predictive = NULL;
    dim = posterior->dimension;
    mean = (bolr_real *) malloc((size_t) dim * sizeof(bolr_real));
    cov = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
    chol = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
    if ((mean == NULL) || (cov == NULL) || (chol == NULL)) { free(mean); free(cov); free(chol); return BOLR_ALLOCATION_FAILED; }
    memcpy(mean, posterior->mean, (size_t) dim * sizeof(bolr_real));
    memcpy(cov, posterior->covariance, (size_t) (dim * dim) * sizeof(bolr_real));
    if (transition->family == BOLR_TRANSITION_ADDITIVE_Q) {
        if ((transition->process_noise.rows != dim) || (transition->process_noise.cols != dim)) { free(mean); free(cov); free(chol); return BOLR_INVALID_SHAPE; }
        for (r = 0; r < dim; ++r) for (c = 0; c < dim; ++c) cov[r * dim + c] += transition->process_noise.data[r * transition->process_noise.row_stride + c * transition->process_noise.col_stride];
    } else if (transition->family == BOLR_TRANSITION_GLOBAL_DISCOUNT) {
        bolr_real scale;
        if ((transition->global_discount <= 0.0) || (transition->global_discount > 1.0)) { free(mean); free(cov); free(chol); return BOLR_INVALID_ARGUMENT; }
        scale = 1.0 / transition->global_discount;
        for (r = 0; r < dim * dim; ++r) cov[r] *= scale;
    } else if (transition->family == BOLR_TRANSITION_BLOCK_DISCOUNT) {
        if (transition->block_discount_scales.length != dim) { free(mean); free(cov); free(chol); return BOLR_INVALID_SHAPE; }
        for (r = 0; r < dim; ++r) for (c = 0; c < dim; ++c) cov[r * dim + c] *= transition->block_discount_scales.data[r * transition->block_discount_scales.stride] * transition->block_discount_scales.data[c * transition->block_discount_scales.stride];
    } else {
        free(mean); free(cov); free(chol); return BOLR_INVALID_ARGUMENT;
    }
    memcpy(chol, cov, (size_t) (dim * dim) * sizeof(bolr_real));
    status = bolr_cholesky_factor((bolr_matrix_view){chol, dim, dim, dim, 1}, 1e-10, 10.0, 8, &chol_diag);
    if (status != BOLR_OK) { free(mean); free(cov); free(chol); return status; }
    status = bolr_gaussian_state_create((bolr_const_vector_view){mean, dim, 1}, (bolr_const_matrix_view){cov, dim, dim, dim, 1}, posterior->state_layout_hash, posterior->model_schema_hash, posterior->allocator, out_predictive);
    if (status == BOLR_OK) (*out_predictive)->step_index = posterior->step_index + 1U;
    if (diagnostics != NULL) {
        diagnostics->process_noise_trace = (transition->family == BOLR_TRANSITION_ADDITIVE_Q) ? trace_of_matrix(cov, dim) - trace_of_matrix(posterior->covariance, dim) : 0.0;
        diagnostics->predictive_covariance_trace = trace_of_matrix(cov, dim);
        diagnostics->minimum_cholesky_diagonal = chol_diag.minimum_diagonal;
        diagnostics->jitter_used = chol_diag.jitter_used;
    }
    free(mean); free(cov); free(chol);
    return status;
}

bolr_status bolr_gaussian_kl(const bolr_gaussian_state *posterior, const bolr_gaussian_state *predictive, bolr_real *out_kl) {
    bolr_real *chol_prior, *chol_post;
    bolr_real *tmp;
    bolr_real trace_term = 0.0, quad_term = 0.0, logdet_prior, logdet_post;
    bolr_index dim, i, j;
    bolr_cholesky_diagnostics diag;
    if ((posterior == NULL) || (predictive == NULL) || (out_kl == NULL)) return BOLR_INVALID_ARGUMENT;
    if (posterior->dimension != predictive->dimension) return BOLR_INVALID_SHAPE;
    dim = posterior->dimension;
    chol_prior = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
    chol_post = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
    tmp = (bolr_real *) malloc((size_t) dim * sizeof(bolr_real));
    if ((chol_prior == NULL) || (chol_post == NULL) || (tmp == NULL)) { free(chol_prior); free(chol_post); free(tmp); return BOLR_ALLOCATION_FAILED; }
    memcpy(chol_prior, predictive->covariance, (size_t) (dim * dim) * sizeof(bolr_real));
    memcpy(chol_post, posterior->covariance, (size_t) (dim * dim) * sizeof(bolr_real));
    if (bolr_cholesky_factor((bolr_matrix_view){chol_prior, dim, dim, dim, 1}, 1e-10, 10.0, 8, &diag) != BOLR_OK) { free(chol_prior); free(chol_post); free(tmp); return BOLR_NOT_POSITIVE_DEFINITE; }
    if (bolr_cholesky_factor((bolr_matrix_view){chol_post, dim, dim, dim, 1}, 1e-10, 10.0, 8, &diag) != BOLR_OK) { free(chol_prior); free(chol_post); free(tmp); return BOLR_NOT_POSITIVE_DEFINITE; }
    for (j = 0; j < dim; ++j) {
        for (i = 0; i < dim; ++i) tmp[i] = posterior->covariance[i * dim + j];
        if (bolr_cholesky_solve((bolr_const_matrix_view){chol_prior, dim, dim, dim, 1}, (bolr_const_vector_view){tmp, dim, 1}, (bolr_vector_view){tmp, dim, 1}) != BOLR_OK) { free(chol_prior); free(chol_post); free(tmp); return BOLR_NUMERICAL_FAILURE; }
        trace_term += tmp[j];
    }
    for (i = 0; i < dim; ++i) tmp[i] = posterior->mean[i] - predictive->mean[i];
    if (bolr_cholesky_solve((bolr_const_matrix_view){chol_prior, dim, dim, dim, 1}, (bolr_const_vector_view){tmp, dim, 1}, (bolr_vector_view){tmp, dim, 1}) != BOLR_OK) { free(chol_prior); free(chol_post); free(tmp); return BOLR_NUMERICAL_FAILURE; }
    for (i = 0; i < dim; ++i) quad_term += (posterior->mean[i] - predictive->mean[i]) * tmp[i];
    bolr_logdet_from_cholesky((bolr_const_matrix_view){chol_prior, dim, dim, dim, 1}, &logdet_prior);
    bolr_logdet_from_cholesky((bolr_const_matrix_view){chol_post, dim, dim, dim, 1}, &logdet_post);
    *out_kl = 0.5 * (trace_term + quad_term - (bolr_real) dim + logdet_prior - logdet_post);
    free(chol_prior); free(chol_post); free(tmp);
    return BOLR_OK;
}
