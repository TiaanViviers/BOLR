#include "bolr/attribution.h"

#include "bolr/linalg.h"

#include <math.h>
#include <stdlib.h>

bolr_status bolr_block_innovation_attribution(
    const bolr_state_layout *layout,
    bolr_const_vector_view predictive_mean,
    bolr_const_matrix_view predictive_covariance,
    bolr_const_vector_view posterior_mean,
    bolr_real epsilon,
    bolr_vector_view euclidean_energy,
    bolr_vector_view mahalanobis_energy,
    bolr_vector_view attribution_weight
) {
    bolr_index block_count;
    bolr_index b;
    bolr_real denom = 0.0;
    if ((layout == NULL) || (epsilon < 0.0)) return BOLR_INVALID_ARGUMENT;
    if ((predictive_mean.length != posterior_mean.length) || (predictive_covariance.rows != predictive_covariance.cols) || (predictive_covariance.rows != predictive_mean.length)) {
        return BOLR_INVALID_SHAPE;
    }
    block_count = bolr_state_layout_block_count(layout);
    if ((euclidean_energy.length != block_count) || (mahalanobis_energy.length != block_count) || (attribution_weight.length != block_count)) return BOLR_INVALID_SHAPE;
    for (b = 0; b < block_count; ++b) {
        bolr_state_block_spec spec;
        bolr_index dim;
        bolr_real *chol;
        bolr_real *delta;
        bolr_real mahal = 0.0;
        bolr_real euclid = 0.0;
        bolr_cholesky_diagnostics diag;
        bolr_index i, j;
        if (bolr_state_layout_block_spec(layout, b, &spec) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
        dim = spec.stop - spec.start;
        chol = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
        delta = (bolr_real *) malloc((size_t) dim * sizeof(bolr_real));
        if ((chol == NULL) || (delta == NULL)) { free(chol); free(delta); return BOLR_ALLOCATION_FAILED; }
        for (i = 0; i < dim; ++i) {
            delta[i] = posterior_mean.data[(spec.start + i) * posterior_mean.stride] - predictive_mean.data[(spec.start + i) * predictive_mean.stride];
            euclid += delta[i] * delta[i];
            for (j = 0; j < dim; ++j) {
                chol[i * dim + j] = predictive_covariance.data[(spec.start + i) * predictive_covariance.row_stride + (spec.start + j) * predictive_covariance.col_stride];
            }
        }
        if (bolr_cholesky_factor((bolr_matrix_view){chol, dim, dim, dim, 1}, 1e-10, 10.0, 8, &diag) != BOLR_OK) {
            free(chol);
            free(delta);
            return BOLR_NOT_POSITIVE_DEFINITE;
        }
        if (bolr_cholesky_solve((bolr_const_matrix_view){chol, dim, dim, dim, 1}, (bolr_const_vector_view){delta, dim, 1}, (bolr_vector_view){delta, dim, 1}) != BOLR_OK) {
            free(chol);
            free(delta);
            return BOLR_NUMERICAL_FAILURE;
        }
        for (i = 0; i < dim; ++i) mahal += delta[i] * delta[i];
        euclidean_energy.data[b * euclidean_energy.stride] = euclid;
        mahalanobis_energy.data[b * mahalanobis_energy.stride] = mahal;
        denom += mahal + epsilon;
        free(chol);
        free(delta);
    }
    for (b = 0; b < block_count; ++b) {
        attribution_weight.data[b * attribution_weight.stride] = (mahalanobis_energy.data[b * mahalanobis_energy.stride] + epsilon) / denom;
    }
    return BOLR_OK;
}
