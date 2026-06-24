#include "bolr/dynamics.h"
#include "bolr/status.h"

bolr_status bolr_additive_transition_covariance(bolr_const_matrix_view posterior_covariance, bolr_const_matrix_view process_noise, bolr_matrix_view output_covariance) {
    bolr_index r, c;
    if ((posterior_covariance.rows != posterior_covariance.cols) || (process_noise.rows != process_noise.cols) || (posterior_covariance.rows != process_noise.rows)) return BOLR_INVALID_SHAPE;
    if ((output_covariance.rows != posterior_covariance.rows) || (output_covariance.cols != posterior_covariance.cols)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < output_covariance.rows; ++r) for (c = 0; c < output_covariance.cols; ++c) output_covariance.data[r * output_covariance.row_stride + c * output_covariance.col_stride] = posterior_covariance.data[r * posterior_covariance.row_stride + c * posterior_covariance.col_stride] + process_noise.data[r * process_noise.row_stride + c * process_noise.col_stride];
    return BOLR_OK;
}

bolr_status bolr_heterogeneous_discount_covariance(bolr_const_matrix_view posterior_covariance, bolr_const_vector_view block_scale, bolr_matrix_view output_covariance) {
    bolr_index r, c;
    if ((posterior_covariance.rows != posterior_covariance.cols) || (posterior_covariance.rows != block_scale.length)) return BOLR_INVALID_SHAPE;
    if ((output_covariance.rows != posterior_covariance.rows) || (output_covariance.cols != posterior_covariance.cols)) return BOLR_INVALID_SHAPE;
    for (r = 0; r < output_covariance.rows; ++r) for (c = 0; c < output_covariance.cols; ++c) output_covariance.data[r * output_covariance.row_stride + c * output_covariance.col_stride] = posterior_covariance.data[r * posterior_covariance.row_stride + c * posterior_covariance.col_stride] * block_scale.data[r * block_scale.stride] * block_scale.data[c * block_scale.stride];
    return BOLR_OK;
}
