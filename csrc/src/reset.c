#include "bolr/reset.h"

bolr_status bolr_apply_partial_reset(
    bolr_const_vector_view mean,
    bolr_const_matrix_view covariance,
    bolr_index start,
    bolr_index stop,
    bolr_real strength,
    bolr_const_vector_view anchor_mean,
    bolr_const_matrix_view anchor_covariance,
    bolr_vector_view out_mean,
    bolr_matrix_view out_covariance
) {
    bolr_index dim;
    bolr_index i, r, c;
    bolr_real a;
    if ((strength < 0.0) || (strength > 1.0)) return BOLR_INVALID_ARGUMENT;
    if ((start < 0) || (stop <= start) || (stop > mean.length)) return BOLR_INVALID_ARGUMENT;
    if ((covariance.rows != covariance.cols) || (covariance.rows != mean.length)) return BOLR_INVALID_SHAPE;
    if ((out_mean.length != mean.length) || (out_covariance.rows != covariance.rows) || (out_covariance.cols != covariance.cols)) return BOLR_INVALID_SHAPE;
    dim = stop - start;
    if ((anchor_mean.length != dim) || (anchor_covariance.rows != dim) || (anchor_covariance.cols != dim)) return BOLR_INVALID_SHAPE;
    for (i = 0; i < mean.length; ++i) out_mean.data[i * out_mean.stride] = mean.data[i * mean.stride];
    for (r = 0; r < covariance.rows; ++r) {
        for (c = 0; c < covariance.cols; ++c) {
            out_covariance.data[r * out_covariance.row_stride + c * out_covariance.col_stride] =
                covariance.data[r * covariance.row_stride + c * covariance.col_stride];
        }
    }
    a = 1.0 - strength;
    for (i = 0; i < dim; ++i) {
        out_mean.data[(start + i) * out_mean.stride] =
            anchor_mean.data[i * anchor_mean.stride] + a * (out_mean.data[(start + i) * out_mean.stride] - anchor_mean.data[i * anchor_mean.stride]);
    }
    for (r = start; r < stop; ++r) {
        for (c = 0; c < covariance.cols; ++c) out_covariance.data[r * out_covariance.row_stride + c * out_covariance.col_stride] *= a;
    }
    for (r = 0; r < covariance.rows; ++r) {
        for (c = start; c < stop; ++c) out_covariance.data[r * out_covariance.row_stride + c * out_covariance.col_stride] *= a;
    }
    for (r = 0; r < dim; ++r) {
        for (c = 0; c < dim; ++c) {
            out_covariance.data[(start + r) * out_covariance.row_stride + (start + c) * out_covariance.col_stride] +=
                (1.0 - a * a) * anchor_covariance.data[r * anchor_covariance.row_stride + c * anchor_covariance.col_stride];
        }
    }
    return BOLR_OK;
}
