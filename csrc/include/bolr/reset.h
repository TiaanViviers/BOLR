#ifndef BOLR_RESET_H
#define BOLR_RESET_H

#include "bolr/array.h"
#include "bolr/status.h"

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
);

#endif
