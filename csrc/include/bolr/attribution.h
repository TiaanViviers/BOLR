#ifndef BOLR_ATTRIBUTION_H
#define BOLR_ATTRIBUTION_H

#include "bolr/array.h"
#include "bolr/state_layout.h"
#include "bolr/status.h"

bolr_status bolr_block_innovation_attribution(
    const bolr_state_layout *layout,
    bolr_const_vector_view predictive_mean,
    bolr_const_matrix_view predictive_covariance,
    bolr_const_vector_view posterior_mean,
    bolr_real epsilon,
    bolr_vector_view euclidean_energy,
    bolr_vector_view mahalanobis_energy,
    bolr_vector_view attribution_weight
);

#endif
