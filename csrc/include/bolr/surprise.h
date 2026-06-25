#ifndef BOLR_SURPRISE_H
#define BOLR_SURPRISE_H

#include "bolr/status.h"
#include "bolr/types.h"

typedef struct {
    int informative;
    bolr_real log_factor_at_predictive_mean;
    bolr_real log_factor_at_posterior_mode;
    bolr_real effective_strength;
    bolr_real information_size;
    bolr_real mahalanobis_update;
    bolr_real gaussian_kl;
    bolr_real objective_improvement;
} bolr_surprise_input;

typedef enum {
    BOLR_SURPRISE_GENERALIZED_LOSS = 1,
    BOLR_SURPRISE_GENERALIZED_LOSS_STRENGTH_NORMALIZED = 2,
    BOLR_SURPRISE_GENERALIZED_LOSS_INFORMATION_NORMALIZED = 3,
    BOLR_SURPRISE_POSTERIOR_MAHALANOBIS = 4,
    BOLR_SURPRISE_POSTERIOR_KL = 5,
    BOLR_SURPRISE_OBJECTIVE_IMPROVEMENT = 6
} bolr_surprise_mode;

bolr_status bolr_surprise_compute(
    bolr_surprise_mode mode,
    const bolr_surprise_input *input,
    bolr_real *out_value,
    bolr_real *out_raw,
    bolr_real *out_normalized,
    bolr_real *out_information_normalized,
    int *out_present
);

#endif
