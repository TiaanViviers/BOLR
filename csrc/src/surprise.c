#include "bolr/surprise.h"

#include <math.h>
#include <stddef.h>

bolr_status bolr_surprise_compute(
    bolr_surprise_mode mode,
    const bolr_surprise_input *input,
    bolr_real *out_value,
    bolr_real *out_raw,
    bolr_real *out_normalized,
    bolr_real *out_information_normalized,
    int *out_present
) {
    bolr_real raw;
    bolr_real normalized;
    bolr_real information_normalized;
    bolr_real value;
    if ((input == NULL) || (out_value == NULL) || (out_raw == NULL) || (out_normalized == NULL) || (out_information_normalized == NULL) || (out_present == NULL)) {
        return BOLR_INVALID_ARGUMENT;
    }
    *out_present = 0;
    raw = -input->log_factor_at_predictive_mean;
    normalized = (input->effective_strength > 0.0) ? raw / fmax(input->effective_strength, 1e-12) : raw;
    information_normalized = (input->information_size > 0.0) ? normalized / fmax(input->information_size, 1.0) : normalized;
    if (!input->informative) {
        *out_value = 0.0;
        *out_raw = 0.0;
        *out_normalized = 0.0;
        *out_information_normalized = 0.0;
        return BOLR_OK;
    }
    switch (mode) {
        case BOLR_SURPRISE_GENERALIZED_LOSS: value = raw; break;
        case BOLR_SURPRISE_GENERALIZED_LOSS_STRENGTH_NORMALIZED: value = normalized; break;
        case BOLR_SURPRISE_GENERALIZED_LOSS_INFORMATION_NORMALIZED: value = information_normalized; break;
        case BOLR_SURPRISE_POSTERIOR_MAHALANOBIS: value = input->mahalanobis_update; break;
        case BOLR_SURPRISE_POSTERIOR_KL: value = input->gaussian_kl; break;
        case BOLR_SURPRISE_OBJECTIVE_IMPROVEMENT: value = input->objective_improvement; break;
        default: return BOLR_INVALID_ARGUMENT;
    }
    if (!isfinite(value)) return BOLR_NUMERICAL_FAILURE;
    *out_value = value;
    *out_raw = raw;
    *out_normalized = normalized;
    *out_information_normalized = information_normalized;
    *out_present = 1;
    return BOLR_OK;
}
