#include "bolr/standardizer.h"

#include <stddef.h>
#include <math.h>

void bolr_standardizer_state_init(const bolr_standardizer_config *config, bolr_standardizer_state *state) {
    if ((config == NULL) || (state == NULL)) return;
    state->schema_version = 1U;
    state->count = 0U;
    state->mean = 0.0;
    state->variance = config->variance_floor;
    state->last_z = 0.0;
    state->last_z_present = 0;
}

bolr_status bolr_standardizer_step(
    const bolr_standardizer_config *config,
    bolr_standardizer_state *state,
    bolr_real value,
    int value_present,
    bolr_standardizer_diagnostics *diagnostics
) {
    bolr_real mean_before;
    bolr_real scale_before;
    bolr_real z;
    bolr_real alpha;
    bolr_real delta;
    bolr_real mean_after;
    bolr_real variance_after;
    if ((config == NULL) || (state == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((config->decay <= 0.0) || (config->decay > 1.0) || (config->variance_floor <= 0.0)) return BOLR_INVALID_ARGUMENT;
    mean_before = state->mean;
    scale_before = sqrt(fmax(state->variance, config->variance_floor));
    if (diagnostics != NULL) {
        diagnostics->value = value;
        diagnostics->mean_before = mean_before;
        diagnostics->scale_before = scale_before;
        diagnostics->missing = value_present ? 0 : 1;
        diagnostics->z_score_present = 0;
        diagnostics->z_score = 0.0;
    }
    if (!value_present) return BOLR_OK;
    if (!isfinite(value)) return BOLR_NONFINITE_INPUT;
    z = (state->count < (uint64_t) config->warmup_count) ? 0.0 : (value - mean_before) / scale_before;
    if (config->clip_enabled) {
        if (z > config->clip_z) z = config->clip_z;
        else if (z < -config->clip_z) z = -config->clip_z;
    }
    alpha = config->decay;
    delta = value - state->mean;
    mean_after = (1.0 - alpha) * state->mean + alpha * value;
    variance_after = (1.0 - alpha) * (state->variance + alpha * delta * delta);
    if (variance_after < config->variance_floor) variance_after = config->variance_floor;
    state->count += 1U;
    state->mean = mean_after;
    state->variance = variance_after;
    state->last_z = z;
    state->last_z_present = 1;
    if (diagnostics != NULL) {
        diagnostics->z_score = z;
        diagnostics->z_score_present = 1;
    }
    return BOLR_OK;
}
