#include "bolr/optimizer.h"
#include "bolr/status.h"

#include <stddef.h>

bolr_status bolr_newton_config_validate(const bolr_newton_config *config) {
    if (config == NULL) return BOLR_INVALID_ARGUMENT;
    if (config->maximum_iterations <= 0) return BOLR_INVALID_ARGUMENT;
    if ((config->gradient_tolerance <= 0.0) || (config->step_tolerance <= 0.0) || (config->objective_tolerance < 0.0)) return BOLR_INVALID_ARGUMENT;
    if ((config->damping_multiplier <= 1.0) || (config->maximum_damping < config->initial_damping) || (config->initial_damping < 0.0)) return BOLR_INVALID_ARGUMENT;
    if ((config->armijo_constant <= 0.0) || (config->armijo_constant >= 1.0)) return BOLR_INVALID_ARGUMENT;
    if ((config->line_search_reduction <= 0.0) || (config->line_search_reduction >= 1.0)) return BOLR_INVALID_ARGUMENT;
    if ((config->maximum_line_search_steps <= 0) || (config->maximum_cholesky_attempts <= 0)) return BOLR_INVALID_ARGUMENT;
    if ((config->cholesky_initial_jitter < 0.0) || (config->cholesky_jitter_multiplier < 1.0)) return BOLR_INVALID_ARGUMENT;
    return BOLR_OK;
}
