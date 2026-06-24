#ifndef BOLR_OPTIMIZER_H
#define BOLR_OPTIMIZER_H

#include "bolr/types.h"

typedef struct {
    bolr_index maximum_iterations;
    bolr_real gradient_tolerance;
    bolr_real step_tolerance;
    bolr_real objective_tolerance;
    bolr_real initial_damping;
    bolr_real damping_multiplier;
    bolr_real maximum_damping;
    bolr_real armijo_constant;
    bolr_real line_search_reduction;
    bolr_index maximum_line_search_steps;
    bolr_real cholesky_initial_jitter;
    bolr_real cholesky_jitter_multiplier;
    bolr_index maximum_cholesky_attempts;
} bolr_newton_config;

typedef struct {
    bolr_status status;
    bolr_index iterations;
    bolr_index objective_evaluations;
    bolr_index gradient_evaluations;
    bolr_index hessian_evaluations;
    bolr_index line_search_evaluations;
    bolr_real initial_objective;
    bolr_real final_objective;
    bolr_real final_gradient_norm;
    bolr_real final_step_norm;
    bolr_real final_damping;
    bolr_real maximum_jitter_used;
    int converged;
    int used_damping;
    int used_jitter;
} bolr_newton_diagnostics;

bolr_status bolr_newton_config_validate(const bolr_newton_config *config);

#endif
