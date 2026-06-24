#include "bolr/inference.h"
#include "bolr/checkpoint.h"
#include "bolr/linalg.h"
#include "bolr/math.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static bolr_status alloc_buffer(const bolr_allocator *allocator, size_t count, bolr_real **out) {
    *out = (bolr_real *) bolr_allocator_calloc(allocator, count, sizeof(bolr_real));
    return (*out == NULL) ? BOLR_ALLOCATION_FAILED : BOLR_OK;
}

bolr_status bolr_inference_workspace_create(
    bolr_index state_dimension,
    bolr_index candidate_count,
    const bolr_allocator *allocator,
    bolr_inference_workspace **out_workspace
) {
    bolr_inference_workspace *workspace;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    if ((out_workspace == NULL) || (state_dimension <= 0) || (candidate_count <= 0)) return BOLR_INVALID_ARGUMENT;
    *out_workspace = NULL;
    workspace = (bolr_inference_workspace *) bolr_allocator_calloc(active, 1U, sizeof(*workspace));
    if (workspace == NULL) return BOLR_ALLOCATION_FAILED;
    workspace->allocator = active;
    workspace->state_dimension = state_dimension;
    workspace->candidate_count = candidate_count;
    if ((alloc_buffer(active, (size_t) state_dimension, &workspace->state_displacement) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) state_dimension, &workspace->prior_solve) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) candidate_count, &workspace->score_vector) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) candidate_count, &workspace->score_gradient) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) candidate_count, &workspace->score_hvp) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) state_dimension, &workspace->parameter_gradient) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) state_dimension, &workspace->parameter_hvp) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) state_dimension, &workspace->newton_step) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) state_dimension, &workspace->trial_state) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) candidate_count, &workspace->trial_scores) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) (state_dimension * state_dimension), &workspace->dense_hessian) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) (state_dimension * state_dimension), &workspace->damped_hessian) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) (state_dimension * state_dimension), &workspace->posterior_covariance) != BOLR_OK) ||
        (alloc_buffer(active, (size_t) state_dimension, &workspace->identity_rhs) != BOLR_OK)) {
        bolr_inference_workspace_destroy(workspace);
        return BOLR_ALLOCATION_FAILED;
    }
    if (bolr_workspace_create(&(bolr_workspace_config){candidate_count, state_dimension, (state_dimension > candidate_count) ? state_dimension : candidate_count}, active, &workspace->score_workspace) != BOLR_OK) {
        bolr_inference_workspace_destroy(workspace);
        return BOLR_ALLOCATION_FAILED;
    }
    *out_workspace = workspace;
    return BOLR_OK;
}

void bolr_inference_workspace_destroy(bolr_inference_workspace *workspace) {
    if (workspace == NULL) return;
    bolr_allocator_free(workspace->allocator, workspace->state_displacement);
    bolr_allocator_free(workspace->allocator, workspace->prior_solve);
    bolr_allocator_free(workspace->allocator, workspace->score_vector);
    bolr_allocator_free(workspace->allocator, workspace->score_gradient);
    bolr_allocator_free(workspace->allocator, workspace->score_hvp);
    bolr_allocator_free(workspace->allocator, workspace->parameter_gradient);
    bolr_allocator_free(workspace->allocator, workspace->parameter_hvp);
    bolr_allocator_free(workspace->allocator, workspace->newton_step);
    bolr_allocator_free(workspace->allocator, workspace->trial_state);
    bolr_allocator_free(workspace->allocator, workspace->trial_scores);
    bolr_allocator_free(workspace->allocator, workspace->dense_hessian);
    bolr_allocator_free(workspace->allocator, workspace->damped_hessian);
    bolr_allocator_free(workspace->allocator, workspace->posterior_covariance);
    bolr_allocator_free(workspace->allocator, workspace->identity_rhs);
    bolr_workspace_destroy(workspace->score_workspace);
    bolr_allocator_free(workspace->allocator, workspace);
}

bolr_index bolr_inference_workspace_state_dimension(const bolr_inference_workspace *workspace) { return (workspace == NULL) ? -1 : workspace->state_dimension; }
bolr_index bolr_inference_workspace_candidate_count(const bolr_inference_workspace *workspace) { return (workspace == NULL) ? -1 : workspace->candidate_count; }

static bolr_real vector_norm(const bolr_real *data, bolr_index length) {
    bolr_real total = 0.0;
    bolr_index i;
    for (i = 0; i < length; ++i) total += data[i] * data[i];
    return sqrt(total);
}

static bolr_real trace_dense(const bolr_real *matrix, bolr_index dim) {
    bolr_index i;
    bolr_real total = 0.0;
    for (i = 0; i < dim; ++i) total += matrix[i * dim + i];
    return total;
}

static bolr_status compute_scores(const bolr_model *model, const bolr_real *theta, bolr_const_vector_view context, bolr_inference_workspace *workspace, bolr_real *output) {
    return bolr_model_forward(model, (bolr_const_vector_view){theta, bolr_model_state_dim(model), 1}, context, (bolr_vector_view){output, bolr_model_score_count(model), 1}, workspace->score_workspace);
}

static bolr_status compute_dynamic_scores(const bolr_model *model, const bolr_real *theta, bolr_const_vector_view context, bolr_inference_workspace *workspace, bolr_real *output) {
    return bolr_model_dynamic_forward(model, (bolr_const_vector_view){theta, bolr_model_state_dim(model), 1}, context, (bolr_vector_view){output, bolr_model_score_count(model), 1}, workspace->score_workspace);
}

static bolr_status prior_solve(const bolr_real *factor, bolr_index dim, const bolr_real *rhs, bolr_real *output) {
    memcpy(output, rhs, (size_t) dim * sizeof(bolr_real));
    return bolr_cholesky_solve((bolr_const_matrix_view){factor, dim, dim, dim, 1}, (bolr_const_vector_view){rhs, dim, 1}, (bolr_vector_view){output, dim, 1});
}

static bolr_status objective_gradient_hessian(
    const bolr_gaussian_state *predictive,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_observation_operator *observation,
    const bolr_real *theta,
    const bolr_real *prior_chol,
    bolr_inference_workspace *workspace,
    bolr_real *out_objective,
    bolr_real *out_gradient,
    bolr_real *out_hessian
) {
    bolr_index dim = predictive->dimension;
    bolr_index n = bolr_model_score_count(model);
    bolr_index i, j;
    bolr_real log_factor = 0.0;
    for (i = 0; i < dim; ++i) workspace->state_displacement[i] = theta[i] - predictive->mean[i];
    if (prior_solve(prior_chol, dim, workspace->state_displacement, workspace->prior_solve) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (compute_scores(model, theta, context, workspace, workspace->score_vector) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (observation->value(observation->context, (bolr_const_vector_view){workspace->score_vector, n, 1}, &log_factor, NULL) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
    if (out_objective != NULL) {
        bolr_real quad = 0.0;
        for (i = 0; i < dim; ++i) quad += workspace->state_displacement[i] * workspace->prior_solve[i];
        *out_objective = 0.5 * quad - log_factor;
    }
    if (out_gradient != NULL) {
        if (observation->gradient(observation->context, (bolr_const_vector_view){workspace->score_vector, n, 1}, (bolr_vector_view){workspace->score_gradient, n, 1}, NULL) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
        if (bolr_model_transpose(model, (bolr_const_vector_view){workspace->score_gradient, n, 1}, context, (bolr_vector_view){workspace->parameter_gradient, dim, 1}, workspace->score_workspace) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
        for (i = 0; i < dim; ++i) out_gradient[i] = workspace->prior_solve[i] - workspace->parameter_gradient[i];
    }
    if (out_hessian != NULL) {
        for (j = 0; j < dim; ++j) {
            for (i = 0; i < dim; ++i) workspace->identity_rhs[i] = (i == j) ? 1.0 : 0.0;
            if (prior_solve(prior_chol, dim, workspace->identity_rhs, workspace->parameter_hvp) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
            if (compute_dynamic_scores(model, workspace->identity_rhs, context, workspace, workspace->score_hvp) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
            if (observation->curvature_hvp(observation->context, (bolr_const_vector_view){workspace->score_vector, n, 1}, (bolr_const_vector_view){workspace->score_hvp, n, 1}, (bolr_vector_view){workspace->score_hvp, n, 1}, NULL) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
            if (bolr_model_transpose(model, (bolr_const_vector_view){workspace->score_hvp, n, 1}, context, (bolr_vector_view){workspace->parameter_gradient, dim, 1}, workspace->score_workspace) != BOLR_OK) return BOLR_NUMERICAL_FAILURE;
            for (i = 0; i < dim; ++i) out_hessian[i * dim + j] = workspace->parameter_hvp[i] + workspace->parameter_gradient[i];
        }
        for (i = 0; i < dim; ++i) {
            for (j = i + 1; j < dim; ++j) {
                bolr_real sym = 0.5 * (out_hessian[i * dim + j] + out_hessian[j * dim + i]);
                out_hessian[i * dim + j] = sym;
                out_hessian[j * dim + i] = sym;
            }
        }
    }
    return BOLR_OK;
}

static bolr_status clone_predictive_if_uninformative(
    const bolr_gaussian_state *predictive,
    const bolr_observation_operator *observation,
    const bolr_model *model,
    bolr_const_vector_view context,
    bolr_inference_workspace *workspace,
    bolr_gaussian_state **out_posterior,
    bolr_laplace_diagnostics *diagnostics
) {
    bolr_real log_factor = 0.0;
    bolr_index n = bolr_model_score_count(model);
    bolr_real *scores = (bolr_real *) malloc((size_t) n * sizeof(bolr_real));
    if (scores == NULL) return BOLR_ALLOCATION_FAILED;
    if (bolr_model_forward(model, (bolr_const_vector_view){predictive->mean, predictive->dimension, 1}, context, (bolr_vector_view){scores, n, 1}, workspace->score_workspace) != BOLR_OK) { free(scores); return BOLR_NUMERICAL_FAILURE; }
    if (observation->value(observation->context, (bolr_const_vector_view){scores, n, 1}, &log_factor, NULL) != BOLR_OK) { free(scores); return BOLR_NUMERICAL_FAILURE; }
    free(scores);
    if (log_factor != 0.0) return BOLR_NUMERICAL_FAILURE;
    if (bolr_gaussian_state_clone(predictive, predictive->allocator, out_posterior) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    if (diagnostics != NULL) {
        memset(diagnostics, 0, sizeof(*diagnostics));
        diagnostics->newton.converged = 1;
        diagnostics->log_factor_at_predictive_mean = 0.0;
        diagnostics->log_factor_at_posterior_mode = 0.0;
        diagnostics->prior_covariance_trace = trace_dense(predictive->covariance, predictive->dimension);
        diagnostics->posterior_covariance_trace = diagnostics->prior_covariance_trace;
    }
    return BOLR_OK;
}

bolr_status bolr_laplace_update(
    const bolr_gaussian_state *predictive,
    const bolr_model *model,
    bolr_const_vector_view context,
    const bolr_observation_operator *observation,
    const bolr_newton_config *config,
    bolr_inference_workspace *workspace,
    bolr_gaussian_state **out_posterior,
    bolr_laplace_diagnostics *diagnostics
) {
    bolr_index dim, iter, i, j;
    bolr_real objective, trial_objective, damping, theta_norm, grad_norm, step_norm, logdet;
    bolr_real *prior_chol;
    bolr_real *theta;
    bolr_real *gradient;
    bolr_real *hessian;
    bolr_real *covariance;
    bolr_cholesky_diagnostics chol_diag;
    bolr_status status;
    if ((predictive == NULL) || (model == NULL) || (observation == NULL) || (config == NULL) || (workspace == NULL) || (out_posterior == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_posterior = NULL;
    status = bolr_newton_config_validate(config); if (status != BOLR_OK) return status;
    if ((workspace->state_dimension != predictive->dimension) || (workspace->candidate_count != bolr_model_score_count(model))) return BOLR_INVALID_SHAPE;
    dim = predictive->dimension;
    theta = workspace->trial_state;
    gradient = workspace->parameter_gradient;
    hessian = workspace->dense_hessian;
    covariance = workspace->posterior_covariance;
    prior_chol = workspace->damped_hessian;
    memcpy(theta, predictive->mean, (size_t) dim * sizeof(bolr_real));
    memcpy(prior_chol, predictive->covariance, (size_t) (dim * dim) * sizeof(bolr_real));
    if (bolr_cholesky_factor((bolr_matrix_view){prior_chol, dim, dim, dim, 1}, config->cholesky_initial_jitter, config->cholesky_jitter_multiplier, config->maximum_cholesky_attempts, &chol_diag) != BOLR_OK) return BOLR_NOT_POSITIVE_DEFINITE;
    status = objective_gradient_hessian(predictive, model, context, observation, theta, prior_chol, workspace, &objective, gradient, hessian);
    if (status != BOLR_OK) {
        if (clone_predictive_if_uninformative(predictive, observation, model, context, workspace, out_posterior, diagnostics) == BOLR_OK) return BOLR_OK;
        return status;
    }
    damping = config->initial_damping;
    if (diagnostics != NULL) memset(diagnostics, 0, sizeof(*diagnostics));
    for (iter = 0; iter < config->maximum_iterations; ++iter) {
        grad_norm = vector_norm(gradient, dim);
        theta_norm = vector_norm(theta, dim);
        if (grad_norm <= config->gradient_tolerance * ((theta_norm > 1.0) ? theta_norm : 1.0)) break;
        memcpy(workspace->damped_hessian, hessian, (size_t) (dim * dim) * sizeof(bolr_real));
        for (i = 0; i < dim; ++i) workspace->damped_hessian[i * dim + i] += damping;
        status = bolr_cholesky_factor((bolr_matrix_view){workspace->damped_hessian, dim, dim, dim, 1}, config->cholesky_initial_jitter, config->cholesky_jitter_multiplier, config->maximum_cholesky_attempts, &chol_diag);
        if (status != BOLR_OK) {
            damping = (damping == 0.0) ? config->damping_multiplier : damping * config->damping_multiplier;
            if (damping > config->maximum_damping) return status;
            continue;
        }
        for (i = 0; i < dim; ++i) workspace->newton_step[i] = -gradient[i];
        status = bolr_cholesky_solve((bolr_const_matrix_view){workspace->damped_hessian, dim, dim, dim, 1}, (bolr_const_vector_view){workspace->newton_step, dim, 1}, (bolr_vector_view){workspace->newton_step, dim, 1});
        if (status != BOLR_OK) return status;
        step_norm = vector_norm(workspace->newton_step, dim);
        if (step_norm <= config->step_tolerance * ((theta_norm > 1.0) ? theta_norm : 1.0)) break;
        for (j = 0; j < config->maximum_line_search_steps; ++j) {
            bolr_real alpha = pow(config->line_search_reduction, (bolr_real) j);
            bolr_real directional = 0.0;
            for (i = 0; i < dim; ++i) {
                workspace->trial_state[i] = theta[i] + alpha * workspace->newton_step[i];
                directional += gradient[i] * workspace->newton_step[i];
            }
            status = objective_gradient_hessian(predictive, model, context, observation, workspace->trial_state, prior_chol, workspace, &trial_objective, NULL, NULL);
            if (status != BOLR_OK) return status;
            if (trial_objective <= objective + config->armijo_constant * alpha * directional) {
                memcpy(theta, workspace->trial_state, (size_t) dim * sizeof(bolr_real));
                objective = trial_objective;
                status = objective_gradient_hessian(predictive, model, context, observation, theta, prior_chol, workspace, NULL, gradient, hessian);
                if (status != BOLR_OK) return status;
                break;
            }
            if (j == config->maximum_line_search_steps - 1) {
                damping = (damping == 0.0) ? config->damping_multiplier : damping * config->damping_multiplier;
                if (damping > config->maximum_damping) return BOLR_NUMERICAL_FAILURE;
            }
        }
    }
    memcpy(workspace->damped_hessian, hessian, (size_t) (dim * dim) * sizeof(bolr_real));
    status = bolr_cholesky_factor((bolr_matrix_view){workspace->damped_hessian, dim, dim, dim, 1}, config->cholesky_initial_jitter, config->cholesky_jitter_multiplier, config->maximum_cholesky_attempts, &chol_diag);
    if (status != BOLR_OK) return status;
    for (j = 0; j < dim; ++j) {
        for (i = 0; i < dim; ++i) workspace->identity_rhs[i] = (i == j) ? 1.0 : 0.0;
        status = bolr_cholesky_solve((bolr_const_matrix_view){workspace->damped_hessian, dim, dim, dim, 1}, (bolr_const_vector_view){workspace->identity_rhs, dim, 1}, (bolr_vector_view){workspace->identity_rhs, dim, 1});
        if (status != BOLR_OK) return status;
        for (i = 0; i < dim; ++i) covariance[i * dim + j] = workspace->identity_rhs[i];
    }
    for (i = 0; i < dim; ++i) for (j = i + 1; j < dim; ++j) { bolr_real sym = 0.5 * (covariance[i * dim + j] + covariance[j * dim + i]); covariance[i * dim + j] = sym; covariance[j * dim + i] = sym; }
    status = bolr_gaussian_state_create((bolr_const_vector_view){theta, dim, 1}, (bolr_const_matrix_view){covariance, dim, dim, dim, 1}, predictive->state_layout_hash, predictive->model_schema_hash, predictive->allocator, out_posterior);
    if (status != BOLR_OK) return status;
    (*out_posterior)->step_index = predictive->step_index;
    if (diagnostics != NULL) {
        diagnostics->newton.status = BOLR_OK;
        diagnostics->newton.iterations = iter;
        diagnostics->newton.converged = 1;
        diagnostics->newton.final_gradient_norm = vector_norm(gradient, dim);
        diagnostics->newton.final_step_norm = step_norm;
        diagnostics->newton.final_damping = damping;
        diagnostics->newton.maximum_jitter_used = chol_diag.jitter_used;
        diagnostics->prior_covariance_trace = trace_dense(predictive->covariance, dim);
        diagnostics->posterior_covariance_trace = trace_dense(covariance, dim);
        bolr_logdet_from_cholesky((bolr_const_matrix_view){workspace->damped_hessian, dim, dim, dim, 1}, &logdet);
        diagnostics->posterior_covariance_log_determinant = -logdet;
        diagnostics->mean_update_norm = 0.0;
        for (i = 0; i < dim; ++i) {
            bolr_real delta = theta[i] - predictive->mean[i];
            diagnostics->mean_update_norm += delta * delta;
        }
        diagnostics->mean_update_norm = sqrt(diagnostics->mean_update_norm);
        diagnostics->mahalanobis_update_norm = 0.0;
        for (i = 0; i < dim; ++i) diagnostics->mahalanobis_update_norm += workspace->state_displacement[i] * workspace->prior_solve[i];
        diagnostics->log_factor_at_predictive_mean = 0.0;
        diagnostics->log_factor_at_posterior_mode = 0.0;
        diagnostics->objective_improvement = 0.0;
        diagnostics->score_mean_min = workspace->score_vector[0];
        diagnostics->score_mean_max = workspace->score_vector[0];
        for (i = 1; i < bolr_model_score_count(model); ++i) {
            if (workspace->score_vector[i] < diagnostics->score_mean_min) diagnostics->score_mean_min = workspace->score_vector[i];
            if (workspace->score_vector[i] > diagnostics->score_mean_max) diagnostics->score_mean_max = workspace->score_vector[i];
        }
        diagnostics->gradient_sum_diagnostic = 0.0;
        for (i = 0; i < bolr_model_score_count(model); ++i) diagnostics->gradient_sum_diagnostic += workspace->score_gradient[i];
        diagnostics->curvature_null_direction_diagnostic = 0.0;
    }
    return BOLR_OK;
}
