#include "bolr/observation.h"
#include "bolr/math.h"
#include "bolr/linalg.h"

#include <math.h>
#include <stdlib.h>

struct bolr_candidate_a_observation {
    const bolr_allocator *allocator;
    bolr_real *target;
    bolr_index dimension;
    bolr_real eta_effective;
};

static bolr_status validate_target(bolr_const_vector_view target) {
    bolr_real sum = 0.0;
    bolr_index i;
    if ((bolr_vector_view_validate(target) != BOLR_OK) || (target.length <= 0)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < target.length; ++i) {
        bolr_real value = target.data[i * target.stride];
        if ((!isfinite(value)) || (value < 0.0)) return BOLR_NONFINITE_INPUT;
        sum += value;
    }
    return (fabs(sum - 1.0) > 1e-9) ? BOLR_INVALID_ARGUMENT : BOLR_OK;
}

bolr_status bolr_candidate_a_log_factor(bolr_const_vector_view scores, bolr_const_vector_view target_probabilities, bolr_real eta_effective, bolr_real *out_value) {
    bolr_vector_view buffer_view;
    bolr_real *buffer;
    bolr_status status;
    bolr_real total = 0.0;
    bolr_index i;
    if ((out_value == NULL) || !isfinite(eta_effective)) return BOLR_INVALID_ARGUMENT;
    status = validate_target(target_probabilities); if (status != BOLR_OK) return status;
    if (scores.length != target_probabilities.length) return BOLR_INVALID_SHAPE;
    buffer = (bolr_real *) malloc((size_t) scores.length * sizeof(bolr_real));
    if (buffer == NULL) return BOLR_ALLOCATION_FAILED;
    buffer_view = (bolr_vector_view){buffer, scores.length, 1};
    status = bolr_log_softmax(scores, buffer_view);
    if (status == BOLR_OK) {
        for (i = 0; i < scores.length; ++i) total += target_probabilities.data[i * target_probabilities.stride] * buffer[i];
        *out_value = eta_effective * total;
    }
    free(buffer);
    return status;
}

bolr_status bolr_candidate_a_score_gradient(bolr_const_vector_view scores, bolr_const_vector_view target_probabilities, bolr_real eta_effective, bolr_vector_view output_gradient) {
    bolr_status status;
    bolr_index i;
    status = validate_target(target_probabilities); if (status != BOLR_OK) return status;
    if ((scores.length != target_probabilities.length) || (scores.length != output_gradient.length)) return BOLR_INVALID_SHAPE;
    status = bolr_softmax(scores, output_gradient); if (status != BOLR_OK) return status;
    for (i = 0; i < scores.length; ++i) output_gradient.data[i * output_gradient.stride] = eta_effective * (target_probabilities.data[i * target_probabilities.stride] - output_gradient.data[i * output_gradient.stride]);
    return BOLR_OK;
}

bolr_status bolr_candidate_a_score_hvp(bolr_const_vector_view scores, bolr_const_vector_view vector, bolr_real eta_effective, bolr_vector_view output_hvp) {
    bolr_status status;
    bolr_index i;
    bolr_real dot;
    status = bolr_softmax(scores, output_hvp); if (status != BOLR_OK) return status;
    if ((vector.length != scores.length) || (output_hvp.length != scores.length)) return BOLR_INVALID_SHAPE;
    status = bolr_dot((bolr_const_vector_view){output_hvp.data, output_hvp.length, output_hvp.stride}, vector, &dot); if (status != BOLR_OK) return status;
    for (i = 0; i < scores.length; ++i) {
        bolr_real q = output_hvp.data[i * output_hvp.stride];
        output_hvp.data[i * output_hvp.stride] = eta_effective * (q * vector.data[i * vector.stride] - q * dot);
    }
    return BOLR_OK;
}

bolr_status bolr_candidate_a_parameter_gradient(const bolr_dense_operator *op, bolr_const_vector_view scores, bolr_const_vector_view target_probabilities, bolr_real eta_effective, bolr_vector_view output_gradient, bolr_workspace *workspace) {
    bolr_vector_view score_grad;
    bolr_status status;
    if ((op == NULL) || (workspace == NULL)) return BOLR_INVALID_ARGUMENT;
    status = bolr_workspace_score_buffer(workspace, scores.length, &score_grad); if (status != BOLR_OK) return status;
    status = bolr_candidate_a_score_gradient(scores, target_probabilities, eta_effective, score_grad); if (status != BOLR_OK) return status;
    return bolr_dense_operator_transpose(op, (bolr_const_vector_view){score_grad.data, score_grad.length, score_grad.stride}, output_gradient, workspace);
}

bolr_status bolr_candidate_a_parameter_hvp(const bolr_dense_operator *op, bolr_const_vector_view scores, bolr_const_vector_view vector, bolr_real eta_effective, bolr_vector_view output_hvp, bolr_workspace *workspace) {
    bolr_vector_view score_vec, score_hvp;
    bolr_status status;
    if ((op == NULL) || (workspace == NULL)) return BOLR_INVALID_ARGUMENT;
    status = bolr_workspace_score_buffer(workspace, scores.length, &score_vec); if (status != BOLR_OK) return status;
    status = bolr_workspace_context_buffer(workspace, scores.length, &score_hvp); if (status != BOLR_OK) return status;
    status = bolr_dense_operator_forward(op, vector, score_vec, workspace); if (status != BOLR_OK) return status;
    status = bolr_candidate_a_score_hvp(scores, (bolr_const_vector_view){score_vec.data, score_vec.length, score_vec.stride}, eta_effective, score_hvp); if (status != BOLR_OK) return status;
    return bolr_dense_operator_transpose(op, (bolr_const_vector_view){score_hvp.data, score_hvp.length, score_hvp.stride}, output_hvp, workspace);
}

static bolr_status observation_value_callback(const void *context, bolr_const_vector_view scores, bolr_real *out_log_factor, void *observation_workspace) {
    const bolr_candidate_a_observation *observation = (const bolr_candidate_a_observation *) context;
    (void) observation_workspace;
    return bolr_candidate_a_log_factor(scores, (bolr_const_vector_view){observation->target, observation->dimension, 1}, observation->eta_effective, out_log_factor);
}

static bolr_status observation_gradient_callback(const void *context, bolr_const_vector_view scores, bolr_vector_view output_gradient, void *observation_workspace) {
    const bolr_candidate_a_observation *observation = (const bolr_candidate_a_observation *) context;
    (void) observation_workspace;
    return bolr_candidate_a_score_gradient(scores, (bolr_const_vector_view){observation->target, observation->dimension, 1}, observation->eta_effective, output_gradient);
}

static bolr_status observation_curvature_hvp_callback(const void *context, bolr_const_vector_view scores, bolr_const_vector_view vector, bolr_vector_view output, void *observation_workspace) {
    const bolr_candidate_a_observation *observation = (const bolr_candidate_a_observation *) context;
    (void) observation_workspace;
    return bolr_candidate_a_score_hvp(scores, vector, observation->eta_effective, output);
}

bolr_status bolr_candidate_a_observation_create(
    bolr_const_vector_view target,
    bolr_real eta,
    bolr_real update_weight,
    const bolr_allocator *allocator,
    bolr_candidate_a_observation **out_observation
) {
    bolr_candidate_a_observation *observation;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index i;
    if ((out_observation == NULL) || (target.length <= 0)) return BOLR_INVALID_ARGUMENT;
    *out_observation = NULL;
    observation = (bolr_candidate_a_observation *) bolr_allocator_calloc(active, 1U, sizeof(*observation));
    if (observation == NULL) return BOLR_ALLOCATION_FAILED;
    observation->allocator = active;
    observation->dimension = target.length;
    observation->eta_effective = eta * update_weight;
    observation->target = (bolr_real *) bolr_allocator_malloc(active, (size_t) target.length * sizeof(bolr_real));
    if (observation->target == NULL) { bolr_candidate_a_observation_destroy(observation); return BOLR_ALLOCATION_FAILED; }
    for (i = 0; i < target.length; ++i) observation->target[i] = target.data[i * target.stride];
    *out_observation = observation;
    return BOLR_OK;
}

void bolr_candidate_a_observation_destroy(bolr_candidate_a_observation *observation) {
    if (observation == NULL) return;
    bolr_allocator_free(observation->allocator, observation->target);
    bolr_allocator_free(observation->allocator, observation);
}

bolr_status bolr_candidate_a_observation_operator(
    const bolr_candidate_a_observation *observation,
    bolr_observation_operator *out_operator
) {
    if ((observation == NULL) || (out_operator == NULL)) return BOLR_INVALID_ARGUMENT;
    out_operator->value = observation_value_callback;
    out_operator->gradient = observation_gradient_callback;
    out_operator->curvature_hvp = observation_curvature_hvp_callback;
    out_operator->context = observation;
    return BOLR_OK;
}
