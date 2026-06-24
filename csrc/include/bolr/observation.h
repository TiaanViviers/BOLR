#ifndef BOLR_OBSERVATION_H
#define BOLR_OBSERVATION_H

#include "bolr/array.h"
#include "bolr/score.h"

typedef struct {
    bolr_status (*value)(
        const void *context,
        bolr_const_vector_view scores,
        bolr_real *out_log_factor,
        void *observation_workspace
    );
    bolr_status (*gradient)(
        const void *context,
        bolr_const_vector_view scores,
        bolr_vector_view output_gradient,
        void *observation_workspace
    );
    bolr_status (*curvature_hvp)(
        const void *context,
        bolr_const_vector_view scores,
        bolr_const_vector_view vector,
        bolr_vector_view output,
        void *observation_workspace
    );
    const void *context;
} bolr_observation_operator;

typedef struct bolr_candidate_a_observation bolr_candidate_a_observation;

bolr_status bolr_candidate_a_log_factor(bolr_const_vector_view scores, bolr_const_vector_view target_probabilities, bolr_real eta_effective, bolr_real *out_value);
bolr_status bolr_candidate_a_score_gradient(bolr_const_vector_view scores, bolr_const_vector_view target_probabilities, bolr_real eta_effective, bolr_vector_view output_gradient);
bolr_status bolr_candidate_a_score_hvp(bolr_const_vector_view scores, bolr_const_vector_view vector, bolr_real eta_effective, bolr_vector_view output_hvp);
bolr_status bolr_candidate_a_parameter_gradient(const bolr_dense_operator *op, bolr_const_vector_view scores, bolr_const_vector_view target_probabilities, bolr_real eta_effective, bolr_vector_view output_gradient, bolr_workspace *workspace);
bolr_status bolr_candidate_a_parameter_hvp(const bolr_dense_operator *op, bolr_const_vector_view scores, bolr_const_vector_view vector, bolr_real eta_effective, bolr_vector_view output_hvp, bolr_workspace *workspace);
bolr_status bolr_candidate_a_observation_create(
    bolr_const_vector_view target,
    bolr_real eta,
    bolr_real update_weight,
    const bolr_allocator *allocator,
    bolr_candidate_a_observation **out_observation
);
void bolr_candidate_a_observation_destroy(bolr_candidate_a_observation *observation);
bolr_status bolr_candidate_a_observation_operator(
    const bolr_candidate_a_observation *observation,
    bolr_observation_operator *out_operator
);

#endif
