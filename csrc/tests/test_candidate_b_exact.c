#include "test_suite.h"

#include "bolr/observation_candidate_b.h"
#include "bolr/status.h"

#include <math.h>

int test_candidate_b_exact(void) {
    bolr_ordered_partition *partition = NULL;
    bolr_candidate_b_exact_observation *observation = NULL;
    bolr_observation_operator op;
    bolr_real scores[] = {1.0, -1.0};
    bolr_real gradient[] = {0.0, 0.0};
    bolr_real vector[] = {1.0, -1.0};
    bolr_real hvp[] = {0.0, 0.0};
    bolr_real value = 0.0;
    bolr_real expected = -log1p(exp(-2.0));
    bolr_index offsets[] = {0, 1, 2};
    bolr_index indices[] = {0, 1};
    bolr_index candidate_to_group[] = {0, 1};
    if (bolr_ordered_partition_create_copy(offsets, indices, 2, candidate_to_group, 2, 0.0, 1.0, 0.0, 1.0, 0, 1.0, NULL, &partition) != BOLR_OK) return 1;
    if (bolr_candidate_b_exact_observation_create(partition, 1, NULL, &observation) != BOLR_OK) return 1;
    if (bolr_candidate_b_exact_observation_operator(observation, &op) != BOLR_OK) return 1;
    if (op.value(op.context, (bolr_const_vector_view){scores, 2, 1}, &value, NULL) != BOLR_OK) return 1;
    if (op.gradient(op.context, (bolr_const_vector_view){scores, 2, 1}, (bolr_vector_view){gradient, 2, 1}, NULL) != BOLR_OK) return 1;
    if (op.curvature_hvp(op.context, (bolr_const_vector_view){scores, 2, 1}, (bolr_const_vector_view){vector, 2, 1}, (bolr_vector_view){hvp, 2, 1}, NULL) != BOLR_OK) return 1;
    if (fabs(value - expected) > 1e-12) return 1;
    if (fabs(gradient[0] + gradient[1]) > 1e-12) return 1;
    if (!(gradient[0] > 0.0 && gradient[1] < 0.0)) return 1;
    if (fabs(hvp[0] + hvp[1]) > 1e-12) return 1;
    bolr_candidate_b_exact_observation_destroy(observation);
    bolr_ordered_partition_destroy(partition);
    return 0;
}
