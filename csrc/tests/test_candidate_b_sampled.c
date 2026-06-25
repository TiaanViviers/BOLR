#include "test_suite.h"

#include "bolr/observation_candidate_b.h"
#include "bolr/status.h"

#include <math.h>

int test_candidate_b_sampled(void) {
    bolr_candidate_b_sampled_observation *observation = NULL;
    bolr_candidate_b_diagnostics diagnostics;
    bolr_observation_operator op;
    bolr_real scores[] = {0.5, 0.1, -0.2, -1.0};
    bolr_real gradient[] = {0.0, 0.0, 0.0, 0.0};
    bolr_index winners[] = {0, 1};
    bolr_index losers[] = {2, 3};
    bolr_real weights[] = {0.25, 0.25};
    if (bolr_candidate_b_sampled_observation_create(4, winners, losers, weights, 2, 1.0, 4, 0, 1, NULL, &observation) != BOLR_OK) return 1;
    if (bolr_candidate_b_sampled_observation_diagnostics(observation, &diagnostics) != BOLR_OK) return 1;
    if ((diagnostics.used_pair_count != 2) || (diagnostics.possible_pair_count != 4)) return 1;
    if (bolr_candidate_b_sampled_observation_operator(observation, &op) != BOLR_OK) return 1;
    if (op.gradient(op.context, (bolr_const_vector_view){scores, 4, 1}, (bolr_vector_view){gradient, 4, 1}, NULL) != BOLR_OK) return 1;
    if (fabs(gradient[0] + gradient[1] + gradient[2] + gradient[3]) > 1e-12) return 1;
    bolr_candidate_b_sampled_observation_destroy(observation);
    return 0;
}
