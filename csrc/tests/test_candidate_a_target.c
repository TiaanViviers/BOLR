#include "test_suite.h"

#include "bolr/status.h"
#include "bolr/target.h"

#include <math.h>

int test_candidate_a_target(void) {
    bolr_candidate_a_target_config config = {1.0, 1.0, 4.0, 0.0, 0.0, 1e-6, 1};
    bolr_candidate_a_target_diagnostics diagnostics;
    bolr_real utilities[] = {2.0, 1.0, -1.0};
    bolr_real target[] = {0.0, 0.0, 0.0};
    bolr_real update_weight = -1.0;
    bolr_real sum = 0.0;
    bolr_real uniform_utilities[] = {1.0, 1.0, 1.0};
    bolr_real uniform_target[] = {0.0, 0.0, 0.0};
    bolr_real uniform_weight = -1.0;
    int i;
    if (bolr_candidate_a_target_build(&config, (bolr_const_vector_view){utilities, 3, 1}, (bolr_vector_view){target, 3, 1}, &update_weight, &diagnostics) != BOLR_OK) return 1;
    for (i = 0; i < 3; ++i) sum += target[i];
    if (fabs(sum - 1.0) > 1e-12) return 1;
    if (update_weight != 1.0) return 1;
    if ((!diagnostics.informative) || (diagnostics.positive_candidate_count != 2)) return 1;
    if (!(target[0] > target[1] && target[1] > target[2])) return 1;
    if (bolr_candidate_a_target_build(&config, (bolr_const_vector_view){uniform_utilities, 3, 1}, (bolr_vector_view){uniform_target, 3, 1}, &uniform_weight, &diagnostics) != BOLR_OK) return 1;
    if ((!diagnostics.all_irrelevant) || (uniform_weight != 0.0)) return 1;
    return 0;
}
