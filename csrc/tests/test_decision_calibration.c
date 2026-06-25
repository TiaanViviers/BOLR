#include "test_suite.h"

#include "bolr/decision.h"

#include <math.h>

int test_decision_calibration(void) {
    bolr_real probability_best[] = {0.6, 0.4, 0.0};
    bolr_real probability_top2[] = {1.0, 1.0, 0.0};
    bolr_real utilities[] = {1.0, 1.0, -0.5};
    bolr_real realized_best[] = {0.0, 0.0, 0.0};
    bolr_real brier_best = 0.0;
    bolr_real brier_top2 = 0.0;
    bolr_index region[] = {0, 2};
    int covered = 0;
    if (bolr_realized_best_distribution((bolr_const_vector_view){utilities, 3, 1}, 0.0, (bolr_vector_view){realized_best, 3, 1}) != BOLR_OK) return 1;
    if (bolr_probability_best_brier((bolr_const_vector_view){probability_best, 3, 1}, (bolr_const_vector_view){utilities, 3, 1}, 0.0, &brier_best) != BOLR_OK) return 1;
    if (bolr_top_k_brier((bolr_const_vector_view){probability_top2, 3, 1}, (bolr_const_vector_view){utilities, 3, 1}, 2, &brier_top2) != BOLR_OK) return 1;
    if (bolr_region_coverage(region, 2, (bolr_const_vector_view){utilities, 3, 1}, 0.0, &covered) != BOLR_OK) return 1;
    if ((fabs(realized_best[0] - 0.5) > 1e-12) || (fabs(realized_best[1] - 0.5) > 1e-12)) return 1;
    if (fabs(brier_top2) > 1e-12) return 1;
    return ((brier_best < 0.0) || (!covered)) ? 1 : 0;
}
