#include "test_suite.h"

#include <stdio.h>

int main(void) {
    int failures = 0;
    failures += test_math();
    failures += test_linalg();
    failures += test_ownership();
    failures += test_state_layout();
    failures += test_score();
    failures += test_candidate_a();
    failures += test_penalty();
    failures += test_dynamics();
    failures += test_golden();
    failures += test_gaussian_state();
    failures += test_gaussian_prediction();
    failures += test_posterior_objective();
    failures += test_newton();
    failures += test_laplace();
    failures += test_static_fit();
    failures += test_inference_ownership();
    if (failures != 0) {
        fprintf(stderr, "bolr_tests: %d failures\n", failures);
        return 1;
    }
    return 0;
}
