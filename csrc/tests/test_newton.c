#include "test_suite.h"

#include "bolr/optimizer.h"
#include "bolr/status.h"

int test_newton(void) {
    bolr_newton_config valid = {8, 1e-6, 1e-9, 1e-12, 1e-3, 10.0, 1e6, 1e-4, 0.5, 8, 1e-10, 10.0, 8};
    bolr_newton_config invalid = valid;
    invalid.maximum_iterations = 0;
    if (bolr_newton_config_validate(&valid) != BOLR_OK) return 1;
    if (bolr_newton_config_validate(&invalid) == BOLR_OK) return 1;
    return 0;
}
