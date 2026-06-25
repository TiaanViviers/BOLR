#include "test_suite.h"

#include "bolr/standardizer.h"

#include <math.h>

int test_standardizer(void) {
    bolr_standardizer_config config = {0.5, 1e-4, 0, 0.0, 0};
    bolr_standardizer_state state;
    bolr_standardizer_diagnostics first;
    bolr_standardizer_diagnostics second;
    bolr_standardizer_state_init(&config, &state);
    if (bolr_standardizer_step(&config, &state, 2.0, 1, &first) != BOLR_OK) return 1;
    if (bolr_standardizer_step(&config, &state, 4.0, 1, &second) != BOLR_OK) return 1;
    if (fabs(first.z_score - 200.0) > 1e-9) return 1;
    if (!(second.z_score > 2.0)) return 1;
    if (bolr_standardizer_step(&config, &state, 0.0, 0, &second) != BOLR_OK) return 1;
    return second.missing ? 0 : 1;
}
