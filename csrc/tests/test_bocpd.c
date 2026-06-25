#include "test_suite.h"

#include "bolr/bocpd.h"

#include <math.h>
#include <stdlib.h>

int test_bocpd(void) {
    bolr_bocpd_config config = {0.2, 8, 0.0, 1.0, 2.0, 1.0, BOLR_BOCPD_MISSING_HOLD};
    bolr_bocpd_state *state = NULL;
    bolr_bocpd_diagnostics diag;
    bolr_real posterior[9];
    size_t encoded_size = 0U;
    void *payload = NULL;
    bolr_bocpd_state *decoded = NULL;
    if (bolr_bocpd_state_create(&config, NULL, &state) != BOLR_OK) return 1;
    if (bolr_bocpd_step(state, 0.1, 1, &diag) != BOLR_OK) return 1;
    if (bolr_bocpd_step(state, 0.2, 1, &diag) != BOLR_OK) return 1;
    if (bolr_bocpd_copy_run_length_posterior(state, (bolr_vector_view){posterior, 9, 1}) != BOLR_OK) return 1;
    if (!(diag.change_probability > 0.0 && diag.change_probability < 1.0)) return 1;
    if (bolr_bocpd_encoded_size(state, &encoded_size) != BOLR_OK) return 1;
    payload = malloc(encoded_size);
    if (payload == NULL) return 1;
    if (bolr_bocpd_encode(state, payload, encoded_size, NULL) != BOLR_OK) return 1;
    if (bolr_bocpd_decode(payload, encoded_size, NULL, &decoded) != BOLR_OK) return 1;
    free(payload);
    bolr_bocpd_state_destroy(decoded);
    bolr_bocpd_state_destroy(state);
    return 0;
}
