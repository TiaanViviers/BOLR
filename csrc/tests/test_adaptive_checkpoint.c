#include "test_suite.h"

#include "bolr/adaptation.h"
#include "bolr/state_layout.h"

#include <stdlib.h>

int test_adaptive_checkpoint(void) {
    bolr_state_layout *layout = NULL;
    bolr_state_block_spec specs[1] = {{"surface", 0, 1, 1, 1, 1, 'C'}};
    bolr_real q[1] = {0.1};
    bolr_adaptive_policy_config config = {
        BOLR_SURPRISE_GENERALIZED_LOSS_STRENGTH_NORMALIZED,
        {0.5, 1e-4, 0, 0.0, 0},
        {0.2, 8, 0.0, 1.0, 2.0, 1.0, BOLR_BOCPD_MISSING_HOLD},
        1.0,
        2.0,
        1e-8,
    };
    bolr_adaptive_block_config block = {"surface", BOLR_ADAPTIVE_BLOCK_ADDITIVE, 4.0, 1.0, 0.0, 1e-6, 1.0, 0, 0, 0.0, 0, 0.0, 0, 0, 1.0, 1};
    bolr_adaptive_policy *policy = NULL;
    bolr_adaptive_state *state = NULL;
    bolr_adaptive_state *decoded = NULL;
    size_t size = 0U;
    void *payload = NULL;
    if (bolr_state_layout_create(specs, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_adaptive_policy_create(layout, (bolr_const_matrix_view){q, 1, 1, 1, 1}, &config, &block, 1, NULL, &policy) != BOLR_OK) return 1;
    if (bolr_adaptive_state_create(policy, NULL, &state) != BOLR_OK) return 1;
    if (bolr_adaptive_state_encoded_size(policy, state, &size) != BOLR_OK) return 1;
    payload = malloc(size);
    if (payload == NULL) return 1;
    if (bolr_adaptive_state_encode(policy, state, payload, size, NULL) != BOLR_OK) return 1;
    if (bolr_adaptive_state_decode(policy, payload, size, NULL, &decoded) != BOLR_OK) return 1;
    free(payload);
    bolr_adaptive_state_destroy(decoded);
    bolr_adaptive_state_destroy(state);
    bolr_adaptive_policy_destroy(policy);
    bolr_state_layout_destroy(layout);
    return 0;
}
