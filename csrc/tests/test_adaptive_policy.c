#include "test_suite.h"

#include "bolr/adaptation.h"
#include "bolr/gaussian.h"
#include "bolr/state_layout.h"

#include <math.h>

int test_adaptive_policy(void) {
    bolr_state_layout *layout = NULL;
    bolr_state_block_spec specs[2] = {
        {"surface", 0, 1, 1, 1, 1, 'C'},
        {"residual", 1, 2, 1, 1, 1, 'C'},
    };
    bolr_real q[4] = {0.2, 0.0, 0.0, 0.1};
    bolr_real mean0[2] = {0.0, 0.0};
    bolr_real cov0[4] = {1.0, 0.0, 0.0, 1.0};
    bolr_real mean1[2] = {1.0, 0.0};
    bolr_real cov1[4] = {0.8, 0.0, 0.0, 0.9};
    bolr_adaptive_policy_config config = {
        BOLR_SURPRISE_GENERALIZED_LOSS_STRENGTH_NORMALIZED,
        {0.5, 1e-4, 0, 0.0, 0},
        {0.2, 8, 0.0, 1.0, 2.0, 1.0, BOLR_BOCPD_MISSING_HOLD},
        1.0,
        2.0,
        1e-8,
    };
    bolr_adaptive_block_config blocks[2] = {
        {"surface", BOLR_ADAPTIVE_BLOCK_ADDITIVE, 4.0, 1.0, 0.0, 1e-6, 1.0, 0, 0, 0.0, 0, 0.0, 0, 0, 2.0, 1},
        {"residual", BOLR_ADAPTIVE_BLOCK_ADDITIVE, 4.0, 1.0, 0.0, 1e-6, 1.0, 0, 0, 0.0, 0, 0.0, 0, 0, 1.0, 1},
    };
    bolr_adaptive_policy *policy = NULL;
    bolr_adaptive_state *state = NULL;
    bolr_gaussian_state *posterior0 = NULL;
    bolr_gaussian_state *predictive = NULL;
    bolr_gaussian_state *posterior1 = NULL;
    bolr_surprise_input surprise = {1, -2.0, -1.5, 1.0, 1.0, 1.0, 0.3, 0.5};
    bolr_adaptation_diagnostics diag = {0};
    bolr_real multipliers[2];
    bolr_real attrib[2];
    bolr_real target[2];
    int32_t reset_flags[2];
    if (bolr_state_layout_create(specs, 2, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_adaptive_policy_create(layout, (bolr_const_matrix_view){q, 2, 2, 2, 1}, &config, blocks, 2, NULL, &policy) != BOLR_OK) return 1;
    if (bolr_adaptive_state_create(policy, NULL, &state) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean0, 2, 1}, (bolr_const_matrix_view){cov0, 2, 2, 2, 1}, bolr_state_layout_schema_hash(layout), 7ULL, NULL, &posterior0) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean1, 2, 1}, (bolr_const_matrix_view){cov1, 2, 2, 2, 1}, bolr_state_layout_schema_hash(layout), 7ULL, NULL, &posterior1) != BOLR_OK) return 1;
    if (bolr_adaptive_policy_predict(policy, state, posterior0, NULL, &predictive, NULL) != BOLR_OK) return 1;
    diag.process_noise_multiplier = multipliers;
    diag.attribution_weight = attrib;
    diag.target_multiplier = target;
    diag.reset_scheduled = reset_flags;
    if (bolr_adaptive_policy_observe(policy, state, predictive, posterior1, &surprise, &diag) != BOLR_OK) return 1;
    bolr_gaussian_state_destroy(predictive);
    bolr_gaussian_state_destroy(posterior0);
    bolr_gaussian_state_destroy(posterior1);
    bolr_adaptive_state_destroy(state);
    bolr_adaptive_policy_destroy(policy);
    bolr_state_layout_destroy(layout);
    return (multipliers[0] > multipliers[1] && diag.activation_value >= 0.0) ? 0 : 1;
}
