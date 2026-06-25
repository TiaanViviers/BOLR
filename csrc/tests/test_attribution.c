#include "test_suite.h"

#include "bolr/attribution.h"
#include "bolr/state_layout.h"

int test_attribution(void) {
    bolr_state_layout *layout = NULL;
    bolr_state_block_spec specs[2] = {
        {"surface", 0, 1, 1, 1, 1, 'C'},
        {"residual", 1, 2, 1, 1, 1, 'C'},
    };
    bolr_real mean0[2] = {0.0, 0.0};
    bolr_real mean1[2] = {1.0, 0.0};
    bolr_real covariance[4] = {1.0, 0.0, 0.0, 1.0};
    bolr_real euclid[2];
    bolr_real mahal[2];
    bolr_real weight[2];
    if (bolr_state_layout_create(specs, 2, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_block_innovation_attribution(
        layout,
        (bolr_const_vector_view){mean0, 2, 1},
        (bolr_const_matrix_view){covariance, 2, 2, 2, 1},
        (bolr_const_vector_view){mean1, 2, 1},
        1e-8,
        (bolr_vector_view){euclid, 2, 1},
        (bolr_vector_view){mahal, 2, 1},
        (bolr_vector_view){weight, 2, 1}
    ) != BOLR_OK) return 1;
    bolr_state_layout_destroy(layout);
    return (weight[0] > weight[1]) ? 0 : 1;
}
