#include "test_suite.h"

#include "bolr/state_layout.h"
#include "bolr/status.h"

int test_state_layout(void) {
    bolr_state_layout *layout = NULL;
    bolr_state_block_spec specs[2] = {
        {"surface", 0, 2, 2, 1, 1, 'C'},
        {"context", 2, 6, 2, 2, 1, 'F'}
    };
    if (bolr_state_layout_create(specs, 2, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_state_layout_total_dimension(layout) != 6) return 1;
    bolr_state_layout_destroy(layout);
    return 0;
}
