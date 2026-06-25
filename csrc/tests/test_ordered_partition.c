#include "test_suite.h"

#include "bolr/partition.h"
#include "bolr/status.h"

int test_ordered_partition(void) {
    bolr_ordered_partition_config config = {{0.2, 0.0, 0.0, BOLR_ORDERED_PARTITION_SCALE_MAD, 1e-6}, 0.0, BOLR_ORDERED_PARTITION_NO_UPDATE, 0.25};
    bolr_ordered_partition *partition = NULL;
    bolr_ordered_partition_diagnostics diagnostics;
    bolr_real utilities[] = {3.0, 2.5, 0.5, -1.0, -2.0};
    bolr_index candidate_to_group[] = {0, 0, 0, 0, 0};
    if (bolr_ordered_partition_build(&config, (bolr_const_vector_view){utilities, 5, 1}, NULL, &partition) != BOLR_OK) return 1;
    if (bolr_ordered_partition_get_diagnostics(partition, &diagnostics) != BOLR_OK) return 1;
    if ((diagnostics.group_count != 3) || (diagnostics.high_group_size != 1) || (diagnostics.middle_group_size != 2) || (diagnostics.low_group_size != 2)) return 1;
    if (bolr_ordered_partition_copy_candidate_to_group(partition, candidate_to_group, 5) != BOLR_OK) return 1;
    if ((candidate_to_group[0] != 0) || (candidate_to_group[1] != 1) || (candidate_to_group[2] != 1) || (candidate_to_group[3] != 2) || (candidate_to_group[4] != 2)) return 1;
    bolr_ordered_partition_destroy(partition);
    return 0;
}
