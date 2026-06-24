#include "test_suite.h"

#include "bolr/checkpoint.h"
#include "bolr/gaussian.h"
#include "bolr/status.h"

#include <math.h>
#include <stdlib.h>

int test_gaussian_state(void) {
    bolr_real mean[] = {0.25, -0.5};
    bolr_real covariance[] = {1.0, 0.1, 0.1, 1.5};
    bolr_real copied_mean[] = {0.0, 0.0};
    bolr_real copied_covariance[] = {0.0, 0.0, 0.0, 0.0};
    bolr_gaussian_state *state = NULL;
    bolr_checkpoint_state *checkpoint = NULL;
    bolr_gaussian_state *restored = NULL;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, 7ULL, 11ULL, NULL, &state) != BOLR_OK) return 1;
    if (bolr_gaussian_state_set(state, (bolr_const_vector_view){mean, 2, 1}, (bolr_const_matrix_view){covariance, 2, 2, 2, 1}, 3ULL) != BOLR_OK) return 1;
    if (bolr_gaussian_state_copy_mean(state, (bolr_vector_view){copied_mean, 2, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_copy_covariance(state, (bolr_matrix_view){copied_covariance, 2, 2, 2, 1}) != BOLR_OK) return 1;
    if ((fabs(copied_mean[0] - mean[0]) > 1e-12) || (fabs(copied_covariance[3] - covariance[3]) > 1e-12)) return 1;
    if (bolr_gaussian_state_export(state, NULL, &checkpoint) != BOLR_OK) return 1;
    if (bolr_gaussian_state_import(checkpoint, NULL, &restored) != BOLR_OK) return 1;
    if ((bolr_gaussian_state_step_index(restored) != 3ULL) || (bolr_gaussian_state_state_layout_hash(restored) != 7ULL)) return 1;
    bolr_gaussian_state_destroy(restored);
    bolr_checkpoint_state_destroy(checkpoint);
    bolr_gaussian_state_destroy(state);
    return 0;
}
