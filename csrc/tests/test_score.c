#include "test_suite.h"

#include "bolr/score.h"
#include "bolr/status.h"

#include <math.h>

int test_score(void) {
    bolr_dense_operator *op = NULL;
    bolr_real design[] = {1.0, 0.0, 0.5, 1.0};
    bolr_real state[] = {2.0, 3.0};
    bolr_real scores[] = {0.0, 0.0};
    bolr_workspace *workspace = NULL;
    bolr_workspace_config config = {4, 4, 4};
    if (bolr_workspace_create(&config, NULL, &workspace) != BOLR_OK) return 1;
    if (bolr_dense_operator_create_copy((bolr_const_matrix_view){design, 2, 2, 2, 1}, NULL, &op) != BOLR_OK) return 1;
    if (bolr_dense_operator_forward(op, (bolr_const_vector_view){state, 2, 1}, (bolr_vector_view){scores, 2, 1}, workspace) != BOLR_OK) return 1;
    bolr_dense_operator_destroy(op);
    bolr_workspace_destroy(workspace);
    return (fabs(scores[1] - 4.0) > 1e-12) ? 1 : 0;
}
