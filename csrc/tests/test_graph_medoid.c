#include "test_suite.h"

#include "bolr/region.h"

#include <math.h>

int test_graph_medoid(void) {
    bolr_grid_graph *graph = NULL;
    bolr_index edges[] = {0, 1, 1, 2};
    bolr_index entry_indices[] = {0, 0, 0};
    bolr_index stop_indices[] = {0, 1, 2};
    bolr_index indices[] = {0, 1, 2};
    bolr_real weights[] = {0.2, 0.6, 0.2};
    bolr_index medoid = -1;
    bolr_real objective = 0.0;
    if (bolr_grid_graph_create(3, edges, 2, entry_indices, stop_indices, NULL, &graph) != BOLR_OK) return 1;
    if (bolr_weighted_graph_medoid(graph, indices, (bolr_const_vector_view){weights, 3, 1}, 3, &medoid, &objective) != BOLR_OK) return 1;
    bolr_grid_graph_destroy(graph);
    return ((medoid != 1) || (!isfinite(objective))) ? 1 : 0;
}
