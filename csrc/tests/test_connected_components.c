#include "test_suite.h"

#include "bolr/gaussian.h"
#include "bolr/prediction.h"
#include "bolr/region.h"
#include "bolr/score.h"
#include "bolr/state_layout.h"

int test_connected_components(void) {
    bolr_state_layout *layout = NULL;
    bolr_model *model = NULL;
    bolr_gaussian_state *state = NULL;
    bolr_posterior_prediction *prediction = NULL;
    bolr_region_set *regions = NULL;
    bolr_grid_graph *graph = NULL;
    bolr_region_summary summary0;
    bolr_region_summary summary1;
    bolr_region_config config = {2, 0.0, 0.4, BOLR_CONSENSUS_THRESHOLD};
    bolr_state_block_spec spec = {"dense", 0, 1, 1, 6, 1, 'C'};
    bolr_real static_scores[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bolr_real design[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    bolr_real mean[] = {0.0};
    bolr_real covariance[] = {1.0};
    bolr_real top2[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    bolr_index edges[] = {0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 1, 4, 2, 5};
    bolr_index entry_indices[] = {0, 0, 0, 1, 1, 1};
    bolr_index stop_indices[] = {0, 1, 2, 0, 1, 2};
    if (bolr_state_layout_create(&spec, 1, NULL, &layout) != BOLR_OK) return 1;
    if (bolr_model_create(layout, (bolr_const_vector_view){static_scores, 6, 1}, NULL, &model) != BOLR_OK) return 1;
    if (bolr_model_add_dense_block_copy(model, "dense", (bolr_const_matrix_view){design, 6, 1, 1, 1}) != BOLR_OK) return 1;
    if (bolr_gaussian_state_create((bolr_const_vector_view){mean, 1, 1}, (bolr_const_matrix_view){covariance, 1, 1, 1, 1}, bolr_model_state_layout_hash(model), bolr_model_schema_hash(model), NULL, &state) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_create(state, model, (bolr_const_vector_view){NULL, 0, 1}, NULL, NULL, &prediction, NULL) != BOLR_OK) return 1;
    if (bolr_posterior_prediction_set_probability_top_k(prediction, 2, (bolr_const_vector_view){top2, 6, 1}) != BOLR_OK) return 1;
    if (bolr_grid_graph_create(6, edges, 7, entry_indices, stop_indices, NULL, &graph) != BOLR_OK) return 1;
    if (bolr_region_set_build(prediction, graph, &config, NULL, &regions) != BOLR_OK) return 1;
    if (bolr_region_set_region_count(regions) != 2) return 1;
    if (bolr_region_set_summary(regions, 0, &summary0) != BOLR_OK) return 1;
    if (bolr_region_set_summary(regions, 1, &summary1) != BOLR_OK) return 1;
    bolr_region_set_destroy(regions);
    bolr_grid_graph_destroy(graph);
    bolr_posterior_prediction_destroy(prediction);
    bolr_gaussian_state_destroy(state);
    bolr_model_destroy(model);
    bolr_state_layout_destroy(layout);
    return ((summary0.candidate_count != 1) || (summary1.candidate_count != 1)) ? 1 : 0;
}
