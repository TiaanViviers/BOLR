#ifndef BOLR_REGION_H
#define BOLR_REGION_H

#include "bolr/allocator.h"
#include "bolr/array.h"
#include "bolr/prediction.h"

typedef struct bolr_grid_graph bolr_grid_graph;
typedef struct bolr_region_set bolr_region_set;

typedef enum {
    BOLR_CONSENSUS_THRESHOLD = 1,
    BOLR_CONSENSUS_TOP_COUNT = 2,
    BOLR_CONSENSUS_CUMULATIVE_MASS = 3
} bolr_consensus_family;

typedef struct {
    bolr_index top_k;
    bolr_real top_fraction;
    bolr_real inclusion_threshold;
    bolr_consensus_family consensus_family;
} bolr_region_config;

typedef struct {
    bolr_index region_id;
    bolr_index candidate_offset;
    bolr_index candidate_count;
    bolr_real inclusion_mass;
    bolr_real probability_best_mass;
    bolr_real maximum_score_mean;
    bolr_real average_score_mean;
    bolr_real inclusion_weighted_score_mean;
    bolr_real average_score_variance;
    bolr_real maximum_score_variance;
    bolr_real inclusion_weighted_variance;
    bolr_index entry_index_min;
    bolr_index entry_index_max;
    bolr_index stop_index_min;
    bolr_index stop_index_max;
    bolr_real graph_diameter;
    bolr_index boundary_edge_count;
    bolr_real compactness;
    bolr_index representative_medoid_index;
} bolr_region_summary;

bolr_status bolr_grid_graph_create(
    bolr_index node_count,
    const bolr_index *edge_index,
    bolr_index edge_count,
    const bolr_index *entry_indices,
    const bolr_index *stop_indices,
    const bolr_allocator *allocator,
    bolr_grid_graph **out_graph
);
void bolr_grid_graph_destroy(bolr_grid_graph *graph);
uint64_t bolr_grid_graph_hash(const bolr_grid_graph *graph);
bolr_index bolr_grid_graph_node_count(const bolr_grid_graph *graph);
bolr_index bolr_grid_graph_edge_count(const bolr_grid_graph *graph);

bolr_status bolr_region_set_build(
    const bolr_posterior_prediction *prediction,
    const bolr_grid_graph *graph,
    const bolr_region_config *config,
    const bolr_allocator *allocator,
    bolr_region_set **out_regions
);
void bolr_region_set_destroy(bolr_region_set *regions);
bolr_index bolr_region_set_region_count(const bolr_region_set *regions);
bolr_index bolr_region_set_top_k(const bolr_region_set *regions);
bolr_status bolr_region_set_copy_inclusion_probability(const bolr_region_set *regions, bolr_vector_view output);
bolr_status bolr_region_set_copy_consensus_indices(const bolr_region_set *regions, bolr_index *output, bolr_index capacity);
bolr_index bolr_region_set_consensus_count(const bolr_region_set *regions);
int bolr_region_set_empty_consensus(const bolr_region_set *regions);
bolr_status bolr_region_set_summary(const bolr_region_set *regions, bolr_index region_index, bolr_region_summary *out_summary);
bolr_status bolr_region_set_copy_region_candidates(const bolr_region_set *regions, bolr_index region_index, bolr_index *output, bolr_index capacity);
bolr_status bolr_weighted_graph_medoid(
    const bolr_grid_graph *graph,
    const bolr_index *indices,
    bolr_const_vector_view weights,
    bolr_index count,
    bolr_index *out_medoid,
    bolr_real *out_objective
);

#endif
