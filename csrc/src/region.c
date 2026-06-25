#include "bolr/region.h"

#include "bolr/linalg.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a_update(uint64_t state, const unsigned char *data, size_t size) {
    size_t i;
    for (i = 0; i < size; ++i) {
        state ^= (uint64_t) data[i];
        state *= 1099511628211ULL;
    }
    return state;
}

static bolr_index top_k_from_config(const bolr_region_config *config, bolr_index candidate_count) {
    bolr_index top_k;
    if (config->top_k > 0) top_k = config->top_k;
    else if (config->top_fraction > 0.0) top_k = (bolr_index) ceil(config->top_fraction * (bolr_real) candidate_count);
    else top_k = 1;
    if (top_k < 1) top_k = 1;
    if (top_k > candidate_count) top_k = candidate_count;
    return top_k;
}

static void stable_sort_probability_desc(const bolr_real *probability, bolr_index *indices, bolr_index count) {
    bolr_index i;
    for (i = 1; i < count; ++i) {
        bolr_index key = indices[i];
        bolr_index j = i;
        while (j > 0) {
            bolr_index prev = indices[j - 1];
            if (probability[prev] > probability[key]) break;
            if ((probability[prev] == probability[key]) && (prev < key)) break;
            indices[j] = prev;
            j -= 1;
        }
        indices[j] = key;
    }
}

static void stable_sort_index_asc(bolr_index *indices, bolr_index count) {
    bolr_index i;
    for (i = 1; i < count; ++i) {
        bolr_index key = indices[i];
        bolr_index j = i;
        while ((j > 0) && (indices[j - 1] > key)) {
            indices[j] = indices[j - 1];
            j -= 1;
        }
        indices[j] = key;
    }
}

static bolr_status build_consensus(
    const bolr_real *inclusion_probability,
    bolr_index candidate_count,
    const bolr_region_config *config,
    bolr_index **out_indices,
    bolr_index *out_count,
    int *out_empty
) {
    bolr_index *indices;
    bolr_index *order;
    bolr_index count = 0;
    bolr_index i;
    if ((inclusion_probability == NULL) || (config == NULL) || (out_indices == NULL) || (out_count == NULL) || (out_empty == NULL) || (candidate_count <= 0)) {
        return BOLR_INVALID_ARGUMENT;
    }
    *out_indices = NULL;
    *out_count = 0;
    *out_empty = 0;
    indices = (bolr_index *) malloc((size_t) candidate_count * sizeof(*indices));
    order = (bolr_index *) malloc((size_t) candidate_count * sizeof(*order));
    if ((indices == NULL) || (order == NULL)) {
        free(indices);
        free(order);
        return BOLR_ALLOCATION_FAILED;
    }
    for (i = 0; i < candidate_count; ++i) order[i] = i;
    stable_sort_probability_desc(inclusion_probability, order, candidate_count);
    if (config->consensus_family == BOLR_CONSENSUS_THRESHOLD) {
        for (i = 0; i < candidate_count; ++i) if (inclusion_probability[i] >= config->inclusion_threshold) indices[count++] = i;
    } else if (config->consensus_family == BOLR_CONSENSUS_TOP_COUNT) {
        count = (bolr_index) round(config->inclusion_threshold);
        if (count < 1) count = 1;
        if (count > candidate_count) count = candidate_count;
        for (i = 0; i < count; ++i) indices[i] = order[i];
    } else {
        bolr_real cumulative = 0.0;
        for (i = 0; i < candidate_count; ++i) {
            indices[count++] = order[i];
            cumulative += inclusion_probability[order[i]];
            if (cumulative >= config->inclusion_threshold) break;
        }
    }
    if (count == 0) {
        indices[0] = order[0];
        count = 1;
        *out_empty = 1;
    }
    stable_sort_index_asc(indices, count);
    free(order);
    *out_indices = indices;
    *out_count = count;
    return BOLR_OK;
}

static bolr_status bfs_component(
    const struct bolr_grid_graph *graph,
    const int *eligible,
    bolr_index start,
    int *visited,
    bolr_index *queue,
    bolr_index *output,
    bolr_index *out_count
) {
    bolr_index head = 0;
    bolr_index tail = 0;
    bolr_index count = 0;
    bolr_index e;
    if ((graph == NULL) || (eligible == NULL) || (visited == NULL) || (queue == NULL) || (output == NULL) || (out_count == NULL)) return BOLR_INVALID_ARGUMENT;
    visited[start] = 1;
    queue[tail++] = start;
    while (head < tail) {
        bolr_index node = queue[head++];
        output[count++] = node;
        for (e = 0; e < graph->edge_count; ++e) {
            bolr_index left = graph->edge_index[2 * e];
            bolr_index right = graph->edge_index[2 * e + 1];
            bolr_index next = -1;
            if ((left == node) && eligible[right] && !visited[right]) next = right;
            else if ((right == node) && eligible[left] && !visited[left]) next = left;
            if (next >= 0) {
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    stable_sort_index_asc(output, count);
    *out_count = count;
    return BOLR_OK;
}

bolr_status bolr_grid_graph_create(
    bolr_index node_count,
    const bolr_index *edge_index,
    bolr_index edge_count,
    const bolr_index *entry_indices,
    const bolr_index *stop_indices,
    const bolr_allocator *allocator,
    bolr_grid_graph **out_graph
) {
    struct bolr_grid_graph *graph;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    size_t edge_bytes;
    size_t node_bytes;
    bolr_index i;
    if ((node_count <= 0) || (edge_count < 0) || (edge_index == NULL) || (entry_indices == NULL) || (stop_indices == NULL) || (out_graph == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_graph = NULL;
    graph = (struct bolr_grid_graph *) bolr_allocator_calloc(active, 1U, sizeof(*graph));
    if (graph == NULL) return BOLR_ALLOCATION_FAILED;
    graph->allocator = active;
    graph->node_count = node_count;
    graph->edge_count = edge_count;
    edge_bytes = (size_t) (2 * edge_count) * sizeof(bolr_index);
    node_bytes = (size_t) node_count * sizeof(bolr_index);
    graph->edge_index = (bolr_index *) bolr_allocator_malloc(active, edge_bytes);
    graph->entry_indices = (bolr_index *) bolr_allocator_malloc(active, node_bytes);
    graph->stop_indices = (bolr_index *) bolr_allocator_malloc(active, node_bytes);
    if ((graph->edge_index == NULL) || (graph->entry_indices == NULL) || (graph->stop_indices == NULL)) {
        bolr_grid_graph_destroy(graph);
        return BOLR_ALLOCATION_FAILED;
    }
    memcpy(graph->edge_index, edge_index, edge_bytes);
    memcpy(graph->entry_indices, entry_indices, node_bytes);
    memcpy(graph->stop_indices, stop_indices, node_bytes);
    for (i = 0; i < edge_count; ++i) {
        bolr_index left = graph->edge_index[2 * i];
        bolr_index right = graph->edge_index[2 * i + 1];
        if ((left < 0) || (left >= node_count) || (right < 0) || (right >= node_count)) {
            bolr_grid_graph_destroy(graph);
            return BOLR_INVALID_ARGUMENT;
        }
    }
    graph->graph_hash = 1469598103934665603ULL;
    graph->graph_hash = fnv1a_update(graph->graph_hash, (const unsigned char *) graph->edge_index, edge_bytes);
    graph->graph_hash = fnv1a_update(graph->graph_hash, (const unsigned char *) graph->entry_indices, node_bytes);
    graph->graph_hash = fnv1a_update(graph->graph_hash, (const unsigned char *) graph->stop_indices, node_bytes);
    *out_graph = graph;
    return BOLR_OK;
}

void bolr_grid_graph_destroy(bolr_grid_graph *opaque) {
    struct bolr_grid_graph *graph = opaque;
    if (graph == NULL) return;
    bolr_allocator_free(graph->allocator, graph->edge_index);
    bolr_allocator_free(graph->allocator, graph->entry_indices);
    bolr_allocator_free(graph->allocator, graph->stop_indices);
    bolr_allocator_free(graph->allocator, graph);
}

uint64_t bolr_grid_graph_hash(const bolr_grid_graph *opaque) {
    const struct bolr_grid_graph *graph = opaque;
    return (graph == NULL) ? 0ULL : graph->graph_hash;
}

bolr_index bolr_grid_graph_node_count(const bolr_grid_graph *opaque) {
    const struct bolr_grid_graph *graph = opaque;
    return (graph == NULL) ? -1 : graph->node_count;
}

bolr_index bolr_grid_graph_edge_count(const bolr_grid_graph *opaque) {
    const struct bolr_grid_graph *graph = opaque;
    return (graph == NULL) ? -1 : graph->edge_count;
}

static bolr_status shortest_path_sum(
    const struct bolr_grid_graph *graph,
    const int *mask,
    const bolr_index *indices,
    bolr_const_vector_view weights,
    bolr_index count,
    bolr_index source_slot,
    bolr_real *out_objective,
    bolr_real *out_max_distance
) {
    bolr_index *queue;
    bolr_index *distance;
    bolr_index head = 0;
    bolr_index tail = 0;
    bolr_index i;
    bolr_index e;
    bolr_real objective = 0.0;
    bolr_real max_distance = 0.0;
    if ((graph == NULL) || (mask == NULL) || (indices == NULL) || (weights.length != count) || (source_slot < 0) || (source_slot >= count) || (out_objective == NULL) || (out_max_distance == NULL)) {
        return BOLR_INVALID_ARGUMENT;
    }
    queue = (bolr_index *) malloc((size_t) count * sizeof(*queue));
    distance = (bolr_index *) malloc((size_t) graph->node_count * sizeof(*distance));
    if ((queue == NULL) || (distance == NULL)) {
        free(queue);
        free(distance);
        return BOLR_ALLOCATION_FAILED;
    }
    for (i = 0; i < graph->node_count; ++i) distance[i] = -1;
    distance[indices[source_slot]] = 0;
    queue[tail++] = indices[source_slot];
    while (head < tail) {
        bolr_index node = queue[head++];
        for (e = 0; e < graph->edge_count; ++e) {
            bolr_index left = graph->edge_index[2 * e];
            bolr_index right = graph->edge_index[2 * e + 1];
            bolr_index next = -1;
            if ((left == node) && mask[right] && (distance[right] < 0)) next = right;
            else if ((right == node) && mask[left] && (distance[left] < 0)) next = left;
            if (next >= 0) {
                distance[next] = distance[node] + 1;
                queue[tail++] = next;
            }
        }
    }
    for (i = 0; i < count; ++i) {
        bolr_index d = distance[indices[i]];
        if (d < 0) {
            free(queue);
            free(distance);
            return BOLR_NUMERICAL_FAILURE;
        }
        objective += weights.data[i * weights.stride] * (bolr_real) d;
        if ((bolr_real) d > max_distance) max_distance = (bolr_real) d;
    }
    free(queue);
    free(distance);
    *out_objective = objective;
    *out_max_distance = max_distance;
    return BOLR_OK;
}

bolr_status bolr_weighted_graph_medoid(
    const bolr_grid_graph *opaque,
    const bolr_index *indices,
    bolr_const_vector_view weights,
    bolr_index count,
    bolr_index *out_medoid,
    bolr_real *out_objective
) {
    const struct bolr_grid_graph *graph = opaque;
    int *mask;
    bolr_index best_slot = 0;
    bolr_real best_objective = INFINITY;
    bolr_index i;
    if ((graph == NULL) || (indices == NULL) || (out_medoid == NULL) || (out_objective == NULL) || (count <= 0) || (weights.length != count)) {
        return BOLR_INVALID_ARGUMENT;
    }
    mask = (int *) calloc((size_t) graph->node_count, sizeof(*mask));
    if (mask == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < count; ++i) {
        if ((indices[i] < 0) || (indices[i] >= graph->node_count)) {
            free(mask);
            return BOLR_INVALID_ARGUMENT;
        }
        mask[indices[i]] = 1;
    }
    for (i = 0; i < count; ++i) {
        bolr_real objective = 0.0;
        bolr_real max_distance = 0.0;
        bolr_status status = shortest_path_sum(graph, mask, indices, weights, count, i, &objective, &max_distance);
        if (status != BOLR_OK) {
            free(mask);
            return status;
        }
        if ((objective < best_objective) || ((objective == best_objective) && (indices[i] < indices[best_slot]))) {
            best_slot = i;
            best_objective = objective;
        }
    }
    free(mask);
    *out_medoid = indices[best_slot];
    *out_objective = best_objective;
    return BOLR_OK;
}

bolr_status bolr_region_set_build(
    const bolr_posterior_prediction *prediction_opaque,
    const bolr_grid_graph *graph_opaque,
    const bolr_region_config *config,
    const bolr_allocator *allocator,
    bolr_region_set **out_regions
) {
    const struct bolr_posterior_prediction *prediction = prediction_opaque;
    const struct bolr_grid_graph *graph = graph_opaque;
    struct bolr_region_set *regions;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index top_k;
    bolr_index top_k_slot = -1;
    bolr_index *consensus = NULL;
    bolr_index consensus_count = 0;
    int empty_consensus = 0;
    int *eligible = NULL;
    int *visited = NULL;
    bolr_index *queue = NULL;
    bolr_index *component_buffer = NULL;
    bolr_index region_count = 0;
    bolr_index total_region_candidates = 0;
    bolr_index i;
    if ((prediction == NULL) || (graph == NULL) || (config == NULL) || (out_regions == NULL)) return BOLR_INVALID_ARGUMENT;
    if (prediction->candidate_count != graph->node_count) return BOLR_INVALID_SHAPE;
    *out_regions = NULL;
    top_k = top_k_from_config(config, prediction->candidate_count);
    for (i = 0; i < prediction->probability_top_k_count; ++i) {
        if (prediction->probability_top_k_keys[i] == top_k) {
            top_k_slot = i;
            break;
        }
    }
    if (top_k_slot < 0) return BOLR_INVALID_ARGUMENT;
    if (build_consensus(prediction->probability_top_k_values[top_k_slot], prediction->candidate_count, config, &consensus, &consensus_count, &empty_consensus) != BOLR_OK) {
        return BOLR_ALLOCATION_FAILED;
    }
    eligible = (int *) calloc((size_t) prediction->candidate_count, sizeof(*eligible));
    visited = (int *) calloc((size_t) prediction->candidate_count, sizeof(*visited));
    queue = (bolr_index *) malloc((size_t) consensus_count * sizeof(*queue));
    component_buffer = (bolr_index *) malloc((size_t) consensus_count * sizeof(*component_buffer));
    if ((eligible == NULL) || (visited == NULL) || (queue == NULL) || (component_buffer == NULL)) {
        free(consensus);
        free(eligible);
        free(visited);
        free(queue);
        free(component_buffer);
        return BOLR_ALLOCATION_FAILED;
    }
    for (i = 0; i < consensus_count; ++i) eligible[consensus[i]] = 1;
    for (i = 0; i < consensus_count; ++i) {
        bolr_index component_count = 0;
        if (visited[consensus[i]]) continue;
        if (bfs_component(graph, eligible, consensus[i], visited, queue, component_buffer, &component_count) != BOLR_OK) {
            free(consensus);
            free(eligible);
            free(visited);
            free(queue);
            free(component_buffer);
            return BOLR_NUMERICAL_FAILURE;
        }
        region_count += 1;
        total_region_candidates += component_count;
    }
    memset(visited, 0, (size_t) prediction->candidate_count * sizeof(*visited));
    regions = (struct bolr_region_set *) bolr_allocator_calloc(active, 1U, sizeof(*regions));
    if (regions == NULL) {
        free(consensus);
        free(eligible);
        free(visited);
        free(queue);
        free(component_buffer);
        return BOLR_ALLOCATION_FAILED;
    }
    regions->allocator = active;
    regions->candidate_count = prediction->candidate_count;
    regions->consensus_count = consensus_count;
    regions->top_k = top_k;
    regions->empty_consensus = empty_consensus;
    regions->region_count = region_count;
    regions->region_candidate_count = total_region_candidates;
    regions->inclusion_probability = (bolr_real *) bolr_allocator_malloc(active, (size_t) prediction->candidate_count * sizeof(bolr_real));
    regions->consensus_indices = (bolr_index *) bolr_allocator_malloc(active, (size_t) consensus_count * sizeof(bolr_index));
    regions->summaries = (bolr_region_summary *) bolr_allocator_calloc(active, (size_t) region_count, sizeof(bolr_region_summary));
    regions->region_candidates = (bolr_index *) bolr_allocator_malloc(active, (size_t) total_region_candidates * sizeof(bolr_index));
    if ((regions->inclusion_probability == NULL) || (regions->consensus_indices == NULL) || (regions->summaries == NULL) || (regions->region_candidates == NULL)) {
        bolr_region_set_destroy(regions);
        free(consensus);
        free(eligible);
        free(visited);
        free(queue);
        free(component_buffer);
        return BOLR_ALLOCATION_FAILED;
    }
    memcpy(regions->inclusion_probability, prediction->probability_top_k_values[top_k_slot], (size_t) prediction->candidate_count * sizeof(bolr_real));
    memcpy(regions->consensus_indices, consensus, (size_t) consensus_count * sizeof(bolr_index));
    {
        bolr_index region_idx = 0;
        bolr_index offset = 0;
        for (i = 0; i < consensus_count; ++i) {
            bolr_index component_count = 0;
            bolr_index j;
            bolr_region_summary *summary;
            bolr_real inclusion_mass = 0.0;
            bolr_real probability_best_mass = 0.0;
            bolr_real maximum_score_mean = -INFINITY;
            bolr_real average_score_mean = 0.0;
            bolr_real weighted_score_mean = 0.0;
            bolr_real average_score_variance = 0.0;
            bolr_real maximum_score_variance = -INFINITY;
            bolr_real weighted_variance = 0.0;
            bolr_real weight_sum = 0.0;
            bolr_real graph_diameter = 0.0;
            bolr_index entry_min = prediction->candidate_count;
            bolr_index entry_max = -1;
            bolr_index stop_min = prediction->candidate_count;
            bolr_index stop_max = -1;
            bolr_index boundary_edges = 0;
            bolr_index internal_edges = 0;
            bolr_index medoid = -1;
            bolr_real medoid_objective = 0.0;
            int *component_mask;
            bolr_real *weights;
            if (visited[consensus[i]]) continue;
            if (bfs_component(graph, eligible, consensus[i], visited, queue, component_buffer, &component_count) != BOLR_OK) {
                bolr_region_set_destroy(regions);
                free(consensus);
                free(eligible);
                free(visited);
                free(queue);
                free(component_buffer);
                return BOLR_NUMERICAL_FAILURE;
            }
            component_mask = (int *) calloc((size_t) graph->node_count, sizeof(*component_mask));
            weights = (bolr_real *) malloc((size_t) component_count * sizeof(*weights));
            if ((component_mask == NULL) || (weights == NULL)) {
                free(component_mask);
                free(weights);
                bolr_region_set_destroy(regions);
                free(consensus);
                free(eligible);
                free(visited);
                free(queue);
                free(component_buffer);
                return BOLR_ALLOCATION_FAILED;
            }
            for (j = 0; j < component_count; ++j) {
                bolr_index node = component_buffer[j];
                bolr_real weight = regions->inclusion_probability[node];
                regions->region_candidates[offset + j] = node;
                component_mask[node] = 1;
                weights[j] = weight;
                inclusion_mass += weight;
                weight_sum += weight;
                if (prediction->probability_best != NULL) probability_best_mass += prediction->probability_best[node];
                average_score_mean += prediction->score_mean[node];
                if (prediction->score_mean[node] > maximum_score_mean) maximum_score_mean = prediction->score_mean[node];
                weighted_score_mean += weight * prediction->score_mean[node];
                average_score_variance += prediction->score_variance[node];
                if (prediction->score_variance[node] > maximum_score_variance) maximum_score_variance = prediction->score_variance[node];
                weighted_variance += weight * prediction->score_variance[node];
                if (graph->entry_indices[node] < entry_min) entry_min = graph->entry_indices[node];
                if (graph->entry_indices[node] > entry_max) entry_max = graph->entry_indices[node];
                if (graph->stop_indices[node] < stop_min) stop_min = graph->stop_indices[node];
                if (graph->stop_indices[node] > stop_max) stop_max = graph->stop_indices[node];
            }
            for (j = 0; j < graph->edge_count; ++j) {
                bolr_index left = graph->edge_index[2 * j];
                bolr_index right = graph->edge_index[2 * j + 1];
                int left_here = component_mask[left];
                int right_here = component_mask[right];
                if (left_here && right_here) internal_edges += 1;
                else if (left_here || right_here) boundary_edges += 1;
            }
            if (bolr_weighted_graph_medoid(graph, component_buffer, (bolr_const_vector_view){weights, component_count, 1}, component_count, &medoid, &medoid_objective) != BOLR_OK) {
                free(component_mask);
                free(weights);
                bolr_region_set_destroy(regions);
                free(consensus);
                free(eligible);
                free(visited);
                free(queue);
                free(component_buffer);
                return BOLR_NUMERICAL_FAILURE;
            }
            for (j = 0; j < component_count; ++j) {
                bolr_real objective = 0.0;
                bolr_real max_distance = 0.0;
                if (shortest_path_sum(graph, component_mask, component_buffer, (bolr_const_vector_view){weights, component_count, 1}, component_count, j, &objective, &max_distance) != BOLR_OK) {
                    free(component_mask);
                    free(weights);
                    bolr_region_set_destroy(regions);
                    free(consensus);
                    free(eligible);
                    free(visited);
                    free(queue);
                    free(component_buffer);
                    return BOLR_NUMERICAL_FAILURE;
                }
                if (max_distance > graph_diameter) graph_diameter = max_distance;
            }
            free(component_mask);
            free(weights);
            summary = &regions->summaries[region_idx];
            summary->region_id = region_idx;
            summary->candidate_offset = offset;
            summary->candidate_count = component_count;
            summary->inclusion_mass = inclusion_mass;
            summary->probability_best_mass = probability_best_mass;
            summary->maximum_score_mean = maximum_score_mean;
            summary->average_score_mean = average_score_mean / (bolr_real) component_count;
            summary->inclusion_weighted_score_mean = weighted_score_mean / ((weight_sum > 0.0) ? weight_sum : 1.0);
            summary->average_score_variance = average_score_variance / (bolr_real) component_count;
            summary->maximum_score_variance = maximum_score_variance;
            summary->inclusion_weighted_variance = weighted_variance / ((weight_sum > 0.0) ? weight_sum : 1.0);
            summary->entry_index_min = entry_min;
            summary->entry_index_max = entry_max;
            summary->stop_index_min = stop_min;
            summary->stop_index_max = stop_max;
            summary->graph_diameter = graph_diameter;
            summary->boundary_edge_count = boundary_edges;
            summary->compactness = (bolr_real) internal_edges / (bolr_real) ((component_count > 1) ? (component_count - 1) : 1);
            summary->representative_medoid_index = medoid;
            offset += component_count;
            region_idx += 1;
        }
    }
    free(consensus);
    free(eligible);
    free(visited);
    free(queue);
    free(component_buffer);
    *out_regions = regions;
    return BOLR_OK;
}

void bolr_region_set_destroy(bolr_region_set *opaque) {
    struct bolr_region_set *regions = opaque;
    if (regions == NULL) return;
    bolr_allocator_free(regions->allocator, regions->inclusion_probability);
    bolr_allocator_free(regions->allocator, regions->consensus_indices);
    bolr_allocator_free(regions->allocator, regions->summaries);
    bolr_allocator_free(regions->allocator, regions->region_candidates);
    bolr_allocator_free(regions->allocator, regions);
}

bolr_index bolr_region_set_region_count(const bolr_region_set *opaque) {
    const struct bolr_region_set *regions = opaque;
    return (regions == NULL) ? -1 : regions->region_count;
}

bolr_index bolr_region_set_top_k(const bolr_region_set *opaque) {
    const struct bolr_region_set *regions = opaque;
    return (regions == NULL) ? -1 : regions->top_k;
}

bolr_status bolr_region_set_copy_inclusion_probability(const bolr_region_set *opaque, bolr_vector_view output) {
    const struct bolr_region_set *regions = opaque;
    if ((regions == NULL) || (output.length != regions->candidate_count)) return BOLR_INVALID_ARGUMENT;
    return bolr_copy((bolr_const_vector_view){regions->inclusion_probability, regions->candidate_count, 1}, output);
}

bolr_status bolr_region_set_copy_consensus_indices(const bolr_region_set *opaque, bolr_index *output, bolr_index capacity) {
    const struct bolr_region_set *regions = opaque;
    if ((regions == NULL) || (output == NULL) || (capacity < regions->consensus_count)) return BOLR_INVALID_ARGUMENT;
    memcpy(output, regions->consensus_indices, (size_t) regions->consensus_count * sizeof(bolr_index));
    return BOLR_OK;
}

bolr_index bolr_region_set_consensus_count(const bolr_region_set *opaque) {
    const struct bolr_region_set *regions = opaque;
    return (regions == NULL) ? -1 : regions->consensus_count;
}

int bolr_region_set_empty_consensus(const bolr_region_set *opaque) {
    const struct bolr_region_set *regions = opaque;
    return (regions == NULL) ? 0 : regions->empty_consensus;
}

bolr_status bolr_region_set_summary(const bolr_region_set *opaque, bolr_index region_index, bolr_region_summary *out_summary) {
    const struct bolr_region_set *regions = opaque;
    if ((regions == NULL) || (out_summary == NULL) || (region_index < 0) || (region_index >= regions->region_count)) return BOLR_INVALID_ARGUMENT;
    *out_summary = regions->summaries[region_index];
    return BOLR_OK;
}

bolr_status bolr_region_set_copy_region_candidates(const bolr_region_set *opaque, bolr_index region_index, bolr_index *output, bolr_index capacity) {
    const struct bolr_region_set *regions = opaque;
    bolr_region_summary summary;
    if ((regions == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    if (bolr_region_set_summary(regions, region_index, &summary) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    if (capacity < summary.candidate_count) return BOLR_INVALID_ARGUMENT;
    memcpy(output, regions->region_candidates + summary.candidate_offset, (size_t) summary.candidate_count * sizeof(bolr_index));
    return BOLR_OK;
}
