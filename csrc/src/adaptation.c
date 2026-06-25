#include "bolr/adaptation.h"

#include "bolr/attribution.h"
#include "bolr/linalg.h"
#include "bolr/math.h"
#include "bolr/reset.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct bolr_adaptive_policy {
    const bolr_allocator *allocator;
    bolr_state_layout *layout;
    bolr_real *base_process_noise;
    bolr_adaptive_policy_config config;
    bolr_adaptive_block_config *block_configs;
    bolr_index block_count;
    uint64_t layout_hash;
    uint64_t configuration_hash;
};

struct bolr_adaptive_state {
    const bolr_allocator *allocator;
    uint32_t schema_version;
    uint64_t configuration_hash;
    uint64_t layout_hash;
    uint64_t step_index;
    bolr_real *block_multipliers;
    bolr_real *block_discounts;
    int64_t *days_since_reset;
    int32_t *pending_reset;
    bolr_real *pending_reset_strength;
    bolr_standardizer_state standardizer;
    bolr_bocpd_state *detector;
    bolr_real last_raw_surprise;
    int last_raw_surprise_present;
    bolr_real last_standardised_surprise;
    int last_standardised_surprise_present;
};

typedef struct {
    uint32_t schema_version;
    uint32_t reserved;
    uint64_t configuration_hash;
    uint64_t layout_hash;
    uint64_t step_index;
    bolr_standardizer_state standardizer;
    uint32_t detector_payload_size;
    uint32_t block_count;
} bolr_adaptive_state_wire_header;

static uint64_t fnv1a_update(uint64_t state, const unsigned char *data, size_t size) {
    size_t i;
    for (i = 0; i < size; ++i) {
        state ^= (uint64_t) data[i];
        state *= 1099511628211ULL;
    }
    return state;
}

static bolr_status alloc_real_array(const bolr_allocator *allocator, bolr_index count, bolr_real **out) {
    bolr_real *data = (bolr_real *) bolr_allocator_calloc(allocator, (size_t) count, sizeof(bolr_real));
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    *out = data;
    return BOLR_OK;
}

static bolr_status alloc_i64_array(const bolr_allocator *allocator, bolr_index count, int64_t **out) {
    int64_t *data = (int64_t *) bolr_allocator_calloc(allocator, (size_t) count, sizeof(int64_t));
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    *out = data;
    return BOLR_OK;
}

static bolr_status alloc_i32_array(const bolr_allocator *allocator, bolr_index count, int32_t **out) {
    int32_t *data = (int32_t *) bolr_allocator_calloc(allocator, (size_t) count, sizeof(int32_t));
    if (data == NULL) return BOLR_ALLOCATION_FAILED;
    *out = data;
    return BOLR_OK;
}

static bolr_status clone_layout(const bolr_state_layout *layout, const bolr_allocator *allocator, bolr_state_layout **out_layout) {
    bolr_index count;
    bolr_index i;
    bolr_state_block_spec *specs;
    bolr_status status;
    if ((layout == NULL) || (out_layout == NULL)) return BOLR_INVALID_ARGUMENT;
    count = bolr_state_layout_block_count(layout);
    specs = (bolr_state_block_spec *) malloc((size_t) count * sizeof(*specs));
    if (specs == NULL) return BOLR_ALLOCATION_FAILED;
    for (i = 0; i < count; ++i) {
        status = bolr_state_layout_block_spec(layout, i, &specs[i]);
        if (status != BOLR_OK) { free(specs); return status; }
    }
    status = bolr_state_layout_create(specs, count, allocator, out_layout);
    free(specs);
    return status;
}

static uint64_t configuration_hash(const bolr_state_layout *layout, bolr_const_matrix_view base_process_noise, const bolr_adaptive_policy_config *config, const bolr_adaptive_block_config *blocks, bolr_index block_count) {
    uint64_t h = 1469598103934665603ULL;
    bolr_index i;
    h = fnv1a_update(h, (const unsigned char *) &config->surprise_mode, sizeof(config->surprise_mode));
    h = fnv1a_update(h, (const unsigned char *) &config->standardizer, sizeof(config->standardizer));
    h = fnv1a_update(h, (const unsigned char *) &config->detector, sizeof(config->detector));
    h = fnv1a_update(h, (const unsigned char *) &config->activation_beta, sizeof(config->activation_beta));
    h = fnv1a_update(h, (const unsigned char *) &config->activation_z0, sizeof(config->activation_z0));
    h = fnv1a_update(h, (const unsigned char *) &config->attribution_epsilon, sizeof(config->attribution_epsilon));
    h = fnv1a_update(h, (const unsigned char *) &block_count, sizeof(block_count));
    {
        uint64_t layout_hash = bolr_state_layout_schema_hash(layout);
        h = fnv1a_update(h, (const unsigned char *) &layout_hash, sizeof(layout_hash));
    }
    for (i = 0; i < base_process_noise.rows; ++i) {
        h = fnv1a_update(h, (const unsigned char *) (base_process_noise.data + i * base_process_noise.row_stride), (size_t) base_process_noise.cols * sizeof(bolr_real));
    }
    for (i = 0; i < block_count; ++i) {
        h = fnv1a_update(h, (const unsigned char *) blocks[i].block_name, strlen(blocks[i].block_name));
        h = fnv1a_update(h, (const unsigned char *) &blocks[i], sizeof(blocks[i]));
    }
    return h;
}

static bolr_real activation_value(bolr_real change_probability, bolr_real z_score, const bolr_adaptive_policy_config *config) {
    bolr_real sigmoid;
    bolr_real out;
    if (bolr_sigmoid(config->activation_beta * (z_score - config->activation_z0), &sigmoid) != BOLR_OK) return change_probability;
    out = (change_probability > sigmoid) ? change_probability : sigmoid;
    if (out < 0.0) out = 0.0;
    if (out > 1.0) out = 1.0;
    return out;
}

static void fill_predict_diags(const bolr_adaptive_policy *policy, const bolr_adaptive_state *state, const bolr_real *active_q, const int32_t *applied_reset, bolr_adaptation_diagnostics *diagnostics) {
    bolr_index b;
    if (diagnostics == NULL) return;
    diagnostics->block_count = policy->block_count;
    diagnostics->activation_value = 0.0;
    diagnostics->change_probability = 0.0;
    diagnostics->informative = 0;
    for (b = 0; b < policy->block_count; ++b) {
        bolr_state_block_spec spec;
        bolr_real trace = 0.0;
        bolr_index i;
        bolr_state_layout_block_spec(policy->layout, b, &spec);
        for (i = spec.start; i < spec.stop; ++i) trace += active_q[i * bolr_state_layout_total_dimension(policy->layout) + i];
        if (diagnostics->process_noise_multiplier != NULL) diagnostics->process_noise_multiplier[b] = state->block_multipliers[b];
        if (diagnostics->active_discount != NULL) diagnostics->active_discount[b] = state->block_discounts[b];
        if (diagnostics->target_multiplier != NULL) diagnostics->target_multiplier[b] = state->block_multipliers[b];
        if (diagnostics->reset_applied != NULL) diagnostics->reset_applied[b] = applied_reset[b];
        if (diagnostics->reset_scheduled != NULL) diagnostics->reset_scheduled[b] = 0;
        if (diagnostics->reset_strength != NULL) diagnostics->reset_strength[b] = state->pending_reset_strength[b];
        if (diagnostics->days_since_reset != NULL) diagnostics->days_since_reset[b] = state->days_since_reset[b];
        if (diagnostics->euclidean_update_energy != NULL) diagnostics->euclidean_update_energy[b] = trace;
        if (diagnostics->mahalanobis_update_energy != NULL) diagnostics->mahalanobis_update_energy[b] = 0.0;
        if (diagnostics->attribution_weight != NULL) diagnostics->attribution_weight[b] = 0.0;
    }
}

bolr_status bolr_adaptive_policy_create(
    const bolr_state_layout *layout,
    bolr_const_matrix_view base_process_noise,
    const bolr_adaptive_policy_config *config,
    const bolr_adaptive_block_config *block_configs,
    bolr_index block_config_count,
    const bolr_allocator *allocator,
    bolr_adaptive_policy **out_policy
) {
    struct bolr_adaptive_policy *policy;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index block_count;
    bolr_index r, c;
    if ((layout == NULL) || (config == NULL) || (block_configs == NULL) || (out_policy == NULL)) return BOLR_INVALID_ARGUMENT;
    block_count = bolr_state_layout_block_count(layout);
    if ((base_process_noise.rows != base_process_noise.cols) || (base_process_noise.rows != bolr_state_layout_total_dimension(layout))) return BOLR_INVALID_SHAPE;
    if (block_config_count != block_count) return BOLR_INVALID_SHAPE;
    *out_policy = NULL;
    policy = (struct bolr_adaptive_policy *) bolr_allocator_calloc(active, 1U, sizeof(*policy));
    if (policy == NULL) return BOLR_ALLOCATION_FAILED;
    policy->allocator = active;
    policy->config = *config;
    policy->block_count = block_count;
    if (clone_layout(layout, active, &policy->layout) != BOLR_OK) { bolr_adaptive_policy_destroy(policy); return BOLR_ALLOCATION_FAILED; }
    policy->base_process_noise = (bolr_real *) bolr_allocator_malloc(active, (size_t) (base_process_noise.rows * base_process_noise.cols) * sizeof(bolr_real));
    policy->block_configs = (bolr_adaptive_block_config *) bolr_allocator_calloc(active, (size_t) block_count, sizeof(*policy->block_configs));
    if ((policy->base_process_noise == NULL) || (policy->block_configs == NULL)) { bolr_adaptive_policy_destroy(policy); return BOLR_ALLOCATION_FAILED; }
    for (r = 0; r < base_process_noise.rows; ++r) {
        for (c = 0; c < base_process_noise.cols; ++c) {
            policy->base_process_noise[r * base_process_noise.cols + c] = base_process_noise.data[r * base_process_noise.row_stride + c * base_process_noise.col_stride];
        }
    }
    memcpy(policy->block_configs, block_configs, (size_t) block_count * sizeof(*policy->block_configs));
    policy->layout_hash = bolr_state_layout_schema_hash(layout);
    policy->configuration_hash = configuration_hash(layout, base_process_noise, config, block_configs, block_count);
    *out_policy = policy;
    return BOLR_OK;
}

void bolr_adaptive_policy_destroy(bolr_adaptive_policy *opaque) {
    struct bolr_adaptive_policy *policy = opaque;
    if (policy == NULL) return;
    bolr_state_layout_destroy(policy->layout);
    bolr_allocator_free(policy->allocator, policy->base_process_noise);
    bolr_allocator_free(policy->allocator, policy->block_configs);
    bolr_allocator_free(policy->allocator, policy);
}

uint64_t bolr_adaptive_policy_configuration_hash(const bolr_adaptive_policy *opaque) {
    const struct bolr_adaptive_policy *policy = opaque;
    return (policy == NULL) ? 0ULL : policy->configuration_hash;
}

bolr_index bolr_adaptive_policy_block_count(const bolr_adaptive_policy *opaque) {
    const struct bolr_adaptive_policy *policy = opaque;
    return (policy == NULL) ? -1 : policy->block_count;
}

bolr_status bolr_adaptive_state_create(const bolr_adaptive_policy *policy_opaque, const bolr_allocator *allocator, bolr_adaptive_state **out_state) {
    const struct bolr_adaptive_policy *policy = policy_opaque;
    struct bolr_adaptive_state *state;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index b;
    if ((policy == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    state = (struct bolr_adaptive_state *) bolr_allocator_calloc(active, 1U, sizeof(*state));
    if (state == NULL) return BOLR_ALLOCATION_FAILED;
    state->allocator = active;
    state->schema_version = 1U;
    state->configuration_hash = policy->configuration_hash;
    state->layout_hash = policy->layout_hash;
    if ((alloc_real_array(active, policy->block_count, &state->block_multipliers) != BOLR_OK) ||
        (alloc_real_array(active, policy->block_count, &state->block_discounts) != BOLR_OK) ||
        (alloc_i64_array(active, policy->block_count, &state->days_since_reset) != BOLR_OK) ||
        (alloc_i32_array(active, policy->block_count, &state->pending_reset) != BOLR_OK) ||
        (alloc_real_array(active, policy->block_count, &state->pending_reset_strength) != BOLR_OK)) {
        bolr_adaptive_state_destroy(state);
        return BOLR_ALLOCATION_FAILED;
    }
    for (b = 0; b < policy->block_count; ++b) {
        state->block_multipliers[b] = 1.0;
        state->block_discounts[b] = 1.0;
        state->days_since_reset[b] = 1000000000LL;
        state->pending_reset[b] = 0;
        state->pending_reset_strength[b] = 0.0;
    }
    bolr_standardizer_state_init(&policy->config.standardizer, &state->standardizer);
    if (bolr_bocpd_state_create(&policy->config.detector, active, &state->detector) != BOLR_OK) {
        bolr_adaptive_state_destroy(state);
        return BOLR_ALLOCATION_FAILED;
    }
    *out_state = state;
    return BOLR_OK;
}

void bolr_adaptive_state_destroy(bolr_adaptive_state *opaque) {
    struct bolr_adaptive_state *state = opaque;
    if (state == NULL) return;
    bolr_bocpd_state_destroy(state->detector);
    bolr_allocator_free(state->allocator, state->block_multipliers);
    bolr_allocator_free(state->allocator, state->block_discounts);
    bolr_allocator_free(state->allocator, state->days_since_reset);
    bolr_allocator_free(state->allocator, state->pending_reset);
    bolr_allocator_free(state->allocator, state->pending_reset_strength);
    bolr_allocator_free(state->allocator, state);
}

uint64_t bolr_adaptive_state_step_index(const bolr_adaptive_state *opaque) {
    const struct bolr_adaptive_state *state = opaque;
    return (state == NULL) ? 0ULL : state->step_index;
}

bolr_status bolr_adaptive_state_copy_block_multipliers(const bolr_adaptive_state *opaque, bolr_vector_view output) {
    const struct bolr_adaptive_state *state = opaque;
    bolr_index i;
    if (state == NULL) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < output.length; ++i) output.data[i * output.stride] = state->block_multipliers[i];
    return BOLR_OK;
}

bolr_status bolr_adaptive_state_copy_block_discounts(const bolr_adaptive_state *opaque, bolr_vector_view output) {
    const struct bolr_adaptive_state *state = opaque;
    bolr_index i;
    if (state == NULL) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < output.length; ++i) output.data[i * output.stride] = state->block_discounts[i];
    return BOLR_OK;
}

bolr_status bolr_adaptive_state_copy_run_length_posterior(const bolr_adaptive_state *opaque, bolr_vector_view output) {
    const struct bolr_adaptive_state *state = opaque;
    if (state == NULL) return BOLR_INVALID_ARGUMENT;
    return bolr_bocpd_copy_run_length_posterior(state->detector, output);
}

bolr_status bolr_adaptive_policy_predict(
    const bolr_adaptive_policy *policy_opaque,
    bolr_adaptive_state *state_opaque,
    const bolr_gaussian_state *posterior,
    bolr_workspace *workspace,
    bolr_gaussian_state **out_predictive,
    bolr_adaptation_diagnostics *diagnostics
) {
    const struct bolr_adaptive_policy *policy = policy_opaque;
    struct bolr_adaptive_state *state = state_opaque;
    bolr_index dim;
    bolr_real *mean = NULL;
    bolr_real *cov = NULL;
    bolr_real *active_q = NULL;
    int32_t *applied_reset = NULL;
    bolr_index r, c, b;
    (void) workspace;
    if ((policy == NULL) || (state == NULL) || (posterior == NULL) || (out_predictive == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((state->configuration_hash != policy->configuration_hash) || (state->layout_hash != policy->layout_hash) || (posterior->state_layout_hash != policy->layout_hash)) return BOLR_SCHEMA_MISMATCH;
    *out_predictive = NULL;
    dim = posterior->dimension;
    mean = (bolr_real *) malloc((size_t) dim * sizeof(bolr_real));
    cov = (bolr_real *) malloc((size_t) (dim * dim) * sizeof(bolr_real));
    active_q = (bolr_real *) calloc((size_t) (dim * dim), sizeof(bolr_real));
    applied_reset = (int32_t *) calloc((size_t) policy->block_count, sizeof(int32_t));
    if ((mean == NULL) || (cov == NULL) || (active_q == NULL) || (applied_reset == NULL)) { free(mean); free(cov); free(active_q); free(applied_reset); return BOLR_ALLOCATION_FAILED; }
    memcpy(mean, posterior->mean, (size_t) dim * sizeof(bolr_real));
    memcpy(cov, posterior->covariance, (size_t) (dim * dim) * sizeof(bolr_real));
    for (b = 0; b < policy->block_count; ++b) {
        bolr_state_block_spec spec;
        bolr_state_layout_block_spec(policy->layout, b, &spec);
        if (state->pending_reset[b]) {
            bolr_real *anchor_mean = (bolr_real *) calloc((size_t) (spec.stop - spec.start), sizeof(bolr_real));
            bolr_real *anchor_cov = (bolr_real *) calloc((size_t) ((spec.stop - spec.start) * (spec.stop - spec.start)), sizeof(bolr_real));
            if ((anchor_mean == NULL) || (anchor_cov == NULL)) { free(mean); free(cov); free(active_q); free(applied_reset); free(anchor_mean); free(anchor_cov); return BOLR_ALLOCATION_FAILED; }
            for (r = 0; r < spec.stop - spec.start; ++r) anchor_cov[r * (spec.stop - spec.start) + r] = cov[(spec.start + r) * dim + (spec.start + r)];
            if (bolr_apply_partial_reset(
                (bolr_const_vector_view){mean, dim, 1},
                (bolr_const_matrix_view){cov, dim, dim, dim, 1},
                spec.start,
                spec.stop,
                state->pending_reset_strength[b],
                (bolr_const_vector_view){anchor_mean, spec.stop - spec.start, 1},
                (bolr_const_matrix_view){anchor_cov, spec.stop - spec.start, spec.stop - spec.start, spec.stop - spec.start, 1},
                (bolr_vector_view){mean, dim, 1},
                (bolr_matrix_view){cov, dim, dim, dim, 1}
            ) != BOLR_OK) { free(mean); free(cov); free(active_q); free(applied_reset); free(anchor_mean); free(anchor_cov); return BOLR_NUMERICAL_FAILURE; }
            applied_reset[b] = 1;
            free(anchor_mean);
            free(anchor_cov);
        }
        for (r = spec.start; r < spec.stop; ++r) {
            for (c = spec.start; c < spec.stop; ++c) {
                active_q[r * dim + c] = state->block_multipliers[b] * policy->base_process_noise[r * dim + c];
                cov[r * dim + c] += active_q[r * dim + c];
            }
        }
    }
    if (bolr_gaussian_state_create(
        (bolr_const_vector_view){mean, dim, 1},
        (bolr_const_matrix_view){cov, dim, dim, dim, 1},
        posterior->state_layout_hash,
        posterior->model_schema_hash,
        policy->allocator,
        out_predictive
    ) != BOLR_OK) { free(mean); free(cov); free(active_q); free(applied_reset); return BOLR_NOT_POSITIVE_DEFINITE; }
    (*out_predictive)->step_index = posterior->step_index + 1U;
    for (b = 0; b < policy->block_count; ++b) state->pending_reset[b] = 0;
    fill_predict_diags(policy, state, active_q, applied_reset, diagnostics);
    free(mean);
    free(cov);
    free(active_q);
    free(applied_reset);
    return BOLR_OK;
}

bolr_status bolr_adaptive_policy_observe(
    const bolr_adaptive_policy *policy_opaque,
    bolr_adaptive_state *state_opaque,
    const bolr_gaussian_state *predictive,
    const bolr_gaussian_state *posterior,
    const bolr_surprise_input *surprise_input,
    bolr_adaptation_diagnostics *diagnostics
) {
    const struct bolr_adaptive_policy *policy = policy_opaque;
    struct bolr_adaptive_state *state = state_opaque;
    bolr_index block_count;
    bolr_real *euclidean = NULL;
    bolr_real *mahal = NULL;
    bolr_real *attrib = NULL;
    bolr_real surprise_value = 0.0;
    bolr_real raw_surprise = 0.0;
    bolr_real normalized_surprise = 0.0;
    bolr_real information_normalized = 0.0;
    int surprise_present = 0;
    bolr_standardizer_diagnostics std_diag;
    bolr_bocpd_diagnostics detector_diag;
    bolr_real z_score = 0.0;
    bolr_real activation;
    bolr_index b;
    if ((policy == NULL) || (state == NULL) || (predictive == NULL) || (posterior == NULL) || (surprise_input == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((state->configuration_hash != policy->configuration_hash) || (predictive->state_layout_hash != policy->layout_hash) || (posterior->state_layout_hash != policy->layout_hash)) return BOLR_SCHEMA_MISMATCH;
    block_count = policy->block_count;
    euclidean = (bolr_real *) malloc((size_t) block_count * sizeof(bolr_real));
    mahal = (bolr_real *) malloc((size_t) block_count * sizeof(bolr_real));
    attrib = (bolr_real *) malloc((size_t) block_count * sizeof(bolr_real));
    if ((euclidean == NULL) || (mahal == NULL) || (attrib == NULL)) { free(euclidean); free(mahal); free(attrib); return BOLR_ALLOCATION_FAILED; }
    if (bolr_block_innovation_attribution(
        policy->layout,
        (bolr_const_vector_view){predictive->mean, predictive->dimension, 1},
        (bolr_const_matrix_view){predictive->covariance, predictive->dimension, predictive->dimension, predictive->dimension, 1},
        (bolr_const_vector_view){posterior->mean, posterior->dimension, 1},
        policy->config.attribution_epsilon,
        (bolr_vector_view){euclidean, block_count, 1},
        (bolr_vector_view){mahal, block_count, 1},
        (bolr_vector_view){attrib, block_count, 1}
    ) != BOLR_OK) { free(euclidean); free(mahal); free(attrib); return BOLR_NUMERICAL_FAILURE; }
    if (bolr_surprise_compute(policy->config.surprise_mode, surprise_input, &surprise_value, &raw_surprise, &normalized_surprise, &information_normalized, &surprise_present) != BOLR_OK) {
        free(euclidean); free(mahal); free(attrib); return BOLR_NUMERICAL_FAILURE;
    }
    if (bolr_standardizer_step(&policy->config.standardizer, &state->standardizer, surprise_value, surprise_present, &std_diag) != BOLR_OK) {
        free(euclidean); free(mahal); free(attrib); return BOLR_NUMERICAL_FAILURE;
    }
    if (bolr_bocpd_step(state->detector, surprise_value, surprise_present, &detector_diag) != BOLR_OK) {
        free(euclidean); free(mahal); free(attrib); return BOLR_NUMERICAL_FAILURE;
    }
    z_score = std_diag.z_score_present ? std_diag.z_score : 0.0;
    activation = activation_value(detector_diag.change_probability, z_score, &policy->config);
    for (b = 0; b < block_count; ++b) {
        const bolr_adaptive_block_config *cfg = &policy->block_configs[b];
        bolr_real attrib_weight = attrib[b];
        if (attrib_weight < cfg->attribution_floor) attrib_weight = cfg->attribution_floor;
        if (cfg->transition_family == BOLR_ADAPTIVE_BLOCK_DISCOUNT) {
            bolr_real minimum_discount = cfg->minimum_discount_present ? cfg->minimum_discount : 1.0;
            bolr_real target_discount = 1.0 - activation * attrib_weight * (1.0 - minimum_discount);
            bolr_real updated = (1.0 - cfg->decay) * target_discount + cfg->decay * state->block_discounts[b];
            if (updated < minimum_discount) updated = minimum_discount;
            if (updated > 1.0) updated = 1.0;
            state->block_discounts[b] = updated;
            state->block_multipliers[b] = 1.0;
        } else if (((cfg->transition_family == BOLR_ADAPTIVE_BLOCK_ADDITIVE) || (cfg->transition_family == BOLR_ADAPTIVE_BLOCK_ZERO_NOISE)) && cfg->adaptive_enabled) {
            bolr_real target = 1.0 + cfg->amplitude * attrib_weight * activation;
            bolr_real decayed = 1.0 + cfg->decay * (state->block_multipliers[b] - 1.0) + (1.0 - cfg->decay) * (target - 1.0);
            if (decayed < cfg->minimum_multiplier) decayed = cfg->minimum_multiplier;
            if (decayed > cfg->maximum_multiplier) decayed = cfg->maximum_multiplier;
            state->block_multipliers[b] = decayed;
        }
        if (cfg->reset_enabled && cfg->reset_threshold_present && cfg->reset_strength_present) {
            if ((activation >= cfg->reset_threshold) && (state->days_since_reset[b] >= (int64_t) cfg->reset_cooldown)) {
                state->pending_reset[b] = 1;
                state->pending_reset_strength[b] = cfg->reset_strength;
                state->days_since_reset[b] = 0;
            }
        }
        state->days_since_reset[b] += state->pending_reset[b] ? 0 : 1;
    }
    state->step_index += 1U;
    state->last_raw_surprise = raw_surprise;
    state->last_raw_surprise_present = surprise_present;
    state->last_standardised_surprise = z_score;
    state->last_standardised_surprise_present = std_diag.z_score_present;
    if (diagnostics != NULL) {
        diagnostics->raw_surprise = raw_surprise;
        diagnostics->normalized_surprise = normalized_surprise;
        diagnostics->information_normalized_surprise = information_normalized;
        diagnostics->standardised_surprise = z_score;
        diagnostics->standardizer_mean_before = std_diag.mean_before;
        diagnostics->standardizer_scale_before = std_diag.scale_before;
        diagnostics->change_probability = detector_diag.change_probability;
        diagnostics->map_run_length = detector_diag.map_run_length;
        diagnostics->expected_run_length = detector_diag.expected_run_length;
        diagnostics->run_length_entropy = detector_diag.run_length_entropy;
        diagnostics->predictive_log_density = detector_diag.predictive_log_density;
        diagnostics->truncation_mass = detector_diag.truncation_mass;
        diagnostics->activation_value = activation;
        diagnostics->informative = surprise_present;
        diagnostics->predictive_log_density_present = detector_diag.predictive_log_density_present;
        diagnostics->block_count = block_count;
        for (b = 0; b < block_count; ++b) {
            if (diagnostics->euclidean_update_energy != NULL) diagnostics->euclidean_update_energy[b] = euclidean[b];
            if (diagnostics->mahalanobis_update_energy != NULL) diagnostics->mahalanobis_update_energy[b] = mahal[b];
            if (diagnostics->attribution_weight != NULL) diagnostics->attribution_weight[b] = attrib[b];
            if (diagnostics->process_noise_multiplier != NULL) diagnostics->process_noise_multiplier[b] = state->block_multipliers[b];
            if (diagnostics->target_multiplier != NULL) diagnostics->target_multiplier[b] = 1.0 + policy->block_configs[b].amplitude * attrib[b] * activation;
            if (diagnostics->active_discount != NULL) diagnostics->active_discount[b] = state->block_discounts[b];
            if (diagnostics->reset_strength != NULL) diagnostics->reset_strength[b] = state->pending_reset_strength[b];
            if (diagnostics->reset_scheduled != NULL) diagnostics->reset_scheduled[b] = state->pending_reset[b];
            if (diagnostics->reset_applied != NULL) diagnostics->reset_applied[b] = 0;
            if (diagnostics->days_since_reset != NULL) diagnostics->days_since_reset[b] = state->days_since_reset[b];
        }
    }
    free(euclidean);
    free(mahal);
    free(attrib);
    return BOLR_OK;
}

bolr_status bolr_adaptive_state_encoded_size(const bolr_adaptive_policy *policy_opaque, const bolr_adaptive_state *state_opaque, size_t *out_size) {
    const struct bolr_adaptive_policy *policy = policy_opaque;
    const struct bolr_adaptive_state *state = state_opaque;
    size_t detector_size = 0U;
    if ((policy == NULL) || (state == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    if (bolr_bocpd_encoded_size(state->detector, &detector_size) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    *out_size = sizeof(bolr_adaptive_state_wire_header)
        + (size_t) policy->block_count * sizeof(bolr_real) * 2U
        + (size_t) policy->block_count * sizeof(int64_t)
        + (size_t) policy->block_count * sizeof(int32_t)
        + (size_t) policy->block_count * sizeof(bolr_real)
        + detector_size
        + sizeof(bolr_real) * 2U
        + sizeof(int32_t) * 2U;
    return BOLR_OK;
}

bolr_status bolr_adaptive_state_encode(const bolr_adaptive_policy *policy_opaque, const bolr_adaptive_state *state_opaque, void *output, size_t output_size, size_t *out_written) {
    const struct bolr_adaptive_policy *policy = policy_opaque;
    const struct bolr_adaptive_state *state = state_opaque;
    bolr_adaptive_state_wire_header header;
    size_t needed;
    size_t detector_written = 0U;
    unsigned char *cursor;
    if (out_written != NULL) *out_written = 0U;
    if ((policy == NULL) || (state == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    if (bolr_adaptive_state_encoded_size(policy, state, &needed) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    if (output_size < needed) return BOLR_INVALID_SHAPE;
    memset(&header, 0, sizeof(header));
    header.schema_version = state->schema_version;
    header.configuration_hash = state->configuration_hash;
    header.layout_hash = state->layout_hash;
    header.step_index = state->step_index;
    header.standardizer = state->standardizer;
    header.block_count = (uint32_t) policy->block_count;
    bolr_bocpd_encoded_size(state->detector, &needed);
    header.detector_payload_size = (uint32_t) needed;
    memcpy(output, &header, sizeof(header));
    cursor = (unsigned char *) output + sizeof(header);
    memcpy(cursor, state->block_multipliers, (size_t) policy->block_count * sizeof(bolr_real)); cursor += (size_t) policy->block_count * sizeof(bolr_real);
    memcpy(cursor, state->block_discounts, (size_t) policy->block_count * sizeof(bolr_real)); cursor += (size_t) policy->block_count * sizeof(bolr_real);
    memcpy(cursor, state->days_since_reset, (size_t) policy->block_count * sizeof(int64_t)); cursor += (size_t) policy->block_count * sizeof(int64_t);
    memcpy(cursor, state->pending_reset, (size_t) policy->block_count * sizeof(int32_t)); cursor += (size_t) policy->block_count * sizeof(int32_t);
    memcpy(cursor, state->pending_reset_strength, (size_t) policy->block_count * sizeof(bolr_real)); cursor += (size_t) policy->block_count * sizeof(bolr_real);
    memcpy(cursor, &state->last_raw_surprise, sizeof(bolr_real)); cursor += sizeof(bolr_real);
    memcpy(cursor, &state->last_standardised_surprise, sizeof(bolr_real)); cursor += sizeof(bolr_real);
    memcpy(cursor, &state->last_raw_surprise_present, sizeof(int32_t)); cursor += sizeof(int32_t);
    memcpy(cursor, &state->last_standardised_surprise_present, sizeof(int32_t)); cursor += sizeof(int32_t);
    if (bolr_bocpd_encode(state->detector, cursor, output_size - (size_t) (cursor - (unsigned char *) output), &detector_written) != BOLR_OK) return BOLR_INVALID_ARGUMENT;
    if (out_written != NULL) *out_written = sizeof(header)
        + (size_t) policy->block_count * sizeof(bolr_real) * 2U
        + (size_t) policy->block_count * sizeof(int64_t)
        + (size_t) policy->block_count * sizeof(int32_t)
        + (size_t) policy->block_count * sizeof(bolr_real)
        + sizeof(bolr_real) * 2U
        + sizeof(int32_t) * 2U
        + detector_written;
    return BOLR_OK;
}

bolr_status bolr_adaptive_state_decode(const bolr_adaptive_policy *policy_opaque, const void *data, size_t data_size, const bolr_allocator *allocator, bolr_adaptive_state **out_state) {
    const struct bolr_adaptive_policy *policy = policy_opaque;
    bolr_adaptive_state_wire_header header;
    struct bolr_adaptive_state *state;
    const unsigned char *cursor;
    if ((policy == NULL) || (data == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    if (data_size < sizeof(header)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    memcpy(&header, data, sizeof(header));
    if ((header.configuration_hash != policy->configuration_hash) || (header.layout_hash != policy->layout_hash) || (header.block_count != (uint32_t) policy->block_count)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (bolr_adaptive_state_create(policy, allocator, &state) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    state->schema_version = header.schema_version;
    state->step_index = header.step_index;
    state->standardizer = header.standardizer;
    cursor = (const unsigned char *) data + sizeof(header);
    memcpy(state->block_multipliers, cursor, (size_t) policy->block_count * sizeof(bolr_real)); cursor += (size_t) policy->block_count * sizeof(bolr_real);
    memcpy(state->block_discounts, cursor, (size_t) policy->block_count * sizeof(bolr_real)); cursor += (size_t) policy->block_count * sizeof(bolr_real);
    memcpy(state->days_since_reset, cursor, (size_t) policy->block_count * sizeof(int64_t)); cursor += (size_t) policy->block_count * sizeof(int64_t);
    memcpy(state->pending_reset, cursor, (size_t) policy->block_count * sizeof(int32_t)); cursor += (size_t) policy->block_count * sizeof(int32_t);
    memcpy(state->pending_reset_strength, cursor, (size_t) policy->block_count * sizeof(bolr_real)); cursor += (size_t) policy->block_count * sizeof(bolr_real);
    memcpy(&state->last_raw_surprise, cursor, sizeof(bolr_real)); cursor += sizeof(bolr_real);
    memcpy(&state->last_standardised_surprise, cursor, sizeof(bolr_real)); cursor += sizeof(bolr_real);
    memcpy(&state->last_raw_surprise_present, cursor, sizeof(int32_t)); cursor += sizeof(int32_t);
    memcpy(&state->last_standardised_surprise_present, cursor, sizeof(int32_t)); cursor += sizeof(int32_t);
    bolr_bocpd_state_destroy(state->detector);
    if (bolr_bocpd_decode(cursor, header.detector_payload_size, allocator, &state->detector) != BOLR_OK) { bolr_adaptive_state_destroy(state); return BOLR_INCOMPATIBLE_CHECKPOINT; }
    *out_state = state;
    return BOLR_OK;
}
