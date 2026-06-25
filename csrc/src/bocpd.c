#include "bolr/bocpd.h"

#include "bolr/linalg.h"
#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct bolr_bocpd_state {
    const bolr_allocator *allocator;
    bolr_bocpd_config config;
    uint32_t schema_version;
    uint64_t step_index;
    bolr_real *log_prob;
    bolr_real *mu;
    bolr_real *kappa;
    bolr_real *alpha;
    bolr_real *beta;
    bolr_real *scratch_log;
    bolr_real *scratch_pred;
};

typedef struct {
    uint32_t schema_version;
    uint32_t reserved;
    uint64_t step_index;
    int64_t max_run_length;
    bolr_bocpd_config config;
} bolr_bocpd_wire_header;

static bolr_real neg_inf(void) { return -INFINITY; }

static bolr_status alloc_array(const bolr_allocator *allocator, size_t count, bolr_real **out) {
    size_t bytes = count * sizeof(bolr_real);
    bolr_real *ptr = (bolr_real *) bolr_allocator_calloc(allocator, count, sizeof(bolr_real));
    (void) bytes;
    if (ptr == NULL) return BOLR_ALLOCATION_FAILED;
    *out = ptr;
    return BOLR_OK;
}

static bolr_real logsumexp_slice(const bolr_real *values, bolr_index length) {
    bolr_real max_value = neg_inf();
    bolr_real total = 0.0;
    bolr_index i;
    for (i = 0; i < length; ++i) if (values[i] > max_value) max_value = values[i];
    if (!isfinite(max_value)) return max_value;
    for (i = 0; i < length; ++i) if (isfinite(values[i])) total += exp(values[i] - max_value);
    return max_value + log(total);
}

static void posterior_update(bolr_real x, bolr_real mu, bolr_real kappa, bolr_real alpha, bolr_real beta, bolr_real *out_mu, bolr_real *out_kappa, bolr_real *out_alpha, bolr_real *out_beta) {
    bolr_real kappa_n = kappa + 1.0;
    *out_mu = (kappa * mu + x) / kappa_n;
    *out_kappa = kappa_n;
    *out_alpha = alpha + 0.5;
    *out_beta = beta + 0.5 * kappa * (x - mu) * (x - mu) / kappa_n;
}

static bolr_real student_t_log_pdf(bolr_real x, bolr_real mu, bolr_real kappa, bolr_real alpha, bolr_real beta) {
    bolr_real nu = 2.0 * alpha;
    bolr_real scale2 = beta * (kappa + 1.0) / (alpha * kappa);
    bolr_real y = (x - mu) * (x - mu) / scale2;
    return lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * (log(nu) + log(M_PI) + log(scale2)) - 0.5 * (nu + 1.0) * log1p(y / nu);
}

static void initial_arrays(struct bolr_bocpd_state *state) {
    bolr_index size = state->config.max_run_length + 1;
    bolr_index i;
    for (i = 0; i < size; ++i) {
        state->log_prob[i] = neg_inf();
        state->mu[i] = state->config.prior_mean;
        state->kappa[i] = state->config.prior_kappa;
        state->alpha[i] = state->config.prior_alpha;
        state->beta[i] = state->config.prior_beta;
    }
    state->log_prob[0] = 0.0;
}

static bolr_status fill_diagnostics(struct bolr_bocpd_state *state, bolr_real predictive_log_density, int predictive_present, bolr_real truncation_mass, bolr_bocpd_diagnostics *diagnostics) {
    bolr_index size = state->config.max_run_length + 1;
    bolr_real sum = 0.0;
    bolr_real entropy = 0.0;
    bolr_real expected = 0.0;
    bolr_real cp = 0.0;
    bolr_index i;
    bolr_index argmax = 0;
    bolr_real argmax_value = neg_inf();
    if (diagnostics == NULL) return BOLR_OK;
    for (i = 0; i < size; ++i) {
        bolr_real p = exp(state->log_prob[i]);
        sum += p;
        expected += ((bolr_real) i) * p;
        if ((p > 0.0) && isfinite(p)) entropy -= p * log(p);
        if (state->log_prob[i] > argmax_value) { argmax_value = state->log_prob[i]; argmax = i; }
    }
    cp = exp(state->log_prob[0]);
    diagnostics->change_probability = cp / sum;
    diagnostics->map_run_length = (bolr_real) argmax;
    diagnostics->expected_run_length = expected / sum;
    diagnostics->run_length_entropy = entropy + log(sum);
    diagnostics->predictive_log_density = predictive_log_density;
    diagnostics->truncation_mass = truncation_mass;
    diagnostics->hazard = state->config.hazard;
    diagnostics->informative = 1;
    diagnostics->predictive_log_density_present = predictive_present;
    diagnostics->missing_policy = 0;
    return BOLR_OK;
}

bolr_status bolr_bocpd_state_create(const bolr_bocpd_config *config, const bolr_allocator *allocator, bolr_bocpd_state **out_state) {
    struct bolr_bocpd_state *state;
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    bolr_index size;
    if ((config == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((config->hazard <= 0.0) || (config->hazard >= 1.0) || (config->max_run_length <= 0) || (config->prior_kappa <= 0.0) || (config->prior_alpha <= 0.0) || (config->prior_beta <= 0.0)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    state = (struct bolr_bocpd_state *) bolr_allocator_calloc(active, 1U, sizeof(*state));
    if (state == NULL) return BOLR_ALLOCATION_FAILED;
    state->allocator = active;
    state->config = *config;
    state->schema_version = 1U;
    size = config->max_run_length + 1;
    if ((alloc_array(active, (size_t) size, &state->log_prob) != BOLR_OK) ||
        (alloc_array(active, (size_t) size, &state->mu) != BOLR_OK) ||
        (alloc_array(active, (size_t) size, &state->kappa) != BOLR_OK) ||
        (alloc_array(active, (size_t) size, &state->alpha) != BOLR_OK) ||
        (alloc_array(active, (size_t) size, &state->beta) != BOLR_OK) ||
        (alloc_array(active, (size_t) size, &state->scratch_log) != BOLR_OK) ||
        (alloc_array(active, (size_t) size, &state->scratch_pred) != BOLR_OK)) {
        bolr_bocpd_state_destroy(state);
        return BOLR_ALLOCATION_FAILED;
    }
    initial_arrays(state);
    *out_state = state;
    return BOLR_OK;
}

void bolr_bocpd_state_destroy(bolr_bocpd_state *opaque) {
    struct bolr_bocpd_state *state = opaque;
    if (state == NULL) return;
    bolr_allocator_free(state->allocator, state->log_prob);
    bolr_allocator_free(state->allocator, state->mu);
    bolr_allocator_free(state->allocator, state->kappa);
    bolr_allocator_free(state->allocator, state->alpha);
    bolr_allocator_free(state->allocator, state->beta);
    bolr_allocator_free(state->allocator, state->scratch_log);
    bolr_allocator_free(state->allocator, state->scratch_pred);
    bolr_allocator_free(state->allocator, state);
}

bolr_status bolr_bocpd_step(bolr_bocpd_state *opaque, bolr_real value, int value_present, bolr_bocpd_diagnostics *diagnostics) {
    struct bolr_bocpd_state *state = opaque;
    bolr_index current_max;
    bolr_index size;
    bolr_real hazard;
    bolr_real log_h;
    bolr_real log_1mh;
    bolr_real normalizer;
    bolr_real predictive_log_density = 0.0;
    int predictive_present = 0;
    bolr_real truncation_mass = 0.0;
    bolr_index r;
    if (state == NULL) return BOLR_INVALID_ARGUMENT;
    if (!value_present && state->config.missing_policy == BOLR_BOCPD_MISSING_HOLD) {
        if (diagnostics != NULL) {
            fill_diagnostics(state, 0.0, 0, 0.0, diagnostics);
            diagnostics->informative = 0;
            diagnostics->missing_policy = BOLR_BOCPD_MISSING_HOLD;
        }
        return BOLR_OK;
    }
    current_max = (state->step_index < (uint64_t) state->config.max_run_length) ? (bolr_index) state->step_index : state->config.max_run_length;
    size = state->config.max_run_length + 1;
    hazard = state->config.hazard;
    log_h = log(hazard);
    log_1mh = log(1.0 - hazard);
    for (r = 0; r < size; ++r) state->scratch_log[r] = neg_inf();
    if (value_present) {
        bolr_real *cp_terms = (bolr_real *) malloc((size_t) (current_max + 1) * sizeof(bolr_real));
        if (cp_terms == NULL) return BOLR_ALLOCATION_FAILED;
        for (r = 0; r <= current_max; ++r) {
            state->scratch_pred[r] = student_t_log_pdf(value, state->mu[r], state->kappa[r], state->alpha[r], state->beta[r]);
            cp_terms[r] = state->log_prob[r] + log_h + state->scratch_pred[r];
        }
        state->scratch_log[0] = logsumexp_slice(cp_terms, current_max + 1);
        predictive_log_density = logsumexp_slice(cp_terms, current_max + 1) - log_h;
        predictive_present = 1;
        free(cp_terms);
        for (r = 0; r <= current_max; ++r) {
            bolr_index target = r + 1;
            bolr_real term = state->log_prob[r] + log_1mh + state->scratch_pred[r];
            if (target <= state->config.max_run_length) state->scratch_log[target] = term;
            else truncation_mass += exp(term);
        }
    } else {
        bolr_real *cp_terms = (bolr_real *) malloc((size_t) (current_max + 1) * sizeof(bolr_real));
        if (cp_terms == NULL) return BOLR_ALLOCATION_FAILED;
        for (r = 0; r <= current_max; ++r) cp_terms[r] = state->log_prob[r] + log_h;
        state->scratch_log[0] = logsumexp_slice(cp_terms, current_max + 1);
        free(cp_terms);
        for (r = 0; r <= current_max; ++r) {
            bolr_index target = r + 1;
            bolr_real term = state->log_prob[r] + log_1mh;
            if (target <= state->config.max_run_length) state->scratch_log[target] = term;
            else truncation_mass += exp(term);
        }
    }
    normalizer = logsumexp_slice(state->scratch_log, size);
    for (r = 0; r < size; ++r) state->scratch_log[r] -= normalizer;
    if (truncation_mass > 0.0) truncation_mass = truncation_mass / exp(normalizer);
    if (value_present) {
        bolr_real *new_mu = state->scratch_pred;
        bolr_real *new_kappa = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        bolr_real *new_alpha = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        bolr_real *new_beta = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        if ((new_kappa == NULL) || (new_alpha == NULL) || (new_beta == NULL)) { free(new_kappa); free(new_alpha); free(new_beta); return BOLR_ALLOCATION_FAILED; }
        for (r = 0; r < size; ++r) {
            new_mu[r] = state->config.prior_mean;
            new_kappa[r] = state->config.prior_kappa;
            new_alpha[r] = state->config.prior_alpha;
            new_beta[r] = state->config.prior_beta;
        }
        posterior_update(value, state->config.prior_mean, state->config.prior_kappa, state->config.prior_alpha, state->config.prior_beta, &new_mu[0], &new_kappa[0], &new_alpha[0], &new_beta[0]);
        for (r = 0; r <= current_max; ++r) {
            bolr_index target = r + 1;
            if (target > state->config.max_run_length) continue;
            posterior_update(value, state->mu[r], state->kappa[r], state->alpha[r], state->beta[r], &new_mu[target], &new_kappa[target], &new_alpha[target], &new_beta[target]);
        }
        memcpy(state->mu, new_mu, (size_t) size * sizeof(bolr_real));
        memcpy(state->kappa, new_kappa, (size_t) size * sizeof(bolr_real));
        memcpy(state->alpha, new_alpha, (size_t) size * sizeof(bolr_real));
        memcpy(state->beta, new_beta, (size_t) size * sizeof(bolr_real));
        free(new_kappa);
        free(new_alpha);
        free(new_beta);
    } else {
        bolr_real *new_mu = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        bolr_real *new_kappa = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        bolr_real *new_alpha = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        bolr_real *new_beta = (bolr_real *) malloc((size_t) size * sizeof(bolr_real));
        if ((new_mu == NULL) || (new_kappa == NULL) || (new_alpha == NULL) || (new_beta == NULL)) { free(new_mu); free(new_kappa); free(new_alpha); free(new_beta); return BOLR_ALLOCATION_FAILED; }
        for (r = 0; r < size; ++r) {
            new_mu[r] = state->config.prior_mean;
            new_kappa[r] = state->config.prior_kappa;
            new_alpha[r] = state->config.prior_alpha;
            new_beta[r] = state->config.prior_beta;
        }
        for (r = 0; r <= current_max; ++r) {
            bolr_index target = r + 1;
            if (target > state->config.max_run_length) continue;
            new_mu[target] = state->mu[r];
            new_kappa[target] = state->kappa[r];
            new_alpha[target] = state->alpha[r];
            new_beta[target] = state->beta[r];
        }
        memcpy(state->mu, new_mu, (size_t) size * sizeof(bolr_real));
        memcpy(state->kappa, new_kappa, (size_t) size * sizeof(bolr_real));
        memcpy(state->alpha, new_alpha, (size_t) size * sizeof(bolr_real));
        memcpy(state->beta, new_beta, (size_t) size * sizeof(bolr_real));
        free(new_mu);
        free(new_kappa);
        free(new_alpha);
        free(new_beta);
    }
    memcpy(state->log_prob, state->scratch_log, (size_t) size * sizeof(bolr_real));
    state->step_index += 1U;
    if (diagnostics != NULL) {
        fill_diagnostics(state, predictive_log_density, predictive_present, truncation_mass, diagnostics);
        diagnostics->missing_policy = value_present ? 0 : BOLR_BOCPD_MISSING_HAZARD_ONLY;
    }
    return BOLR_OK;
}

bolr_status bolr_bocpd_copy_run_length_posterior(const bolr_bocpd_state *opaque, bolr_vector_view output) {
    const struct bolr_bocpd_state *state = opaque;
    bolr_index size;
    bolr_index i;
    if (state == NULL) return BOLR_INVALID_ARGUMENT;
    size = state->config.max_run_length + 1;
    if (output.length != size) return BOLR_INVALID_SHAPE;
    for (i = 0; i < size; ++i) output.data[i * output.stride] = exp(state->log_prob[i]);
    return BOLR_OK;
}

bolr_index bolr_bocpd_max_run_length(const bolr_bocpd_state *opaque) {
    const struct bolr_bocpd_state *state = opaque;
    return (state == NULL) ? -1 : state->config.max_run_length;
}

uint64_t bolr_bocpd_step_index(const bolr_bocpd_state *opaque) {
    const struct bolr_bocpd_state *state = opaque;
    return (state == NULL) ? 0ULL : state->step_index;
}

bolr_status bolr_bocpd_encoded_size(const bolr_bocpd_state *opaque, size_t *out_size) {
    const struct bolr_bocpd_state *state = opaque;
    size_t size;
    if ((state == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    size = sizeof(bolr_bocpd_wire_header) + (size_t) (state->config.max_run_length + 1) * 5U * sizeof(bolr_real);
    *out_size = size;
    return BOLR_OK;
}

bolr_status bolr_bocpd_encode(const bolr_bocpd_state *opaque, void *output, size_t output_size, size_t *out_written) {
    const struct bolr_bocpd_state *state = opaque;
    bolr_bocpd_wire_header header;
    size_t needed;
    unsigned char *cursor;
    bolr_index size;
    if (out_written != NULL) *out_written = 0U;
    if ((state == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    bolr_bocpd_encoded_size(state, &needed);
    if (output_size < needed) return BOLR_INVALID_SHAPE;
    memset(&header, 0, sizeof(header));
    header.schema_version = state->schema_version;
    header.step_index = state->step_index;
    header.max_run_length = state->config.max_run_length;
    header.config = state->config;
    memcpy(output, &header, sizeof(header));
    cursor = (unsigned char *) output + sizeof(header);
    size = state->config.max_run_length + 1;
    memcpy(cursor, state->log_prob, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(cursor, state->mu, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(cursor, state->kappa, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(cursor, state->alpha, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(cursor, state->beta, (size_t) size * sizeof(bolr_real));
    if (out_written != NULL) *out_written = needed;
    return BOLR_OK;
}

bolr_status bolr_bocpd_decode(const void *data, size_t data_size, const bolr_allocator *allocator, bolr_bocpd_state **out_state) {
    bolr_bocpd_wire_header header;
    bolr_bocpd_state *state = NULL;
    const unsigned char *cursor;
    bolr_index size;
    if ((data == NULL) || (out_state == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_state = NULL;
    if (data_size < sizeof(header)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    memcpy(&header, data, sizeof(header));
    if (bolr_bocpd_state_create(&header.config, allocator, &state) != BOLR_OK) return BOLR_ALLOCATION_FAILED;
    state->schema_version = header.schema_version;
    state->step_index = header.step_index;
    size = state->config.max_run_length + 1;
    cursor = (const unsigned char *) data + sizeof(header);
    if (data_size < sizeof(header) + (size_t) size * 5U * sizeof(bolr_real)) { bolr_bocpd_state_destroy(state); return BOLR_INCOMPATIBLE_CHECKPOINT; }
    memcpy(state->log_prob, cursor, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(state->mu, cursor, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(state->kappa, cursor, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(state->alpha, cursor, (size_t) size * sizeof(bolr_real)); cursor += (size_t) size * sizeof(bolr_real);
    memcpy(state->beta, cursor, (size_t) size * sizeof(bolr_real));
    *out_state = state;
    return BOLR_OK;
}
