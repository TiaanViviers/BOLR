#include "bolr/math.h"

#include <math.h>
#include <stddef.h>

static int is_finite_vector(bolr_const_vector_view input) {
    bolr_index i;
    for (i = 0; i < input.length; ++i) {
        if (!isfinite(input.data[i * input.stride])) return 0;
    }
    return 1;
}

bolr_status bolr_logsumexp(bolr_const_vector_view input, bolr_real *out) {
    bolr_status status;
    bolr_real max_value;
    bolr_real total;
    bolr_index i;
    if (out == NULL) return BOLR_INVALID_ARGUMENT;
    status = bolr_vector_view_validate(input);
    if (status != BOLR_OK) return status;
    if (input.length <= 0) return BOLR_INVALID_SHAPE;
    if (!is_finite_vector(input)) return BOLR_NONFINITE_INPUT;
    max_value = input.data[0];
    for (i = 1; i < input.length; ++i) {
        bolr_real value = input.data[i * input.stride];
        if (value > max_value) max_value = value;
    }
    total = 0.0;
    for (i = 0; i < input.length; ++i) total += exp(input.data[i * input.stride] - max_value);
    *out = max_value + log(total);
    return BOLR_OK;
}

bolr_status bolr_softmax(bolr_const_vector_view input, bolr_vector_view output) {
    bolr_real log_norm;
    bolr_status status = bolr_logsumexp(input, &log_norm);
    bolr_index i;
    if (status != BOLR_OK) return status;
    status = bolr_mutable_vector_view_validate(output);
    if (status != BOLR_OK) return status;
    if (input.length != output.length) return BOLR_INVALID_SHAPE;
    for (i = 0; i < input.length; ++i) output.data[i * output.stride] = exp(input.data[i * input.stride] - log_norm);
    return BOLR_OK;
}

bolr_status bolr_log_softmax(bolr_const_vector_view input, bolr_vector_view output) {
    bolr_real log_norm;
    bolr_status status = bolr_logsumexp(input, &log_norm);
    bolr_index i;
    if (status != BOLR_OK) return status;
    status = bolr_mutable_vector_view_validate(output);
    if (status != BOLR_OK) return status;
    if (input.length != output.length) return BOLR_INVALID_SHAPE;
    for (i = 0; i < input.length; ++i) output.data[i * output.stride] = input.data[i * input.stride] - log_norm;
    return BOLR_OK;
}

bolr_status bolr_sigmoid(bolr_real input, bolr_real *out) {
    if ((out == NULL) || !isfinite(input)) return BOLR_INVALID_ARGUMENT;
    if (input >= 0.0) {
        bolr_real z = exp(-input);
        *out = 1.0 / (1.0 + z);
    } else {
        bolr_real z = exp(input);
        *out = z / (1.0 + z);
    }
    return BOLR_OK;
}

bolr_status bolr_softplus(bolr_real input, bolr_real *out) {
    if ((out == NULL) || !isfinite(input)) return BOLR_INVALID_ARGUMENT;
    if (input > 0.0) *out = input + log1p(exp(-input));
    else *out = log1p(exp(input));
    return BOLR_OK;
}

bolr_status bolr_normal_cdf(bolr_real input, bolr_real *out) {
    if ((out == NULL) || !isfinite(input)) return BOLR_INVALID_ARGUMENT;
    *out = 0.5 * (1.0 + erf(input / sqrt(2.0)));
    return BOLR_OK;
}
