#ifndef BOLR_MATH_H
#define BOLR_MATH_H

#include "bolr/array.h"
#include "bolr/status.h"

bolr_status bolr_logsumexp(bolr_const_vector_view input, bolr_real *out);
bolr_status bolr_softmax(bolr_const_vector_view input, bolr_vector_view output);
bolr_status bolr_log_softmax(bolr_const_vector_view input, bolr_vector_view output);
bolr_status bolr_sigmoid(bolr_real input, bolr_real *out);
bolr_status bolr_softplus(bolr_real input, bolr_real *out);
bolr_status bolr_normal_cdf(bolr_real input, bolr_real *out);

#endif
