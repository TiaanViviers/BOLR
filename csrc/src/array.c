#include "bolr/array.h"
#include "bolr/status.h"

#include <stddef.h>

static bolr_status validate_length(bolr_index length, bolr_index stride) {
    if (length < 0) return BOLR_INVALID_SHAPE;
    if ((length > 0) && (stride <= 0)) return BOLR_INVALID_SHAPE;
    return BOLR_OK;
}

bolr_status bolr_vector_view_validate(bolr_const_vector_view view) {
    if ((view.length > 0) && (view.data == NULL)) return BOLR_INVALID_ARGUMENT;
    return validate_length(view.length, view.stride);
}

bolr_status bolr_mutable_vector_view_validate(bolr_vector_view view) {
    if ((view.length > 0) && (view.data == NULL)) return BOLR_INVALID_ARGUMENT;
    return validate_length(view.length, view.stride);
}

bolr_status bolr_matrix_view_validate(bolr_const_matrix_view view) {
    if ((view.rows < 0) || (view.cols < 0)) return BOLR_INVALID_SHAPE;
    if ((view.rows > 0) && (view.cols > 0) && (view.data == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((view.rows > 0) && (view.row_stride <= 0)) return BOLR_INVALID_SHAPE;
    if ((view.cols > 0) && (view.col_stride <= 0)) return BOLR_INVALID_SHAPE;
    return BOLR_OK;
}

bolr_status bolr_mutable_matrix_view_validate(bolr_matrix_view view) {
    if ((view.rows < 0) || (view.cols < 0)) return BOLR_INVALID_SHAPE;
    if ((view.rows > 0) && (view.cols > 0) && (view.data == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((view.rows > 0) && (view.row_stride <= 0)) return BOLR_INVALID_SHAPE;
    if ((view.cols > 0) && (view.col_stride <= 0)) return BOLR_INVALID_SHAPE;
    return BOLR_OK;
}
