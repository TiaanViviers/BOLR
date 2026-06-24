#ifndef BOLR_ARRAY_H
#define BOLR_ARRAY_H

#include "bolr/types.h"

typedef struct {
    const bolr_real *data;
    bolr_index length;
    bolr_index stride;
} bolr_const_vector_view;

typedef struct {
    bolr_real *data;
    bolr_index length;
    bolr_index stride;
} bolr_vector_view;

typedef struct {
    const bolr_real *data;
    bolr_index rows;
    bolr_index cols;
    bolr_index row_stride;
    bolr_index col_stride;
} bolr_const_matrix_view;

typedef struct {
    bolr_real *data;
    bolr_index rows;
    bolr_index cols;
    bolr_index row_stride;
    bolr_index col_stride;
} bolr_matrix_view;

bolr_status bolr_vector_view_validate(bolr_const_vector_view view);
bolr_status bolr_mutable_vector_view_validate(bolr_vector_view view);
bolr_status bolr_matrix_view_validate(bolr_const_matrix_view view);
bolr_status bolr_mutable_matrix_view_validate(bolr_matrix_view view);

#endif
