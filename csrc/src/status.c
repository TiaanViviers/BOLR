#include "bolr/status.h"

const char *bolr_status_string(bolr_status status) {
    switch (status) {
        case BOLR_OK: return "BOLR_OK";
        case BOLR_INVALID_ARGUMENT: return "BOLR_INVALID_ARGUMENT";
        case BOLR_INVALID_SHAPE: return "BOLR_INVALID_SHAPE";
        case BOLR_NONFINITE_INPUT: return "BOLR_NONFINITE_INPUT";
        case BOLR_ALLOCATION_FAILED: return "BOLR_ALLOCATION_FAILED";
        case BOLR_DIMENSION_OVERFLOW: return "BOLR_DIMENSION_OVERFLOW";
        case BOLR_NOT_POSITIVE_DEFINITE: return "BOLR_NOT_POSITIVE_DEFINITE";
        case BOLR_SINGULAR_MATRIX: return "BOLR_SINGULAR_MATRIX";
        case BOLR_NUMERICAL_FAILURE: return "BOLR_NUMERICAL_FAILURE";
        case BOLR_SCHEMA_MISMATCH: return "BOLR_SCHEMA_MISMATCH";
        case BOLR_VERSION_MISMATCH: return "BOLR_VERSION_MISMATCH";
        case BOLR_INCOMPATIBLE_CHECKPOINT: return "BOLR_INCOMPATIBLE_CHECKPOINT";
        case BOLR_UNSUPPORTED_OPERATION: return "BOLR_UNSUPPORTED_OPERATION";
        case BOLR_ALREADY_CLOSED: return "BOLR_ALREADY_CLOSED";
        default: return "BOLR_UNKNOWN_STATUS";
    }
}
