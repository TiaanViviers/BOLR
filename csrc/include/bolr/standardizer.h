#ifndef BOLR_STANDARDIZER_H
#define BOLR_STANDARDIZER_H

#include "bolr/status.h"
#include "bolr/types.h"

typedef struct {
    bolr_real decay;
    bolr_real variance_floor;
    bolr_index warmup_count;
    bolr_real clip_z;
    int clip_enabled;
} bolr_standardizer_config;

typedef struct {
    uint32_t schema_version;
    uint64_t count;
    bolr_real mean;
    bolr_real variance;
    bolr_real last_z;
    int last_z_present;
} bolr_standardizer_state;

typedef struct {
    bolr_real value;
    bolr_real z_score;
    bolr_real mean_before;
    bolr_real scale_before;
    int missing;
    int z_score_present;
} bolr_standardizer_diagnostics;

void bolr_standardizer_state_init(const bolr_standardizer_config *config, bolr_standardizer_state *state);
bolr_status bolr_standardizer_step(
    const bolr_standardizer_config *config,
    bolr_standardizer_state *state,
    bolr_real value,
    int value_present,
    bolr_standardizer_diagnostics *diagnostics
);

#endif
