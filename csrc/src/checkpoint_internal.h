#ifndef BOLR_CHECKPOINT_INTERNAL_H
#define BOLR_CHECKPOINT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/checkpoint_sections.h"
#include "bolr/replay.h"
#include "internal.h"

#define BOLR_CHECKPOINT_HEADER_SIZE 180U
#define BOLR_CHECKPOINT_DIRECTORY_ENTRY_SIZE 44U
#define BOLR_CHECKPOINT_SECTION_SCHEMA_MAJOR 1U
#define BOLR_CHECKPOINT_SECTION_SCHEMA_MINOR 0U

typedef struct {
    uint32_t section_type;
    uint32_t section_flags;
    uint64_t element_count;
    uint64_t payload_offset;
    uint64_t payload_length;
    uint32_t section_crc32;
} bolr_checkpoint_directory_entry;

typedef struct {
    uint32_t section_type;
    uint32_t section_flags;
    uint64_t element_count;
    const unsigned char *payload;
    size_t payload_length;
} bolr_checkpoint_parsed_section;

bolr_status bolr_replay_checkpoint_export_internal(const bolr_replay_engine *engine, const bolr_allocator *allocator, bolr_replay_checkpoint **out_checkpoint);
void bolr_replay_checkpoint_destroy_pending(bolr_replay_checkpoint *checkpoint);

#endif
