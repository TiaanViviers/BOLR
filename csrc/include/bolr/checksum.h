#ifndef BOLR_CHECKSUM_H
#define BOLR_CHECKSUM_H

#include <stddef.h>
#include <stdint.h>

uint32_t bolr_crc32(const void *data, size_t length);

#endif
