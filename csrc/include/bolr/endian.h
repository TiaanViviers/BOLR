#ifndef BOLR_ENDIAN_H
#define BOLR_ENDIAN_H

#include <stddef.h>
#include <stdint.h>

#include "bolr/status.h"

bolr_status bolr_platform_validate(void);

bolr_status bolr_encode_u16_le(void *buf, size_t cap, size_t *cursor, uint16_t v);
bolr_status bolr_encode_u32_le(void *buf, size_t cap, size_t *cursor, uint32_t v);
bolr_status bolr_encode_u64_le(void *buf, size_t cap, size_t *cursor, uint64_t v);
bolr_status bolr_encode_i64_le(void *buf, size_t cap, size_t *cursor, int64_t v);
bolr_status bolr_encode_f64_le(void *buf, size_t cap, size_t *cursor, double v);
bolr_status bolr_encode_bytes(void *buf, size_t cap, size_t *cursor, const void *data, size_t length);

bolr_status bolr_decode_u16_le(const void *buf, size_t cap, size_t *cursor, uint16_t *out);
bolr_status bolr_decode_u32_le(const void *buf, size_t cap, size_t *cursor, uint32_t *out);
bolr_status bolr_decode_u64_le(const void *buf, size_t cap, size_t *cursor, uint64_t *out);
bolr_status bolr_decode_i64_le(const void *buf, size_t cap, size_t *cursor, int64_t *out);
bolr_status bolr_decode_f64_le(const void *buf, size_t cap, size_t *cursor, double *out);
bolr_status bolr_decode_bytes(const void *buf, size_t cap, size_t *cursor, void *out, size_t length);

#endif
