#include "bolr/endian.h"

#include <string.h>

bolr_status bolr_platform_validate(void) {
    union {
        uint16_t value;
        unsigned char bytes[2];
    } probe = {1U};
    return (probe.bytes[0] == 1U) ? BOLR_OK : BOLR_UNSUPPORTED_OPERATION;
}

static bolr_status ensure_capacity(size_t cap, size_t cursor, size_t need) {
    if (cursor > cap) return BOLR_CHECKPOINT_TRUNCATED;
    if ((cap - cursor) < need) return BOLR_CHECKPOINT_TRUNCATED;
    return BOLR_OK;
}

bolr_status bolr_encode_u16_le(void *buf, size_t cap, size_t *cursor, uint16_t v) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(v)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy((unsigned char *) buf + pos, &v, sizeof(v));
    *cursor = pos + sizeof(v);
    return BOLR_OK;
}

bolr_status bolr_encode_u32_le(void *buf, size_t cap, size_t *cursor, uint32_t v) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(v)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy((unsigned char *) buf + pos, &v, sizeof(v));
    *cursor = pos + sizeof(v);
    return BOLR_OK;
}

bolr_status bolr_encode_u64_le(void *buf, size_t cap, size_t *cursor, uint64_t v) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(v)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy((unsigned char *) buf + pos, &v, sizeof(v));
    *cursor = pos + sizeof(v);
    return BOLR_OK;
}

bolr_status bolr_encode_i64_le(void *buf, size_t cap, size_t *cursor, int64_t v) {
    return bolr_encode_u64_le(buf, cap, cursor, (uint64_t) v);
}

bolr_status bolr_encode_f64_le(void *buf, size_t cap, size_t *cursor, double v) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(v)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy((unsigned char *) buf + pos, &v, sizeof(v));
    *cursor = pos + sizeof(v);
    return BOLR_OK;
}

bolr_status bolr_encode_bytes(void *buf, size_t cap, size_t *cursor, const void *data, size_t length) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((length > 0U) && (data == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, length) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    if (length > 0U) memcpy((unsigned char *) buf + pos, data, length);
    *cursor = pos + length;
    return BOLR_OK;
}

bolr_status bolr_decode_u16_le(const void *buf, size_t cap, size_t *cursor, uint16_t *out) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(*out)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy(out, (const unsigned char *) buf + pos, sizeof(*out));
    *cursor = pos + sizeof(*out);
    return BOLR_OK;
}

bolr_status bolr_decode_u32_le(const void *buf, size_t cap, size_t *cursor, uint32_t *out) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(*out)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy(out, (const unsigned char *) buf + pos, sizeof(*out));
    *cursor = pos + sizeof(*out);
    return BOLR_OK;
}

bolr_status bolr_decode_u64_le(const void *buf, size_t cap, size_t *cursor, uint64_t *out) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(*out)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy(out, (const unsigned char *) buf + pos, sizeof(*out));
    *cursor = pos + sizeof(*out);
    return BOLR_OK;
}

bolr_status bolr_decode_i64_le(const void *buf, size_t cap, size_t *cursor, int64_t *out) {
    uint64_t raw;
    bolr_status status = bolr_decode_u64_le(buf, cap, cursor, &raw);
    if (status != BOLR_OK) return status;
    *out = (int64_t) raw;
    return BOLR_OK;
}

bolr_status bolr_decode_f64_le(const void *buf, size_t cap, size_t *cursor, double *out) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, sizeof(*out)) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    memcpy(out, (const unsigned char *) buf + pos, sizeof(*out));
    *cursor = pos + sizeof(*out);
    return BOLR_OK;
}

bolr_status bolr_decode_bytes(const void *buf, size_t cap, size_t *cursor, void *out, size_t length) {
    size_t pos;
    if ((buf == NULL) || (cursor == NULL)) return BOLR_INVALID_ARGUMENT;
    if ((length > 0U) && (out == NULL)) return BOLR_INVALID_ARGUMENT;
    pos = *cursor;
    if (ensure_capacity(cap, pos, length) != BOLR_OK) return BOLR_CHECKPOINT_TRUNCATED;
    if (length > 0U) memcpy(out, (const unsigned char *) buf + pos, length);
    *cursor = pos + length;
    return BOLR_OK;
}
