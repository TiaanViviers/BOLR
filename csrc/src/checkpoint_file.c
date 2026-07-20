#define _GNU_SOURCE
#include "bolr/checkpoint_file.h"

#include "bolr/checkpoint_codec.h"
#include "internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bolr_checkpoint_io_hooks g_hooks;
static int g_hooks_custom = 0;

static int posix_open(const char *path, int flags, mode_t mode, void *ctx) {
    (void) ctx;
    return open(path, flags, mode);
}

static ssize_t posix_write(int fd, const void *buf, size_t n, void *ctx) {
    (void) ctx;
    return write(fd, buf, n);
}

static ssize_t posix_read(int fd, void *buf, size_t n, void *ctx) {
    (void) ctx;
    return read(fd, buf, n);
}

static int posix_fsync(int fd, void *ctx) {
    (void) ctx;
    return fsync(fd);
}

static int posix_close(int fd, void *ctx) {
    (void) ctx;
    return close(fd);
}

static int posix_rename(const char *oldpath, const char *newpath, void *ctx) {
    (void) ctx;
    return rename(oldpath, newpath);
}

static int posix_unlink(const char *path, void *ctx) {
    (void) ctx;
    return unlink(path);
}

static int posix_fstat(int fd, struct stat *st, void *ctx) {
    (void) ctx;
    return fstat(fd, st);
}

static const bolr_checkpoint_io_hooks g_default_hooks = {
    posix_open,
    posix_write,
    posix_read,
    posix_fsync,
    posix_close,
    posix_rename,
    posix_unlink,
    posix_fstat,
    NULL
};

static const bolr_checkpoint_io_hooks *active_hooks(void) {
    return g_hooks_custom ? &g_hooks : &g_default_hooks;
}

void bolr_checkpoint_io_hooks_set(const bolr_checkpoint_io_hooks *hooks) {
    if (hooks == NULL) {
        g_hooks_custom = 0;
        memset(&g_hooks, 0, sizeof(g_hooks));
        return;
    }
    g_hooks = *hooks;
    g_hooks_custom = 1;
}

void bolr_checkpoint_io_hooks_reset(void) { bolr_checkpoint_io_hooks_set(NULL); }

bolr_checkpoint_file_options bolr_checkpoint_file_options_default(void) {
    bolr_checkpoint_file_options options;
    options.fsync_file = 1;
    options.fsync_directory = 1;
    options.file_mode = 0600U;
    options.replace_existing = 1;
    return options;
}

static bolr_status write_all(int fd, const void *buf, size_t n, const bolr_checkpoint_io_hooks *hooks) {
    const unsigned char *cursor = (const unsigned char *) buf;
    size_t remaining = n;
    while (remaining > 0U) {
        ssize_t written = hooks->write_fn(fd, cursor, remaining, hooks->ctx);
        if (written < 0) {
            if (errno == EINTR) continue;
            return BOLR_CHECKPOINT_IO_ERROR;
        }
        if (written == 0) return BOLR_CHECKPOINT_IO_ERROR;
        cursor += (size_t) written;
        remaining -= (size_t) written;
    }
    return BOLR_OK;
}

static bolr_status read_all(int fd, void *buf, size_t n, const bolr_checkpoint_io_hooks *hooks) {
    unsigned char *cursor = (unsigned char *) buf;
    size_t remaining = n;
    while (remaining > 0U) {
        ssize_t got = hooks->read_fn(fd, cursor, remaining, hooks->ctx);
        if (got < 0) {
            if (errno == EINTR) continue;
            return BOLR_CHECKPOINT_IO_ERROR;
        }
        if (got == 0) return BOLR_CHECKPOINT_TRUNCATED;
        cursor += (size_t) got;
        remaining -= (size_t) got;
    }
    return BOLR_OK;
}

bolr_status bolr_replay_checkpoint_write_atomic(const bolr_replay_engine *engine, const char *path, const bolr_checkpoint_file_options *options) {
    bolr_checkpoint_file_options active_options = (options == NULL) ? bolr_checkpoint_file_options_default() : *options;
    const bolr_checkpoint_io_hooks *hooks = active_hooks();
    const bolr_allocator *allocator = ((const struct bolr_replay_engine *) engine)->allocator;
    void *bytes = NULL;
    size_t size = 0U;
    char temp_path[4096];
    int fd = -1;
    bolr_status status;
    if ((engine == NULL) || (path == NULL)) return BOLR_INVALID_ARGUMENT;
    if (!active_options.replace_existing) {
        int existing_fd = hooks->open_fn(path, O_RDONLY, 0, hooks->ctx);
        if (existing_fd >= 0) {
            hooks->close_fn(existing_fd, hooks->ctx);
            return BOLR_CHECKPOINT_IO_ERROR;
        }
        if (errno != ENOENT) return BOLR_CHECKPOINT_IO_ERROR;
    }
    status = bolr_replay_checkpoint_encode_buffer(engine, allocator, &bytes, &size);
    if (status != BOLR_OK) return status;
    if (snprintf(temp_path, sizeof(temp_path), "%s.XXXXXX", path) >= (int) sizeof(temp_path)) {
        bolr_allocator_free(allocator, bytes);
        return BOLR_INVALID_ARGUMENT;
    }
    fd = mkstemp(temp_path);
    if (fd < 0) {
        bolr_allocator_free(allocator, bytes);
        return BOLR_CHECKPOINT_IO_ERROR;
    }
    status = write_all(fd, bytes, size, hooks);
    if ((status == BOLR_OK) && active_options.fsync_file) {
        if (hooks->fsync_fn(fd, hooks->ctx) != 0) status = BOLR_CHECKPOINT_IO_ERROR;
    }
    if (hooks->close_fn(fd, hooks->ctx) != 0) status = BOLR_CHECKPOINT_IO_ERROR;
    fd = -1;
    if (status != BOLR_OK) {
        hooks->unlink_fn(temp_path, hooks->ctx);
        bolr_allocator_free(allocator, bytes);
        return status;
    }
    if (hooks->rename_fn(temp_path, path, hooks->ctx) != 0) {
        hooks->unlink_fn(temp_path, hooks->ctx);
        bolr_allocator_free(allocator, bytes);
        return BOLR_CHECKPOINT_IO_ERROR;
    }
    if (active_options.fsync_directory) {
        char dir_path[4096];
        const char *slash = strrchr(path, '/');
        if (slash != NULL) {
            size_t dir_len = (size_t) (slash - path);
            if (dir_len >= sizeof(dir_path)) dir_len = sizeof(dir_path) - 1U;
            memcpy(dir_path, path, dir_len);
            dir_path[dir_len] = '\0';
            fd = hooks->open_fn(dir_path, O_RDONLY | O_DIRECTORY, 0, hooks->ctx);
            if (fd >= 0) {
                if (hooks->fsync_fn(fd, hooks->ctx) != 0) status = BOLR_CHECKPOINT_IO_ERROR;
                hooks->close_fn(fd, hooks->ctx);
            }
        }
    }
    bolr_allocator_free(allocator, bytes);
    return status;
}

bolr_status bolr_replay_checkpoint_read_file(const char *path, const bolr_replay_restore_context *context, const bolr_allocator *allocator, bolr_replay_engine **out_engine) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    const bolr_checkpoint_io_hooks *hooks = active_hooks();
    int fd;
    struct stat st;
    void *bytes;
    bolr_status status;
    if ((path == NULL) || (context == NULL) || (out_engine == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_engine = NULL;
    fd = hooks->open_fn(path, O_RDONLY, 0, hooks->ctx);
    if (fd < 0) return BOLR_CHECKPOINT_IO_ERROR;
    if (hooks->fstat_fn(fd, &st, hooks->ctx) != 0) {
        hooks->close_fn(fd, hooks->ctx);
        return BOLR_CHECKPOINT_IO_ERROR;
    }
    if (st.st_size <= 0) {
        hooks->close_fn(fd, hooks->ctx);
        return BOLR_CHECKPOINT_TRUNCATED;
    }
    bytes = bolr_allocator_malloc(active, (size_t) st.st_size);
    if (bytes == NULL) {
        hooks->close_fn(fd, hooks->ctx);
        return BOLR_ALLOCATION_FAILED;
    }
    status = read_all(fd, bytes, (size_t) st.st_size, hooks);
    hooks->close_fn(fd, hooks->ctx);
    if (status != BOLR_OK) {
        bolr_allocator_free(active, bytes);
        return status;
    }
    status = bolr_replay_checkpoint_decode(bytes, (size_t) st.st_size, context, active, out_engine);
    bolr_allocator_free(active, bytes);
    return status;
}
