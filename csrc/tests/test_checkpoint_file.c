#include "test_suite.h"
#include "test_checkpoint_fixture.h"

#include "bolr/checkpoint_codec.h"
#include "bolr/checkpoint_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    size_t fail_after_bytes;
    size_t written;
} failing_write_state;

static int test_open(const char *path, int flags, mode_t mode, void *ctx) {
    (void) ctx;
    return open(path, flags, mode);
}

static ssize_t test_write(int fd, const void *buf, size_t n, void *ctx) {
    failing_write_state *state = (failing_write_state *) ctx;
    if ((state != NULL) && (state->fail_after_bytes > 0U) && ((state->written + n) > state->fail_after_bytes)) {
        errno = EIO;
        return -1;
    }
    {
        ssize_t got = write(fd, buf, n);
        if ((got > 0) && (state != NULL)) state->written += (size_t) got;
        return got;
    }
}

static ssize_t test_read(int fd, void *buf, size_t n, void *ctx) {
    (void) ctx;
    return read(fd, buf, n);
}

static int test_fsync(int fd, void *ctx) {
    (void) ctx;
    return fsync(fd);
}

static int test_close(int fd, void *ctx) {
    (void) ctx;
    return close(fd);
}

static int test_rename(const char *oldpath, const char *newpath, void *ctx) {
    (void) ctx;
    return rename(oldpath, newpath);
}

static int test_unlink(const char *path, void *ctx) {
    (void) ctx;
    return unlink(path);
}

static int test_fstat(int fd, struct stat *st, void *ctx) {
    (void) ctx;
    return fstat(fd, st);
}

int test_checkpoint_file(void) {
    bolr_test_checkpoint_fixture fixture;
    bolr_replay_restore_context ctx;
    bolr_replay_engine *restored = NULL;
    bolr_checkpoint_file_options options;
    failing_write_state fail_state;
    bolr_checkpoint_io_hooks hooks;
    char path[512];
    unsigned char *original = NULL;
    unsigned char *after_fail = NULL;
    size_t original_size = 0U;
    size_t after_size = 0U;
    int rc = 1;

    snprintf(path, sizeof(path), "build/l4b23-debug-gcc/bolr_checkpoint_test_%d.cp", (int) getpid());
    (void) system("mkdir -p build/l4b23-debug-gcc");
    bolr_checkpoint_io_hooks_reset();
    if (bolr_test_checkpoint_fixture_create(&fixture) != BOLR_OK) goto cleanup;
    ctx = bolr_test_checkpoint_restore_context(fixture.model);
    options = bolr_checkpoint_file_options_default();
    options.fsync_directory = 0;
    options.replace_existing = 1;

    if (bolr_replay_checkpoint_write_atomic(fixture.engine, path, &options) != BOLR_OK) goto cleanup;
    if (bolr_replay_checkpoint_read_file(path, &ctx, NULL, &restored) != BOLR_OK) goto cleanup;
    if (bolr_replay_engine_phase(restored) != BOLR_REPLAY_PHASE_READY) goto cleanup;
    bolr_replay_engine_destroy(restored);
    restored = NULL;

    options.replace_existing = 0;
    if (bolr_replay_checkpoint_write_atomic(fixture.engine, path, &options) != BOLR_CHECKPOINT_IO_ERROR) goto cleanup;

    original = (unsigned char *) malloc(65536U);
    after_fail = (unsigned char *) malloc(65536U);
    if ((original == NULL) || (after_fail == NULL)) goto cleanup;
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) goto cleanup;
        original_size = (size_t) read(fd, original, 65536U);
        close(fd);
        if (original_size == 0U) goto cleanup;
    }

    memset(&fail_state, 0, sizeof(fail_state));
    fail_state.fail_after_bytes = 32U;
    hooks.open_fn = test_open;
    hooks.write_fn = test_write;
    hooks.read_fn = test_read;
    hooks.fsync_fn = test_fsync;
    hooks.close_fn = test_close;
    hooks.rename_fn = test_rename;
    hooks.unlink_fn = test_unlink;
    hooks.fstat_fn = test_fstat;
    hooks.ctx = &fail_state;
    bolr_checkpoint_io_hooks_set(&hooks);

    options.replace_existing = 1;
    if (bolr_replay_checkpoint_write_atomic(fixture.engine, path, &options) != BOLR_CHECKPOINT_IO_ERROR) goto cleanup;
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) goto cleanup;
        after_size = (size_t) read(fd, after_fail, 65536U);
        close(fd);
    }
    if ((after_size != original_size) || (memcmp(original, after_fail, original_size) != 0)) goto cleanup;
    rc = 0;

cleanup:
    bolr_checkpoint_io_hooks_reset();
    bolr_replay_engine_destroy(restored);
    free(after_fail);
    free(original);
    unlink(path);
    bolr_test_checkpoint_fixture_destroy(&fixture);
    return rc;
}
