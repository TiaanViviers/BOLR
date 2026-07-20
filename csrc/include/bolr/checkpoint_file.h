#ifndef BOLR_CHECKPOINT_FILE_H
#define BOLR_CHECKPOINT_FILE_H

#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "bolr/allocator.h"
#include "bolr/checkpoint_codec.h"
#include "bolr/replay.h"
#include "bolr/status.h"

typedef struct {
    int fsync_file;
    int fsync_directory;
    uint32_t file_mode;
    int replace_existing;
} bolr_checkpoint_file_options;

bolr_checkpoint_file_options bolr_checkpoint_file_options_default(void);

bolr_status bolr_replay_checkpoint_write_atomic(const bolr_replay_engine *engine, const char *path, const bolr_checkpoint_file_options *options);
bolr_status bolr_replay_checkpoint_read_file(const char *path, const bolr_replay_restore_context *context, const bolr_allocator *allocator, bolr_replay_engine **out_engine);

typedef struct bolr_checkpoint_io_hooks {
    int (*open_fn)(const char *path, int flags, mode_t mode, void *ctx);
    ssize_t (*write_fn)(int fd, const void *buf, size_t n, void *ctx);
    ssize_t (*read_fn)(int fd, void *buf, size_t n, void *ctx);
    int (*fsync_fn)(int fd, void *ctx);
    int (*close_fn)(int fd, void *ctx);
    int (*rename_fn)(const char *oldpath, const char *newpath, void *ctx);
    int (*unlink_fn)(const char *path, void *ctx);
    int (*fstat_fn)(int fd, struct stat *st, void *ctx);
    void *ctx;
} bolr_checkpoint_io_hooks;

void bolr_checkpoint_io_hooks_set(const bolr_checkpoint_io_hooks *hooks);
void bolr_checkpoint_io_hooks_reset(void);

#endif
