#ifndef BOLR_VERSION_H
#define BOLR_VERSION_H

#include <stdint.h>

uint32_t bolr_abi_version_major(void);
uint32_t bolr_abi_version_minor(void);
uint32_t bolr_abi_version_patch(void);

const char *bolr_library_version(void);
const char *bolr_build_compiler(void);
const char *bolr_linalg_backend(void);

#endif
