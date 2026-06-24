#include "bolr/version.h"
#include "bolr/types.h"

#include <float.h>

#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "BOLR requires IEEE-754 binary64 double support."
#endif

uint32_t bolr_abi_version_major(void) { return BOLR_ABI_VERSION_MAJOR; }
uint32_t bolr_abi_version_minor(void) { return BOLR_ABI_VERSION_MINOR; }
uint32_t bolr_abi_version_patch(void) { return BOLR_ABI_VERSION_PATCH; }

const char *bolr_library_version(void) { return "1.0.0"; }
const char *bolr_build_compiler(void) {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#else
    return "unknown";
#endif
}
const char *bolr_linalg_backend(void) { return "builtin_c"; }
