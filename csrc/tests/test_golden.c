#include "test_suite.h"

#include "bolr/version.h"

int test_golden(void) { return (bolr_abi_version_major() != 1U) ? 1 : 0; }
