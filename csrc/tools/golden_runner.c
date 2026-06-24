#include "bolr/version.h"

#include <stdio.h>

int main(void) {
    printf("bolr_golden_runner abi=%u.%u.%u\n", bolr_abi_version_major(), bolr_abi_version_minor(), bolr_abi_version_patch());
    return 0;
}
