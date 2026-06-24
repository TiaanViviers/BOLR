#include "bolr/version.h"

#include <stdio.h>

int main(void) {
    printf("{\"abi_major\":%u,\"abi_minor\":%u,\"abi_patch\":%u,\"compiler\":\"%s\",\"linalg\":\"%s\"}\n",
        bolr_abi_version_major(), bolr_abi_version_minor(), bolr_abi_version_patch(), bolr_build_compiler(), bolr_linalg_backend());
    return 0;
}
