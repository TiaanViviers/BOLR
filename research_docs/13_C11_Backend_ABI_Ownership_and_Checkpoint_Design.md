# C11 Backend ABI Ownership and Checkpoint Design

## Scope

Phase L1 establishes the pure-C11 backend foundation. The production backend is a standalone C library built with GNU Make on Linux/POSIX toolchains. Python is a client of the C ABI and does not own the backend design.

## Build and layout

The C subtree lives under `csrc/` with:

- public headers in `csrc/include/bolr`
- implementation sources in `csrc/src`
- C-native tests in `csrc/tests`
- small tools in `csrc/tools`
- benchmark scaffolding in `csrc/benchmarks`

The authoritative build entry point is `csrc/Makefile`. It exposes:

- `all`
- `static`
- `shared`
- `tests`
- `test`
- `golden`
- `bench`
- `debug`
- `release`
- `sanitize`
- `install`
- `uninstall`
- `clean`
- `distclean`
- `print-config`

## ABI and error model

The ABI is versioned with:

- `BOLR_ABI_VERSION_MAJOR`
- `BOLR_ABI_VERSION_MINOR`
- `BOLR_ABI_VERSION_PATCH`

All public functions return explicit status codes. The library does not call `exit`, does not print errors directly, and does not rely on a global mutable error buffer.

## Ownership model

Two ownership styles are fixed:

- caller-owned numerical buffers through vector and matrix views
- library-owned persistent handles with typed destructors

Persistent handles include workspaces, state layouts, checkpoint-state shells, and score/model objects. Python never calls a generic libc `free`; it always routes closure through typed BOLR destructors.

## Python lifecycle handling

The Python adapter uses `ctypes`, explicit `.close()`, a `_closed` guard, context-manager support, and `weakref.finalize` as a fallback finalizer. Closed handles raise clear Python exceptions.

## Allocator and workspace rules

The C backend exposes a pluggable allocator interface and checked size helpers for overflow-safe allocation. Reusable temporaries are kept in opaque workspaces. Numerical outputs do not borrow workspace memory after return.

## Checkpoint-ready separation

Runtime objects may contain allocators, pointers, and caches. Checkpoint state is reserved as a logical portable layer and excludes raw addresses, callback pointers, file handles, and Python objects. Encoding/decoding APIs are reserved now; final file serialization remains deferred.

## RNG policy

Phase L1 is deterministic and does not depend on RNGs. A future-compatible `bolr_rng` interface is reserved, while generation routines currently return `BOLR_UNSUPPORTED_OPERATION`. Deferred Ziggurat integration remains a later phase.

## Numerical scope

Phase L1 ports:

- stable math primitives
- dense linear algebra wrappers
- Cholesky with explicit diagnostics
- state-layout metadata
- dense, context, graph, and composite score kernels
- Candidate A log-factor, gradient, and HVP kernels
- quadratic penalties
- additive and discount transitions

Full inference and replay remain later phases.

## Validation procedure

Primary validation includes:

- `make -C csrc debug`
- `make -C csrc test`
- `make -C csrc sanitize`
- Python `tests/c_backend`
- full Python-suite regression

Golden equivalence is driven from Python fixtures rather than native `.npz` parsing inside C.
