# Phase L4B2.3: C11 Checkpoint Codec and Atomic Persistence

Phase L4B2.3 persists the validated L4B2.2 in-memory replay state in a portable, versioned binary format without changing replay mathematics.

## Scope

- ABI bump to `1.8.0`
- Checkpoint format `v1.0` (`BOLRCP01`)
- Explicit little-endian scalar codec
- IEEE CRC32 checksums (corruption detection only; not authentication)
- Section directory with deterministic section-type ordering
- Ready and pending (`AWAITING_OUTCOME`) encode/decode
- Atomic POSIX checkpoint write (`mkstemp` + write + optional `fsync` + `rename`)
- Safe file read through the same byte decoder
- Injectable file I/O hooks for failure testing
- Python ctypes byte and file wrappers
- Golden fixtures under `tests/fixtures/golden/`

## Versions

| Version | Value |
| --- | --- |
| ABI | `1.8.0` |
| Checkpoint format | major `1`, minor `0` |
| Magic | `BOLRCP01` (8 bytes) |

ABI and checkpoint versions are independent. Unsupported checkpoint major versions fail. Unknown optional sections may be skipped; unknown required sections fail.

## Logical source of truth

The portable codec serialises the extended in-memory replay checkpoint:

- posterior Gaussian state
- transition or adaptive state
- RNG state
- phase and completed-day index
- pending predictive Gaussian
- pending decision
- pending score context
- ranking / decision configuration retained from `begin_day`
- pending posterior score summaries and rank summaries when present
- optional pending region summary when a region policy was used

Replay phases, observation mathematics, and `begin_day` / `finish_day` causality are unchanged. A restored `AWAITING_OUTCOME` checkpoint finishes without resampling or re-running ranking.

## Header layout (180 bytes)

Little-endian fields in order:

```text
magic[8]
format_major u16
format_minor u16
header_size u32
directory_entry_size u32
section_count u32
flags u32
total_file_size u64
directory_offset u64
payload_offset u64
payload_size u64
payload_crc32 u32
header_crc32 u32          # CRC of header with this field zeroed
abi_major u16
abi_minor u16
replay_phase u32
completed_day_index i64
model_schema_hash u64
state_layout_hash u64
candidate_grid_hash u64
replay_config_hash u64
decision_config_hash u64
monte_carlo_config_hash u64
adaptive_policy_hash u64
checkpoint_id u64
reserved[32] = 0
```

Header flags:

```text
ADAPTIVE_PRESENT
RNG_STATEFUL
PENDING_DAY
RANK_SUMMARY
REGION_SUMMARY
THOMPSON_DECISION
```

## Directory entry (44 bytes)

```text
section_type u32
schema_major u16
schema_minor u16
section_flags u32
payload_offset u64
payload_length u64
element_count u64
section_crc32 u32
reserved u32 = 0
```

Section flags: `REQUIRED`, `OPTIONAL`. `COMPRESSED` and `ENCRYPTED` are rejected in v1.

## Core sections

| ID | Name | Role |
| --- | --- | --- |
| `0x0001` | `REPLAY_METADATA` | phase, dimensions, hashes |
| `0x0002` | `GAUSSIAN_POSTERIOR` | mean + row-major covariance |
| `0x0003` | `ADAPTIVE_STATE` | wrapped adaptive blob when adaptive |
| `0x0006` | `RNG_STATE` | PCG32 + metadata |
| `0x0008` | `DECISION_CONFIG` | decision family / region rules |
| `0x0009` | `MONTE_CARLO_CONFIG` | sample/chunk/retention/top-k |
| `0x000A` | `PENDING_DAY_METADATA` | decision id / selection |
| `0x000B` | `PENDING_SCORE_CONTEXT` | daily context vector |
| `0x000C` | `PENDING_PREDICTIVE_GAUSSIAN` | predictive state for `finish_day` |
| `0x000D` | `PENDING_POSTERIOR_PREDICTION` | score mean/variance |
| `0x000E` | `PENDING_RANK_SUMMARY` | MC ranking summaries |
| `0x000F` | `PENDING_REGION_SET` | optional region summary |
| `0x0010` | `PENDING_DECISION` | immutable issued decision |
| `0x0011` | `PROVENANCE` | informative library/ABI metadata |
| `0x0012` | `TRANSITION_CONFIG` | fixed-transition process noise |

v1 encodes adaptive state as a single `ADAPTIVE_STATE` section (nested validated adaptive blob). Separate `BOCPD_STATE` / `STANDARDIZER_STATE` section IDs are reserved but not required emitters in v1.

## Ready vs pending

**READY** requires metadata, posterior, decision/MC configs, RNG, provenance, and either `TRANSITION_CONFIG` or `ADAPTIVE_STATE`. It must not include pending-day sections.

**AWAITING_OUTCOME** requires the ready sections plus pending metadata, score context, predictive Gaussian, posterior prediction, decision, and rank summary when ranking was used.

## Restore context and limits

Decode validates:

- magic, versions, sizes, checksums
- section directory bounds and non-overlap
- required/duplicate section policy
- model / layout / adaptive policy hashes against `bolr_replay_restore_context`
- `bolr_checkpoint_limits` allocation caps

Partial engines never escape on failure.

## Atomic file persistence

`bolr_replay_checkpoint_write_atomic`:

1. encode into memory
2. create a temporary file in the destination directory
3. write with `EINTR` / short-write handling
4. optional file `fsync`
5. close
6. `rename` over the destination
7. optional directory `fsync`

Default file mode is `0600`. If `replace_existing` is false and the destination exists, the write fails without mutating the existing file.

## Python API

```python
payload = replay.encode_checkpoint()
replay = CReplayEngine.decode_checkpoint(payload, restore_context)
replay.write_checkpoint(path, durable=True)
replay = CReplayEngine.read_checkpoint(path, restore_context)
```

Typed exceptions:

- `CCheckpointCorruptionError`
- `CCheckpointVersionError`
- `CCheckpointCompatibilityError`
- `CCheckpointLimitError`
- `CCheckpointIOError`

## Golden fixtures

```text
tests/fixtures/golden/c_checkpoint_ready_v1.bin
tests/fixtures/golden/c_checkpoint_pending_v1.bin
tests/fixtures/golden/c_checkpoint_v1.json
```

These are compatibility contracts. Do not regenerate casually.

## Platform assumptions

Validated when encoding/decoding:

- `CHAR_BIT == 8`
- fixed-width integer sizes
- `sizeof(double) == 8` (IEEE-754 binary64)

Unsupported platforms return an error rather than writing incompatible bytes.

## Known limitations / deferred

- no compression or encryption
- no keyed-RNG section emitter (stateful RNG only in v1)
- adaptive BOCPD/standardizer not emitted as separate top-level sections
- CRC32 is not cryptographic authentication
- durable claims require the configured `fsync` sequence; platform power-loss behaviour varies
- full native historical YM replay remains Phase L5

## Validation

```bash
make -C csrc BUILD_DIR=build/l4b23-debug-gcc clean test CC=gcc
make -C csrc BUILD_DIR=build/l4b23-debug-clang clean test CC=clang
make -C csrc BUILD_DIR=build/l4b23-sanitize-gcc clean sanitize CC=gcc
make -C csrc BUILD_DIR=build/l4b23-release-gcc clean release CC=gcc
PYTHONPATH=. pytest -q tests/c_backend/test_c_checkpoint_codec.py \
  tests/c_backend/test_c_checkpoint_corruption.py \
  tests/c_backend/test_c_checkpoint_files.py \
  tests/c_backend/test_c_replay_file_restart.py
PYTHONPATH=. pytest -q tests/c_backend
PYTHONPATH=. pytest -q
```
