# LMCache-Inspired KV Cache Expansion for Native Windows

Status: CPU LRU/ARC and RAM + filesystem LRU/ARC remain implemented and
validated in the v0.27.1 release wheel; the feature remains experimental and
opt-in.

Last reviewed: 2026-08-21

Target baseline: vLLM 0.27.1, CPython 3.13, PyTorch 2.13.0+cu130,
CUDA 13.0, native Windows

## v0.27.1 Release Update

The implementation was rebuilt on upstream vLLM 0.27.1 in
`E:\vllm-windows-build-v2`. The release wheel retains the Windows DMA,
shared-mmap, filesystem namespace, and non-Triton block-table fixes described
below. The wheel is `vllm-0.27.1-cp313-cp313-win_amd64.whl` (239,025,534
bytes; SHA-256
`7c13ed44e94694478bdd4f5fcca23e2d66ba1e8fa9bccad9fddb8651d1b2447b`).

The installed wheel served Qwen3.5 9B GPTQ on the RTX 3090 and passed CPU LRU,
forced filesystem eviction/restore, and fresh-process persistent reuse. Each
offload restore reused 1,056 of 1,452 prompt tokens and reproduced exact token
IDs. The focused source regression suite passed 80 tests with one expected
platform-gated skip. These results validate this project's local tiers; they
do not claim that LMCache itself runs natively on Windows.

## Executive Summary

[LMCache](https://github.com/LMCache/LMCache) is a substantial KV-cache
management system for vLLM. It can retain and reuse KV data in CPU memory,
local storage, remote storage, and other serving processes. Its most relevant
benefit is reducing prefill work and time-to-first-token (TTFT) when requests
reuse long prompt prefixes.

LMCache cannot currently be installed as a normal dependency of this native
Windows distribution. LMCache officially targets Linux, publishes Linux
wheels, and uses Linux facilities such as POSIX shared memory, `/dev/shm`,
`mmap` behavior, `shm_open`, `fcntl`, NUMA operations, and Linux-oriented
high-performance storage and networking libraries. Python 3.13 is **not** the
problem: LMCache supports it, and its current PyTorch baseline aligns with this
project's PyTorch 2.13.0.

The recommended direction is not a wholesale LMCache port. It is to:

1. Use vLLM's existing connector and tiering interfaces rather than re-create
   the entire LMCache package.
2. Harden only the native paths that fail on Windows.
3. Keep the feature disabled by default and require explicit capacity/storage
   choices.
4. Validate ordinary KV cache first, then evaluate this project's compressed
   KV modes separately.
5. Treat remote/distributed features as later projects with their own evidence.

A key local-source finding improved the starting point: vLLM 0.25.1 already
contains `OffloadingConnector` and `SimpleCPUOffloadConnector`, and both are
present in the Windows wheel. The simple connector was used to prove the first
Windows DMA path. The full connector was then made portable by adapting shared
mmap, filesystem I/O, cache paths, and file-backed restore behavior. No new
external connector was needed.

## Validated v0.25.1 Result

On 2026-07-19, the existing `SimpleCPUOffloadConnector` first proved that a
real local Qwen3 14B AWQ-4bit model could reuse prompt KV on the RTX 3090. The
upstream `cuMemcpyBatchAsync` path caused an illegal CUDA memory access on
Windows. A Windows-only fallback using standard per-region `cudaMemcpyAsync`
passed direct GPU-to-CPU and CPU-to-GPU checks and restored 2,400 of 2,414
prompt tokens with identical output.

The work then moved to vLLM 0.25.1's full `OffloadingConnector`. The final
wheel passed:

| Validation | Result |
|---|---|
| Low-level GPU matrix | 24/24 cases passed: both directions, ordinary/shared memory, multiple page sizes, block-size factor 1/3, and multiple KV groups |
| CPU policies | LRU and ARC each restored 1,440 of 1,451 prompt tokens with exact baseline output |
| Forced filesystem restore | LRU and ARC each restored 1,440 tokens after enough unrelated prefixes were stored to evict the target from the 102-block RAM tier |
| Persistent restart | A new engine process reused 1,440 tokens from the existing filesystem cache with exact output |
| Final packaged wheel | Repeated the persistent model smoke test from the exact release wheel |

The forced LRU filesystem run measured about 1.36 seconds cold and 1.02 seconds
for disk restore; the ARC run measured about 1.37 seconds cold and 0.76 seconds
for disk restore. These focused eager-mode timings are functional evidence,
not a general performance benchmark. They demonstrate Windows allocation,
shared mmap, asynchronous store, GPU prefix-cache reset, RAM eviction,
filesystem lookup/load, CPU-to-GPU restore, and continued generation together.

Final release artifact:

```text
vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
size:   293080424 bytes
SHA256: 0C4F9B2E36482523FC7B4C092D711AC49B4265EF9F36A7AEEFFF9A667C875339
```

The feature is still experimental because the current evidence is one Windows
machine/model family and focused correctness runs. It is disabled by default,
and the filesystem tier deliberately has no implicit root or automatic quota.

## Terminology: Cache Format Versus Cache Tier

Two independent settings must remain separate.

**KV cache format** controls how active KV data is represented:

- `auto`
- `fp8`
- `isoquant3`, `isoquant4`
- `planarquant3`, `planarquant4`
- `turboquant25`, `turboquant35`
- upstream TurboQuant formats

**KV cache tier** controls where reusable or evicted KV data is retained:

- GPU only
- GPU plus system RAM
- GPU plus system RAM and NVMe
- potentially a shared local or remote service later

The intended user-facing model is therefore:

```text
KV cache format: auto / fp8 / isoquant4 / ...
KV offload tier: disabled / system RAM / system RAM + NVMe
CPU cache limit: 8 GiB / 16 GiB / ...
Filesystem cache root: explicit dedicated directory (no automatic quota yet)
```

Offloading is not another quantization recipe. It complements the cache format
selected by the user.

## What LMCache Provides

LMCache's advertised capability set includes several distinct levels:

- Reuse of KV data for repeated prompt prefixes.
- GPU-to-CPU and CPU-to-GPU cache movement.
- Local disk and pluggable storage backends.
- Cache sharing among engines and processes.
- A standalone multiprocess cache service.
- Asynchronous storage and loading.
- Distributed peer-to-peer transfer and disaggregated prefill workflows.
- NIXL, GDS, RDMA, and other datacenter-oriented paths.
- CacheBlend and non-prefix reuse research.
- Metrics, tracing, and operational tooling.
- Optional storage serialization/compression, including TurboQuant serde.

These features do not all have equal value for a native Windows workstation.
The local tiers and repeated-prefix reuse are the highest-value subset. The
Linux cluster features are not an appropriate first target.

## What Users Could Expect If the Windows Work Succeeds

The first request for a prefix would still perform normal prefill. Once its KV
blocks are stored, a later request with the same prefix could restore those
blocks from CPU memory instead of recomputing every prefix token.

Expected benefits:

- Lower TTFT for repeated long system prompts.
- Lower TTFT for RAG requests that repeatedly use the same documents.
- Reuse of long conversation prefixes across compatible requests.
- More reusable prefixes retained than GPU memory alone can hold.
- Less recomputation after the GPU prefix-cache pool evicts an entry.
- Potential persistence across server restarts after an NVMe tier is added.
- Potentially smaller offloaded entries when compatible compressed KV formats
  are supported.

What it would **not** do:

- It would not increase decode tokens per second by itself.
- It would not make the current PyTorch fallback in local Multi-TurboQuant
  encode/decode faster.
- It would not improve the mathematical accuracy of a compression method.
- It would not automatically benefit unique, short prompts.
- It would not make arbitrarily large active contexts free. Attention still
  needs the active request's usable KV data, and continuous CPU/GPU transfers
  can become slower than computation.
- It would not necessarily help the first request; storing KV adds some work.

The feature is successful only if a cache hit saves more time than locating and
transferring the cached blocks costs.

## Current LMCache Compatibility Assessment

The assessment was refreshed against
[LMCache v0.5.4](https://github.com/LMCache/LMCache/releases/tag/v0.5.4),
released 2026-08-20. Its tagged
[`pyproject.toml`](https://github.com/LMCache/LMCache/blob/v0.5.4/pyproject.toml)
is the source for the Python, Torch, and CUDA rows below.

| Area | LMCache v0.5.4 | This project | Result |
|---|---|---|---|
| Operating system | Linux | Native Windows | Primary incompatibility |
| Python | 3.10-3.13 | 3.13.14 | Compatible |
| PyTorch baseline | 2.13.0 | 2.13.0+cu130 | Aligned |
| Published wheels | manylinux | win_amd64 required | No compatible wheel |
| Published CUDA packages | CUDA 13.0 primary, CUDA 12.9 secondary | CUDA 13.0 | CUDA aligned; OS wheel is not |
| vLLM connector API | Supported | Present in local vLLM 0.27.1 source | Architecturally compatible |
| Recommended MP transport | POSIX SHM/CUDA IPC | Native Windows | Requires redesign |
| NIXL/GDS/RDMA paths | Linux/datacenter oriented | Local Windows workstation | Defer or omit |

The lack of a Windows wheel is not merely a packaging omission. Examples of
porting boundaries found in LMCache include:

- POSIX and Linux headers in the native allocator.
- `mmap`, `munmap`, `shm_open`, `fcntl`, Linux hugepage, and NUMA behavior.
- POSIX shared memory in the multiprocess implementation.
- Linux-oriented compiler flags and native dependency assumptions.
- NIXL and GDS packages without an equivalent native Windows distribution.

LMCache contains a Python fallback for a meaningful portion of its memory-copy
operations. That makes a limited port technically plausible, but package
requirements, imports, multiprocess behavior, and performance would still need
Windows-specific work.

Running LMCache in WSL does not transparently add it to a native Windows vLLM
process. Its shared-memory and CUDA-IPC assumptions require both vLLM and
LMCache to run in the same compatible Linux environment. Running the whole
stack in WSL is a valid Linux deployment, but it bypasses the purpose of this
native Windows project.

## Relevant Capabilities Already in vLLM 0.24

The build workspace is:

```text
E:\vllm-windows-build-v2
```

The vLLM source under that workspace already contains:

- `KVTransferConfig`
- `kv_connector_module_path` for external connectors
- `LMCacheConnectorV1`
- `LMCacheMPConnector`
- `--kv-offloading-size`
- `--kv-offloading-backend native|lmcache`
- `OffloadingConnector`
- `SimpleCPUOffloadConnector`
- CPU offload cache policies, including LRU and ARC in the full connector
- connector metrics and asynchronous transfer scheduling

The relevant factory and implementations live under:

```text
vllm/distributed/kv_transfer/kv_connector/
vllm/distributed/kv_transfer/kv_connector/v1/
vllm/v1/kv_offload/
vllm/v1/simple_kv_offload/
```

### Full native `OffloadingConnector`

The normal `native` backend maps to `OffloadingConnector`. It already provides
many of the ideas we wanted: block-based CPU storage, asynchronous operations,
metrics, and LRU/ARC policies. Its `SharedOffloadRegion`, however, currently
uses:

```text
/dev/shm/vllm_offload_<instance>.mmap
mmap.MAP_SHARED
mmap.PROT_READ | mmap.PROT_WRITE
mmap.madvise(...)
```

Those assumptions were Linux-specific. The v9 patch now selects a temporary
file plus `mmap.ACCESS_WRITE` on Windows, skips `madvise`, and waits by path so
a joining process cannot block the creator's initial resize.

### `SimpleCPUOffloadConnector`

When `VLLM_USE_SIMPLE_KV_OFFLOAD=1`, the same `native` configuration selects
`SimpleCPUOffloadConnector` instead. It:

- Requires vLLM prefix caching to be enabled.
- Allocates CPU KV blocks with ordinary PyTorch CPU tensors.
- Registers those buffers with `cudaHostRegister` when pinning is available.
- Uses asynchronous CUDA streams.
- Uses `cuMemcpyBatchAsync` resolved through CUDA Python bindings.
- Maintains an LRU-style CPU block pool.
- Understands ordinary and hybrid cache storage according to its current
  source design.
- Is included in the v0.25.1 Windows wheel.

The initial Windows experiment now proves the basic path, with a Windows DMA
fallback. Items still requiring broader validation include:

- Performance of the per-region `cudaMemcpyAsync` fallback versus a future
  safe Windows batch-copy implementation.
- `cudaHostRegister` success for the requested allocation.
- Cleanup and unregistration behavior on normal shutdown, cancellation, and
  failure.
- Correct handling of Windows threads and vLLM worker lifecycle.
- Correct transfer of every cache layout used by this build.
- Behavior under memory pressure and a small/disabled Windows pagefile.

## Recommended Architecture

The target architecture should grow in independent tiers:

```text
Request prefix hash and compatibility namespace
                    |
             GPU prefix cache
                    |
          asynchronous block transfer
                    |
        pinned system-RAM block store
                    |
         optional NVMe block store
                    |
  optional shared local/remote service later
```

The scheduler should always choose the fastest valid source:

1. Use a GPU-resident match when available.
2. Otherwise load a compatible CPU-resident match.
3. Otherwise load a compatible NVMe-resident match, if enabled.
4. Otherwise run normal prefill and optionally store the result.

Each slower tier must remain optional. A failure in a secondary tier should
fall back safely to recomputation rather than corrupt a request or crash the
server.

## Cache-Key and Namespace Correctness

Cache keys are a correctness boundary, not just a performance detail.

This project's custom cache formats are represented physically as
`torch.uint8`. That means `isoquant3`, `isoquant4`, `planarquant*`, local
TurboQuant, upstream TurboQuant, and possibly other packed layouts can look
identical if a key records only the PyTorch dtype. Loading bytes encoded by one
scheme into another scheme would be invalid.

Every reusable-cache namespace or entry must identify at least:

- Model identity and a stable model revision/fingerprint.
- Token-prefix hash and hash algorithm/version.
- Tokenizer and relevant prompt/template identity where necessary.
- vLLM cache block size and hash block size.
- Tensor layout and cache schema version.
- Original vLLM `kv_cache_dtype` string, not only `torch.dtype`.
- Compression recipe and recipe version.
- Number of KV heads and head dimension.
- Tensor-parallel/world-size/rank configuration.
- Adapter/LoRA identity when adapters affect model outputs.
- Connector storage-format version.

Until the namespace is complete, every cache format should use a separate
storage namespace or directory. Storage from different models or cache formats
must never be mixed merely because both use `uint8`.

Persistent entries also require a versioned manifest and atomic publication.
Partially written or stale entries must be rejected and safely deleted.

## Interaction With Multi-TurboQuant

The local Multi-TurboQuant modes keep the standard vLLM cache shape and store
packed bytes in `torch.uint8`; active blocks are decoded before attention. This
suggests that a raw byte-oriented offloader may eventually copy and restore
their physical buffers without interpreting the compression.

That is promising, but unproven. The safe sequence is:

1. Validate offloading with `kv_cache_dtype=auto`.
2. Confirm cache-hit output against an offload-disabled deterministic baseline.
3. Inspect byte-for-byte round trips for one packed method.
4. Test `isoquant4` first with a strictly isolated namespace.
5. Add one format at a time and retain independent compatibility tests.

Do not initially stack LMCache's TurboQuant serialization on this project's
already packed cache. LMCache's TurboQuant serde compresses data for storage
and restores ordinary KV tensors before attention. It is different from this
project's persistent compressed attention cache and would introduce a second
compression/decompression layer.

Potential longer-term synergy is real: if packed cache bytes can be stored
directly, CPU capacity increases and PCIe traffic decreases. That must be
demonstrated through correctness and transfer benchmarks rather than assumed.

## Implementation Plan

### Phase 0: Preserve a Reproducible Baseline

Status: completed and retained as an ongoing release requirement.

- Work in `E:\vllm-windows-build-v2` as the build/prototype workspace.
- Keep the tracked repository as the source of patches, tests, launcher code,
  and documentation.
- Record exact vLLM, Python, PyTorch, CUDA, Triton, driver, model, GPU, cache
  dtype, and launch settings for each result.
- Capture offload-disabled cold and warm-prefix baselines before changing code.
- Use deterministic decoding for correctness comparisons.

### Phase 1: Test Existing Simple CPU Offload Without Reimplementing It

Status: completed for the Qwen3 14B `auto`-dtype test on 2026-07-19.

Start with the existing wheel/source path:

```text
VLLM_USE_SIMPLE_KV_OFFLOAD=1
enable_prefix_caching=True
kv_offloading_backend="native"
kv_offloading_size=<small GiB value>
kv_cache_dtype="auto"
```

The exact launcher wiring must be confirmed against `LLM()` in the pinned vLLM
version. Upstream `vllm serve` already exposes the corresponding CLI options.

Initial capacity should be deliberately small, such as 1-2 GiB, so allocation,
pinning, lookup, eviction, and cleanup can be observed without committing a
large host buffer. Increase only after the lifecycle is stable.

Phase 1 success criteria:

- Server starts on native Windows.
- CPU buffers allocate and pin successfully, or clearly fall back when pinning
  is unavailable.
- First request reports a miss and stores blocks.
- Repeated compatible prefix reports a hit and restores blocks.
- Generated tokens match the no-offload baseline.
- Repeated long-prefix TTFT improves enough to exceed transfer overhead.
- Shutdown, client cancellation, and cache reset complete without a crash,
  leaked pinned memory, or a stuck worker.

### Phase 2: Windows Hardening and Launcher Support

Status: completed for v0.25.1. The launcher exposes the full native
`OffloadingConnector` as an experimental opt-in and keeps it disabled by
default.

Suggested configuration surface:

```text
--kv-offload disabled|cpu-lru|cpu-arc|fs-lru|fs-arc
--kv-offload-cpu-gb <GiB>
--kv-offload-fs-root <directory>
--kv-offload-read-threads <count>
--kv-offload-write-threads <count>
```

The launcher translates this Windows-facing interface into the exact vLLM
`KVTransferConfig` arguments for the pinned release, validates incompatible
combinations before model loading, and automatically enables prefix caching.
It logs the selected mode/capacity and warns about unbounded filesystem use.
The upstream connector metrics retain:

- Selected connector.
- CPU capacity.
- Cache format and namespace.
- Whether host pinning succeeded.
- Cache hit/miss/store/load/eviction counters.
- Bytes moved and transfer time where available.

If the existing simple connector needs fixes, prefer small Windows guards and
portable fallbacks over forking the complete LMCache stack.

### Phase 3: Cache-Format Compatibility

Status: ordinary `auto` KV is validated. Compressed active-cache combinations
remain future work and must not be inferred from the ordinary-cache result.

Run the same test suite with:

1. `auto`
2. an upstream, supported low-precision format
3. `isoquant4`
4. the remaining local packed formats one at a time

For each method, verify:

- Physical bytes before store and after restore.
- Tensor shape, stride, dtype, block ordering, and layer ordering.
- Output tokens/logits against an offload-disabled baseline.
- Prefix hits across different request lengths and partial final blocks.
- No cross-format cache collision.
- Memory savings and PCIe bytes actually match expectations.

### Phase 4: Windows NVMe Tier

Status: the first filesystem/NVMe-capable tier is implemented and validated for
forced eviction/restore. It uses atomic temporary-write then replace,
configuration-hashed namespaces, bounded read/write thread pools, and LRU/ARC
in the RAM primary tier.

Requirements:

- Configurable cache root. An automatic byte quota is still required.
- Atomic temporary-write then publish behavior.
- Versioned manifest and compatibility namespace.
- Checksums or another corruption detector.
- LRU/ARC-style eviction.
- Cleanup of incomplete entries after a crash.
- Safe behavior with NTFS, long paths, antivirus scanning, and low free space.
- Bounded concurrency so disk I/O does not starve inference.
- A clear distinction between volatile cache and user model files.

The current E: drive had roughly 702 GiB free at investigation time, but disk
capacity is dynamic and must always be checked at runtime.

### Phase 5: Persistence and Local Sharing

Status: reopen/reuse after a server-process restart is validated with a fixed
`PYTHONHASHSEED=0`. Simultaneous multiprocess sharing and administration remain
future work.

Remaining possibilities:

- A Windows named-pipe or TCP local cache service.
- Shared cache access between the RTX 3060 and RTX 3090 engines.
- Per-model quotas and cache administration endpoints.
- Explicit invalidation and cache inspection tools.

Sharing requires leases, process-crash recovery, atomic ownership, and strict
namespace validation. It should not begin with a direct translation of POSIX
shared-memory code.

### Phase 6: Optional Remote Backends

Portable client libraries could later support a remote object store, Redis-like
service, or a purpose-built cache server. This is useful only after local cache
semantics are stable. Network transport, authentication, integrity, admission
control, and eviction become part of the security and correctness boundary.

## Feature Scope Compared With LMCache

| Capability | v0.25.1 status | Later possibility | Out of current scope |
|---|:---:|:---:|:---:|
| In-process CPU RAM offload | Validated |  |  |
| Repeated-prefix KV reuse | Validated |  |  |
| LRU/ARC RAM capacity policies | Validated |  |  |
| Hit/miss/transfer metrics | Included | More launcher/UI exposure |  |
| Asynchronous CPU/GPU transfer | Validated through vLLM |  |  |
| Filesystem/NVMe persistence | Validated, experimental | Quota/administration |  |
| Local multiprocess sharing |  | Yes |  |
| Sharing between both local GPUs |  | Yes |  |
| Remote storage |  | Separate future project |  |
| Compressed raw-byte offload |  | Yes, after validation |  |
| CacheBlend/non-prefix reuse |  | Research only |  |
| Disaggregated prefill cluster |  |  | Yes |
| NIXL/RDMA/InfiniBand |  |  | Yes |
| GPUDirect Storage parity |  |  | Yes |
| Full LMCache API/feature parity |  |  | Yes |

The goal is to capture the high-value native Windows workstation use cases,
not to promise a Windows clone of every LMCache datacenter feature.

## Test Hardware

Hardware observed on 2026-07-17:

| GPU | VRAM | Recommended role |
|---|---:|---|
| NVIDIA GeForce RTX 3060 | 12 GiB | Unit/integration development, small-model correctness, constrained-memory tests |
| NVIDIA GeForce RTX 3090 | 24 GiB | Qwen3 14B baseline, long-prefix testing, final performance measurements |

At the time of inspection the RTX 3060 had about 10.1 GiB free and the RTX
3090 was free. Those numbers are only a snapshot.

The 3060 is sufficient to begin. Tests must explicitly select GPU 0 so they do
not unexpectedly consume the 3090 while it is in use. The 3090 is preferable
for final performance results because it has enough room for a proven
full-attention model and a useful GPU KV pool.

The machine had approximately 64 GiB of installed system RAM and 46.6 GiB free
at inspection time. Begin with a small CPU-cache allocation rather than the
connector's larger defaults.

## Available Hugging Face Test Models

Model root:

```text
E:\vllm-windows-build-v2\models
```

The following are real Hugging Face/Safetensors directories with configuration,
tokenizer files, index files, and all referenced shards present:

| Model | Weight size | Notes |
|---|---:|---|
| `Qwen3.5-9B-abliterated-GPTQ-4bit` | 7.15 GiB | Smallest local generative model; hybrid Qwen3.5 architecture; tight but plausible on 3060 |
| `Qwen3-14B-abliterated-AWQ-4bit` | 9.29 GiB | Proven project baseline; conventional full attention; preferred first meaningful cache test on 3090 |
| `Qwen3.5-27B-abliterated-GPTQ-3bit` | 13.71 GiB | Later stress test |
| `Qwen3.5-27B-abliterated-GPTQ-4bit` | 16.57 GiB | Conditional-generation configuration; later compatibility test |

The 9B and 14B directories contain `turboquant_kv.json` metadata.

Recommended model order:

1. Use a small full-attention Hugging Face model for the fastest connector
   correctness loop if one is added locally.
2. Use Qwen3 14B on the 3090 for the first meaningful, already-proven baseline.
3. Use Qwen3.5 9B for hybrid-cache-manager compatibility and constrained 3060
   testing.
4. Use the 27B models only after the basic path is stable.

`E:\LL STUDIO` is not part of this test inventory. Almost all models there are
GGUF files intended for LM Studio/llama.cpp. The only Safetensors model found
there was an embedding model, not an appropriate generative KV-cache test.

## Test Matrix

Every functional run should compare offload disabled versus enabled using the
same model, prompt, seed, decoding settings, cache format, and GPU.

### Correctness

- Cold request with no matching prefix.
- Exact repeated prefix.
- Shared prefix followed by different suffixes.
- Prefix ending on a complete cache block.
- Prefix ending on a partial block.
- Requests shorter and longer than the CPU-cache capacity.
- Cache eviction followed by recomputation.
- Cache reset while idle.
- Client cancellation during prefill, decode, store, and load.
- Normal shutdown and immediate restart.
- Invalid/stale namespace rejection.
- Concurrent requests with overlapping and unrelated prefixes.

Validation should compare generated token IDs and, where practical, selected
logits. A speedup with different output is a failure.

### Performance

Record at minimum:

- Cold TTFT.
- Warm GPU-prefix-cache TTFT.
- Warm CPU-offload-cache TTFT.
- Prefill tokens per second.
- Decode tokens per second.
- GPU memory used and GPU KV capacity.
- CPU memory committed and pinned.
- Store/load bytes and duration.
- Cache hits, misses, evictions, and rejected entries.
- CPU usage and disk throughput for later tiers.

Test prompt sizes should include short prompts, a crossover range, and long
prefixes. The exact crossover point where RAM restore beats recomputation is a
benchmark result, not a fixed assumption.

### Stability

- Run repeated hit/miss cycles long enough to detect pinned-memory leaks.
- Repeatedly start and stop the server.
- Disconnect clients during active transfers.
- Exhaust the configured cache and force eviction.
- Run with the 3090 unavailable to ensure GPU selection remains isolated.
- Test with conservative Windows pagefile settings because this project
  explicitly supports systems sensitive to committed host memory.

Current LMCache issue reports involving vLLM 0.24 client disconnects and MP
server restarts reinforce the need for these lifecycle tests even if the
Windows implementation does not reuse LMCache's affected code directly.

## Wheel and Distribution Implications

### v0.25.1 requires the rebuilt wheel

The first simple-connector DMA proof changed only packaged Python and could be
reassembled without recompiling. The finished v0.25.1 feature also includes the
new upstream base, native `fs_io_C.pyd`, C++ filesystem changes, and other
compiled Windows fixes, so the full v0.25.1 native build was required.

The final artifact contract is:

```text
vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
293080424 bytes
SHA256 0C4F9B2E36482523FC7B4C092D711AC49B4265EF9F36A7AEEFFF9A667C875339
```

The installer, launcher marker, assembler, wheel validator, tests, and release
documentation are updated together. The launcher remains disabled-by-default
even though the implementation is present in the wheel.

### A rebuilt vLLM wheel is needed when

- vLLM's packaged Python implementation is patched directly,
- internal connector APIs are changed,
- C++ or CUDA transfer code is added,
- the Windows shared-memory implementation is integrated inside vLLM, or
- a release needs to ship modified internal files rather than an external
  module.

### External connector alternative

vLLM 0.25.1 supports `kv_connector_module_path`. A future external connector can
therefore live in this repository or in a separate pure-Python package while
the design is experimental. That can avoid rebuilding the compiled wheel.

If it later includes native C++/CUDA acceleration, it could become a separate
version-pinned extension wheel or be folded into the vLLM Windows wheel. Any
native artifact must match CPython 3.13, PyTorch 2.11, CUDA 12.8, and win_amd64.

## Repository and Build Workflow

Source build and test work belongs in the separate v2 workspace:

```text
E:\vllm-windows-build-v2\vllm-source-v0.25.1
```

Durable changes must then be represented in the tracked repository:

- Update the versioned Windows patch when vLLM internals change.
- Update `vllm_launcher.py` and `launch.bat` for user-facing configuration.
- Add focused unit and integration tests under `tests/`.
- Update installer dependencies only when actually required.
- Rebuild and verify the wheel only when packaged vLLM files change.
- Document experimental status and safe defaults.
- Never silently enable host-memory reservation in an existing installation.

The feature remains opt-in while broader workload, lifecycle, compressed-cache,
and multi-GPU-machine coverage is accumulated.

## Licensing and Credit

LMCache is Apache-2.0 licensed. vLLM source is also Apache-2.0, while this
repository uses the MIT license for its own material.

Architectural ideas may be independently adapted and LMCache should be credited
as the inspiration. If actual LMCache or vLLM source is copied or modified:

- Preserve SPDX and copyright headers.
- Preserve the applicable Apache-2.0 license and NOTICE obligations.
- Mark substantial modifications where appropriate.
- Keep attribution in documentation and source comments.
- Avoid implying that the Windows adaptation is an official LMCache release or
  is supported by the LMCache maintainers.

This project does not need to avoid code reuse merely to retain credit. It does
need to keep the provenance and licensing of reused files explicit.

## Risks and Guardrails

### Correctness risks

- Cache-format collisions among layouts that all use `torch.uint8`.
- Partial-block or hybrid-cache-group ordering mistakes.
- Loading stale data from a different model, adapter, or build.
- Publishing partially written persistent entries.
- Reusing GPU blocks before an asynchronous store completes.

### Stability risks

- Pinned host-memory leaks.
- Failure to unregister CUDA host buffers.
- Client cancellation during active transfers.
- Worker exit or restart while a background thread owns CUDA objects.
- Windows pagefile/commit failures.
- Driver support differences for CUDA batch-copy APIs.

### Performance risks

- CPU/GPU transfer slower than prefix recomputation.
- Storing one-use prefixes that are never read again.
- Excessive hashing or bookkeeping for short requests.
- Disk I/O competing with model loading or Windows antivirus scanning.
- Double compression when combining incompatible serde and active-cache modes.

Guardrails:

- Disabled by default.
- Explicit RAM capacity and filesystem root; add an automatic disk limit before
  treating filesystem mode as a general default.
- Safe recomputation fallback on miss or tier failure.
- Versioned namespaces.
- Admission threshold so one-use blocks can be skipped.
- Clear telemetry instead of silent behavior.
- Exact dependency/version pinning during development.

## Decision Record

The current decisions and completed milestones are:

1. Do not vendor or install the complete LMCache package in the native Windows
   release today.
2. Development could begin on the RTX 3060, but final low-level and real-model
   validation ran on the available RTX 3090.
3. vLLM's existing `SimpleCPUOffloadConnector` was tested before the full
   connector; no separate replacement connector was needed.
4. In-process pinned CPU RAM was proven first, followed by the filesystem tier.
5. Begin with `kv_cache_dtype=auto`.
6. Treat cache tier as separate from cache compression format.
7. Require a cache-format-aware namespace before testing packed `uint8` modes.
8. Test `isoquant4` first after the ordinary cache path is proven.
9. Do not stack LMCache TurboQuant serde on already compressed active caches in
   the initial design.
10. Filesystem/NVMe persistence was added only after RAM offloading worked and
    is still experimental because it lacks an automatic quota.
11. Do not target full LMCache feature parity; prioritize native Windows
    workstation use cases.
12. Preserve Apache-2.0 attribution and notices for any adapted source.

## Next Actions

Following the v0.25.1 implementation, later work should:

1. Run longer latency distributions and sustained hit/evict cycles rather than
   relying on focused correctness timings.
2. Stress shutdown, client cancellation, disk-full behavior, and pinned-memory
   cleanup.
3. Add an explicit filesystem byte quota, cleanup tooling, and clearer cache
   administration before considering a broader default.
4. Test compressed active-cache formats one at a time with exact layout and
   output checks.
5. Keep Qwen3.5 hybrid-cache behavior covered as upstream registry/config APIs
   change; the 0.27.1 model path already works in targeted tests.
6. Benchmark and optimize the Windows DMA choices only if profiling identifies
   transfer submission overhead as material.

## Primary References

- [LMCache repository](https://github.com/LMCache/LMCache)
- [LMCache v0.5.1 release](https://github.com/LMCache/LMCache/releases/tag/v0.5.1)
- [LMCache installation requirements](https://docs.lmcache.ai/getting_started/installation.html)
- [LMCache vLLM quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)
- [LMCache TurboQuant serde design](https://github.com/LMCache/LMCache/blob/v0.5.1/docs/design/v1/distributed/serde/turboquant.md)
- [LMCache packaging configuration](https://github.com/LMCache/LMCache/blob/v0.5.1/pyproject.toml)
- [LMCache issue #3688: vLLM 0.24 disconnect crash report](https://github.com/LMCache/LMCache/issues/3688)
- [LMCache issue #4069: MP restart/native worker crash report](https://github.com/LMCache/LMCache/issues/4069)
- [Project architecture](architecture.md)
- [Project TurboQuant notes](turboquant.md)
