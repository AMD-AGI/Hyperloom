# Troubleshooting

> **Release selection:** [the Windows releases](releases.md) by Python/CUDA version, installer, and source-build guidance.

Common errors when building or running the native Windows vLLM releases.

## Runtime errors

### Blackwell FP8/NVFP4 startup failures (v0.27.1) <a id="blackwell-fp8-nvfp4"></a>

> **0.29.0 and later (including the current 0.33.0):** these releases include
> the issue #16 FP8 Marlin fallback fixes, so the Python edit below is not
> needed. If an FP8/NVFP4 model still fails on Blackwell, try
> `--linear-backend marlin` in the launcher and report the full traceback.
> Native Blackwell FP4 acceleration is still not available on Windows. The
> rest of this section applies to the 0.27.1 wheel.

**The published v0.27.1 wheel needs a workaround for the mixed FP8/NVFP4
model in [issue #16](https://github.com/aivrar/vllm-windows-build/issues/16).**
The reporter confirmed `unsloth/Qwen3.8-27B-NVFP4` serving on an RTX PRO 5000
Blackwell (48 GB, SM 12.0), Windows 11 Pro, after applying the workaround below.
Compilation, CUDA graph capture, and a `/v1/chat/completions` request succeeded.
This confirmation applies to that model and configuration.

The unmodified installation can fail with:

```text
cutlass_scaled_mm_sm80_epilogue ... ScalarType::Char
```

The wheel omits native SM120 CUTLASS FP8/NVFP4 kernels because their aligned
by-value parameters fail to compile with MSVC. Its FP8 backend selection still
allows the omitted implementation, then dispatch falls through to an INT8
kernel that rejects FP8 inputs. Reducing context length does not fix this
dtype error. The model contains FP8 layers as well as NVFP4 layers.

#### Apply the existing-wheel workaround

These instructions are for **this repository's vLLM 0.27.1 Windows wheel**.
Keep its pinned Python 3.13 / PyTorch 2.13.0+cu130 environment. Stop the server
and use the same Python environment that runs it. For the portable installation,
replace `python` in the commands below with `python\python.exe` from the repo
directory.

1. **Check the environment and locate the installed package.**

   ```bat
   python -c "import importlib.metadata, importlib.util, sys; print(sys.executable); print(importlib.metadata.version('vllm')); print(importlib.util.find_spec('vllm').origin)"
   ```

   The distribution version should be `0.27.1`. The final line locates
   `vllm\__init__.py`; use that `vllm` directory for the edit below.

2. **Fix the two missing LM-head attributes.** Back up
   `model_executor\layers\quantization\compressed_tensors\schemes\compressed_tensors_w8a8_fp8.py`
   inside that installed `vllm` directory before editing it.

   Inside `create_weights`, insert the two partition-size assignments after
   `layer.logical_widths`, keeping the existing indentation. The block should
   then read:

   ```python
           output_size_per_partition = sum(output_partition_sizes)
           layer.logical_widths = output_partition_sizes
           layer.input_size_per_partition = input_size_per_partition
           layer.output_size_per_partition = output_size_per_partition
           layer.weight_block_size = None
   ```

   This fixes `AttributeError: 'ParallelLMHead' object has no attribute
   'output_size_per_partition'`; the input attribute is required too. It matches
   the fix proposed in upstream [PR #52451](https://github.com/vllm-project/vllm/pull/52451).
   If the assignments are already present, do not add duplicates. If the
   surrounding code differs, check the version and package path before editing.

3. **Force the FP8 Marlin fallback before starting the server.** In Command Prompt:

   ```bat
   set "VLLM_TEST_FORCE_FP8_MARLIN=1"
   ```

   Or in PowerShell:

   ```powershell
   $env:VLLM_TEST_FORCE_FP8_MARLIN = '1'
   ```

   Rerun your server command from the same terminal. For the direct API server,
   a launch example is:

   ```bat
   python -m vllm.entrypoints.openai.api_server --model D:\Qwen3.8-27B-NVFP4 --linear-backend marlin --host 127.0.0.1 --port 8080
   ```

   Substitute your local model directory. In v0.27.1, `--linear-backend marlin`
   alone does not bypass the FP8 capability gate; keep the environment setting.
   These version-specific instructions should not be assumed to apply to later
   vLLM releases.

Check that startup selects `MarlinFP8ScaledMMLinearKernel` and
`MarlinNvFp4LinearKernel`, then verify readiness and a completed chat request.
The reporter's working configuration removed `--enforce-eager`; eager mode is
not required by this workaround. See their
[confirmation](https://github.com/aivrar/vllm-windows-build/issues/16#issuecomment-5611096282).

**No wheel rebuild is needed for these steps.** The published wheel, installer,
and `vllm-windows-v10.patch` do not include the Python correction or an automatic
backend-selection fix. Pulling the repo or reinstalling that wheel alone does
not apply this workaround; reinstalling the package can overwrite the edit.
To undo the workaround, stop the server, restore the backed-up Python file,
and remove the environment setting from your terminal or launch script.

#### What this confirms and what remains limited

- Marlin uses FP8 or FP4 weights with 16-bit activations (W8A16/W4A16). This
  workaround does not activate native Blackwell FP8/FP4 Tensor Core computation.
- The Marlin warning that the GPU does not support FP4 is misleading on
  Blackwell: the limitation here is the wheel's available kernels. A runtime
  flag cannot enable omitted native kernels; FlashInfer is not included as a
  supported Windows replacement in this release.
- Local RTX 3090 tests passed six synthetic FP8 Marlin cases and twelve LM-head
  cases. Full-model serving with CUDA graphs was confirmed by the Blackwell
  reporter. These checks do not establish reference-model accuracy, every
  context length, other models, or MTP/speculative decoding.
- The reporter's first `Avg generation throughput: 15.5 tokens/s` log entry
  includes a logging interval during the first request, which also triggered
  JIT compilation. It is not an established steady decode rate or performance
  ceiling. Measure warm requests and compare matching model, token lengths,
  concurrency, and speculative-decoding settings.

### RTX 20xx / SM 7.5 reports an unsupported architecture or stalls during startup

The v0.27.1 release wheel includes SM 7.5 code, so this message does
not by itself mean that an RTX 2080/2080 Ti is excluded. Turing cannot use every
newer attention backend or CUDA-graph path. Start with the compatibility profile
and small resource limits:

```bat
launch.bat --model cyankiwi/Qwen3.5-9B-AWQ-4bit --gpu-id 0 --turing-compat
```

That profile selects `TRITON_ATTN`, a float16 KV cache, 32-token blocks, eager
execution, 0.89 GPU utilization, one sequence, and a 2,048-token batch cap.
These defaults were confirmed to load Qwen3.5-9B AWQ on the issue #14
reporter's 11-GB RTX 2080 Ti. If it still fails, try `--max-model-len 2048`.

`gpu_memory_utilization` is the maximum fraction vLLM is allowed to use. The
old 0.6 default gave that 8.42-GiB model a negative KV-cache budget. Raise the
value when VRAM is actually free; reduce context, sequences, and batched tokens
to reduce the cache requirement. This is the launcher path for issue
[#14](https://github.com/aivrar/vllm-windows-build/issues/14).

### An old installer says runtime 0.26.0 but expected 0.26.0+cu128

The wheel metadata is `0.26.0+cu128`, but upstream vLLM intentionally exposes
the base module version `0.26.0`. The original v0.26 verifier compared those two
different version fields directly and falsely reported installation failure.
Pull the current repository and rerun `install.bat`; the verifier now checks
the distribution build tag and runtime base version separately.
The current 0.33.0 runtime and distribution both report `0.33.0`.

### A `.gguf` model fails with UTF-8 or invalid JSON errors

Direct GGUF files are not supported by this launcher. The old model scanner
listed them, after which Transformers attempted to read the binary GGUF file as
JSON and produced a misleading UTF-8 error. Use a Hugging Face-format model
directory or repository ID with `config.json` and compatible Safetensors, AWQ,
or GPTQ weights. The current launcher rejects `.gguf` paths immediately with
this explanation.

### Filesystem KV cache is growing or is not reused after a restart

The experimental `fs-lru`/`fs-arc` tier has no automatic disk quota. Point
`--kv-offload-fs-root` at a dedicated directory, monitor it, and remove that
directory when you want to clear the persistent blocks.

Cross-process reuse also requires `PYTHONHASHSEED=0` to be present before the
Python process starts. `launch.bat` sets it automatically. For a direct launch:

```bat
set PYTHONHASHSEED=0
python vllm_launcher.py --model E:\models\Qwen3-14B-AWQ-4bit --kv-offload fs-lru --kv-offload-fs-root E:\vllm-kv-cache
```

The model identifier, tensor/cache configuration, and block size participate
in the cache namespace. A changed configuration correctly creates a different
namespace rather than reusing incompatible data.

### Illegal memory access while loading offloaded KV blocks

Install the current release wheel. The Windows patch routes restores from
file-backed mmap through native CUDA DMA; the earlier Triton host-pointer route
could fault for some grouped block shapes. If the problem persists, verify the
installed wheel hash and run `python verify_install.py` before collecting a
minimal reproduction.

### `ValueError: low is out of bounds for int32` during request sampling

This was the follow-up failure in issue #10. vLLM asks NumPy for a random seed
across the full signed 64-bit range. NumPy's legacy `randint` API defaults to a
C `long`, which is still 32-bit on 64-bit Windows, so the lower bound was
rejected before a request could run.

Pull the latest repository and rerun `install.bat`. The patched call now
requests `dtype=np.int64` explicitly. This failure is separate from the
original Python/Triton error in issue #10: changing to Python 3.10, 3.11, or
3.12 alone is not a proven fix for this Windows integer-width bug.

### A small model generates at only `0.01`-`0.3` tokens/s

Check the startup line beginning with `non-default arg`. If it contains
`kv_cache_dtype: 'isoquant4'` (or another local Multi-TurboQuant dtype), the
engine is using the current unfused PyTorch encode/decode fallback. That path
reduces KV-cache memory but is intentionally 30-300× slower than the normal
`auto` baseline. It is for offline or memory-constrained workloads, not a
Hello World performance test.

Use the fast baseline first:

```python
llm = LLM(
    model=r"E:\models\Qwen2.5-0.5B-Instruct",
    dtype="float16",
    kv_cache_dtype="auto",
    max_model_len=512,
    gpu_memory_utilization=0.5,
)
params = SamplingParams(temperature=0.0, max_tokens=32, seed=0)
```

Measure a second request in the same process after one-time JIT and CUDA-graph
setup. `enforce_eager=True` can help isolate a graph/compile compatibility
problem, but it disables those optimizations and should not be treated as the
throughput default. In vLLM's progress display, the final `output` rate is the
generation rate; a very low `input` rate can simply reflect the full elapsed
request time.

### `Get-FileHash` is not recognized

This was issue #9. Some Windows PowerShell environments do not expose the
`Get-FileHash` cmdlet, so the old pre-Python integrity check stopped immediately
after downloading the embedded Python archive.

Pull the latest repository and rerun `install.bat`. Bootstrap hashing now uses
the .NET SHA-256 implementation in `verify_bootstrap.ps1`; ZIP extraction uses
`expand_zip.ps1` instead of `Expand-Archive`. No manual PowerShell module install
is required. Windows PowerShell 3 or newer is required because the downloader
uses `Invoke-WebRequest`.

### Wheel SHA-256 shows a blank `Actual:` value

This was issue #8. The old batch installer parsed `Get-FileHash` through a
`for /f` command substitution; on some Windows configurations the command
succeeded but its stdout was not captured, leaving the digest blank and
rejecting a valid wheel.

Pull the latest repository and rerun `install.bat`. Hashing now runs through
`verify_artifact.py`, reports size and SHA-256 directly, and replaces stale or
truncated wheels automatically. The historical v0.25.1 wheel affected by the original report is 293,080,424 bytes
with SHA-256:

```text
0C4F9B2E36482523FC7B4C092D711AC49B4265EF9F36A7AEEFFF9A667C875339
```

### `Failed to find Python libs` / `Python.h not found`

Triton compiles a small CUDA driver helper the first time some kernels are
loaded. With the portable `install.bat` setup, this fails if the embedded
Python directory has no CPython development files:

```text
triton\windows_utils.py:302: UserWarning: Failed to find Python libs.
cuda_utils.c:15: error: include file 'Python.h' not found
```

This is not a model architecture problem, even if vLLM wraps it as
`Error in inspecting model architecture`.

Fix: pull the latest repo and run either `launch.bat` or `install.bat`.
Both paths now repair an existing portable Python 3.13 install by
copying `Include\Python.h` and `libs\python313.lib` from the Python
3.13.14 NuGet package. If your `python\` directory is from an older
major/minor Python version, delete `python\` and rerun `install.bat`.

### `FAILED: Triton CUDA runtime check failed.`

`install.bat` verifies that Triton can initialize its CUDA driver helper
before reporting success. If this check fails:

- Confirm the NVIDIA driver is installed and the GPU is visible.
- Pull the latest repo and rerun `install.bat` so the portable Python
  dev files are present.
- If the portable `python\` directory came from an older release, delete
  it and rerun `install.bat`.

The launcher pins Triton to its bundled CUDA helper toolkit when present,
so a mismatched system `CUDA_PATH` should not be needed for wheel installs.

### `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'`

The original v0.24.0 Windows wheel omitted Python files generated by the
FlashAttention build. Text-only models could start successfully, while
Qwen3-VL failed during multimodal profiling when it imported the rotary
implementation.

Pull the latest repository and rerun `install.bat`; it force-reinstalls the
corrected wheel even though its version is unchanged. For a manual install,
force-reinstall the wheel attached to the `v0.33.0-win-cu130`
release. Its SHA256 is:

```text
28C45D24E246D515D1AE87E8CA286CF7D3B1DADBD8AEC4FEB685B01B93627321
```

Verify the repaired import with:

```bat
python -c "from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb; print('FlashAttention rotary OK')"
```

### `ImportError: DLL load failed while importing _C: The specified module could not be found.`

vLLM's compiled `_C.pyd` extension can't find its CUDA / torch DLLs.
The fix is to add the CUDA bin and the torch lib directories to the
Python DLL search path **before** importing vLLM:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin")
os.add_dll_directory(r"E:\path\to\venv\Lib\site-packages\torch\lib")
import vllm
```

The test scripts in `tests/` already do this. If you're embedding vLLM
in your own code, copy the pattern.

### `OSError: The paging file is too small for this operation to complete. (os error 1455)` <a id="oserror-1455"></a>

Windows uses *commit charge* (RAM + pagefile) to back any process
allocation. If your pagefile is small or set to zero, even a process
with plenty of free physical RAM can fail to allocate large buffers
because the system commit limit is exhausted.

This shows up most often when loading large model weights — the
embedding tensor of a 14B model is ~1.5 GB and needs a contiguous
allocation.

**Fix 1**: enable a system pagefile.
1. Win+R → `sysdm.cpl` → Advanced → Performance → Settings → Advanced → Virtual memory → Change
2. Uncheck "Automatically manage paging file size for all drives"
3. Pick a drive with ≥16 GB free, choose "System managed size"
4. **Reboot**

**Fix 2**: this build's custom safetensors reader (in
`vllm/model_executor/model_loader/weight_utils.py`) bypasses the
problem by using `numpy.memmap` (file-backed, no commit charge) plus
chunked GPU streaming. It's already enabled — if you're seeing this
error you may have an older patched build. Re-apply
`vllm-windows-v12.patch` to a clean upstream v0.33.0 tree, or install the
published wheel.

### `torch.OutOfMemoryError: CUDA out of memory. ... X GiB is free` <a id="oom-with-free-gpu"></a>

PyTorch's caching allocator may be unable to satisfy an allocation even when
the GPU appears to have free memory. On Windows, PyTorch reports
`expandable_segments not supported on this platform`, so the commonly shared
allocator setting does not fix this condition. Clear it if it is present:

```bat
set PYTORCH_CUDA_ALLOC_CONF=
```

Then lower vLLM's reservation and concurrency settings:

1. Lower `gpu_memory_utilization` (try 0.5, then 0.4).
2. Reduce `max_num_seqs` and `max_model_len`.
3. Close other GPU processes and keep the Windows pagefile enabled.
4. Use a compressed KV dtype only when its documented throughput trade-off is
   acceptable.

### `ValueError: not enough values to unpack (expected 2, got 1)` in `torch.unique`

You're running a mismatched PyTorch + vLLM combo. The v0.33.0 release wheel
in this repo expects Python 3.13 and PyTorch 2.14.0+cu130. Reinstall with
`install.bat`, or in a manual venv reinstall:

```bat
pip install torch==2.14.0 torchaudio==2.11.0 torchvision==0.29.0 ^
    --index-url https://download.pytorch.org/whl/cu130
pip install --force-reinstall dist-v0.33.0\vllm-0.33.0-cp313-cp313-win_amd64.whl
```

### `pyo3_runtime.PanicException: Python API call failed`

safetensors crashed inside vLLM's engine context. This is the same
root cause as OSError 1455 — Windows commit limit. Enable a pagefile
or use the patched safetensors reader. See above.

### `Segmentation fault` during model loading

Same root cause family as the two errors above. Apply the
[OSError 1455](#oserror-1455) fix.

## Build errors

### `CMake Error: Generator: Visual Studio 17 2022 does not match the generator used previously: Ninja`

There are stale CMake caches in `vllm-source/.deps/` from a previous
build that used a different generator. Clean them:

```bat
rmdir /s /q vllm-source\.deps
del /s /q vllm-source\build
```

Then re-run `build.bat`.

### `cl : error C2018: unknown character 'or' / 'and' / 'not'`

MSVC doesn't accept `or` / `and` / `not` as keywords by default. The
patch fixes every known instance - make sure `vllm-windows-v12.patch`
applied cleanly:

```bat
cd vllm-source
git checkout v0.33.0
git apply --check ..\vllm-windows-v12.patch
```

If the check fails, the patch is partially applied or the source has
been modified. Reset:

```bat
cd vllm-source
git checkout v0.33.0
git reset --hard v0.33.0
git apply ..\vllm-windows-v12.patch
```

### `nvcc fatal: Unsupported gpu architecture 'compute_120'`

Your CUDA toolkit doesn't support Blackwell (SM 12.0). Either:
- Lower `TORCH_CUDA_ARCH_LIST` (drop `12.0` from the list)
- Or install CUDA 13.0 Update 2 to reproduce the current build. CUDA 12.8 was
  the minimum used by the older cu128 releases.

### `error C1061: compiler limit: blocks nested too deeply`

You hit MSVC's nested-block limit on the auto-generated Marlin kernel
selector. The patch converts these from `else if` chains to flat `if`
chains, which avoids the limit. If you still see this, the patch isn't
applied — see the previous section.

### `catastrophic error: out of memory` from nvcc

nvcc ran out of memory during template instantiation. This is most
common on the Marlin kernel files. Lower `MAX_JOBS`:

```bat
set MAX_JOBS=2
```

Or `MAX_JOBS=1` on a 16 GB RAM machine.

### `is not able to compile a simple test program. ... rc /fo ... no such file or directory`

The Windows SDK Resource Compiler (`rc.exe`) isn't on PATH. The build
needs both MSVC and the Windows SDK 10.0.19041 or newer. Open a
"Developer Command Prompt for VS 2022" and run `where rc.exe` — if
nothing is found, install the Windows SDK component in the VS Installer.

### `ImportError: cannot import name 'X' from 'vllm.v1...'`

The patch is partially applied. Run:

```bat
cd vllm-source
git status
```

to see what's modified. If the modifications don't match the patch's
expected changes, reset and re-apply:

```bat
cd vllm-source
git checkout .
cd ..
build.bat
```

## Runtime warnings (safe to ignore)

### `WARNING: Distributed backend nccl is not available; falling back to fake.`

NCCL doesn't ship with PyTorch on Windows. The patch wires up
`FakeProcessGroup` so single-GPU operation still works. This warning
is expected.

### `triton kernels are Linux-only. Pure PyTorch fallbacks will be used on Windows.`

This is from `multi_turboquant`. The build uses `triton-windows`
(separate package, ships only Triton) so Triton itself does work. The
warning is misleading — your build does have Triton kernels.

### `Failed to compute shorthash for libnvrtc.so`

`libnvrtc.so` is a Linux library; Windows uses `nvrtc64_*.dll`. Harmless.
