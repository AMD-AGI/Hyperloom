# Tests

Before publishing a rebuilt wheel, validate its complete ZIP payload and RECORD:

```powershell
python tests/test_wheel_contents.py E:/vllm-windows-build-v2/dist-v0.27.1/vllm-0.27.1-cp313-cp313-win_amd64.whl
```

This check includes the generated FlashAttention rotary and CuteDSL modules,
native/Rust extensions, issue #10's explicit NumPy `int64` request-seed fix,
and the Windows DMA/mmap/filesystem/block-table fixes required by tiered KV
offload. It also validates ZIP integrity and every wheel RECORD entry.

Run the installer/hash and concurrent engine-dispatch regressions with:

```powershell
python -m unittest tests.test_verify_artifact tests.test_bootstrap_helpers tests.test_engine_dispatcher tests.test_release_contract tests.test_launcher_configuration -v
```

The bootstrap suite specifically runs the PowerShell helpers without
`Get-FileHash`/`Expand-Archive`, checks paths with spaces and overwrite repair,
and verifies that a ZIP path-traversal entry is rejected.
The launcher configuration suite verifies the general defaults, the exact
RTX 2080 Ti profile, explicit override precedence, and early GGUF rejection
without importing vLLM or reserving GPU memory.

All six Multi-TurboQuant write/decode paths have a small CUDA integration test:

```powershell
python tests/test_multi_turboquant_integration.py
```

Windows process cleanup and Unix-socket guards are covered by:

```powershell
python -m unittest tests.test_windows_runtime_guards -v
```

Issue #7's Qwen3-VL import and CUDA rotary path can be exercised against an
isolated `pip --target` installation with:

```powershell
python tests/test_issue7_flash_attn.py --package-root $env:TEMP\vllm-issue7-wheeltest
```

End-to-end test scripts for the Windows vLLM 0.27.1 release with
Multi-TurboQuant and experimental native KV offload.

## Setup

All tests need two environment variables:

```bat
set VLLM_MODEL_PATH=E:\path\to\Qwen3-14B-AWQ-4bit
set VLLM_PYTHON=E:\vllm-windows-build\venv\Scripts\python.exe
```

Keep the Windows pagefile enabled. Do not set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the Windows PyTorch build
reports that mode as unsupported.

## test_windows_kv_offload.py - real-model RAM offload

Run the baseline and offload modes in separate processes. The test generates a
deterministic long prompt, clears the GPU prefix cache, and requires the
offload run to restore cached tokens from pinned system RAM without changing
the generated token IDs.

This legacy focused test exercises `SimpleCPUOffloadConnector`. Prefer the
tiering harness below for release validation.

```bat
set VLLM_MODEL_PATH=E:\vllm-windows-build-v2\models\Qwen3-14B-abliterated-AWQ-4bit
set CUDA_VISIBLE_DEVICES=0

%VLLM_PYTHON% tests\test_windows_kv_offload.py --mode baseline
%VLLM_PYTHON% tests\test_windows_kv_offload.py --mode offload
```

`CUDA_VISIBLE_DEVICES` uses CUDA's ordinal order, which can differ from
`nvidia-smi` numbering on mixed-GPU machines. Confirm the `Visible GPU` line
before relying on a result.

## test_windows_kv_tiering.py - v0.27.1 release matrix

This is the main real-model harness for `OffloadingConnector`. It requires the
installed `0.27.1` wheel, exactly one visible RTX 3090, and a local
Hugging Face model. Run each mode in a fresh process.

```bat
set VLLM_PYTHON=E:\vllm-windows-build-v2\venv313\Scripts\python.exe
set VLLM_MODEL_PATH=E:\vllm-windows-build-v2\models\Qwen3-14B-abliterated-AWQ-4bit
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=GPU-36feccef-50ef-2eaf-5c0c-5448e28a4d8a

%VLLM_PYTHON% tests\test_windows_kv_tiering.py --model "%VLLM_MODEL_PATH%" --mode baseline
%VLLM_PYTHON% tests\test_windows_kv_tiering.py --model "%VLLM_MODEL_PATH%" --mode cpu-lru
%VLLM_PYTHON% tests\test_windows_kv_tiering.py --model "%VLLM_MODEL_PATH%" --mode cpu-arc
```

The UUID above records the release machine; replace it with the target 3090's
UUID from `nvidia-smi -L`. For filesystem tests, set the hash seed before
Python starts and use a fresh dedicated directory:

```bat
set PYTHONHASHSEED=0
%VLLM_PYTHON% tests\test_windows_kv_tiering.py ^
    --model "%VLLM_MODEL_PATH%" --mode fs-lru ^
    --cpu-cache-mib 512 --fs-root E:\vllm-kv-test\lru ^
    --result-json E:\vllm-kv-test\fs-lru.json

%VLLM_PYTHON% tests\test_windows_kv_tiering.py ^
    --model "%VLLM_MODEL_PATH%" --mode fs-lru ^
    --cpu-cache-mib 512 --fs-root E:\vllm-kv-test\lru ^
    --reuse-existing-cache --reference-json E:\vllm-kv-test\fs-lru.json
```

The release run passed CPU LRU/ARC, forced filesystem LRU/ARC restore, and a
fresh-process persistent restore. The forced restore reused 1,440 prompt
tokens and required exact generated token IDs; the final wheel repeated the
persistent check successfully. Filesystem cache directories have no automatic
quota and should be removed after disposable test runs.

## test_v19.py — smoke test

The fastest way to confirm the build works. Loads a model, generates a
couple of short prompts, prints them.

```bat
%VLLM_PYTHON% tests\test_v19.py
```

Should finish in well under 60 seconds for a 14B AWQ-4bit model.

## test_tq_diag.py — per-method hang diagnostic

Runs a single TQ method with a 90-second `faulthandler` watchdog. If
`generate()` stalls past the timeout, the watchdog dumps every
thread's Python stack trace and hard-exits, so you can tell a real
hang apart from a slow-but-terminating PyTorch-fallback decode.

```bat
set TQ_METHOD=isoquant3
%VLLM_PYTHON% tests\test_tq_diag.py
```

Select a method via `TQ_METHOD`: `isoquant3`, `isoquant4`,
`planarquant3`, `planarquant4`, `turboquant25`, `turboquant35`. Adjust
the watchdog with `TQ_HANG_TIMEOUT=120` (seconds) if you need to let a
slow method terminate.

Expected timings for 5 decoded tokens on an RTX 3090 with Qwen3-14B
AWQ-4bit: `turboquant25/35` finish in ~5-7s, the iso/planar family in
~40-55s. Anything past 90s is a real hang.

## test_tq_real.py — Multi-TurboQuant correctness sweep

Runs the FP16 baseline plus all 6 Multi-TurboQuant compression methods
in fresh subprocesses (so each gets a clean GPU state). Verifies that
each method:

1. Loads, runs, and generates output without crashing
2. Produces output **different** from the FP16 baseline (proves the
   compression noise is actually entering the inference pipeline)
3. Produces output **unique** from the other TQ methods (proves each
   method has its own numerical signature)

```bat
%VLLM_PYTHON% tests\test_tq_real.py
```

Takes ~15-20 minutes total (FP16 ~30s; each TQ method ~3 min). Output
is a table:

```
Method          Differs from FP16    Differs from others
isoquant3       DIFFERS              UNIQUE
isoquant4       DIFFERS              UNIQUE
planarquant3    DIFFERS              UNIQUE
planarquant4    DIFFERS              UNIQUE
turboquant25    DIFFERS              UNIQUE
turboquant35    DIFFERS              UNIQUE
```

If any method shows `IDENTICAL (placebo!)` it means the dispatch isn't
hitting the real encode/decode path — likely a bug in the integration.

## test_tq_thorough.py — full benchmark

Like `test_tq_real.py` but with more rigour:
- 8 prompts batched (uses `max_num_seqs=8`)
- 80 tokens of generation each
- Captures KV cache token counts to prove memory is actually smaller
- Captures load time, generation time, throughput per method
- Coherence checks on each output (length, repetition, printability)

```bat
%VLLM_PYTHON% tests\test_tq_thorough.py
```

Takes ~60-80 minutes for all 7 methods. Produces a results table with
real numbers.

## Notes

- `test_tq_real.py` and `test_tq_thorough.py` spawn each method in a
  fresh subprocess to make sure GPU state from one run doesn't bleed
  into the next. This is slower but eliminates cross-test interference.
- The Multi-TurboQuant comparison tests use `enforce_eager=True` to hold
  graph/compile behavior constant across methods and `temperature=0.0` so
  output differences reflect KV math rather than RNG. The normal FP16 smoke
  test keeps vLLM's optimized `enforce_eager=False` default unless
  `VLLM_ENFORCE_EAGER=1` is set for troubleshooting.
- The default settings (`max_model_len=512`, `gpu_memory_utilization=0.5`)
  fit comfortably on a single 24 GB RTX 3090 with the 14B AWQ-4bit
  model. Larger models or smaller GPUs will need adjustment.
