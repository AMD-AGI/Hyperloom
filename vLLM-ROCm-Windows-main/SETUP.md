# Installation guide - vLLM ROCm on Windows, from zero to chat

Target machine: a Windows 11 PC with an AMD Radeon RDNA3 GPU (RX 7800 XT / 7900 XT / 7900 XTX...).
Total time: **about 45 minutes** (excluding downloads). Nothing complicated to type:
3 copy-pasted commands, everything else is clicks or automated scripts.

> **Golden rule**: if your Windows username contains a non-ASCII character (e.g. a
> French or Spanish accent), everything ML-related must live under an **accent-free
> path** (e.g. `C:\AI`). This guide applies that rule everywhere. It is pitfall #1:
> it breaks xgrammar/triton with cryptic DLL errors.

---

## Step 0 - Check the machine (2 min)

1. **GPU driver**: up-to-date AMD Adrenalin from https://www.amd.com/en/support
   (install, reboot if asked).
2. **Disk space**: about 15 GB free on C: (ROCm 4 GB, torch 2 GB, model 6 GB, caches).
3. Identify your GPU in Task Manager, Performance tab: "RX 7800 XT" or similar = OK.

**If your CPU is an AMD APU with an integrated GPU** (most Ryzen desktop chips): the
discrete card is **device 1**, the iGPU is device 0. This guide forces
`HIP_VISIBLE_DEVICES=1`, so nothing to do.

## Step 1 - Install the HIP SDK (5 min)

1. Download: https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/quick-start.html
   - take the **HIP SDK** (not full ROCm), version 7.2 or newer.
2. Run the installer with default options. Reboot if asked.

> Note: this SDK acts as a safety net (headers, runtime). The repo's build tooling can
> also use the pip-installed SDK from step 3; both paths work.

## Step 2 - Visual Studio Build Tools + Python (10 min)

1. **Build Tools**: https://visualstudio.microsoft.com/downloads/ under
   "Tools for Visual Studio", download **Build Tools for Visual Studio** (free).
   In the installer, check **"Desktop development with C++"** (brings MSVC + Windows SDK).
2. **Python 3.12**: https://www.python.org/downloads/ - 3.12.x, and during setup check
   **"Add python.exe to PATH"**.

Verify (Win key, type `powershell`, Enter):

```powershell
python --version    # must print Python 3.12.x
```

## Step 3 - Create the Python environment (15 min, mostly downloads)

In PowerShell:

```powershell
# ASCII working directories
mkdir C:\AI, C:\AI\tmp -Force

# Isolated venv (do NOT use the system Python)
python -m venv C:\AI\vllm-venv
C:\AI\vllm-venv\Scripts\python -m pip install -U pip setuptools wheel

# PyTorch with native Windows ROCm (AMD "TheRock" builds)
C:\AI\vllm-venv\Scripts\pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ "torch==2.10.0+rocm7.13.0a20260508" "rocm[libraries,devel]"

# Dependencies
C:\AI\vllm-venv\Scripts\pip install conch-triton-kernels llguidance "xgrammar==0.2.8" "triton-windows<3.7" huggingface_hub
```

The first two installs download about 5 GB. Go grab a coffee.

**Check the GPU**:

```powershell
C:\AI\vllm-venv\Scripts\python -c "import torch; print(torch.__version__); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Expected: a `2.10.0+rocm...` version, then one line per GPU. Note the index of your discrete
card: if your CPU has an integrated GPU (most Ryzen 7000/9000 desktop CPUs and APUs), the iGPU
is usually device 0 and the discrete card device 1; with a single GPU the card is device 0.
On `RuntimeError`: check the pip index (`gfx110X-all`) and your driver.

## Step 4 - The repo, then everything is automated (5 min)

```powershell
cd $env:USERPROFILE\Desktop   # or anywhere; the repo can live wherever you like
git clone https://github.com/ThePie88/vLLM-ROCm-Windows.git
cd vLLM-ROCm-Windows
git clone --depth 1 --branch v0.19.1 https://github.com/vllm-project/vllm.git vllm
```

Then the install sequence (copy-paste as one block):

```powershell
$env:TMP="C:\AI\tmp"; $env:TEMP="C:\AI\tmp"
$py = "C:\AI\vllm-venv\Scripts\python.exe"

# vLLM with no kernels (empty) - the kernels are built natively in step 5
cd vllm; $env:VLLM_TARGET_DEVICE="empty"; & $py -m pip install -e . --no-build-isolation; cd ..

# Plugin + automatic patching (bootstrap, shims, vLLM source patches)
& $py -m pip install -e windows_rocm_plugin
& $py tools\patch_vllm.py vllm
```

> No git? Install it from https://git-scm.com/download/win, or download both repos as
> ZIP files and extract them with the same layout.

## Step 5 - Build the native kernels (3 min)

```powershell
cd experiments\vllm_c_ext
$env:HIP_PATH="C:\AI\vllm-venv\Lib\site-packages\_rocm_sdk_devel"
$env:PATH="C:\AI\vllm-venv\Scripts;"+$env:PATH
cmd /c build_run.bat
```

Expected: `BUILD_OK` at the end (about 1 minute). This build is what delivers real GPTQ
throughput (without it, decode speed collapses; see gfx1101-validation.md).

## Step 6 - First smoke test (2 min)

```powershell
cd ..\..\run
$env:HIP_VISIBLE_DEVICES="1"   # the discrete card's index from Step 3; skip this line with a single GPU
$env:ROCM_HOME="C:\AI\vllm-venv\Lib\site-packages\_rocm_sdk_devel"
$env:TMP="C:\AI\tmp"; $env:TEMP="C:\AI\tmp"
& C:\AI\vllm-venv\Scripts\python.exe first_token.py
```

Expected: the OPT-125m model generates a few tokens, then `FIRST_TOKEN_OK`.

## Step 7 - Benchmark (5 min)

```powershell
$env:VLLM_BENCH_MODEL="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
& C:\AI\vllm-venv\Scripts\python.exe bench.py
```

First run downloads the model (about 5.5 GB). Expected on an RX 7800 XT: **~82 tok/s**
decode, 28 ms TTFT, 9.4 GiB VRAM.

## Step 8 - Chat in your browser

One command from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File localserve.ps1
```

The script: starts the vLLM server, waits until it is ready, serves the chat UI on
http://127.0.0.1:8080, and stops the server cleanly when you close the UI.
Ctrl+C to stop everything.

---

## Troubleshooting (the 6 known failures)

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to load dynamic shared library ...xgrammar_bindings.dll` | accented path | keep venv + caches under `C:\AI` (this guide does) - never inside the user profile |
| `'hip/hip_runtime.h' file not found` (Triton JIT build) | `ROCM_HOME` unresolved | exported by the plugin; manually: `$env:ROCM_HOME="C:\AI\vllm-venv\Lib\site-packages\_rocm_sdk_devel"` |
| `hipErrorInvalidImage` on first run | job landed on the iGPU | `HIP_VISIBLE_DEVICES=1` (localserve.ps1 does it) |
| `10 x call to 'atomicAdd' is ambiguous` at build | HIP 7.13 + stale guard | fixed by PR #2; update to latest main if it comes back |
| `No valid patches in input` on manual `git apply` | UTF-16 patch file | fixed (kvarn.patch re-encoded); use `tools/patch_vllm.py`, it handles everything |
| Slow or failing HF download | network to huggingface.co | retry; the cache resumes where it stopped |

## Going further

- **Upstream README** (the second half of this repo's README.md): every advanced feature
  (KVarN KV-quant, CK FMHA prefill, GEMV autotune, batch sweep) and the code
  documentation. This guide only covers setup and basic use.
- **gfx1101 validation** (gfx1101-validation.md): the full procedure, the 5 bugs
  hit and their fixes, benchmark numbers, methodology.
- **fp8 KV cache on Qwen2.5** (kv-fp8-audit.md): fixed via calibrated K-offset removal —
  calibrate with `run/kv_scale_probe.py` (`VLLM_KV_PROBE_MEAN`), enable with
  `VLLM_WIN_KV_OFFSETS`; benchmarks and design in the audit file.
- **Upcoming optimizations**: hipGraph decode (neutral on this card), torch.compile,
  Triton autotune for gfx1101 - follow the repo issues.
