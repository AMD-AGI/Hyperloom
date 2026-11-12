# CLAUDE.md: working rules for this repo

AI coding agents (Claude Code and others) load this file automatically as instructions when
they work in this repository, on any machine. It describes the project, so everything here has
to hold for every clone; personal preferences do not belong here. Human contributors: see
[CONTRIBUTING.md](CONTRIBUTING.md). Changes to this file are reviewed like code.

Native Windows + AMD ROCm (RDNA3) port of vLLM. vLLM is pinned at tag `v0.19.1` (commit
`b1388b1`; pip reports it as `0.19.2.dev0+gb1388b1fb`, setuptools-scm's string for a locally
modified tree at that tag, not a different base).

## Environment

- Python: the interpreter that has the ROCm torch wheel and this plugin installed. SETUP.md
  creates a venv at `C:\AI\vllm-venv`; a system Python works too if that is where torch-rocm is
  installed. Find out which one it is before running anything: it must `import torch` and
  report a `+rocm` version.
- **Run scripts from `run/`, never from the repo root**: the gitignored `vllm/` clone at the
  root shadows the installed `vllm` package as a namespace package otherwise.
- GPU index: Ryzen desktop CPUs and APUs with an integrated GPU usually expose it as HIP device
  0 and the discrete card as device 1; with a single GPU (an Intel CPU, or no iGPU) the
  discrete card is device 0. List them with
  `python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"`
  and set `HIP_VISIBLE_DEVICES` to the discrete card when there is more than one. Symptom of a
  job landing on an iGPU: "device kernel image is invalid". `localserve.ps1` picks the
  discrete card by itself.
- Engine runs are in-process only: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (every `run/` script
  sets it itself). The plugin's config-time hooks require it.
- Non-ASCII paths (an accented Windows user name, a venv under it) break xgrammar/tvm_ffi
  library loading. `windows_rocm_plugin/sitecustomize.py`, installed into the active
  interpreter's site-packages by `tools/patch_vllm.py`, carries that fix together with the
  torch.distributed shim. Never hand-edit the installed copy: patch_vllm.py overwrites it.
  Keeping the venv and caches under an accent-free path is the belt-and-braces alternative.
- On Windows, running out of VRAM does not fail: it spills into shared system memory and every
  timing becomes meaningless. Before a GPU run make sure no leftover `python.exe` holds the
  card, and watch the "shared GPU memory" counter during benchmarks.

## Where things live

- `windows_rocm_plugin/`: pip-installed platform plugin (monkey-patches; the normal way to
  change engine behaviour). Modules are wired in `vllm_windows_rocm/platform.py`.
- `vllm/`: gitignored clone at the pin. Direct edits there MUST be captured as
  `patches/vllm/*.patch` or they are lost on re-clone.
- `patches/vllm/`: applied by `python tools/patch_vllm.py vllm` in **alphabetical filename
  order**; if patch B depends on patch A's context, name B to sort after A. After editing the
  `vllm/` tree: regenerate the patch and verify it reverse-applies before committing.
- `run/`: all bench/probe drivers (see the `run/README.md` table).
- `experiments/vllm_c_ext/`: native kernel builds (`vllm_win_cache_C.pyd`,
  `vllm_win_attn_C.pyd`), built by `build_run.bat` / `build_attn_c.py`.

## Behaviour-changing work

- fp8 KV cache requires the two-step calibration (probe, then offsets file, then
  `VLLM_WIN_KV_OFFSETS`); never bench fp8 without it. Do not combine `VLLM_WIN_KV_OFFSETS`
  with `VLLM_WIN_KV_KSCALE`.
- Every engine-behaviour change ships with an executable check: a bit-exact kernel check
  (`run/kv_pc_kernel_check.py` style) for Triton kernels; `bench.py` / `precision_check.py`
  for engine paths. Reading or grepping the code is not a check.
- `agree_frac` is run-to-run noisy (GPTQ int4 GEMM nondeterminism, amplified by greedy
  cascades). Deterministic evidence = offline attention error metrics + text coherence.
  Never report agree_frac as primary proof.
- Docs stay in sync in the same change: README.md (features/status), SETUP.md (install),
  run/README.md (scripts), patches/README.md (patch set).

## Git

- Commits: imperative subject; the body explains why. Say when an AI assistant co-authored the
  change (a `Co-Authored-By:` trailer).
- Commit or push only when the person you are working for asks.
- Pull requests: link the issue, include measured numbers (before/after) and the commands that
  produced them; the rest is in CONTRIBUTING.md.

## Tooling gotchas (Windows)

- Git Bash wants forward slashes and POSIX syntax; Windows PowerShell 5.1 has no `&&`.
- `gh pr create/edit` with inline bodies: backticks execute as command substitution in bash,
  so pass `--body-file` instead.
- `git -C vllm apply` needs `--ignore-whitespace` (CRLF vs LF), which patch_vllm.py already does.
- vLLM has file names long enough to hit the 260-character path limit: a second checkout or
  worktree of it in a deep directory needs `git -c core.longpaths=true`.
