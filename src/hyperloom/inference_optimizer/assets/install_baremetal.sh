#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install Hyperloom for bare-metal hosts (no-Docker / bare-host orchestrator).
#
# Runs Hyperloom directly on a host that already provides the ROCm framework
# base (ROCm runtime + a ROCm-built torch + a serving framework). For bare-metal
# installs, the script can optionally install SGLang or vLLM ROCm framework layers.
#
# Phase 1  base preflight  — ROCm / GPU arch / ROCm torch / serving framework
# Phase 2  framework       — optional bare-metal SGLang/vLLM install
# Phase 3  ROCm hotfix     — install ROCclr HIP runtime + roctracer profiler fix
# Phase 4  credentials     — resolve Anthropic/DeepSeek LLM creds into .env
# Phase 5  runtime env     — persist bare-metal runtime vars into .env
#
# Scope: bare-metal base setup only. Open-source deps and the optimizer runtime
# (io pkg, Magpie, InferenceX, kernel-agent Ray/GEAK/TraceLens, fa) are installed
# by the inference_optimizer skill, not here. It STOPS before launching.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_script_dir}/../../../.." && pwd)}"

ENV_TEMPLATE="${REPO_ROOT}/.env.template"
DOTENV="${REPO_ROOT}/.env"
HYPERLOOM_SKILL_PATH="${HYPERLOOM_SKILL_PATH:-${REPO_ROOT}/src/hyperloom/inference_optimizer/SKILL.md}"

HYPERLOOM_WHEEL_REPO="${HYPERLOOM_WHEEL_REPO:-AMD-AGI/Hyperloom}"
HYPERLOOM_WHEEL_TAG="${HYPERLOOM_WHEEL_TAG:-v1.0.0}"
ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR:-/opt/rocm/lib}"
ROCM_PROFILER_HOTFIX_ASSET="${ROCM_PROFILER_HOTFIX_ASSET:-rocm-profiler-hotfix-libs.tar.gz}"
# Installed name must outrank the vendor library in ldconfig's ordering.
ROCM_PROFILER_HOTFIX_SUFFIX="${ROCM_PROFILER_HOTFIX_SUFFIX:--hotfix}"

DEFAULT_OPENAI_BASE_URL="${DEFAULT_OPENAI_BASE_URL:-https://your-openai-compatible-gateway.example.com/v1}"
OPENAI_API_KEY_PLACEHOLDER="ak-your-api-key-here"

# Phase 1 probes every serving framework Hyperloom can benchmark, not just the
# two it can install. An atom image ships neither sglang nor vllm, so gating the
# preflight on those alone made every `--framework atom` host fail setup.
FRAMEWORKS="${FRAMEWORKS:-sglang,vllm,atom}"
INSTALL_FRAMEWORK="none"
# Track whether the operator explicitly picked a framework env (via $FRAMEWORK_ENV
# or --framework-env). When unset, vLLM defaults to isolated (its wheel pins a
# torch that would clash with the host stack); others default to shared.
_FRAMEWORK_ENV_WAS_SET="${FRAMEWORK_ENV+x}"
FRAMEWORK_ENV="${FRAMEWORK_ENV:-shared}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
# Framework versions track docs/compatibility.rst (SGLang v0.5.18, ROCm 7.2.4).
# vLLM installs 0.27.1+rocm723 from the wheels.vllm.ai pip index, matching the
# vllm/vllm-openai-rocm:v0.27.1 Docker image. The rocm723 variant puts the
# vLLM ROCm layer at 7.2.3, one patch level above the SGLang stack. AITER_REF
# can pin ROCm/aiter to a released tag; when unset, the installer selects the
# newest tag compatible with the already-installed ROCm torch/triton stack.
SGLANG_REF="${SGLANG_REF:-v0.5.18}"
_SGLANG_ROCM_PYPI_VERSION_WAS_SET="${SGLANG_ROCM_PYPI_VERSION+x}"
_AITER_REF_WAS_SET="${AITER_REF+x}"
SGLANG_ROCM_EXTRA="${SGLANG_ROCM_EXTRA:-rocm724}"
if [ -z "$_SGLANG_ROCM_PYPI_VERSION_WAS_SET" ]; then
  case "$SGLANG_ROCM_EXTRA" in
    rocm700) SGLANG_ROCM_PYPI_VERSION="7.0.0" ;;
    rocm724) SGLANG_ROCM_PYPI_VERSION="7.2.4" ;;
    *)       SGLANG_ROCM_PYPI_VERSION="7.2.0" ;;
  esac
fi
SGLANG_ROCM_PYPI_VERSION="${SGLANG_ROCM_PYPI_VERSION:-7.2.4}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:-}"
VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
VLLM_ROCM_VARIANT="${VLLM_ROCM_VARIANT:-rocm723}"
VLLM_ROCM_INDEX="${VLLM_ROCM_INDEX:-https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}}"
_VLLM_VENV_ROOT_WAS_SET="${VLLM_VENV_ROOT+x}"
VLLM_VENV_ROOT="${VLLM_VENV_ROOT:-/opt/hyperloom/vllm-venv}"
REQUIRE_FRAMEWORKS=0
SKIP_BASE_CHECK=0
DRY_RUN=0
CHECK_ONLY=0
VERIFY_HOTFIX_ONLY=0
ASSUME_YES=0
USER_DATA_PATH_ARG=""
DEPS_ROOT_ARG=""

usage() {
  cat <<'EOF'
Usage: src/hyperloom/inference_optimizer/assets/install_baremetal.sh [options]

Set up a bare-metal host with ROCm + ROCm torch for Hyperloom. Verifies the base,
optionally installs SGLang/vLLM, resolves credentials, and writes the combined
runtime env. Stops BEFORE launching.

Options:
  --user-data-path PATH  Writable artifact root (default: /workspace/hyperloom)
  --deps-root PATH       Directory for auto-cloned dependency checkouts
  --frameworks LIST      Comma list to verify in Phase 1 (default:
                         sglang,vllm,atom). Phase 1 passes when at least one
                         entry imports.
  --install-framework FW Install a missing bare-metal framework layer.
                         Supported: none, sglang, vllm. Default: none.
  --framework-env MODE   Install target for framework packages: shared or
                         isolated. Default: shared, except vLLM which defaults
                         to isolated so it never replaces the shared ROCm
                         torch stack.
  --vllm-venv-root PATH  Isolated vLLM venv path (default:
                         /opt/hyperloom/vllm-venv).
  --require-frameworks   Treat a missing requested framework as fatal
  --skip-base-check      Skip Phase 1 base preflight
  --check-only           Verify only; do not clone/install/mutate
  --dry-run              Print planned actions without cloning/installing/writing
  --verify-hotfix        Re-check the ROCm profiler hotfix only; exits non-zero
                         when the loader or torch/lib no longer resolves to it
  --yes, -y              Non-interactive; fail fast on missing credentials
  -h, --help             Show this help

Credential resolution (highest precedence first): env > .env > interactive
prompt (TTY + not --yes). Configure Anthropic
(ANTHROPIC_BASE_URL+ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN) or a Claude
subscription token (CLAUDE_CODE_OAUTH_TOKEN, from "claude setup-token"),
matching runtime credential rules. A dual-protocol gateway such as DeepSeek
additionally sets OPENAI_BASE_URL+OPENAI_API_KEY on the same host; retired
DEEPSEEK_* values are migrated automatically.
Env overrides honored: REPO_ROOT,
USER_DATA_PATH, HYPERLOOM_DEPS_ROOT / HYPERLOOM_CACHE_DIR,
PYTHON, INFERENCE_OPTIMIZER_FORCE_PYTHON,
SGLANG_REPO, SGLANG_REF, SGLANG_ROOT, SGLANG_ROCM_PYPI_VERSION,
SGLANG_ROCM_EXTRA, SGLANG_BUILD_RUST_EXTS, AITER_REPO, AITER_REF, AITER_ROOT, ROCM_PATH, HIP_PATH,
LD_LIBRARY_PATH, VLLM_VERSION, VLLM_ROCM_VARIANT, VLLM_ROCM_INDEX,
VLLM_VENV_ROOT, HYPERLOOM_WHEEL_REPO, HYPERLOOM_WHEEL_TAG.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --user-data-path)   [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --user-data-path requires a value" >&2; exit 2; }; shift; USER_DATA_PATH_ARG="${1:-}" ;;
    --deps-root)        [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --deps-root requires a value" >&2; exit 2; }; shift; DEPS_ROOT_ARG="${1:-}" ;;
    --frameworks)       [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --frameworks requires a value" >&2; exit 2; }; shift; FRAMEWORKS="${1:-}" ;;
    --install-framework)
      [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --install-framework requires a value" >&2; exit 2; }
      shift
      INSTALL_FRAMEWORK="${1:-}"
      case "$INSTALL_FRAMEWORK" in
        none|sglang|vllm) ;;
        *) echo "[install-baremetal] ERROR: --install-framework must be one of: none, sglang, vllm" >&2; exit 2 ;;
      esac
      ;;
    --framework-env)
      [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --framework-env requires a value" >&2; exit 2; }
      shift
      FRAMEWORK_ENV="${1:-}"
      _FRAMEWORK_ENV_WAS_SET="x"
      case "$FRAMEWORK_ENV" in
        shared|isolated) ;;
        *) echo "[install-baremetal] ERROR: --framework-env must be one of: shared, isolated" >&2; exit 2 ;;
      esac
      ;;
    --vllm-venv-root)
      [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --vllm-venv-root requires a value" >&2; exit 2; }
      shift
      VLLM_VENV_ROOT="${1:-}"
      _VLLM_VENV_ROOT_WAS_SET="x"
      ;;
    --require-frameworks) REQUIRE_FRAMEWORKS=1 ;;
    --skip-base-check)  SKIP_BASE_CHECK=1 ;;
    --check-only)       CHECK_ONLY=1 ;;
    --dry-run)          DRY_RUN=1 ;;
    --verify-hotfix)    VERIFY_HOTFIX_ONLY=1 ;;
    --yes|-y)           ASSUME_YES=1 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "[install-baremetal] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[install-baremetal] $*"; }
warn() { echo "[install-baremetal WARN] $*" >&2; }
die() { echo "[install-baremetal ERROR] $*" >&2; exit 1; }

IMAGE_HINT="Provision the ROCm framework base first (run inside an AMD ROCm \
SGLang/vLLM image such as lmsysorg/sglang-rocm:v0.5.18-rocm724-mi30x|mi35x-* or \
vllm/vllm-openai-rocm:v0.27.1, or install an equivalent ROCm torch + \
framework stack), then re-run."

is_interactive() { [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ] && [ -t 1 ]; }

# Resolve a Python interpreter, mirroring install.sh: prefer the canonical ROCm
# venv (/opt/venv) unless INFERENCE_OPTIMIZER_FORCE_PYTHON=1 pins $PYTHON.
resolve_python() {
  if [ -x "/opt/venv/bin/python" ] && [ "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-0}" != "1" ]; then
    echo "/opt/venv/bin/python"; return 0
  fi
  if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then echo "$PYTHON"; return 0; fi
  if [ -x "/venv/bin/python" ]; then echo "/venv/bin/python"; return 0; fi
  command -v python3 2>/dev/null && return 0
  return 1
}

# Import-probe a module. `import importlib.util` (not `import importlib`): a bare
# interpreter does not auto-load the util submodule.
_py_has() { "$1" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$2') else 1)" 2>/dev/null; }

# Backs up declared run mode when setup runs inside a container.
running_in_container() { [ -f /.dockerenv ]; }

# The interpreter that owns a given framework. vLLM lives in its own venv under
# FRAMEWORK_ENV=isolated; every other engine uses the shared interpreter. Every
# framework probe goes through here so preflight, framework resolution and the
# profiler hotfix can never disagree about where an engine is installed.
framework_probe_python() {
  local fw="$1" default_py="$2"
  if [ "$fw" = "vllm" ] && [ "$FRAMEWORK_ENV" = "isolated" ] && [ -x "${VLLM_VENV_ROOT}/bin/python" ]; then
    printf '%s' "${VLLM_VENV_ROOT}/bin/python"
  else
    printf '%s' "$default_py"
  fi
}

# Print the serving framework to record for downstream skills, or nothing when
# none is importable. Walks $FRAMEWORKS in order — the same list Phase 1 probes
# — so an engine that passes preflight is always the one written to .env.
resolve_installed_framework() {
  if [ "$INSTALL_FRAMEWORK" = "sglang" ] || [ "$INSTALL_FRAMEWORK" = "vllm" ]; then
    printf '%s' "$INSTALL_FRAMEWORK"; return 0
  fi
  local py fw probe_py _rif_arr
  py="$(resolve_python)" || return 0
  IFS=',' read -r -a _rif_arr <<< "$FRAMEWORKS"
  for fw in "${_rif_arr[@]}"; do
    fw="$(echo "$fw" | tr -d '[:space:]')"; [ -z "$fw" ] && continue
    probe_py="$(framework_probe_python "$fw" "$py")"
    if _py_has "$probe_py" "$fw"; then printf '%s' "$fw"; return 0; fi
  done
  return 0
}

python_venv_root() {
  local py="$1" bin_dir venv_dir
  bin_dir="$(cd "$(dirname "$py")" 2>/dev/null && pwd)" || return 1
  venv_dir="$(cd "${bin_dir}/.." 2>/dev/null && pwd)" || return 1
  [ -f "${venv_dir}/pyvenv.cfg" ] || return 1
  printf '%s\n' "$venv_dir"
}

export_virtualenv_for_python() {
  local py="$1" venv_dir
  if venv_dir="$(python_venv_root "$py")"; then
    export VIRTUAL_ENV="$venv_dir"
    case ":$PATH:" in
      *":${venv_dir}/bin:"*) ;;
      *) export PATH="${venv_dir}/bin:$PATH" ;;
    esac
    log "VIRTUAL_ENV=${VIRTUAL_ENV}"
  fi
}

check_torch_rocm_shared_libs() {
  local py="$1" lib missing
  command -v ldd >/dev/null 2>&1 || return 0
  lib="$("$py" - <<'PY' 2>/dev/null || true
from pathlib import Path
try:
    import torch
    root = Path(torch.__file__).resolve().parent / "lib"
    for name in ("libtorch_hip.so", "libc10_hip.so"):
        candidate = root / name
        if candidate.exists():
            print(candidate)
            break
except Exception:
    pass
PY
)"
  [ -n "$lib" ] || return 0
  missing="$(ldd "$lib" 2>/dev/null | grep 'not found' || true)"
  if [ -n "$missing" ]; then
    warn "ROCm torch shared libraries are missing for ${lib}:"
    echo "$missing" >&2
    warn "Set ROCM_PATH/HIP_PATH and LD_LIBRARY_PATH to a matching ROCm user-space stack."
    return 1
  fi
  return 0
}

check_rocm_toolchain_alignment() {
  local hip_version="$1" hip_major hipcc_path hipcc_root hipconfig_path hipcc_version header
  hip_major="${hip_version%%.*}"
  [ -n "$hip_major" ] || return 0
  hipcc_path="$(command -v hipcc 2>/dev/null || true)"
  if [ -z "$hipcc_path" ]; then
    warn "hipcc not found; AITER/source builds need a ROCm compiler toolchain."
    return 0
  fi
  hipconfig_path="$(dirname "$hipcc_path")/hipconfig"
  [ -x "$hipconfig_path" ] || hipconfig_path="$(command -v hipconfig 2>/dev/null || true)"
  hipcc_root=""
  hipcc_version=""
  if [ -n "$hipconfig_path" ]; then
    hipcc_root="$($hipconfig_path --path 2>/dev/null || true)"
    hipcc_version="$($hipconfig_path --version 2>/dev/null || true)"
  fi
  [ -d "$hipcc_root" ] || hipcc_root="$(cd "$(dirname "$hipcc_path")/.." 2>/dev/null && pwd)" || return 0
  log "hipcc: ${hipcc_path} (sdk=${hipcc_root})"
  if [ -n "${ROCM_PATH:-}" ] && [ "$hipcc_root" != "$(cd "$ROCM_PATH" 2>/dev/null && pwd)" ]; then
    case "$hipcc_version" in
      "$hip_version"*) ;;
      *) warn "hipcc SDK ${hipcc_root} (${hipcc_version:-unknown}) differs from ROCM_PATH=${ROCM_PATH} and torch hip=${hip_version}." ;;
    esac
  fi
  if [ "$hip_major" -ge 7 ] 2>/dev/null; then
    header="${hipcc_root}/include/hip/hip_runtime_api.h"
    if [ ! -f "$header" ] || ! grep -q 'hipDeviceAttributePciChipId' "$header" 2>/dev/null; then
      warn "hipcc headers at ${hipcc_root} do not look compatible with torch hip=${hip_version}."
      warn "Set ROCM_PATH/HIP_PATH/PATH to a ROCm ${hip_major}.x toolchain before installing AITER."
      return 1
    fi
  fi
  return 0
}

# Human GPU label from arch, refined by rocm-smi product name when available.
detect_gpu_label() {
  local gfx="$1" product=""
  if command -v rocm-smi >/dev/null 2>&1; then
    product="$(rocm-smi --showproductname 2>/dev/null | grep -oiE 'MI[0-9]{3}[A-Z]?' | head -1)"
  fi
  if [ -n "$product" ]; then echo "$product"; return 0; fi
  case "$gfx" in
    gfx942) echo "MI300X" ;;
    gfx950) echo "MI355X" ;;
    gfx1151) echo "Radeon 8060S" ;;
    *) echo "MI300X" ;;
  esac
}

DETECTED_GPU="MI300X"

base_preflight() {
  local rc=0
  log "Phase 1: base preflight"

  local os_name=""
  [ -r /etc/os-release ] && os_name="$(. /etc/os-release 2>/dev/null; echo "${NAME:-} ${VERSION_ID:-}")"
  log "OS: ${os_name:-unknown}"

  if command -v rocm-smi >/dev/null 2>&1; then
    local rocm_ver=""
    [ -r /opt/rocm/.info/version ] && rocm_ver="$(cat /opt/rocm/.info/version 2>/dev/null)"
    log "ROCm: present${rocm_ver:+ (version ${rocm_ver})}"
  else
    warn "ROCm: rocm-smi not found. ${IMAGE_HINT}"; rc=1
  fi

  local gfx=""
  command -v rocminfo >/dev/null 2>&1 && gfx="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)"
  DETECTED_GPU="$(detect_gpu_label "$gfx")"
  case "$gfx" in
    gfx942) log "GPU: ${DETECTED_GPU} (${gfx})" ;;
    gfx950) log "GPU: ${DETECTED_GPU} (${gfx})" ;;
    gfx1151) log "GPU: ${DETECTED_GPU} (${gfx})" ;;
    "")     warn "GPU arch: not detected via rocminfo" ;;
    *)      warn "GPU arch ${gfx} untested (supported: gfx942/gfx950/gfx1151)" ;;
  esac

  local py
  if ! py="$(resolve_python)"; then die "no usable Python found (set PYTHON or provide /opt/venv). ${IMAGE_HINT}"; fi
  log "Python: ${py} ($(${py} --version 2>&1))"

  local torch_report tv thip
  torch_report="$("${py}" - <<'PY' 2>/dev/null || true
try:
    import torch
    print(f"{torch.__version__}|{getattr(torch.version,'hip',None) or ''}")
except Exception as exc:
    print(f"ERR|{type(exc).__name__}: {str(exc)[:120]}")
PY
)"
  tv="${torch_report%%|*}"; thip="${torch_report#*|}"
  if [ "$tv" = "ERR" ] || [ -z "$torch_report" ]; then
    warn "torch: NOT importable (${thip:-unknown}). ${IMAGE_HINT}"; rc=1
  elif [ -z "$thip" ]; then
    warn "torch: ${tv} is NOT a ROCm build (hip=None); will crash on GPU. ${IMAGE_HINT}"; rc=1
  else
    log "torch: ${tv} (hip=${thip}) ROCm OK"
    check_torch_rocm_shared_libs "$py" || rc=1
    check_rocm_toolchain_alignment "$thip" || rc=1
    # The torch/triton pin only has to hold when this run is about to build a
    # framework layer against it. An image that already ships a working engine
    # (atom, or a prebuilt sglang/vllm) is allowed to carry its own triton.
    if [ "$INSTALL_FRAMEWORK" = "none" ]; then
      check_torch_triton_alignment "$py" || true
    else
      check_torch_triton_alignment "$py" || rc=1
    fi
  fi

  local any_fw=0 sglang_ok=0 fw
  IFS=',' read -r -a _fw_arr <<< "$FRAMEWORKS"
  for fw in "${_fw_arr[@]}"; do
    fw="$(echo "$fw" | tr -d '[:space:]')"; [ -z "$fw" ] && continue
    local probe_py; probe_py="$(framework_probe_python "$fw" "$py")"
    if _py_has "$probe_py" "$fw"; then
      if [ "$probe_py" != "$py" ]; then
        log "framework ${fw}: OK (isolated: ${probe_py})"
      else
        log "framework ${fw}: OK"
      fi
      any_fw=1
      [ "$fw" = "sglang" ] && sglang_ok=1
    elif [ "$REQUIRE_FRAMEWORKS" -eq 1 ]; then
      warn "framework ${fw}: MISSING (required). ${IMAGE_HINT}"; rc=1
    else
      warn "framework ${fw}: missing (skip if unused)"
    fi
  done
  if [ "$any_fw" -eq 0 ]; then
    if [ "$INSTALL_FRAMEWORK" = "none" ]; then
      warn "no serving framework importable from '${FRAMEWORKS}'. ${IMAGE_HINT}"
      rc=1
    else
      warn "no serving framework importable from '${FRAMEWORKS}'; will attempt --install-framework ${INSTALL_FRAMEWORK}"
    fi
  fi

  # vLLM's ROCm stack owns triton/aiter in isolated mode; SGLang owns sgl_kernel,
  # so only look for it once SGLang is the engine actually in play — an atom or
  # vLLM host has no use for it and should not be told a dependency is missing.
  local m dep_py deps="triton aiter"
  if [ "$sglang_ok" -eq 1 ] || [ "$INSTALL_FRAMEWORK" = "sglang" ]; then
    deps="$deps sgl_kernel"
  fi
  for m in $deps; do
    dep_py="$py"
    if [ "$m" != "sgl_kernel" ] && [ "$FRAMEWORK_ENV" = "isolated" ] \
       && [ -x "${VLLM_VENV_ROOT}/bin/python" ] \
       && printf '%s' ",${FRAMEWORKS}," | grep -q ",vllm,"; then
      dep_py="${VLLM_VENV_ROOT}/bin/python"
    fi
    _py_has "$dep_py" "$m" && log "runtime dep ${m}: OK" || warn "runtime dep ${m}: missing (some phases may degrade)"
  done

  [ "$rc" -ne 0 ] && die "base preflight failed. Fix the items above, or pass --skip-base-check to override."
  log "Phase 1: base preflight OK"
}

# Return the shared dependency root used for bare-metal framework sources.
framework_deps_root() {
  printf '%s' "${HYPERLOOM_CACHE_DIR:-${HYPERLOOM_DEPS_ROOT:-${REPO_ROOT}/.cache}}"
}

# Return the first visible AMD GPU arch, e.g. gfx942.
detect_rocm_gfx_arch() {
  command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1
}

# Print the installed distribution version for pip constraints.
installed_dist_version() {
  local py="$1" dist="$2"
  "$py" - "$dist" <<'PY'
import importlib.metadata as metadata
import sys
try:
    print(metadata.version(sys.argv[1]))
except metadata.PackageNotFoundError:
    sys.exit(1)
PY
}

torch_required_triton_version() {
  local py="$1"
  "$py" <<'PY'
import importlib.metadata as metadata
import re

try:
    requirements = metadata.requires("torch") or []
except metadata.PackageNotFoundError:
    raise SystemExit(0)

for requirement in requirements:
    if not requirement.lower().startswith("triton"):
        continue
    match = re.search(r"==\s*([^;\s]+)", requirement)
    if match:
        print(match.group(1))
        break
PY
}

check_torch_triton_alignment() {
  local py="$1" required current
  required="$(torch_required_triton_version "$py" 2>/dev/null || true)"
  [ -n "$required" ] || return 0
  current="$(installed_dist_version "$py" triton 2>/dev/null || true)"
  if [ "$current" != "$required" ]; then
    warn "triton ${current:-missing} does not match torch requirement ${required}; reinstall the torch-pinned ROCm Triton before installing a framework layer"
    return 1
  fi
}

# Create a temporary constraint file that prevents pip from swapping ROCm torch
# for PyPI CUDA torch while installing framework Python extras.
write_rocm_torch_constraints() {
  local py="$1" file="$2" torch_version
  torch_version="$(installed_dist_version "$py" torch 2>/dev/null || true)"
  if [ -z "$torch_version" ]; then
    die "torch is not installed; install ROCm torch before installing a framework"
  fi
  "$py" - <<'PY' || die "installed torch is not a ROCm build; refusing to install framework"
import torch
raise SystemExit(0 if getattr(torch.version, "hip", None) else 1)
PY
  check_torch_triton_alignment "$py" || die "installed triton does not match torch; refusing to install framework with inconsistent ROCm dependencies"
  printf 'torch==%s\n' "$torch_version" > "$file"
  if installed_dist_version "$py" triton >/dev/null 2>&1; then
    printf 'triton==%s\n' "$(installed_dist_version "$py" triton)" >> "$file"
  fi
}

# Resolve the in-tree ROCm kernel build directory for a SGLang checkout.
# Legacy tags (<= v0.5.16): sgl-kernel/setup_rocm.py
# v0.5.17+ tags: python/sglang/kernels/aot/setup_rocm.py
sglang_kernel_rocm_build_dir() {
  local sglang_root="$1"
  if [ -f "${sglang_root}/sgl-kernel/setup_rocm.py" ]; then
    printf '%s' "${sglang_root}/sgl-kernel"
    return 0
  fi
  if [ -f "${sglang_root}/python/sglang/kernels/aot/setup_rocm.py" ]; then
    printf '%s' "${sglang_root}/python/sglang/kernels/aot"
    return 0
  fi
  return 1
}

install_sglang_kernel_rocm() {
  local py="$1" sglang_root="$2" arch="$3" kernel_dir=""
  kernel_dir="$(sglang_kernel_rocm_build_dir "$sglang_root")" \
    || die "no in-tree ROCm sglang-kernel build path under ${sglang_root} (checked sgl-kernel/ and python/sglang/kernels/aot/)"
  log "building in-tree ROCm kernel from ${kernel_dir} (arch=${arch})"
  (cd "$kernel_dir" && AMDGPU_TARGET="$arch" "$py" setup_rocm.py install)
}

# Install SGLang from source for Python versions not supported by AMD wheels.
install_sglang_from_source() {
  local py="$1" deps_root="$2" sglang_root arch constraint_file
  sglang_root="${SGLANG_ROOT:-${deps_root}/sglang}"
  arch="${PYTORCH_ROCM_ARCH:-$(detect_rocm_gfx_arch)}"
  [ -n "$arch" ] || arch="gfx942"

  log "installing SGLang from source at ${sglang_root} (ref=${SGLANG_REF}, arch=${arch})"
  if [ ! -d "${sglang_root}/.git" ]; then
    mkdir -p "$(dirname "$sglang_root")"
    git clone --recursive --branch "$SGLANG_REF" "$SGLANG_REPO" "$sglang_root"
  else
    git -C "$sglang_root" fetch --all --tags --prune
    git -C "$sglang_root" checkout "$SGLANG_REF"
    git -C "$sglang_root" submodule sync
    git -C "$sglang_root" submodule update --init --recursive
  fi

  "$py" -m pip install --upgrade pip wheel setuptools
  install_sglang_kernel_rocm "$py" "$sglang_root" "$arch"
  if [ -f "${sglang_root}/python/pyproject_other.toml" ]; then
    cp "${sglang_root}/python/pyproject_other.toml" "${sglang_root}/python/pyproject.toml"
  fi
  constraint_file="$(mktemp)"
  write_rocm_torch_constraints "$py" "$constraint_file"
  # ROCm editable installs only need multimodal Rust crates for VLM serving.
  export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
  log "SGLANG_BUILD_RUST_EXTS=${SGLANG_BUILD_RUST_EXTS}"
  "$py" -m pip install --constraint "$constraint_file" -e "${sglang_root}/python[srt_hip]"
  rm -f "$constraint_file"
}

checkout_aiter_ref() {
  local aiter_root="$1" ref="$2"
  git -C "$aiter_root" checkout "$ref"
  git -C "$aiter_root" submodule sync
  git -C "$aiter_root" submodule update --init --recursive
}

ensure_aiter_checkout() {
  local aiter_root="$1"
  mkdir -p "$(dirname "$aiter_root")"
  if [ ! -d "${aiter_root}/.git" ]; then
    git clone "$AITER_REPO" "$aiter_root"
  fi
  git -C "$aiter_root" fetch --all --tags --prune
}

list_aiter_tags_newest_first() {
  local aiter_root="$1"
  git -C "$aiter_root" tag -l 'v*' | sort -V -r
}

install_aiter_ref_with_constraints() {
  local py="$1" aiter_root="$2" ref="$3" constraint_file="$4" aiter_use_system_triton
  checkout_aiter_ref "$aiter_root" "$ref"
  aiter_use_system_triton="${AITER_USE_SYSTEM_TRITON:-1}"
  case "$aiter_use_system_triton" in
    0|1) ;;
    *) warn "AITER_USE_SYSTEM_TRITON must be 0 or 1"; return 1 ;;
  esac
  AITER_USE_SYSTEM_TRITON="$aiter_use_system_triton" "$py" -m pip install --constraint "$constraint_file" \
    --config-settings editable_mode=compat -e "$aiter_root" || return 1
  "$py" -c "import aiter" >/dev/null || return 1
  check_torch_triton_alignment "$py" || return 1
}

install_compatible_aiter() {
  local py="$1" aiter_root="$2" constraint_file ref tried=0
  constraint_file="$(mktemp)"
  write_rocm_torch_constraints "$py" "$constraint_file"

  ensure_aiter_checkout "$aiter_root"
  if [ -n "$AITER_REF" ]; then
    log "installing AITER ${AITER_REF} with existing torch/triton constraints"
    install_aiter_ref_with_constraints "$py" "$aiter_root" "$AITER_REF" "$constraint_file" \
      || { rm -f "$constraint_file"; die "AITER ${AITER_REF} installed or imported unsuccessfully with the current torch/triton constraints"; }
    rm -f "$constraint_file"
    return 0
  fi

  while IFS= read -r ref; do
    [ -n "$ref" ] || continue
    tried=$((tried + 1))
    log "trying AITER ${ref} with existing torch/triton constraints"
    if install_aiter_ref_with_constraints "$py" "$aiter_root" "$ref" "$constraint_file"; then
      AITER_REF="$ref"
      export AITER_REF
      rm -f "$constraint_file"
      log "selected AITER_REF=${AITER_REF}"
      return 0
    fi
    warn "AITER ${ref} did not install/import with current torch/triton; trying older tag"
  done < <(list_aiter_tags_newest_first "$aiter_root")

  rm -f "$constraint_file"
  [ "$tried" -gt 0 ] || die "no AITER tags found in ${AITER_REPO}"
  die "no AITER tag installed and imported successfully with the current torch/triton constraints"
}

# Install the AMD SGLang wheel only when its dependency set matches this Python.
# ROCm target is overridable so hosts pinned to an older driver (e.g. amdgpu
# 6.3.x, which supports up to ROCm 7.0 user space) can select a matching wheel.
install_sglang_from_wheel() {
  local py="$1"
  log "installing amd-sglang ROCm ${SGLANG_ROCM_PYPI_VERSION} wheel (extra=${SGLANG_ROCM_EXTRA})"
  "$py" -m pip uninstall -y sglang-kernel sgl-kernel sglang amd-sglang || true
    "$py" -m pip install \
      "amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}]" \
      -i "https://pypi.amd.com/rocm-${SGLANG_ROCM_PYPI_VERSION}/simple" \
      --extra-index-url https://pypi.org/simple
}

# Install the AMD SGLang ROCm wheel and its source-only AITER dependency.
install_sglang_framework() {
  local py deps_root aiter_root py_mm
  py="$(resolve_python)" || die "no usable Python found for SGLang install"
  deps_root="$(framework_deps_root)"
  aiter_root="${AITER_ROOT:-${deps_root}/aiter}"
  py_mm="$("$py" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  log "Phase 2: installing SGLang ROCm framework layer"
  log "framework python: ${py}"
  log "AITER_ROOT=${aiter_root}"
  if [ -n "$AITER_REF" ]; then
    log "AITER_REF=${AITER_REF}"
  else
    log "AITER_REF=auto (newest tag compatible with installed torch/triton)"
  fi
  log "SGLANG_ROCM_EXTRA=${SGLANG_ROCM_EXTRA}"
  log "SGLANG_ROCM_PYPI_VERSION=${SGLANG_ROCM_PYPI_VERSION}"

  if [ "$SGLANG_ROCM_EXTRA" = "rocm700" ] && [ "$py_mm" != "3.10" ]; then
    die "SGLANG_ROCM_EXTRA=rocm700 currently supports Python 3.10 AMD wheels only; Python ${py_mm} would use source install and can pull mismatched ROCm 7.2 Triton."
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    _py_has "$py" sglang && log "sglang import OK" || warn "sglang missing (check-only; would install amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}])"
    if _py_has "$py" aiter; then
      log "aiter import OK"
    elif [ -n "$AITER_REF" ]; then
      warn "aiter missing (check-only; would clone/install ${AITER_REPO}@${AITER_REF})"
    else
      warn "aiter missing (check-only; would auto-select newest compatible tag from ${AITER_REPO})"
    fi
    _py_has "$py" sgl_kernel && log "sgl_kernel import OK" || warn "sgl_kernel missing (check-only; installed by amd-sglang)"
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$py_mm" = "3.10" ]; then
      log "would run: ${py} -m pip install 'amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}]' -i https://pypi.amd.com/rocm-${SGLANG_ROCM_PYPI_VERSION}/simple --extra-index-url https://pypi.org/simple"
    else
      local sglang_root="${SGLANG_ROOT:-${deps_root}/sglang}" kernel_dir=""
      log "would clone/build SGLang source ${SGLANG_REPO}@${SGLANG_REF} under ${sglang_root}"
      if kernel_dir="$(sglang_kernel_rocm_build_dir "$sglang_root" 2>/dev/null)"; then
        log "would build in-tree ROCm kernel via ${kernel_dir}/setup_rocm.py"
      else
        log "would build in-tree ROCm kernel via setup_rocm.py after clone (sgl-kernel/ legacy or python/sglang/kernels/aot/ for v0.5.17+)"
      fi
      log "would install SGLang source with [srt_hip] runtime dependencies under current torch/triton constraints"
      log "would set SGLANG_BUILD_RUST_EXTS=${SGLANG_BUILD_RUST_EXTS:-none} (editable install; override to build multimodal Rust crates)"
    fi
    if [ -n "$AITER_REF" ]; then
      log "would clone/update ${AITER_REPO}@${AITER_REF} at ${aiter_root} and install it with current torch/triton constraints"
    else
      log "would auto-select the newest compatible AITER tag from ${AITER_REPO} and install it with current torch/triton constraints"
    fi
    return 0
  fi

  if ! _py_has "$py" sglang || ! _py_has "$py" sgl_kernel; then
    if [ "$py_mm" = "3.10" ]; then
      install_sglang_from_wheel "$py"
    else
      warn "amd-sglang ROCm 7.2 wheel currently pulls cp310 torch; Python ${py_mm} uses source install instead"
      install_sglang_from_source "$py" "$deps_root"
    fi
  else
    log "sglang + sgl_kernel already importable; skipping amd-sglang install"
  fi

  if ! "$py" -c "import aiter" >/dev/null 2>&1; then
    install_compatible_aiter "$py" "$aiter_root"
  else
    log "aiter already importable; skipping AITER source install"
  fi

  "$py" -c "import sglang" >/dev/null || die "sglang not importable after install"
  "$py" -c "import sgl_kernel" >/dev/null || die "sgl_kernel not importable after install"
  "$py" -c "import aiter" >/dev/null || die "aiter not importable after install"
  export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
  log "SGLang framework install complete (SGLANG_USE_AITER=${SGLANG_USE_AITER})"
}

# Verify that the installed vLLM package resolves to a ROCm runtime.
verify_vllm_rocm() {
  local py="$1"
  "$py" - <<'PY'
import sys

import torch

if not getattr(torch.version, "hip", None):
    print("torch is not a ROCm build", file=sys.stderr)
    raise SystemExit(1)

import vllm  # noqa: F401

try:
    from vllm.platforms import current_platform
except Exception as exc:
    print(f"could not import vllm platform detector: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

is_rocm = False
checker = getattr(current_platform, "is_rocm", None)
if callable(checker):
    try:
        is_rocm = bool(checker())
    except Exception:
        is_rocm = False
platform_text = f"{current_platform!r} {current_platform.__class__.__name__}".lower()
if "rocm" in platform_text:
    is_rocm = True

print(f"vllm platform: {current_platform!r}")
if not is_rocm:
    print("vLLM is importable but did not report a ROCm platform", file=sys.stderr)
    raise SystemExit(1)
PY
}

# Symlink the isolated-venv `vllm` console script into the shared venv's bin
# dir. Magpie runs a bare `vllm serve` under a PATH that only lists the shared
# bin; the symlink's target shebang still pins the isolated python, so the
# shared env stays uncontaminated. Idempotent; non-fatal on failure.
link_vllm_into_shared_bin() {
  local base_py="$1" iso_py="$2" shared_bin iso_vllm
  shared_bin="$(cd "$(dirname "$base_py")" 2>/dev/null && pwd)" || { warn "cannot resolve shared venv bin dir; skipping vllm symlink"; return 0; }
  iso_vllm="${VLLM_VENV_ROOT}/bin/vllm"
  [ -x "$iso_vllm" ] || { warn "isolated vllm not found at ${iso_vllm}; skipping symlink"; return 0; }
  if [ "$shared_bin" = "$(cd "$(dirname "$iso_py")" 2>/dev/null && pwd)" ]; then
    return 0
  fi
  if ln -sfnT "$iso_vllm" "${shared_bin}/vllm" 2>/dev/null || ln -sfn "$iso_vllm" "${shared_bin}/vllm" 2>/dev/null; then
    log "linked isolated vllm into shared bin: ${shared_bin}/vllm -> ${iso_vllm}"
  else
    warn "failed to symlink ${iso_vllm} into ${shared_bin}; Magpie may not find 'vllm'"
  fi
}

# Install vLLM from the official ROCm wheel index without replacing ROCm torch.
install_vllm_framework() {
  local py base_py py_mm constraint_file package_spec rocm_torch_ver
  base_py="$(resolve_python)" || die "no usable Python found for vLLM install"
  py="$base_py"
  if [ "$FRAMEWORK_ENV" = "isolated" ]; then
    py="${VLLM_VENV_ROOT}/bin/python"
  fi
  if [ "$FRAMEWORK_ENV" = "shared" ]; then
    py_mm="$("$py" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  else
    py_mm="$("$base_py" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  fi
  case "$VLLM_VERSION" in
    *+*) package_spec="vllm==${VLLM_VERSION}" ;;
    *) package_spec="vllm==${VLLM_VERSION}+${VLLM_ROCM_VARIANT}" ;;
  esac

  log "Phase 2: installing vLLM ROCm framework layer"
  log "framework env: ${FRAMEWORK_ENV}"
  log "framework python: ${py}"
  [ "$FRAMEWORK_ENV" = "isolated" ] && log "VLLM_VENV_ROOT=${VLLM_VENV_ROOT}"
  log "VLLM_VERSION=${VLLM_VERSION}"
  log "VLLM_ROCM_VARIANT=${VLLM_ROCM_VARIANT}"
  log "VLLM_ROCM_INDEX=${VLLM_ROCM_INDEX}"

  if [ "$CHECK_ONLY" -eq 1 ]; then
    if [ "$FRAMEWORK_ENV" = "isolated" ] && [ ! -x "$py" ]; then
      warn "vllm isolated env missing (check-only; would create ${VLLM_VENV_ROOT} and install ${package_spec})"
    elif _py_has "$py" vllm; then
      verify_vllm_rocm "$py" && log "vllm import OK (ROCm platform)"
    else
      warn "vllm missing (check-only; would install ${package_spec} from ${VLLM_ROCM_INDEX})"
    fi
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$py_mm" != "3.12" ]; then
      warn "vLLM ROCm wheels require Python 3.12; current Python is ${py_mm}"
    fi
    if [ "$FRAMEWORK_ENV" = "isolated" ]; then
      log "would create/update isolated vLLM venv at ${VLLM_VENV_ROOT}"
      log "would pin ROCm torch from ${VLLM_ROCM_INDEX}/torch/ before installing vLLM"
      log "would run: ${py} -m pip install --upgrade --extra-index-url ${VLLM_ROCM_INDEX} ${package_spec}"
    else
      log "would write ROCm torch/triton constraints"
      log "would run: ${py} -m pip install --upgrade --constraint <rocm-constraints> --extra-index-url ${VLLM_ROCM_INDEX} ${package_spec}"
    fi
    log "would verify vLLM reports a ROCm platform after install"
    return 0
  fi

  [ "$py_mm" = "3.12" ] || die "vLLM ROCm wheels require Python 3.12; current Python is ${py_mm}"
  if [ "$FRAMEWORK_ENV" = "isolated" ]; then
    mkdir -p "$(dirname "$VLLM_VENV_ROOT")"
    if [ ! -x "$py" ]; then
      "$base_py" -m venv "$VLLM_VENV_ROOT"
    fi
    "$py" -m pip install --upgrade pip wheel setuptools
    # Pin the ROCm torch first so the vLLM install below cannot fall back to a
    # CUDA torch from PyPI. The snapshot torch carries a local version tag (e.g.
    # +git<sha>) that PyPI lacks, so an exact pin forces the ROCm build; vLLM then
    # reuses the already-satisfied torch.
    rocm_torch_ver="$("$py" - "$VLLM_ROCM_INDEX" <<'PY'
import re, sys, urllib.request, urllib.error

base = sys.argv[1].rstrip("/") + "/torch/"
try:
    html = urllib.request.urlopen(base, timeout=30).read().decode()
except urllib.error.HTTPError as e:
    print("index unreachable: HTTP %s at %s" % (e.code, base), file=sys.stderr)
    raise SystemExit(0)
except Exception as e:
    print("index unreachable: %s at %s" % (e, base), file=sys.stderr)
    raise SystemExit(0)
# Match the plain "+local" form (e.g. 2.10.0+git8514f05); URL-encoded %2B
# anchors are skipped so the pinned spec stays pip-usable.
matches = re.findall(r"torch-([0-9][0-9A-Za-z.]*\+[0-9A-Za-z.]+)-cp", html)
if not matches:
    print("index reachable but no torch wheel matched at %s" % base, file=sys.stderr)
print(matches[-1] if matches else "")
PY
)"
    if [ -n "$rocm_torch_ver" ]; then
      log "pinning isolated ROCm torch==${rocm_torch_ver}"
      "$py" -m pip install --upgrade --extra-index-url "$VLLM_ROCM_INDEX" "torch==${rocm_torch_ver}" \
        || die "failed to install ROCm torch==${rocm_torch_ver} into ${VLLM_VENV_ROOT}"
    else
      warn "could not resolve a ROCm torch version from ${VLLM_ROCM_INDEX}/torch/; vLLM install will rely on its own torch pin"
    fi
    if ! "$py" -m pip install --upgrade \
      --extra-index-url "$VLLM_ROCM_INDEX" \
      "$package_spec"; then
      die "vLLM isolated install failed. Try a VLLM_VERSION/VLLM_ROCM_VARIANT available from ${VLLM_ROCM_INDEX}."
    fi
    verify_vllm_rocm "$py" || die "vLLM isolated install completed but did not verify as a ROCm build"
    # Isolated venv holds `vllm` only under $VLLM_VENV_ROOT/bin, but Magpie's
    # benchmark wrapper runs a bare `vllm serve` with a YAML-fixed PATH that
    # lists the shared venv bin only. Symlink the isolated vllm into the shared
    # bin so `which vllm` resolves; its shebang keeps using the isolated python.
    link_vllm_into_shared_bin "$base_py" "$py"
    log "vLLM framework install complete (isolated: ${VLLM_VENV_ROOT})"
    return 0
  fi

  constraint_file="$(mktemp)"
  write_rocm_torch_constraints "$py" "$constraint_file"
  if ! "$py" -m pip install --upgrade \
    --constraint "$constraint_file" \
    --extra-index-url "$VLLM_ROCM_INDEX" \
    "$package_spec"; then
    rm -f "$constraint_file"
    die "vLLM install failed; ROCm torch/triton constraints prevented an unsafe dependency change. Try a VLLM_VERSION/VLLM_ROCM_VARIANT matching the installed ROCm torch, or install vLLM in a separate environment."
  fi
  rm -f "$constraint_file"
  if ! verify_vllm_rocm "$py"; then
    "$py" -m pip uninstall -y vllm >/dev/null 2>&1 || true
    die "vLLM installed but did not verify as a ROCm build; removed vllm package"
  fi
  log "vLLM framework install complete"
}

# Dispatch the optional bare-metal framework installer.
install_requested_framework() {
  case "$INSTALL_FRAMEWORK" in
    none) log "Phase 2: framework install skipped (--install-framework none)" ;;
    sglang) install_sglang_framework ;;
    vllm) install_vllm_framework ;;
  esac
}

rocm_profiler_hotfix_compatible() {
  local py hip
  py="$(resolve_python 2>/dev/null)" || { warn "cannot resolve Python; skipping ROCm profiler hotfix"; return 1; }
  hip="$("$py" - <<'PY' 2>/dev/null || true
try:
    import torch
    print(getattr(torch.version, "hip", None) or "")
except Exception:
    pass
PY
)"
  case "$hip" in
    7.2*) log "torch.version.hip=${hip}; ROCm profiler hotfix is eligible" ;;
    "") warn "torch ROCm runtime not importable; skipping ROCm profiler hotfix" ; return 1 ;;
    *) warn "torch.version.hip=${hip}; ROCm profiler hotfix is validated for ROCm 7.2 stacks, skipping" ; return 1 ;;
  esac

  # Gate on any importable engine; torch/lib sync narrows to sglang/vllm only.
  local found="" fw probe_py _hotfix_arr
  IFS=',' read -r -a _hotfix_arr <<< "$FRAMEWORKS"
  for fw in "${_hotfix_arr[@]}"; do
    fw="$(echo "$fw" | tr -d '[:space:]')"; [ -z "$fw" ] && continue
    probe_py="$(framework_probe_python "$fw" "$py")"
    _py_has "$probe_py" "$fw" && found="${found:+${found} }${fw}"
  done
  [ -n "$found" ] || { warn "no serving framework importable from '${FRAMEWORKS}'; skipping ROCm profiler hotfix"; return 1; }
  log "framework imports: ${found}"

  # Container images: sglang needs the overlay; vLLM ships its own workaround.
  local run_mode
  run_mode="$(read_dotenv_var HYPERLOOM_RUN_MODE | tr -d '[:space:]')"
  if running_in_container || [ "$run_mode" = "docker" ]; then
    case " ${found} " in
      *" sglang "*) log "container run with sglang; ROCm profiler hotfix is eligible" ;;
      *) warn "container run without sglang (found: ${found}); skipping ROCm profiler hotfix" ; return 1 ;;
    esac
  else
    case " ${found} " in
      *" sglang "*) ;;
      *) warn "bare-metal run without sglang (found: ${found}); applying the hotfix anyway, unlike the container path" ;;
    esac
  fi
}

download_rocm_profiler_hotfix_libs() {
  local tmp_dir archive url
  tmp_dir="$(mktemp -d)"
  command -v curl >/dev/null 2>&1 || {
    rm -rf "$tmp_dir"
    warn "curl not found; cannot download ROCm profiler hotfix asset"
    return 1
  }
  # Public release asset; no auth needed on an open-source repo.
  url="https://github.com/${HYPERLOOM_WHEEL_REPO}/releases/download/${HYPERLOOM_WHEEL_TAG}/${ROCM_PROFILER_HOTFIX_ASSET}"
  archive="${tmp_dir}/${ROCM_PROFILER_HOTFIX_ASSET}"
  log "downloading ROCm profiler hotfix asset ${ROCM_PROFILER_HOTFIX_ASSET} from ${url}" >&2
  if ! curl -fSL -o "$archive" "$url" >&2; then
    rm -rf "$tmp_dir"
    warn "failed to download ${ROCM_PROFILER_HOTFIX_ASSET} from ${url}"
    return 1
  fi
  if ! tar -xzf "$archive" -C "$tmp_dir"; then
    rm -rf "$tmp_dir"
    warn "failed to extract ${ROCM_PROFILER_HOTFIX_ASSET}"
    return 1
  fi
  if ! find "$tmp_dir" -maxdepth 1 -type f -name 'libamdhip64.so.7.*' | grep -q . \
     || ! find "$tmp_dir" -maxdepth 1 -type f -name 'libroctracer64.so.4.*' | grep -q .; then
    rm -rf "$tmp_dir"
    warn "${ROCM_PROFILER_HOTFIX_ASSET} does not contain the expected ROCm hotfix libraries"
    return 1
  fi
  printf '%s\n' "$tmp_dir"
}

# The loader looks up the versioned SONAME, not the plain file name.
_soname_of() {
  local f="$1" s=""
  [ -f "$f" ] || return 1
  if command -v readelf >/dev/null 2>&1; then
    s="$(readelf -d "$f" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p' | head -n 1)"
  fi
  if [ -z "$s" ] && command -v objdump >/dev/null 2>&1; then
    s="$(objdump -p "$f" 2>/dev/null | awk '/SONAME/ {print $2; exit}')"
  fi
  [ -n "$s" ] || return 1
  printf '%s\n' "$s"
}

# Base is the highest-versioned non-hotfix file, i.e. the vendor library. Reading
# it from the soname link instead would rank against an already-installed hotfix.
_hotfix_installed_name() {
  local target_dir="$1" soname="$2" asset="$3" base tail
  base="$(find "$target_dir" -maxdepth 1 -type f -name "${soname}.*" -printf '%f\n' 2>/dev/null \
    | grep -v -- "$ROCM_PROFILER_HOTFIX_SUFFIX" | sort -V | tail -n 1)"
  [ -n "$base" ] || return 1
  tail="$(basename "$asset")"
  printf '%s%s.%s\n' "$base" "$ROCM_PROFILER_HOTFIX_SUFFIX" "${tail#"${soname%%.so.*}.so."}"
}

install_rocm_profiler_hotfix_ranked() {
  local source_dir="$1" target_dir="$2" hip_lib="$3" tracer_lib="$4"
  local name soname installed unversioned

  for name in "$hip_lib" "$tracer_lib"; do
    soname="$(_soname_of "${source_dir}/${name}")" \
      || { warn "cannot read SONAME of ${name}"; return 1; }
    installed="$(_hotfix_installed_name "$target_dir" "$soname" "${source_dir}/${name}")" \
      || { warn "cannot derive a ranked name for ${soname}"; return 1; }
    log "installing ${name} as ${installed}"
    install -m 0644 "${source_dir}/${name}" "${target_dir}/${installed}" || return 1
    unversioned="${soname%%.so.*}.so"
    ln -sfnT "$installed" "${target_dir}/${soname}" || return 1
    ln -sfnT "$soname" "${target_dir}/${unversioned}" || return 1
  done
  command -v ldconfig >/dev/null 2>&1 && { ldconfig || return 1; }
  return 0
}

rocm_profiler_hotfix_ranked_applied() {
  local target_dir="$1" source_dir="$2" hip_lib="$3" tracer_lib="$4"
  local name soname resolved expected
  for name in "$hip_lib" "$tracer_lib"; do
    soname="$(_soname_of "${source_dir}/${name}")" || return 1
    expected="$(_hotfix_installed_name "$target_dir" "$soname" "${source_dir}/${name}")" || return 1
    resolved="$(readlink -f "${target_dir}/${soname}" 2>/dev/null || true)"
    [ "$(basename "$resolved")" = "$expected" ] || return 1
    cmp -s "${source_dir}/${name}" "$resolved" || return 1
  done
}

backup_rocm_profiler_hotfix_links() {
  local target_dir="$1" backup_dir="$2" path
  install -d "$backup_dir" || return 1
  for path in \
    "${target_dir}/libamdhip64.so" \
    "${target_dir}/libamdhip64.so.7" \
    "${target_dir}/libroctracer64.so" \
    "${target_dir}/libroctracer64.so.4"; do
    [ -e "$path" ] || [ -L "$path" ] || continue
    cp -a "$path" "$backup_dir"/ || return 1
  done
}

rollback_rocm_profiler_hotfix_ranked() {
  local backup_dir="$1" target_dir="$2" path
  [ -d "$backup_dir" ] || return 1
  for path in "${target_dir}"/*"${ROCM_PROFILER_HOTFIX_SUFFIX}"*; do
    [ -e "$path" ] && rm -f "$path"
  done
  cp -a "${backup_dir}"/. "$target_dir"/ || return 1
  command -v ldconfig >/dev/null 2>&1 && { ldconfig || return 1; }
  return 0
}

# Compares content, not paths: /opt/rocm is itself a symlink.
verify_rocm_profiler_hotfix_links() {
  local target_dir="$1" source_dir="$2" hip_lib="$3" tracer_lib="$4"
  local name soname resolved py args=""

  log "verifying ROCm profiler hotfix links in ${target_dir}"
  for name in "$hip_lib" "$tracer_lib"; do
    soname="$(_soname_of "${source_dir}/${name}")" || { warn "cannot read SONAME of ${name}"; return 1; }
    resolved="$(readlink -f "${target_dir}/${soname}" 2>/dev/null || true)"
    log "${soname} -> ${resolved:-<unresolved>}"
    cmp -s "${source_dir}/${name}" "$resolved" \
      || { warn "${soname} does not resolve to the hotfix build"; return 1; }
    log "ldconfig: $(ldconfig -p 2>/dev/null | awk -v s="$soname" '$1 == s {print $NF; exit}')"
    args="${args:+${args} }${soname}=${source_dir}/${name}"
  done

  py="$(resolve_python)" || return 1
  # shellcheck disable=SC2086
  "$py" - $args <<'PY'
import ctypes
import hashlib
import pathlib
import sys

wanted = dict(arg.split("=", 1) for arg in sys.argv[1:])
digests = {
    soname: hashlib.sha256(pathlib.Path(ref).read_bytes()).hexdigest()
    for soname, ref in wanted.items()
}
for soname in wanted:
    ctypes.CDLL(soname)
    print(f"loaded by soname: {soname}")

stems = {soname.split(".so")[0]: soname for soname in wanted}
mapped = {}
with open("/proc/self/maps", encoding="utf-8") as fh:
    for line in fh:
        path = line.rstrip("\n").rsplit(" ", 1)[-1]
        if not path.startswith("/"):
            continue
        base = path.rsplit("/", 1)[-1]
        for stem, soname in stems.items():
            if base.startswith(stem):
                mapped.setdefault(soname, path)

failed = []
for soname, path in mapped.items():
    got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    ok = got == digests[soname]
    print(f"{soname}: mapped {path} -> {'hotfix' if ok else 'NOT the hotfix'}")
    if not ok:
        failed.append(soname)
missing = sorted(set(wanted) - set(mapped))
if missing:
    print("not mapped: " + ", ".join(missing))
if failed:
    raise SystemExit("the loader picked a non-hotfix library: " + ", ".join(failed))
PY
}

# Confirms which profiler libs the framework interpreter actually mapped.
verify_rocm_profiler_hotfix_loaded() {
  local source_dir="$1" hip_lib="$2" tracer_lib="$3"
  local default_py pairs py torch_dir rc=0

  default_py="$(resolve_python)" || return 1
  pairs="$(collect_framework_torch_lib_dirs "$default_py")"
  [ -n "$pairs" ] || { log "no framework torch/lib to check"; return 0; }

  while IFS="$(printf '\t')" read -r py torch_dir; do
    log "checking mapped profiler libs for ${py}"
    "$py" - "${source_dir}/${hip_lib}" "${source_dir}/${tracer_lib}" <<'PY' || rc=1
import hashlib
import pathlib
import sys

refs = [pathlib.Path(p) for p in sys.argv[1:]]
expected = {}
for ref in refs:
    stem = ref.name.split(".so")[0]
    expected[stem] = hashlib.sha256(ref.read_bytes()).hexdigest()

import torch  # noqa: E402

torch.zeros(1)
mapped = {}
with open("/proc/self/maps", encoding="utf-8") as fh:
    for line in fh:
        path = line.rstrip("\n").rsplit(" ", 1)[-1]
        if not path.startswith("/"):
            continue
        name = path.rsplit("/", 1)[-1]
        for stem in expected:
            if name.startswith(stem):
                mapped.setdefault(stem, path)

failed = []
for stem, want in expected.items():
    path = mapped.get(stem)
    if path is None:
        print(f"{stem}: not mapped (framework may load it lazily)")
        continue
    got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    print(f"{stem}: {path} -> {'hotfix' if got == want else 'NOT the hotfix'}")
    if got != want:
        failed.append(stem)
if failed:
    raise SystemExit("mapped profiler libs are not the hotfix: " + ", ".join(failed))
PY
  done <<< "$pairs"
  return "$rc"
}

resolve_torch_lib_dir() {
  local py="$1"
  "$py" - <<'PY' 2>/dev/null || true
import pathlib

try:
    import torch

    print(pathlib.Path(torch.__file__).resolve().parent / "lib")
except Exception:
    pass
PY
}

ROCM_PROFILER_HOTFIX_TORCH_LIBS="libamdhip64.so libroctracer64.so"
ROCM_PROFILER_HOTFIX_SYNC_FRAMEWORKS="sglang vllm"
ROCM_PROFILER_HOTFIX_TORCH_BACKUP_DIR=".profiler_hotfix_backup"
ROCM_PROFILER_HOTFIX_FINGERPRINT=".fingerprint"
ROCM_PROFILER_HOTFIX_SONAME_MARKER=".hotfix_soname_links"

_sha256_file() {
  local f="$1"
  [ -f "$f" ] || { printf 'absent'; return 0; }
  sha256sum "$f" | awk '{print $1}'
}

_fingerprint_field() {
  local fp="$1" kind="$2" name="$3"
  grep "^${kind}:${name}:" "$fp" 2>/dev/null | cut -d: -f3- || true
}

_write_backup_fingerprint() {
  local fp="$1" backup_dir="$2" rocm_lib_dir="$3" name src tmp="${fp}.tmp"
  rm -f "$tmp"
  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    if [ -e "${backup_dir}/${name}" ]; then
      printf 'vendor:%s:%s\n' "$name" "$(_sha256_file "${backup_dir}/${name}")"
    else
      printf 'vendor:%s:absent\n' "$name"
    fi
    src="$(readlink -f "${rocm_lib_dir}/${name}" 2>/dev/null)" || src=""
    printf 'hotfix:%s:%s\n' "$name" "$(_sha256_file "$src")"
  done > "$tmp" || return 1
  mv -f "$tmp" "$fp" || return 1
}

# Vendor snapshot is stale only when torch's on-disk libs are neither the
# recorded vendor, the recorded hotfix, nor the current hotfix source.
_torch_vendor_stale() {
  local backup_dir="$1" torch_lib_dir="$2" rocm_lib_dir="$3"
  local fp="${backup_dir}/${ROCM_PROFILER_HOTFIX_FINGERPRINT}" name src torch_sha vendor_sha hotfix_sha

  [ -f "$fp" ] || return 0
  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    vendor_sha="$(_fingerprint_field "$fp" vendor "$name")"
    hotfix_sha="$(_fingerprint_field "$fp" hotfix "$name")"
    [ -n "$vendor_sha" ] || return 0
    if [ ! -e "${torch_lib_dir}/${name}" ]; then
      [ "$vendor_sha" = "absent" ] || return 0
      continue
    fi
    torch_sha="$(_sha256_file "${torch_lib_dir}/${name}")"
    src="$(readlink -f "${rocm_lib_dir}/${name}" 2>/dev/null)" || src=""
    if [ -f "$src" ] && [ "$torch_sha" = "$(_sha256_file "$src")" ]; then
      if [ -e "${backup_dir}/${name}" ]; then
        [ "$(_sha256_file "${backup_dir}/${name}")" = "$vendor_sha" ] || return 0
      elif [ "$vendor_sha" != "absent" ]; then
        return 0
      fi
      continue
    fi
    [ "$torch_sha" = "$hotfix_sha" ] && continue
    [ "$torch_sha" = "$vendor_sha" ] && continue
    return 0
  done
  return 1
}

_framework_torch_lib_missing() {
  local default_py="$1" fw probe_py dir
  for fw in $ROCM_PROFILER_HOTFIX_SYNC_FRAMEWORKS; do
    probe_py="$(framework_probe_python "$fw" "$default_py")"
    _py_has "$probe_py" "$fw" || continue
    dir="$(resolve_torch_lib_dir "$probe_py")"
    [ -n "$dir" ] && [ -d "$dir" ] || return 0
  done
  return 1
}

collect_framework_torch_lib_dirs() {
  local default_py="$1" fw probe_py dir seen=""
  for fw in $ROCM_PROFILER_HOTFIX_SYNC_FRAMEWORKS; do
    probe_py="$(framework_probe_python "$fw" "$default_py")"
    _py_has "$probe_py" "$fw" || continue
    dir="$(resolve_torch_lib_dir "$probe_py")"
    [ -n "$dir" ] && [ -d "$dir" ] || continue
    case " ${seen} " in *" ${dir} "*) continue ;; esac
    seen="${seen:+${seen} }${dir}"
    printf '%s\t%s\n' "$probe_py" "$dir"
  done
}

verify_torch_runtime() {
  local py="$1"
  "$py" - <<'PY'
import torch

if not getattr(torch.version, "hip", None):
    raise SystemExit("torch.version.hip is empty")
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
PY
}

ensure_torch_lib_backup() {
  local backup_dir="$1" torch_lib_dir="$2" rocm_lib_dir="$3"
  local name staging fp="${backup_dir}/${ROCM_PROFILER_HOTFIX_FINGERPRINT}"
  local old_backup old_fp vendor_sha hotfix_sha torch_sha src

  if [ -d "$backup_dir" ]; then
    _torch_vendor_stale "$backup_dir" "$torch_lib_dir" "$rocm_lib_dir" || return 0
    log "torch vendor libs changed; refreshing ${backup_dir}"
    old_backup="${backup_dir}.preserve"
    rm -rf "$old_backup"
    mv "$backup_dir" "$old_backup" || return 1
    old_fp="${old_backup}/${ROCM_PROFILER_HOTFIX_FINGERPRINT}"

    staging="${backup_dir}.staging"
    rm -rf "$staging"
    install -d "$staging" || { rm -rf "$old_backup"; return 1; }
    for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
      [ -e "${torch_lib_dir}/${name}" ] || continue
      vendor_sha="$(_fingerprint_field "$old_fp" vendor "$name")"
      hotfix_sha="$(_fingerprint_field "$old_fp" hotfix "$name")"
      torch_sha="$(_sha256_file "${torch_lib_dir}/${name}")"
      src="$(readlink -f "${rocm_lib_dir}/${name}" 2>/dev/null)" || src=""
      if [ "$torch_sha" = "$hotfix_sha" ] \
          || { [ -f "$src" ] && [ "$torch_sha" = "$(_sha256_file "$src")" ]; }; then
        [ -e "${old_backup}/${name}" ] \
          && cp -a "${old_backup}/${name}" "${staging}/${name}" || true
      else
        cp -a "${torch_lib_dir}/${name}" "${staging}/${name}" \
          || { rm -rf "$staging" "$old_backup"; return 1; }
      fi
    done
    mv -T "$staging" "$backup_dir" || { rm -rf "$staging" "$old_backup"; return 1; }
    rm -rf "$old_backup"
    _write_backup_fingerprint "$fp" "$backup_dir" "$rocm_lib_dir"
    return 0
  fi

  staging="${backup_dir}.staging"
  rm -rf "$staging"
  install -d "$staging" || return 1
  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    [ -e "${torch_lib_dir}/${name}" ] || continue
    cp -a "${torch_lib_dir}/${name}" "${staging}/${name}" || { rm -rf "$staging"; return 1; }
  done
  mv -T "$staging" "$backup_dir" || { rm -rf "$staging"; return 1; }
  _write_backup_fingerprint "$fp" "$backup_dir" "$rocm_lib_dir"
}

restore_torch_libs() {
  local backup_dir="$1" torch_lib_dir="$2" name
  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    # What we installed is a symlink, and cp would write through it.
    rm -f "${torch_lib_dir}/${name}" || return 1
    if [ -e "${backup_dir}/${name}" ]; then
      cp -a "${backup_dir}/${name}" "${torch_lib_dir}/${name}" || return 1
    fi
  done
}

# torch/lib is outside every ldconfig search path, so aliases here survive apt.
link_torch_soname_aliases() {
  local torch_lib_dir="$1" src_dir="$2"
  local backup_dir="${torch_lib_dir}/${ROCM_PROFILER_HOTFIX_TORCH_BACKUP_DIR}"
  local marker="${backup_dir}/${ROCM_PROFILER_HOTFIX_SONAME_MARKER}"
  local name src soname target

  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    src="$(readlink -f "${src_dir}/${name}" 2>/dev/null)" || src=""
    [ -f "$src" ] || continue
    soname="$(_soname_of "$src")" || continue
    [ "$soname" = "$name" ] && continue
    target="${torch_lib_dir}/${soname}"
    install -d "$backup_dir" || return 1
    # A real file here was shipped by torch; keep it before shadowing.
    if [ -e "$target" ] && [ ! -L "$target" ] && [ ! -e "${backup_dir}/${soname}" ]; then
      cp -a "$target" "${backup_dir}/${soname}" || return 1
    fi
    ln -sfnT "$name" "$target" || return 1
    grep -qxF "$soname" "$marker" 2>/dev/null || printf '%s\n' "$soname" >> "$marker"
    log "torch soname alias ${soname} -> ${name} (${torch_lib_dir})"
  done
}

_torch_soname_aliases_ok() {
  local torch_lib_dir="$1" src_dir="$2" name src soname
  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    src="$(readlink -f "${src_dir}/${name}" 2>/dev/null)" || src=""
    [ -f "$src" ] || continue
    soname="$(_soname_of "$src")" || continue
    [ "$soname" = "$name" ] && continue
    [ "$(readlink "${torch_lib_dir}/${soname}" 2>/dev/null || true)" = "$name" ] || return 1
  done
}

restore_torch_soname_aliases() {
  local backup_dir="$1" torch_lib_dir="$2"
  local marker="${backup_dir}/${ROCM_PROFILER_HOTFIX_SONAME_MARKER}" soname
  [ -f "$marker" ] || return 0
  while read -r soname; do
    [ -n "$soname" ] || continue
    # The alias we installed is a symlink, and cp would write through it.
    rm -f "${torch_lib_dir}/${soname}" || return 1
    if [ -e "${backup_dir}/${soname}" ]; then
      cp -a "${backup_dir}/${soname}" "${torch_lib_dir}/${soname}" || return 1
    fi
  done < "$marker"
  rm -f "$marker"
}

sync_one_torch_lib() {
  local rocm_lib_dir="$1" py="$2" torch_lib_dir="$3"
  local backup_dir="${torch_lib_dir}/${ROCM_PROFILER_HOTFIX_TORCH_BACKUP_DIR}"
  local name src dst pending=""

  for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
    src="$(readlink -f "${rocm_lib_dir}/${name}" 2>/dev/null)" || src=""
    if [ ! -f "$src" ]; then
      if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
        log "${name} not installed under ${rocm_lib_dir} yet; nothing to sync in this mode"
        return 0
      fi
      warn "${name} unresolved under ${rocm_lib_dir}"
      return 1
    fi
    if [ ! -L "${torch_lib_dir}/${name}" ] \
        || [ "$(readlink -f "${torch_lib_dir}/${name}" 2>/dev/null)" != "$src" ]; then
      pending="${pending:+${pending} }${name}"
    fi
  done
  if [ -z "$pending" ] && _torch_soname_aliases_ok "$torch_lib_dir" "$rocm_lib_dir"; then
    log "torch profiler libs already in sync (${torch_lib_dir})"
    return 0
  fi

  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    log "would link ${pending:-<no file changes>} from ${rocm_lib_dir} into ${torch_lib_dir}"
    log "would link versioned soname aliases in ${torch_lib_dir}"
    return 0
  fi

  ensure_torch_lib_backup "$backup_dir" "$torch_lib_dir" "$rocm_lib_dir" \
    || { warn "cannot back up ${torch_lib_dir}"; return 1; }

  [ -n "$pending" ] && log "linking ${pending} into torch lib (${torch_lib_dir})"
  for name in $pending; do
    src="$(readlink -f "${rocm_lib_dir}/${name}")"
    dst="${torch_lib_dir}/${name}"
    # Link rather than copy: a byte-identical copy is a second inode, so glibc
    # maps both it and the ROCm original and each gets its own HIP runtime.
    if ! ln -sfnT "$src" "$dst"; then
      warn "failed to link ${dst} -> ${src}"
      restore_torch_libs "$backup_dir" "$torch_lib_dir" \
        || die "cannot restore ${torch_lib_dir} from ${backup_dir}"
      return 1
    fi
  done

  link_torch_soname_aliases "$torch_lib_dir" "$rocm_lib_dir" || {
    warn "failed to link soname aliases in ${torch_lib_dir}"
    restore_torch_soname_aliases "$backup_dir" "$torch_lib_dir" || true
    restore_torch_libs "$backup_dir" "$torch_lib_dir" \
      || die "cannot restore ${torch_lib_dir} from ${backup_dir}"
    return 1
  }

  if ! verify_torch_runtime "$py"; then
    warn "torch did not come up after the sync; restoring ${torch_lib_dir}"
    restore_torch_soname_aliases "$backup_dir" "$torch_lib_dir" || true
    restore_torch_libs "$backup_dir" "$torch_lib_dir" \
      || die "cannot restore ${torch_lib_dir} from ${backup_dir}"
    for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
      if [ -e "${backup_dir}/${name}" ] || [ -e "${torch_lib_dir}/${name}" ]; then
        cmp -s "${backup_dir}/${name}" "${torch_lib_dir}/${name}" \
          || die "restored ${torch_lib_dir} does not match ${backup_dir}"
      fi
    done
    warn "vendor libraries restored; continuing without the torch lib sync"
    return 1
  fi
  log "torch profiler libs synced into ${torch_lib_dir}"
}

# torch searches its bundled torch/lib first, but only for the versioned soname.
sync_rocm_profiler_libs_to_torch_lib() {
  local rocm_lib_dir="${1:-${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR}}"
  local default_py pairs py dir rc=0

  default_py="$(resolve_python)" || { warn "cannot resolve Python"; return 1; }
  pairs="$(collect_framework_torch_lib_dirs "$default_py")"
  if [ -z "$pairs" ]; then
    if _framework_torch_lib_missing "$default_py"; then
      warn "serving framework importable but torch/lib not resolved; skipping torch lib sync"
      return 1
    fi
    log "no torch/lib to sync for ${ROCM_PROFILER_HOTFIX_SYNC_FRAMEWORKS}"
    return 0
  fi

  while IFS="$(printf '\t')" read -r py dir; do
    sync_one_torch_lib "$rocm_lib_dir" "$py" "$dir" || rc=1
  done <<< "$pairs"
  return "$rc"
}

verify_rocm_profiler_torch_lib_sync() {
  local rocm_lib_dir="${1:-${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR}}"
  local default_py pairs dir name src rc=0

  default_py="$(resolve_python)" || return 1
  pairs="$(collect_framework_torch_lib_dirs "$default_py")"
  [ -n "$pairs" ] || return 0

  while IFS="$(printf '\t')" read -r _py dir; do
    for name in $ROCM_PROFILER_HOTFIX_TORCH_LIBS; do
      src="$(readlink -f "${rocm_lib_dir}/${name}" 2>/dev/null)" || src=""
      [ -f "$src" ] || continue
      cmp -s "$src" "${dir}/${name}" || { warn "torch ${name} differs in ${dir}"; rc=1; }
    done
    _torch_soname_aliases_ok "$dir" "$rocm_lib_dir" \
      || { warn "versioned soname aliases missing in ${dir}"; rc=1; }
  done <<< "$pairs"
  return "$rc"
}

sync_torch_profiler_libs() {
  local target_dir="$1" verify="${2:-0}" rc=0
  sync_rocm_profiler_libs_to_torch_lib "$target_dir" \
    || { warn "torch lib sync reported issues"; rc=1; }
  if [ "$verify" -eq 1 ]; then
    verify_rocm_profiler_torch_lib_sync "$target_dir" \
      || { warn "torch lib sync verification reported issues"; rc=1; }
  fi
  return "$rc"
}

apply_rocm_profiler_hotfix() {
  local target_dir="${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR}"
  local extract_dir backup_dir hip_lib tracer_lib

  log "Phase 3: applying ROCm profiler hotfix"
  log "ROCM_PROFILER_HOTFIX_ASSET=${ROCM_PROFILER_HOTFIX_ASSET}"

  [ -d "$target_dir" ] || { warn "ROCm library directory not found (${target_dir}); skipping profiler hotfix"; return 0; }
  rocm_profiler_hotfix_compatible || return 0

  if [ "$CHECK_ONLY" -eq 1 ]; then
    log "check-only: ROCm profiler hotfix release asset will not be downloaded"
    log "current libamdhip64.so -> $(readlink -f "${target_dir}/libamdhip64.so" 2>/dev/null || echo missing)"
    log "current libroctracer64.so -> $(readlink -f "${target_dir}/libroctracer64.so" 2>/dev/null || echo missing)"
    sync_torch_profiler_libs "$target_dir" || true
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "would download ${ROCM_PROFILER_HOTFIX_ASSET} from ${HYPERLOOM_WHEEL_REPO}@${HYPERLOOM_WHEEL_TAG}"
    log "would back up current ROCm profiler links under ${target_dir}/.profiler_hotfix_backup_<timestamp>"
    log "would install the hotfix as <current version>${ROCM_PROFILER_HOTFIX_SUFFIX} and run ldconfig"
    sync_torch_profiler_libs "$target_dir" || true
    return 0
  fi

  extract_dir="$(download_rocm_profiler_hotfix_libs)" \
    || { warn "could not obtain ROCm profiler hotfix libraries; skipping"; return 0; }
  hip_lib="$(basename "$(find "$extract_dir" -maxdepth 1 -type f -name 'libamdhip64.so.*' | sort | tail -n 1)")"
  tracer_lib="$(basename "$(find "$extract_dir" -maxdepth 1 -type f -name 'libroctracer64.so.*' | sort | tail -n 1)")"
  if rocm_profiler_hotfix_ranked_applied "$target_dir" "$extract_dir" "$hip_lib" "$tracer_lib"; then
    log "ROCm profiler hotfix already applied (${hip_lib}, ${tracer_lib})"
    verify_rocm_profiler_hotfix_links "$target_dir" "$extract_dir" "$hip_lib" "$tracer_lib" \
      || warn "existing ROCm profiler hotfix verification reported issues"
    sync_torch_profiler_libs "$target_dir" 1 || true
    verify_rocm_profiler_hotfix_loaded "$extract_dir" "$hip_lib" "$tracer_lib" \
      || warn "mapped profiler libs are not the hotfix"
    rm -rf "$extract_dir"
    return 0
  fi

  backup_dir="${target_dir}/.profiler_hotfix_backup_$(date -u +%Y%m%dT%H%M%SZ)"
  backup_rocm_profiler_hotfix_links "$target_dir" "$backup_dir" \
    || { rm -rf "$extract_dir"; die "cannot back up ${target_dir} profiler links"; }
  log "backed up current ROCm profiler links to ${backup_dir}"
  if ! install_rocm_profiler_hotfix_ranked "$extract_dir" "$target_dir" "$hip_lib" "$tracer_lib" \
     || ! verify_rocm_profiler_hotfix_links "$target_dir" "$extract_dir" "$hip_lib" "$tracer_lib"; then
    warn "ROCm profiler hotfix failed; attempting rollback from ${backup_dir}"
    if rollback_rocm_profiler_hotfix_ranked "$backup_dir" "$target_dir"; then
      warn "rollback succeeded; continuing without ROCm profiler hotfix"
      rm -rf "$extract_dir"
      return 0
    fi
    rm -rf "$extract_dir"
    die "ROCm profiler hotfix failed and rollback did not complete"
  fi
  if sync_torch_profiler_libs "$target_dir"; then
    log "ROCm profiler hotfix applied"
  else
    warn "ROCm profiler hotfix partially applied (ROCm links only; torch/lib sync failed)"
  fi
  verify_rocm_profiler_hotfix_loaded "$extract_dir" "$hip_lib" "$tracer_lib" \
    || warn "mapped profiler libs are not the hotfix; profiling runs on the vendor libraries"
  rm -rf "$extract_dir"
}

verify_rocm_profiler_hotfix_only() {
  local target_dir="${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR}" hip_lib tracer_lib rc=0

  log "verifying ROCm profiler hotfix state"
  hip_lib="$(basename "$(find "$target_dir" -maxdepth 1 -type f \
    -name "libamdhip64.so.*${ROCM_PROFILER_HOTFIX_SUFFIX}*" 2>/dev/null | sort -V | tail -n 1)")"
  tracer_lib="$(basename "$(find "$target_dir" -maxdepth 1 -type f \
    -name "libroctracer64.so.*${ROCM_PROFILER_HOTFIX_SUFFIX}*" 2>/dev/null | sort -V | tail -n 1)")"
  if [ -z "$hip_lib" ] || [ -z "$tracer_lib" ]; then
    warn "no ${ROCM_PROFILER_HOTFIX_SUFFIX} libraries under ${target_dir}; run setup without --verify-hotfix first"
    return 1
  fi

  verify_rocm_profiler_hotfix_links "$target_dir" "$target_dir" "$hip_lib" "$tracer_lib" || rc=1
  verify_rocm_profiler_torch_lib_sync "$target_dir" || rc=1
  verify_rocm_profiler_hotfix_loaded "$target_dir" "$hip_lib" "$tracer_lib" || rc=1
  if [ "$rc" -eq 0 ]; then
    log "ROCm profiler hotfix verified"
  else
    warn "ROCm profiler hotfix verification failed; re-run setup to repair it"
  fi
  return "$rc"
}

# Re-read framework env persisted by a prior setup run so Phase 3 targets the
# same interpreter/venv on re-runs with --install-framework none.
restore_persisted_framework_env() {
  if [ -z "$_FRAMEWORK_ENV_WAS_SET" ]; then
    local saved_fw
    saved_fw="$(read_dotenv_var HYPERLOOM_FRAMEWORK_ENV)"
    if [ -n "$saved_fw" ]; then
      FRAMEWORK_ENV="$saved_fw"
      log "restored framework env from .env (${FRAMEWORK_ENV})"
    elif [ "$INSTALL_FRAMEWORK" = "vllm" ]; then
      FRAMEWORK_ENV="isolated"
      log "vLLM selected; defaulting to isolated framework env"
    fi
  fi
  if [ -z "${_VLLM_VENV_ROOT_WAS_SET:-}" ]; then
    local saved_root
    saved_root="$(read_dotenv_var VLLM_VENV_ROOT)"
    if [ -n "$saved_root" ]; then
      VLLM_VENV_ROOT="$saved_root"
    fi
  fi
  return 0
}

read_dotenv_var() {
  local name="$1"
  [ -f "$DOTENV" ] || return 0
  # `|| true` keeps a no-match grep from tripping pipefail/set -e when this is
  # used inside a ${VAR:-$(read_dotenv_var ...)} default expansion.
  { grep -E "^[[:space:]]*(export[[:space:]]+)?${name}=" "$DOTENV" 2>/dev/null || true; } | tail -n 1 \
    | sed -E "s/^[[:space:]]*(export[[:space:]]+)?${name}=//; s/^[\"']//; s/[\"']$//"
}

# Upsert KEY=VALUE into .env, matching an optional leading-whitespace / export
# prefix so a pre-existing line is replaced (never duplicated). Pure-bash.
upsert_dotenv_var() {
  local key="$1" value="$2" tmp found=0 line stripped
  tmp="$(mktemp)"
  if [ -f "$DOTENV" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      stripped="${line#"${line%%[![:space:]]*}"}"
      stripped="${stripped#export }"
      case "$stripped" in
        "${key}="*) printf '%s=%s\n' "$key" "$value" >> "$tmp"; found=1 ;;
        *) printf '%s\n' "$line" >> "$tmp" ;;
      esac
    done < "$DOTENV"
  fi
  [ "$found" -eq 0 ] && printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mv "$tmp" "$DOTENV"
  chmod 600 "$DOTENV" 2>/dev/null || true
}

remove_dotenv_var() {
  local key="$1" tmp line stripped
  [ -f "$DOTENV" ] || return 0
  tmp="$(mktemp)"
  while IFS= read -r line || [ -n "$line" ]; do
    stripped="${line#"${line%%[![:space:]]*}"}"
    stripped="${stripped#export }"
    case "$stripped" in
      "${key}="*) ;;
      *) printf '%s\n' "$line" >> "$tmp" ;;
    esac
  done < "$DOTENV"
  mv "$tmp" "$DOTENV"
  chmod 600 "$DOTENV" 2>/dev/null || true
}

# 1 when the Anthropic and OpenAI sides are two protocols of ONE gateway, so
# the OpenAI side must survive the otherwise "Anthropic-only" .env cleanup.
# Decided in resolve_credentials from the resolved URLs, NOT from whether a
# legacy migration happened, so a hand-written two-sided config counts too.
DUAL_PROTOCOL_GATEWAY=0
# 1 once a retired DEEPSEEK_* configuration has been translated, so the legacy
# keys are dropped from .env only after credential validation passes.
LEGACY_DEEPSEEK_MIGRATED=0

# Return the authority (host[:port]) of a URL, or the empty string.
url_authority() {
  local rest="${1#*://}"
  printf '%s' "${rest%%/*}"
}

# Translate a retired DEEPSEEK_* configuration into the standard variables.
# DeepSeek is a dual-protocol gateway, not a third provider: /anthropic speaks
# the Anthropic API and /v1 speaks OpenAI chat-completions, both authenticated
# with the same key. Endpoint and model derivation matches
# hyperloom.common.llm_config.deepseek_compat_env; ANTHROPIC_AUTH_TOKEN is
# deliberately not written here because .env only ever persists the API-key
# spelling (see remove_dotenv_var ANTHROPIC_AUTH_TOKEN below).
migrate_legacy_deepseek_env() {
  local key url model base lowered anthropic_url openai_url
  key="${DEEPSEEK_API_KEY:-$(read_dotenv_var DEEPSEEK_API_KEY || true)}"
  url="${DEEPSEEK_BASE_URL:-$(read_dotenv_var DEEPSEEK_BASE_URL || true)}"
  model="${DEEPSEEK_MODEL:-$(read_dotenv_var DEEPSEEK_MODEL || true)}"
  if [ -z "$key" ] && [ -z "$url" ]; then
    return 0
  fi
  # Adopt the gateway whole or not at all. Anything already on the Anthropic
  # side means the retired variables are stale leftovers: half-adopting them
  # would send an explicit Anthropic credential to DeepSeek's host.
  if [ -n "${ANTHROPIC_BASE_URL:-$(read_dotenv_var ANTHROPIC_BASE_URL || true)}" ] \
     || [ -n "${ANTHROPIC_API_KEY:-$(read_dotenv_var ANTHROPIC_API_KEY || true)}" ] \
     || [ -n "${ANTHROPIC_AUTH_TOKEN:-$(read_dotenv_var ANTHROPIC_AUTH_TOKEN || true)}" ] \
     || [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-$(read_dotenv_var CLAUDE_CODE_OAUTH_TOKEN || true)}" ]; then
    warn "DEEPSEEK_* is retired and ignored here: the Anthropic side is already configured"
    return 0
  fi

  # Match the trailing segment case-insensitively (AMD spells it /Anthropic)
  # and swap it with ${base%/*}, which drops the final segment whatever its
  # casing. Same algorithm as dual_protocol_endpoint_pair().
  base="${url%/}"
  lowered="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    "")           anthropic_url="https://api.deepseek.com/anthropic"; openai_url="https://api.deepseek.com/v1" ;;
    */anthropic)  anthropic_url="$base"; openai_url="${base%/*}/v1" ;;
    */v1)         anthropic_url="${base%/*}/anthropic"; openai_url="$base" ;;
    https://api.deepseek.com|http://api.deepseek.com)
                  anthropic_url="${base}/anthropic"; openai_url="${base}/v1" ;;
    *)            anthropic_url="$base"; openai_url="${base}/v1" ;;
  esac
  model="${model:-deepseek-v4-pro}"

  # The guard above already proved the Anthropic side is free, so these are
  # unconditional. ANTHROPIC_AUTH_TOKEN is not written: .env only ever persists
  # the API-key spelling (see remove_dotenv_var ANTHROPIC_AUTH_TOKEN below).
  export ANTHROPIC_BASE_URL="$anthropic_url"
  [ -n "$key" ] && export ANTHROPIC_API_KEY="$key"
  [ -n "${CLAUDE_MODEL:-}" ] || export CLAUDE_MODEL="$model"
  # GEAKv4 follows whichever Claude model is actually in effect.
  [ -n "${GEAK_CLAUDE_MODEL:-}" ] || export GEAK_CLAUDE_MODEL="${CLAUDE_MODEL:-$model}"
  # The OpenAI side is adopted only when it is entirely free; otherwise some
  # other gateway already runs there and keeps its own key and model.
  if [ -z "${OPENAI_BASE_URL:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_BASE_URL="$openai_url"
    [ -n "$key" ] && export OPENAI_API_KEY="$key"
    [ -n "${CODEX_MODEL:-}" ] || export CODEX_MODEL="$model"
  fi
  unset DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
  LEGACY_DEEPSEEK_MIGRATED=1
  warn "DEEPSEEK_* is retired; migrated to ANTHROPIC_BASE_URL=${anthropic_url} + OPENAI_BASE_URL=${openai_url}"
}

# Resolve the Anthropic entrypoint (plus the OpenAI side of a dual-protocol
# gateway). Mirrors runtime credential validation.
resolve_credentials() {
  log "Phase 4: credentials"
  local anthropic_key anthropic_token anthropic_url
  local oauth_token dv_oauth_token
  local dv_anthropic_key dv_anthropic_token dv_anthropic_url
  local has_url=0 has_key=0 setup_env_authoritative=0 setup_llm_mode=""

  if [ ! -f "$DOTENV" ] && [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if [ -f "$ENV_TEMPLATE" ]; then
      cp "$ENV_TEMPLATE" "$DOTENV"; chmod 600 "$DOTENV" 2>/dev/null || true
      log "created ${DOTENV} from .env.template"
    else
      warn "no .env and no .env.template at ${ENV_TEMPLATE}; expecting LLM credentials from shell env"
    fi
  fi

  # Retire DEEPSEEK_* before anything else reads credentials, so the rest of
  # this function only ever sees the two protocol sides.
  migrate_legacy_deepseek_env

  # .env fallbacks (used only for values missing from process env).
  dv_anthropic_key="$(read_dotenv_var ANTHROPIC_API_KEY || true)"
  dv_anthropic_token="$(read_dotenv_var ANTHROPIC_AUTH_TOKEN || true)"
  dv_oauth_token="$(read_dotenv_var CLAUDE_CODE_OAUTH_TOKEN || true)"
  dv_anthropic_url="$(read_dotenv_var ANTHROPIC_BASE_URL || true)"
  setup_llm_mode="$(read_dotenv_var HYPERLOOM_LLM_MODE || true)"
  setup_llm_mode="$(echo "$setup_llm_mode" | tr '[:upper:]' '[:lower:]')"
  # ``deepseek`` was a mode of its own; it is now an Anthropic-side endpoint value.
  [ "$setup_llm_mode" = "deepseek" ] && setup_llm_mode="anthropic"
  if [ "${HYPERLOOM_SETUP_ENV_AUTHORITATIVE:-}" = "1" ]; then
    setup_env_authoritative=1
  fi
  if [ "$setup_env_authoritative" -eq 1 ] && [ -z "$setup_llm_mode" ]; then
    if [ -n "$dv_anthropic_key" ] || [ -n "$dv_anthropic_token" ] || [ -n "$dv_oauth_token" ] || [ -n "$dv_anthropic_url" ]; then
      setup_llm_mode="anthropic"
    fi
  fi
  if [ "$setup_env_authoritative" -eq 1 ] && [ -n "$setup_llm_mode" ]; then
    export HYPERLOOM_SETUP_LLM_MODE="$setup_llm_mode"
  fi

  # Precedence: process env > .env.
  anthropic_key="${ANTHROPIC_API_KEY:-$dv_anthropic_key}"
  anthropic_token="${ANTHROPIC_AUTH_TOKEN:-$dv_anthropic_token}"
  oauth_token="${CLAUDE_CODE_OAUTH_TOKEN:-$dv_oauth_token}"
  anthropic_url="${ANTHROPIC_BASE_URL:-$dv_anthropic_url}"

  # A dual-protocol gateway serves both protocols from ONE host, so its OpenAI
  # side belongs to the same credential and must survive the Anthropic-only
  # cleanup below. Decide from the resolved endpoints rather than from whether
  # a legacy migration ran, so a hand-written two-sided config counts too.
  local dual_url dual_secret
  if [ "$LEGACY_DEEPSEEK_MIGRATED" -eq 1 ] || [ "$setup_env_authoritative" -eq 0 ]; then
    dual_url="${OPENAI_BASE_URL:-$(read_dotenv_var OPENAI_BASE_URL || true)}"
    dual_secret="${OPENAI_API_KEY:-$(read_dotenv_var OPENAI_API_KEY || true)}"
  else
    # .env is the source of truth the operator just confirmed, so an ambient
    # shell value is not allowed to keep a stale OpenAI side alive.
    dual_url="$(read_dotenv_var OPENAI_BASE_URL || true)"
    dual_secret="$(read_dotenv_var OPENAI_API_KEY || true)"
  fi
  # Same host but a DIFFERENT path: two protocol routes of one gateway. An
  # identical URL on both sides is a stale copy, not a second protocol -- the
  # OpenAI SDK would append /chat/completions to an Anthropic base.
  DUAL_PROTOCOL_GATEWAY=0
  if [ -n "$anthropic_url" ] && [ -n "$dual_url" ] && [ "$dual_url" != "$anthropic_url" ] \
     && [ "$(url_authority "$anthropic_url")" = "$(url_authority "$dual_url")" ]; then
    DUAL_PROTOCOL_GATEWAY=1
    export OPENAI_BASE_URL="$dual_url"
    [ -n "$dual_secret" ] && export OPENAI_API_KEY="$dual_secret"
  fi

  # In the interactive setup flow, .env is the source of truth the user just
  # confirmed. Clear unsupported credential families before downstream scripts
  # source env with "env wins". A dual-protocol gateway keeps its OpenAI side:
  # both sides are the same gateway, not a second provider.
  if [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "anthropic" ]; then
    unset LLM_GATEWAY_KEY
    if [ "$DUAL_PROTOCOL_GATEWAY" -eq 0 ]; then
      unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_CUSTOM_HEADERS
    fi
  fi

  if [ -z "$anthropic_key" ] && [ -z "$anthropic_token" ] && [ -z "$oauth_token" ] && is_interactive; then
    read -rsp "[install-baremetal] Enter Anthropic API key (or leave blank if already configured): " anthropic_key; echo >&2
  fi

  # Reject one provider's base URL paired with only the other provider's key,
  # matching the CLI preflight.
  local _x_akey _x_aend _x_conflict=""
  _x_akey="${anthropic_key:-${anthropic_token:-${oauth_token:-}}}"
  _x_aend="${anthropic_url:-}"
  # A subscription token only validates against Anthropic itself, so it implies
  # the official endpoint and needs no ANTHROPIC_BASE_URL.
  if [ -z "$_x_aend" ] && [ -n "$oauth_token" ]; then
    _x_aend="https://api.anthropic.com"
  fi
  if [ -n "${OPENAI_BASE_URL:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -n "$_x_akey" ]; then
    _x_conflict="OPENAI_BASE_URL is set without an OPENAI_API_KEY, while an Anthropic-side key is configured"
  elif [ -n "$anthropic_url" ] && [ -z "$_x_akey" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    _x_conflict="ANTHROPIC_BASE_URL is set without an Anthropic-side key, while an OPENAI_API_KEY is configured"
  elif [ -n "${OPENAI_BASE_URL:-}" ] && [ -n "$_x_akey" ] && [ -z "$_x_aend" ]; then
    _x_conflict="an Anthropic-side key is configured without ANTHROPIC_BASE_URL, while the OpenAI side points at OPENAI_BASE_URL"
  # Only an explicit ANTHROPIC_BASE_URL signals a gateway-shaped deploy whose
  # OPENAI_API_KEY is likely a gateway key missing its own URL.
  elif [ -n "$anthropic_url" ] && [ -n "${OPENAI_API_KEY:-}" ] && [ -z "${OPENAI_BASE_URL:-}" ]; then
    _x_conflict="OPENAI_API_KEY is configured without OPENAI_BASE_URL, while the Anthropic side points at ANTHROPIC_BASE_URL"
  fi
  if [ -n "$_x_conflict" ]; then
    if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
      warn "conflicting LLM credentials: ${_x_conflict} (continuing: --check-only / --dry-run)"
    else
      die "Conflicting LLM credentials: ${_x_conflict}. Give each side its own base URL and key, or drop the other provider's key."
    fi
  fi

  { [ -n "$anthropic_url" ] || [ -n "$oauth_token" ]; } && has_url=1
  { [ -n "$anthropic_key" ] || [ -n "$anthropic_token" ] || [ -n "$oauth_token" ]; } && has_key=1
  if [ "$has_url" -eq 0 ] || [ "$has_key" -eq 0 ]; then
    if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
      warn "LLM credentials not fully resolved (continuing: --check-only / --dry-run)"
    else
      die "no usable LLM endpoint: configure ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN, or a Claude subscription token (CLAUDE_CODE_OAUTH_TOKEN). A dual-protocol gateway such as DeepSeek also sets OPENAI_BASE_URL + OPENAI_API_KEY."
    fi
  fi

  # Export resolved credentials and persist them to .env for the downstream
  # inference_optimizer skill install and CLI preflight.
  [ -n "$anthropic_key" ] && export ANTHROPIC_API_KEY="$anthropic_key"
  [ -n "$anthropic_token" ] && export ANTHROPIC_AUTH_TOKEN="$anthropic_token"
  # Subscription token stays in its own variable; mirroring it into an API-key
  # slot would move the run onto API billing.
  [ -n "$oauth_token" ] && export CLAUDE_CODE_OAUTH_TOKEN="$oauth_token"
  [ -n "$anthropic_url" ] && export ANTHROPIC_BASE_URL="$anthropic_url"

  # Persist resolved values to .env (skip on check-only / dry-run).
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    local persist_anthropic_key="${anthropic_key:-$anthropic_token}"
    if [ -n "$persist_anthropic_key" ]; then
      upsert_dotenv_var ANTHROPIC_API_KEY "$persist_anthropic_key"
    else
      remove_dotenv_var ANTHROPIC_API_KEY
    fi
    if [ -n "$anthropic_url" ]; then
      upsert_dotenv_var ANTHROPIC_BASE_URL "$anthropic_url"
    else
      remove_dotenv_var ANTHROPIC_BASE_URL
    fi
    # Persisted under its own key, never folded into an API-key slot.
    if [ -n "$oauth_token" ]; then
      upsert_dotenv_var CLAUDE_CODE_OAUTH_TOKEN "$oauth_token"
    else
      remove_dotenv_var CLAUDE_CODE_OAUTH_TOKEN
    fi
    if [ "$DUAL_PROTOCOL_GATEWAY" -eq 1 ]; then
      [ -n "${OPENAI_BASE_URL:-}" ] && upsert_dotenv_var OPENAI_BASE_URL "$OPENAI_BASE_URL"
      [ -n "${OPENAI_API_KEY:-}" ] && upsert_dotenv_var OPENAI_API_KEY "$OPENAI_API_KEY"
      [ -n "${CLAUDE_MODEL:-}" ] && upsert_dotenv_var CLAUDE_MODEL "$CLAUDE_MODEL"
      [ -n "${CODEX_MODEL:-}" ] && upsert_dotenv_var CODEX_MODEL "$CODEX_MODEL"
    elif [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "anthropic" ]; then
      # Anthropic-only deployment: scrub the other provider's entries. The
      # custom header belongs to that side, so it leaves with its URL and key --
      # a dual-protocol gateway takes the branch above and keeps all three,
      # which is the only place a header-authenticated OpenAI route is stored.
      remove_dotenv_var OPENAI_API_KEY
      remove_dotenv_var OPENAI_BASE_URL
      remove_dotenv_var OPENAI_CUSTOM_HEADERS
    fi
    # Drop the retired keys only now: dropping them inside the migration would
    # discard the operator's only copy if validation above had bailed out.
    if [ "$LEGACY_DEEPSEEK_MIGRATED" -eq 1 ]; then
      remove_dotenv_var DEEPSEEK_API_KEY
      remove_dotenv_var DEEPSEEK_BASE_URL
      remove_dotenv_var DEEPSEEK_MODEL
    fi
    remove_dotenv_var LLM_GATEWAY_KEY
    remove_dotenv_var ANTHROPIC_AUTH_TOKEN
    # Legacy gateway key: not read, purged if present.
    remove_dotenv_var SAFE_API_KEY
    log "credentials written to ${DOTENV}"
  fi
}

# Persist bare-metal runtime env to .env (single source of truth). PATH-class
# values are NOT written here; preflight derives them from ROCM_PATH /
# VIRTUAL_ENV / VLLM_VENV_ROOT at launch (_derive_runtime_paths).
write_runtime_dotenv() {
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then log "would update runtime env: ${DOTENV}"; return 0; fi
  # FRAMEWORK for downstream demo skills; empty when none is importable. A stale
  # value from an earlier install on a re-imaged host would point the skills at
  # an engine that is no longer there, so drop it rather than leave it behind.
  local detected_framework; detected_framework="$(resolve_installed_framework)"
  if [ -n "$detected_framework" ]; then
    log "detected serving framework: ${detected_framework}"
    upsert_dotenv_var FRAMEWORK "$detected_framework"
  else
    warn "no serving framework detected; clearing FRAMEWORK in ${DOTENV}"
    remove_dotenv_var FRAMEWORK
  fi

  upsert_dotenv_var USER_DATA_PATH "$USER_DATA_PATH"
  [ -n "${PYTHON:-}" ] && upsert_dotenv_var PYTHON "$PYTHON"
  [ -n "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-}" ] && upsert_dotenv_var INFERENCE_OPTIMIZER_FORCE_PYTHON "$INFERENCE_OPTIMIZER_FORCE_PYTHON"
  [ -n "${VIRTUAL_ENV:-}" ] && upsert_dotenv_var VIRTUAL_ENV "$VIRTUAL_ENV"
  [ -n "${ROCM_PATH:-}" ] && upsert_dotenv_var ROCM_PATH "$ROCM_PATH"
  [ -n "${HIP_PATH:-}" ] && upsert_dotenv_var HIP_PATH "$HIP_PATH"
  [ -n "${SGLANG_ROCM_EXTRA:-}" ] && upsert_dotenv_var SGLANG_ROCM_EXTRA "$SGLANG_ROCM_EXTRA"
  [ -n "${SGLANG_ROCM_PYPI_VERSION:-}" ] && upsert_dotenv_var SGLANG_ROCM_PYPI_VERSION "$SGLANG_ROCM_PYPI_VERSION"
  [ -n "${AITER_REF:-}" ] && upsert_dotenv_var AITER_REF "$AITER_REF"
  [ -n "${KERNEL_OPT_BACKEND_ORDER:-}" ] && upsert_dotenv_var KERNEL_OPT_BACKEND_ORDER "$KERNEL_OPT_BACKEND_ORDER"
  [ -n "${HYPERLOOM_WHEEL_REPO:-}" ] && upsert_dotenv_var HYPERLOOM_WHEEL_REPO "$HYPERLOOM_WHEEL_REPO"
  [ -n "${HYPERLOOM_WHEEL_TAG:-}" ] && upsert_dotenv_var HYPERLOOM_WHEEL_TAG "$HYPERLOOM_WHEEL_TAG"
  [ -n "${HYPERLOOM_SKILL_PATH:-}" ] && upsert_dotenv_var HYPERLOOM_SKILL_PATH "$HYPERLOOM_SKILL_PATH"
  [ -n "${SGLANG_USE_AITER:-}" ] && upsert_dotenv_var SGLANG_USE_AITER "$SGLANG_USE_AITER"
  upsert_dotenv_var HYPERLOOM_FRAMEWORK_ENV "$FRAMEWORK_ENV"
  if [ "$FRAMEWORK_ENV" = "isolated" ] && [ "$INSTALL_FRAMEWORK" = "vllm" ]; then
    upsert_dotenv_var VLLM_VENV_ROOT "$VLLM_VENV_ROOT"
    upsert_dotenv_var VLLM_PYTHON "${VLLM_VENV_ROOT}/bin/python"
  fi
  log "updated ${DOTENV} with bare-metal runtime env"
}

print_next_steps() {
  local framework_hint
  framework_hint="$INSTALL_FRAMEWORK"
  if [ "$framework_hint" = "none" ]; then
    # Nothing was installed, so the prompt must name the engine this host
    # actually has. Falling back to a hardcoded sglang sent atom-only hosts
    # down a framework that is not present.
    framework_hint="$(resolve_installed_framework)"
    [ -n "$framework_hint" ] || framework_hint="<none detected — install or pick one>"
  fi
  cat <<EOF

[install-baremetal] install complete.

Open this folder in Cursor as the workspace:
  ${REPO_ROOT}

Then paste this into Cursor Chat and fill in your workload:

@${HYPERLOOM_SKILL_PATH}

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: ${framework_hint}
- GPU: ${DETECTED_GPU}
- TP: 1
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
EOF
}

main() {
  restore_persisted_framework_env
  case "$FRAMEWORK_ENV" in
    shared|isolated) ;;
    *) die "FRAMEWORK_ENV must be one of: shared, isolated" ;;
  esac
  if [ "$FRAMEWORK_ENV" = "isolated" ] && [ "$INSTALL_FRAMEWORK" = "sglang" ]; then
    die "--framework-env isolated is currently supported for vLLM only"
  fi

  local user_data
  # Precedence: --user-data-path > process env > .env > default. The .env value
  # is honored so the setup skill's written USER_DATA_PATH is not silently lost.
  user_data="${USER_DATA_PATH_ARG:-${USER_DATA_PATH:-$(read_dotenv_var USER_DATA_PATH)}}"
# Container images ship a writable /workspace; a bare-metal host off root has
# neither it nor permission to create it, so the mkdir below would abort.
_default_workspace_root() {
  # The nearest existing ancestor decides: -w is false for a path that does not
  # exist yet, which would divert root off a /workspace it can still create.
  _ws_probe=/workspace
  while [ ! -e "$_ws_probe" ] && [ "$_ws_probe" != / ]; do _ws_probe=$(dirname "$_ws_probe"); done
  if [ -w "$_ws_probe" ]; then printf '%s' /workspace/hyperloom; else printf '%s' "$(pwd -P)/session"; fi
}
  user_data="${user_data:-$(_default_workspace_root)}"
  export USER_DATA_PATH="$user_data"
  export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"

  if [ -n "$DEPS_ROOT_ARG" ]; then
    export HYPERLOOM_DEPS_ROOT="$DEPS_ROOT_ARG"
    export HYPERLOOM_CACHE_DIR="$DEPS_ROOT_ARG"
  fi

  log "REPO_ROOT=${REPO_ROOT}"
  log "USER_DATA_PATH=${USER_DATA_PATH}"
  [ "$DRY_RUN" -eq 1 ] && log "mode: dry-run"
  [ "$CHECK_ONLY" -eq 1 ] && log "mode: check-only"

  local py_for_env
  if py_for_env="$(resolve_python 2>/dev/null)"; then
    export_virtualenv_for_python "$py_for_env"
  fi

  if [ "$VERIFY_HOTFIX_ONLY" -eq 1 ]; then
    verify_rocm_profiler_hotfix_only
    return $?
  fi

  if [ "$SKIP_BASE_CHECK" -eq 1 ]; then
    warn "skipping Phase 1 base preflight (--skip-base-check)"
    DETECTED_GPU="$(detect_gpu_label "$(command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)")"
  else
    base_preflight
  fi

  install_requested_framework
  apply_rocm_profiler_hotfix
  if [ "$INSTALL_FRAMEWORK" != "none" ] && [ "$SKIP_BASE_CHECK" -eq 0 ]; then
    base_preflight
  fi

  resolve_credentials

  write_runtime_dotenv

  if [ "$DRY_RUN" -eq 1 ]; then log "done (dry-run: no changes made)"; return 0; fi
  if [ "$CHECK_ONLY" -eq 1 ]; then log "done (check-only: verification pass complete)"; return 0; fi

  print_next_steps
}

main "$@"
