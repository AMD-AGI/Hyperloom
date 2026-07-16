#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Install Hyperloom for bare-metal hosts (no-Docker / bare-host orchestrator).
#
# Runs Hyperloom directly on a host that already provides the ROCm framework
# base (ROCm runtime + a ROCm-built torch + a serving framework). For bare-metal
# installs, the script can optionally install SGLang or vLLM ROCm framework layers.
#
# Phase 1  base preflight  — ROCm / GPU arch / ROCm torch / serving framework
# Phase 2  framework       — optional bare-metal SGLang/vLLM install
# Phase 3  ROCm hotfix     — install ROCclr HIP runtime + roctracer profiler fix
# Phase 4  credentials     — resolve LLM gateway creds (single-gateway SAFE_API_KEY
#                            or split Anthropic/OpenAI keys) into .env
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
HYPERLOOM_WHEEL_TAG="${HYPERLOOM_WHEEL_TAG:-v0.8}"
ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR:-/opt/rocm/lib}"
ROCM_PROFILER_HOTFIX_ASSET="${ROCM_PROFILER_HOTFIX_ASSET:-rocm-profiler-hotfix-libs.tar.gz}"

DEFAULT_OPENAI_BASE_URL="${DEFAULT_OPENAI_BASE_URL:-}"
SAFE_API_KEY_PLACEHOLDER="ak-your-api-key-here"

FRAMEWORKS="sglang,vllm"
INSTALL_FRAMEWORK="none"
# Track whether the operator explicitly picked a framework env (via $FRAMEWORK_ENV
# or --framework-env). When unset, vLLM defaults to isolated (its wheel pins a
# torch that would clash with the host stack); others default to shared.
_FRAMEWORK_ENV_WAS_SET="${FRAMEWORK_ENV+x}"
FRAMEWORK_ENV="${FRAMEWORK_ENV:-shared}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
# Framework versions track docs/compatibility.md (SGLang v0.5.12,
# ROCm 7.2). vLLM uses the wheels.vllm.ai pip snapshot instead of the
# v0.21.0-rocm720 Docker image (no matching pip snapshot exists); 0.22.0+rocm722
# is the nearest published ROCm 7.2 wheel. AITER_REF must be a pinned commit
# SHA by default; set AITER_ALLOW_UNPINNED=1 to restore the legacy auto-select
# flow when doing local compatibility exploration.
SGLANG_REF="${SGLANG_REF:-v0.5.12}"
_SGLANG_ROCM_PYPI_VERSION_WAS_SET="${SGLANG_ROCM_PYPI_VERSION+x}"
SGLANG_ROCM_EXTRA="${SGLANG_ROCM_EXTRA:-rocm720}"
if [ -z "$_SGLANG_ROCM_PYPI_VERSION_WAS_SET" ]; then
  case "$SGLANG_ROCM_EXTRA" in
    rocm700) SGLANG_ROCM_PYPI_VERSION="7.0.0" ;;
    *)       SGLANG_ROCM_PYPI_VERSION="7.2.0" ;;
  esac
fi
SGLANG_ROCM_PYPI_VERSION="${SGLANG_ROCM_PYPI_VERSION:-7.2.0}"
SGLANG_ROCM_INDEX_URL="${SGLANG_ROCM_INDEX_URL:-}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:-}"
AITER_ALLOW_UNPINNED="${AITER_ALLOW_UNPINNED:-0}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
VLLM_ROCM_VARIANT="${VLLM_ROCM_VARIANT:-rocm722}"
VLLM_ROCM_INDEX="${VLLM_ROCM_INDEX:-https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}}"
VLLM_VENV_ROOT="${VLLM_VENV_ROOT:-/opt/hyperloom/vllm-venv}"
REQUIRE_FRAMEWORKS=0
SKIP_BASE_CHECK=0
DRY_RUN=0
CHECK_ONLY=0
ASSUME_YES=0
USER_DATA_PATH_ARG=""
DEPS_ROOT_ARG=""
SAFE_API_KEY_ARG=""
OPENAI_BASE_URL_ARG=""

usage() {
  cat <<'EOF'
Usage: src/hyperloom/inference_optimizer/assets/install_baremetal.sh [options]

Set up a bare-metal host with ROCm + ROCm torch for Hyperloom. Verifies the base,
optionally installs SGLang/vLLM, resolves credentials, and writes the combined
runtime env. Stops BEFORE launching.

Options:
  --safe-api-key KEY     LLM gateway key (ak-...); overrides env / .env
  --openai-base-url URL  LLM gateway endpoint; overrides env / .env
  --user-data-path PATH  Writable artifact root (default: /workspace/hyperloom)
  --deps-root PATH       Directory for auto-cloned dependency checkouts
  --frameworks LIST      Comma list to verify in Phase 1 (default: sglang,vllm)
  --install-framework FW Install a missing bare-metal framework layer.
                         Supported: none, sglang, vllm. Default: none.
  --framework-env MODE   Install target for framework packages: shared or
                         isolated. Default: shared. Use isolated for vLLM to
                         avoid replacing the shared ROCm torch stack.
  --vllm-venv-root PATH  Isolated vLLM venv path (default:
                         /opt/hyperloom/vllm-venv).
  --require-frameworks   Treat a missing requested framework as fatal
  --skip-base-check      Skip Phase 1 base preflight
  --check-only           Verify only; do not clone/install/mutate
  --dry-run              Print planned actions without cloning/installing/writing
  --yes, -y              Non-interactive; fail fast on missing credentials
  -h, --help             Show this help

Credential resolution (highest precedence first): flags > env > .env >
interactive prompt (TTY + not --yes). SAFE_API_KEY is not required: a split
Anthropic/OpenAI setup (ANTHROPIC_BASE_URL+ANTHROPIC_API_KEY and/or
OPENAI_BASE_URL+OPENAI_API_KEY) is accepted, matching cli.py credential rules.
Env overrides honored: REPO_ROOT,
USER_DATA_PATH, HYPERLOOM_RUNTIME_DIR, HYPERLOOM_DEPS_ROOT / _OPEN_SOURCE_ROOT,
PYTHON, INFERENCE_OPTIMIZER_FORCE_PYTHON, TRACELENS_INTERNAL_ROOT,
SGLANG_REPO, SGLANG_REF, SGLANG_ROOT, SGLANG_ROCM_PYPI_VERSION,
SGLANG_ROCM_EXTRA, SGLANG_ROCM_INDEX_URL, AITER_REPO, AITER_REF,
AITER_ALLOW_UNPINNED, AITER_ROOT, ROCM_PATH, HIP_PATH, LD_LIBRARY_PATH,
VLLM_VERSION, VLLM_ROCM_VARIANT, VLLM_ROCM_INDEX, VLLM_VENV_ROOT,
HYPERLOOM_WHEEL_REPO, HYPERLOOM_WHEEL_TAG.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --safe-api-key)     [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --safe-api-key requires a value" >&2; exit 2; }; shift; SAFE_API_KEY_ARG="${1:-}" ;;
    --openai-base-url)  [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --openai-base-url requires a value" >&2; exit 2; }; shift; OPENAI_BASE_URL_ARG="${1:-}" ;;
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
    --vllm-venv-root)   [ "$#" -ge 2 ] || { echo "[install-baremetal] ERROR: --vllm-venv-root requires a value" >&2; exit 2; }; shift; VLLM_VENV_ROOT="${1:-}" ;;
    --require-frameworks) REQUIRE_FRAMEWORKS=1 ;;
    --skip-base-check)  SKIP_BASE_CHECK=1 ;;
    --check-only)       CHECK_ONLY=1 ;;
    --dry-run)          DRY_RUN=1 ;;
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
SGLang/vLLM image such as primussafe/sglang:*-rocm*-mi30x|mi35x-profilerfix, or \
install an equivalent ROCm torch + framework stack), then re-run."

is_interactive() { [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ] && [ -t 1 ]; }

is_full_git_sha() {
  [[ "${1:-}" =~ ^[0-9a-fA-F]{40}$ ]]
}

require_pinned_aiter_ref() {
  if [ "${AITER_ALLOW_UNPINNED:-0}" = "1" ]; then
    return 0
  fi
  if [ -z "${AITER_REF:-}" ]; then
    die "AITER_REF must be set to a 40-character commit SHA before installing AITER (set AITER_ALLOW_UNPINNED=1 only for local compatibility exploration)"
  fi
  if ! is_full_git_sha "$AITER_REF"; then
    die "AITER_REF must be a 40-character commit SHA, got '${AITER_REF}' (set AITER_ALLOW_UNPINNED=1 to allow tags/auto-select)"
  fi
}

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

# Print the serving framework (sglang|vllm), or nothing when none is importable.
resolve_installed_framework() {
  if [ "$INSTALL_FRAMEWORK" = "sglang" ] || [ "$INSTALL_FRAMEWORK" = "vllm" ]; then
    printf '%s' "$INSTALL_FRAMEWORK"; return 0
  fi
  local py; py="$(resolve_python)" || return 0
  if _py_has "$py" sglang; then printf 'sglang'; return 0; fi
  local vllm_py="$py"
  if [ "$FRAMEWORK_ENV" = "isolated" ] && [ -x "${VLLM_VENV_ROOT}/bin/python" ]; then
    vllm_py="${VLLM_VENV_ROOT}/bin/python"
  fi
  if _py_has "$vllm_py" vllm; then printf 'vllm'; return 0; fi
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
  local hip_version="$1" hip_major hipcc_path hipcc_root header
  hip_major="${hip_version%%.*}"
  [ -n "$hip_major" ] || return 0
  hipcc_path="$(command -v hipcc 2>/dev/null || true)"
  if [ -z "$hipcc_path" ]; then
    warn "hipcc not found; AITER/source builds need a ROCm compiler toolchain."
    return 0
  fi
  hipcc_root="$(cd "$(dirname "$hipcc_path")/.." 2>/dev/null && pwd)" || return 0
  log "hipcc: ${hipcc_path}"
  if [ -n "${ROCM_PATH:-}" ] && [ "$hipcc_root" != "$(cd "$ROCM_PATH" 2>/dev/null && pwd)" ]; then
    warn "hipcc root ${hipcc_root} differs from ROCM_PATH=${ROCM_PATH}; JIT builds may use the wrong ROCm headers."
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
    "")     warn "GPU arch: not detected via rocminfo" ;;
    *)      warn "GPU arch ${gfx} untested (supported: gfx942/gfx950)" ;;
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
    check_torch_triton_alignment "$py" || rc=1
  fi

  local any_fw=0 fw
  IFS=',' read -r -a _fw_arr <<< "$FRAMEWORKS"
  for fw in "${_fw_arr[@]}"; do
    fw="$(echo "$fw" | tr -d '[:space:]')"; [ -z "$fw" ] && continue
    local probe_py="$py"
    if [ "$fw" = "vllm" ] && [ "$FRAMEWORK_ENV" = "isolated" ] && [ -x "${VLLM_VENV_ROOT}/bin/python" ]; then
      probe_py="${VLLM_VENV_ROOT}/bin/python"
    fi
    if _py_has "$probe_py" "$fw"; then
      if [ "$probe_py" != "$py" ]; then
        log "framework ${fw}: OK (isolated: ${probe_py})"
      else
        log "framework ${fw}: OK"
      fi
      any_fw=1
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

  local m
  for m in triton aiter sgl_kernel; do
    _py_has "$py" "$m" && log "runtime dep ${m}: OK" || warn "runtime dep ${m}: missing (some phases may degrade)"
  done

  [ "$rc" -ne 0 ] && die "base preflight failed. Fix the items above, or pass --skip-base-check to override."
  log "Phase 1: base preflight OK"
}

# Return the shared dependency root used for bare-metal framework sources.
framework_deps_root() {
  printf '%s' "${HYPERLOOM_OPEN_SOURCE_ROOT:-${HYPERLOOM_DEPS_ROOT:-/opt/hyperloom/open-source-repos}}"
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
    warn "triton ${current:-missing} does not match torch requirement ${required}; reinstall the torch-pinned ROCm Triton before installing SGLang"
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
  (cd "${sglang_root}/sgl-kernel" && AMDGPU_TARGET="$arch" "$py" setup_rocm.py install)
  if [ -f "${sglang_root}/python/pyproject_other.toml" ]; then
    cp "${sglang_root}/python/pyproject_other.toml" "${sglang_root}/python/pyproject.toml"
  fi
  constraint_file="$(mktemp)"
  write_rocm_torch_constraints "$py" "$constraint_file"
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
  local py="$1" aiter_root="$2" ref="$3" constraint_file="$4"
  checkout_aiter_ref "$aiter_root" "$ref"
  "$py" -m pip install --constraint "$constraint_file" \
    --config-settings editable_mode=compat -e "$aiter_root" || return 1
  "$py" -c "import aiter" >/dev/null
}

install_compatible_aiter() {
  local py="$1" aiter_root="$2" constraint_file ref tried=0
  constraint_file="$(mktemp)"
  write_rocm_torch_constraints "$py" "$constraint_file"

  require_pinned_aiter_ref
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
  local py="$1" pip_args
  log "installing amd-sglang ROCm ${SGLANG_ROCM_PYPI_VERSION} wheel (extra=${SGLANG_ROCM_EXTRA})"
  "$py" -m pip uninstall -y sglang-kernel sgl-kernel sglang amd-sglang || true
  pip_args=("amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}]")
  if [ -n "${SGLANG_ROCM_INDEX_URL:-}" ]; then
    pip_args+=(-i "$SGLANG_ROCM_INDEX_URL" --extra-index-url https://pypi.org/simple)
  fi
  "$py" -m pip install "${pip_args[@]}"
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
  elif [ "${AITER_ALLOW_UNPINNED:-0}" = "1" ]; then
    log "AITER_REF=auto (AITER_ALLOW_UNPINNED=1; newest tag compatible with installed torch/triton)"
  else
    log "AITER_REF unset (install will require a pinned commit SHA before cloning AITER)"
  fi
  log "SGLANG_ROCM_EXTRA=${SGLANG_ROCM_EXTRA}"
  log "SGLANG_ROCM_PYPI_VERSION=${SGLANG_ROCM_PYPI_VERSION}"
  if [ -n "${SGLANG_ROCM_INDEX_URL:-}" ]; then
    log "SGLANG_ROCM_INDEX_URL=${SGLANG_ROCM_INDEX_URL}"
  else
    log "SGLANG_ROCM_INDEX_URL unset (using the default pip index)"
  fi

  if [ "$SGLANG_ROCM_EXTRA" = "rocm700" ] && [ "$py_mm" != "3.10" ]; then
    die "SGLANG_ROCM_EXTRA=rocm700 currently supports Python 3.10 AMD wheels only; Python ${py_mm} would use source install and can pull mismatched ROCm 7.2 Triton."
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    _py_has "$py" sglang && log "sglang import OK" || warn "sglang missing (check-only; would install amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}])"
    if _py_has "$py" aiter; then
      log "aiter import OK"
    elif [ -n "$AITER_REF" ]; then
      warn "aiter missing (check-only; would clone/install ${AITER_REPO}@${AITER_REF})"
    elif [ "${AITER_ALLOW_UNPINNED:-0}" = "1" ]; then
      warn "aiter missing (check-only; would auto-select newest compatible tag from ${AITER_REPO})"
    else
      warn "aiter missing (check-only; install requires AITER_REF=<40-char commit SHA>)"
    fi
    _py_has "$py" sgl_kernel && log "sgl_kernel import OK" || warn "sgl_kernel missing (check-only; installed by amd-sglang)"
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$py_mm" = "3.10" ]; then
      if [ -n "${SGLANG_ROCM_INDEX_URL:-}" ]; then
        log "would run: ${py} -m pip install 'amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}]' -i ${SGLANG_ROCM_INDEX_URL} --extra-index-url https://pypi.org/simple"
      else
        log "would run: ${py} -m pip install 'amd-sglang[all-hip,${SGLANG_ROCM_EXTRA}]'"
      fi
    else
      log "would clone/build SGLang source ${SGLANG_REPO}@${SGLANG_REF} under ${SGLANG_ROOT:-${deps_root}/sglang}"
      log "would install SGLang source with [srt_hip] runtime dependencies under current torch/triton constraints"
    fi
    if [ -n "$AITER_REF" ]; then
      log "would clone/update ${AITER_REPO}@${AITER_REF} at ${aiter_root} and install it with current torch/triton constraints"
    elif [ "${AITER_ALLOW_UNPINNED:-0}" = "1" ]; then
      log "would auto-select the newest compatible AITER tag from ${AITER_REPO} and install it with current torch/triton constraints"
    else
      log "would require AITER_REF=<40-char commit SHA> before cloning ${AITER_REPO}"
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

  if ! _py_has "$py" aiter; then
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
import re, sys, urllib.request

base = sys.argv[1].rstrip("/") + "/torch/"
try:
    html = urllib.request.urlopen(base, timeout=30).read().decode()
except Exception:
    raise SystemExit(0)
# Match the plain "+local" form (e.g. 2.10.0+git8514f05); URL-encoded %2B
# anchors are skipped so the pinned spec stays pip-usable.
matches = re.findall(r"torch-([0-9][0-9A-Za-z.]*\+[0-9A-Za-z.]+)-cp", html)
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

rocm_profiler_hotfix_applied() {
  local target_dir="$1" hip_lib="$2" tracer_lib="$3"
  [ "$(basename "$(readlink -f "${target_dir}/libamdhip64.so" 2>/dev/null || true)")" = "$hip_lib" ] \
    && [ "$(basename "$(readlink -f "${target_dir}/libroctracer64.so" 2>/dev/null || true)")" = "$tracer_lib" ]
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

  # Probe vLLM in the isolated venv when FRAMEWORK_ENV=isolated, mirroring
  # resolve_installed_framework, so an isolated vLLM install still qualifies.
  local vllm_py="$py"
  if [ "$FRAMEWORK_ENV" = "isolated" ] && [ -x "${VLLM_VENV_ROOT}/bin/python" ]; then
    vllm_py="${VLLM_VENV_ROOT}/bin/python"
  fi
  local found=""
  _py_has "$py" sglang && found="sglang"
  _py_has "$vllm_py" vllm && found="${found:+${found} }vllm"
  [ -n "$found" ] || { warn "neither sglang nor vllm is importable; skipping ROCm profiler hotfix"; return 1; }
  log "framework imports: ${found}"
}

download_rocm_profiler_hotfix_libs() {
  local tmp_dir archive
  tmp_dir="$(mktemp -d)"
  command -v gh >/dev/null 2>&1 || {
    rm -rf "$tmp_dir"
    warn "gh CLI not found; cannot download ROCm profiler hotfix asset"
    return 1
  }
  gh auth status >/dev/null 2>&1 || {
    rm -rf "$tmp_dir"
    warn "gh is not authenticated; cannot download ROCm profiler hotfix asset"
    return 1
  }
  log "downloading ROCm profiler hotfix asset ${ROCM_PROFILER_HOTFIX_ASSET} from ${HYPERLOOM_WHEEL_REPO}@${HYPERLOOM_WHEEL_TAG}" >&2
  if ! gh release download "$HYPERLOOM_WHEEL_TAG" -R "$HYPERLOOM_WHEEL_REPO" \
    -p "$ROCM_PROFILER_HOTFIX_ASSET" -D "$tmp_dir" >&2; then
    rm -rf "$tmp_dir"
    warn "failed to download ${ROCM_PROFILER_HOTFIX_ASSET} from ${HYPERLOOM_WHEEL_REPO}@${HYPERLOOM_WHEEL_TAG}"
    return 1
  fi
  archive="${tmp_dir}/${ROCM_PROFILER_HOTFIX_ASSET}"
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

backup_rocm_profiler_hotfix_targets() {
  local target_dir="$1" backup_dir="$2" path real
  install -d "$backup_dir"
  for path in \
    "${target_dir}/libamdhip64.so" \
    "${target_dir}/libamdhip64.so.7" \
    "${target_dir}/libroctracer64.so" \
    "${target_dir}/libroctracer64.so.4"; do
    [ -e "$path" ] || [ -L "$path" ] || continue
    cp -a "$path" "$backup_dir"/
    real="$(readlink -f "$path" 2>/dev/null || true)"
    if [ -n "$real" ] && [ -e "$real" ]; then
      cp -a "$real" "$backup_dir"/
    fi
  done
}

install_rocm_profiler_hotfix_libs() {
  local source_dir="$1" target_dir="$2" hip_lib="$3" tracer_lib="$4"
  install -m 0644 "${source_dir}/${hip_lib}" "${target_dir}/${hip_lib}" || return 1
  install -m 0644 "${source_dir}/${tracer_lib}" "${target_dir}/${tracer_lib}" || return 1
  ln -sfnT "$hip_lib" "${target_dir}/libamdhip64.so.7" || return 1
  ln -sfnT libamdhip64.so.7 "${target_dir}/libamdhip64.so" || return 1
  ln -sfnT "$tracer_lib" "${target_dir}/libroctracer64.so.4" || return 1
  ln -sfnT libroctracer64.so.4 "${target_dir}/libroctracer64.so" || return 1
}

rollback_rocm_profiler_hotfix_targets() {
  local backup_dir="$1" target_dir="$2"
  [ -d "$backup_dir" ] || return 1
  if compgen -G "${backup_dir}/libamdhip64.so*" >/dev/null; then
    cp -a "${backup_dir}"/libamdhip64.so* "$target_dir"/ || return 1
  fi
  if compgen -G "${backup_dir}/libroctracer64.so*" >/dev/null; then
    cp -a "${backup_dir}"/libroctracer64.so* "$target_dir"/ || return 1
  fi
  return 0
}

verify_rocm_profiler_hotfix() {
  local target_dir="$1" hip_lib="$2" tracer_lib="$3" py
  log "verifying ROCm profiler hotfix links"
  ls -l "${target_dir}/libamdhip64.so" "${target_dir}/libamdhip64.so.7" \
        "${target_dir}/libroctracer64.so" "${target_dir}/libroctracer64.so.4" || return 1
  [ "$(basename "$(readlink -f "${target_dir}/libamdhip64.so")")" = "$hip_lib" ] \
    || { warn "libamdhip64.so does not point to ${hip_lib}"; return 1; }
  [ "$(basename "$(readlink -f "${target_dir}/libroctracer64.so")")" = "$tracer_lib" ] \
    || { warn "libroctracer64.so does not point to ${tracer_lib}"; return 1; }

  py="$(resolve_python)" || return 1
  "$py" - "$target_dir" <<'PY'
import ctypes
import os
import sys

target_dir = sys.argv[1]
for name in ("libamdhip64.so", "libroctracer64.so"):
    path = os.path.join(target_dir, name)
    print(f"{path} -> {os.path.realpath(path)}")
    ctypes.CDLL(path)
    print(f"loaded: {path}")

import torch

hip = getattr(torch.version, "hip", None)
print(f"torch.version.hip={hip}")
if not hip:
    raise SystemExit("torch.version.hip is empty after ROCm profiler hotfix")
PY
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
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "would download ${ROCM_PROFILER_HOTFIX_ASSET} from ${HYPERLOOM_WHEEL_REPO}@${HYPERLOOM_WHEEL_TAG}"
    log "would back up current ROCm libraries under ${target_dir}/.profiler_hotfix_backup_<timestamp>"
    log "would install hotfix libraries and update /opt/rocm libamdhip64/libroctracer64 symlinks"
    return 0
  fi

  extract_dir="$(download_rocm_profiler_hotfix_libs)" \
    || { warn "could not obtain ROCm profiler hotfix libraries; skipping"; return 0; }
  hip_lib="$(basename "$(find "$extract_dir" -maxdepth 1 -type f -name 'libamdhip64.so.*' | sort | tail -n 1)")"
  tracer_lib="$(basename "$(find "$extract_dir" -maxdepth 1 -type f -name 'libroctracer64.so.*' | sort | tail -n 1)")"
  if rocm_profiler_hotfix_applied "$target_dir" "$hip_lib" "$tracer_lib"; then
    log "ROCm profiler hotfix already applied (${hip_lib}, ${tracer_lib})"
    verify_rocm_profiler_hotfix "$target_dir" "$hip_lib" "$tracer_lib" || warn "existing ROCm profiler hotfix verification reported issues"
    rm -rf "$extract_dir"
    return 0
  fi
  backup_dir="${target_dir}/.profiler_hotfix_backup_$(date -u +%Y%m%dT%H%M%SZ)"
  backup_rocm_profiler_hotfix_targets "$target_dir" "$backup_dir"
  log "backed up current ROCm profiler libraries to ${backup_dir}"
  if ! install_rocm_profiler_hotfix_libs "$extract_dir" "$target_dir" "$hip_lib" "$tracer_lib" \
     || ! verify_rocm_profiler_hotfix "$target_dir" "$hip_lib" "$tracer_lib"; then
    warn "ROCm profiler hotfix failed; attempting rollback from ${backup_dir}"
    if rollback_rocm_profiler_hotfix_targets "$backup_dir" "$target_dir"; then
      warn "rollback succeeded; continuing without ROCm profiler hotfix"
      rm -rf "$extract_dir"
      return 0
    fi
    rm -rf "$extract_dir"
    die "ROCm profiler hotfix failed and rollback did not complete"
  fi
  rm -rf "$extract_dir"
  log "ROCm profiler hotfix applied"
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

# Resolve LLM gateway credentials, accepting either the AMD single-gateway pair
# (SAFE_API_KEY + OPENAI_BASE_URL) or split Anthropic/OpenAI entrypoints. Mirrors
# inference_optimizer/cli.py::_validate_credentials: a usable endpoint needs at
# least one base URL and at least one key, so SAFE_API_KEY is no longer mandatory.
resolve_credentials() {
  log "Phase 4: credentials"
  local safe_key openai_key anthropic_key anthropic_token openai_url anthropic_url
  local dv_safe dv_openai_key dv_anthropic_key dv_anthropic_token dv_openai_url dv_anthropic_url
  local has_url=0 has_key=0 setup_env_authoritative=0 setup_llm_mode=""

  if [ ! -f "$DOTENV" ] && [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if [ -f "$ENV_TEMPLATE" ]; then
      cp "$ENV_TEMPLATE" "$DOTENV"; chmod 600 "$DOTENV" 2>/dev/null || true
      log "created ${DOTENV} from .env.template"
    else
      warn "no .env and no .env.template at ${ENV_TEMPLATE}; expecting LLM credentials from flags or shell env"
    fi
  fi

  # .env fallbacks (used only for values missing from flags / process env).
  dv_safe="$(read_dotenv_var SAFE_API_KEY || true)"
  [ "$dv_safe" = "$SAFE_API_KEY_PLACEHOLDER" ] && dv_safe=""
  dv_openai_key="$(read_dotenv_var OPENAI_API_KEY || true)"
  dv_anthropic_key="$(read_dotenv_var ANTHROPIC_API_KEY || true)"
  dv_anthropic_token="$(read_dotenv_var ANTHROPIC_AUTH_TOKEN || true)"
  dv_openai_url="$(read_dotenv_var OPENAI_BASE_URL || true)"
  dv_anthropic_url="$(read_dotenv_var ANTHROPIC_BASE_URL || true)"
  setup_llm_mode="$(read_dotenv_var HYPERLOOM_LLM_MODE || true)"
  setup_llm_mode="$(echo "$setup_llm_mode" | tr '[:upper:]' '[:lower:]')"
  if [ "${HYPERLOOM_SETUP_ENV_AUTHORITATIVE:-}" = "1" ]; then
    setup_env_authoritative=1
  fi
  if [ "$setup_env_authoritative" -eq 1 ] && [ -z "$setup_llm_mode" ]; then
    if [ -n "$dv_anthropic_key" ] || [ -n "$dv_anthropic_token" ] || [ -n "$dv_anthropic_url" ]; then
      setup_llm_mode="anthropic"
    elif [ -n "$(read_dotenv_var DEEPSEEK_API_KEY || true)" ] || [ -n "$(read_dotenv_var DEEPSEEK_BASE_URL || true)" ]; then
      setup_llm_mode="deepseek"
    elif [ -n "$dv_openai_key" ] || [ -n "$dv_safe" ] || [ -n "$dv_openai_url" ]; then
      setup_llm_mode="openai"
    fi
  fi
  if [ "$setup_env_authoritative" -eq 1 ] && [ -n "$setup_llm_mode" ]; then
    export HYPERLOOM_SETUP_LLM_MODE="$setup_llm_mode"
  fi

  # Precedence: flags > process env > .env (flags exist only for the single-gateway pair).
  safe_key="${SAFE_API_KEY_ARG:-${SAFE_API_KEY:-$dv_safe}}"
  openai_key="${OPENAI_API_KEY:-$dv_openai_key}"
  anthropic_key="${ANTHROPIC_API_KEY:-$dv_anthropic_key}"
  anthropic_token="${ANTHROPIC_AUTH_TOKEN:-$dv_anthropic_token}"
  openai_url="${OPENAI_BASE_URL_ARG:-${OPENAI_BASE_URL:-$dv_openai_url}}"
  anthropic_url="${ANTHROPIC_BASE_URL:-$dv_anthropic_url}"

  # In the interactive setup flow, .env is the source of truth the user just
  # confirmed. Do not let stale OpenAI/SaFE values leak into the downstream
  # installers, because those scripts source env with "env wins" and may persist
  # or propagate the wrong provider back into runtime env files.
  if [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "anthropic" ]; then
    safe_key=""
    openai_key=""
    openai_url=""
    unset SAFE_API_KEY LLM_GATEWAY_KEY
    unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_CUSTOM_HEADERS
  elif [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "deepseek" ]; then
    safe_key=""
    openai_key=""
    openai_url=""
    anthropic_key=""
    anthropic_token=""
    anthropic_url=""
    unset SAFE_API_KEY LLM_GATEWAY_KEY
    unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_CUSTOM_HEADERS
    unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
  fi

  # Prompt for the single-gateway key only when no key of any kind is available.
  if [ -z "$safe_key" ] && [ -z "$openai_key" ] && [ -z "$anthropic_key" ] \
     && [ -z "$anthropic_token" ] && is_interactive; then
    read -rsp "[install-baremetal] Enter SAFE_API_KEY (ak-...) or leave blank if using ANTHROPIC/OPENAI keys: " safe_key; echo >&2
  fi

  # Single-gateway convenience: use an operator-provided default endpoint only
  # when no base URL is set at all. The public installer intentionally carries
  # no private gateway URL.
  if [ -n "$DEFAULT_OPENAI_BASE_URL" ] && [ -z "$openai_url" ] && [ -z "$anthropic_url" ]; then
    openai_url="$DEFAULT_OPENAI_BASE_URL"
    warn "no LLM base URL set; defaulting OPENAI_BASE_URL to ${openai_url}"
  fi

  { [ -n "$openai_url" ] || [ -n "$anthropic_url" ]; } && has_url=1
  { [ -n "$safe_key" ] || [ -n "$openai_key" ] || [ -n "$anthropic_key" ] || [ -n "$anthropic_token" ]; } && has_key=1
  if [ "$has_url" -eq 0 ] || [ "$has_key" -eq 0 ]; then
    if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
      warn "LLM credentials not fully resolved (continuing: --check-only / --dry-run)"
    else
      die "no usable LLM endpoint: need a base URL (OPENAI_BASE_URL or ANTHROPIC_BASE_URL) and a key (SAFE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN). Single gateway: --safe-api-key + --openai-base-url. Split: export ANTHROPIC_BASE_URL+ANTHROPIC_API_KEY and/or OPENAI_BASE_URL+OPENAI_API_KEY."
    fi
  fi

  # Export resolved credentials and persist them to .env for the downstream
  # inference_optimizer skill install and CLI preflight.
  [ -n "$safe_key" ] && export SAFE_API_KEY="$safe_key"
  [ -n "$openai_key" ] && export OPENAI_API_KEY="$openai_key"
  [ -n "$anthropic_key" ] && export ANTHROPIC_API_KEY="$anthropic_key"
  [ -n "$anthropic_token" ] && export ANTHROPIC_AUTH_TOKEN="$anthropic_token"
  [ -n "$openai_url" ] && export OPENAI_BASE_URL="$openai_url"
  [ -n "$anthropic_url" ] && export ANTHROPIC_BASE_URL="$anthropic_url"

  # Persist resolved values to .env (skip on check-only / dry-run).
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "deepseek" ]; then
      [ -n "$(read_dotenv_var DEEPSEEK_API_KEY || true)" ] && upsert_dotenv_var DEEPSEEK_API_KEY "$(read_dotenv_var DEEPSEEK_API_KEY || true)"
      [ -n "$(read_dotenv_var DEEPSEEK_BASE_URL || true)" ] && upsert_dotenv_var DEEPSEEK_BASE_URL "$(read_dotenv_var DEEPSEEK_BASE_URL || true)"
      remove_dotenv_var OPENAI_API_KEY
      remove_dotenv_var OPENAI_BASE_URL
      remove_dotenv_var ANTHROPIC_API_KEY
      remove_dotenv_var ANTHROPIC_BASE_URL
    else
      local persist_anthropic_key="${anthropic_key:-$anthropic_token}"
      local persist_openai_key="${openai_key:-$safe_key}"
      if [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "anthropic" ]; then
        persist_openai_key=""
        openai_url=""
      elif [ "$setup_env_authoritative" -eq 1 ] && [ "$setup_llm_mode" = "openai" ]; then
        persist_anthropic_key=""
        anthropic_url=""
      fi
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
      if [ -n "$persist_openai_key" ]; then
        upsert_dotenv_var OPENAI_API_KEY "$persist_openai_key"
      else
        remove_dotenv_var OPENAI_API_KEY
      fi
      if [ -n "$openai_url" ]; then
        upsert_dotenv_var OPENAI_BASE_URL "$openai_url"
      else
        remove_dotenv_var OPENAI_BASE_URL
      fi
    fi
    remove_dotenv_var SAFE_API_KEY
    remove_dotenv_var LLM_GATEWAY_KEY
    remove_dotenv_var ANTHROPIC_AUTH_TOKEN
    remove_dotenv_var OPENAI_CUSTOM_HEADERS
    log "credentials written to ${DOTENV}"
  fi
}

# Persist bare-metal runtime env to .env (single source of truth). PATH-class
# values are NOT written here; preflight derives them from ROCM_PATH /
# VIRTUAL_ENV / VLLM_VENV_ROOT at launch (_derive_runtime_paths).
write_runtime_dotenv() {
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then log "would update runtime env: ${DOTENV}"; return 0; fi
  # FRAMEWORK for downstream demo skills; empty when none is importable.
  local detected_framework; detected_framework="$(resolve_installed_framework)"
  if [ -n "$detected_framework" ]; then
    log "detected serving framework: ${detected_framework}"
  else
    warn "no serving framework detected; leaving FRAMEWORK unset in ${DOTENV}"
  fi

  upsert_dotenv_var USER_DATA_PATH "$USER_DATA_PATH"
  [ -n "${PYTHON:-}" ] && upsert_dotenv_var PYTHON "$PYTHON"
  [ -n "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-}" ] && upsert_dotenv_var INFERENCE_OPTIMIZER_FORCE_PYTHON "$INFERENCE_OPTIMIZER_FORCE_PYTHON"
  [ -n "${VIRTUAL_ENV:-}" ] && upsert_dotenv_var VIRTUAL_ENV "$VIRTUAL_ENV"
  [ -n "${ROCM_PATH:-}" ] && upsert_dotenv_var ROCM_PATH "$ROCM_PATH"
  [ -n "${HIP_PATH:-}" ] && upsert_dotenv_var HIP_PATH "$HIP_PATH"
  [ -n "${SGLANG_ROCM_EXTRA:-}" ] && upsert_dotenv_var SGLANG_ROCM_EXTRA "$SGLANG_ROCM_EXTRA"
  [ -n "${SGLANG_ROCM_PYPI_VERSION:-}" ] && upsert_dotenv_var SGLANG_ROCM_PYPI_VERSION "$SGLANG_ROCM_PYPI_VERSION"
  [ -n "${SGLANG_ROCM_INDEX_URL:-}" ] && upsert_dotenv_var SGLANG_ROCM_INDEX_URL "$SGLANG_ROCM_INDEX_URL"
  [ -n "${AITER_REF:-}" ] && upsert_dotenv_var AITER_REF "$AITER_REF"
  [ "${AITER_ALLOW_UNPINNED:-0}" = "1" ] && upsert_dotenv_var AITER_ALLOW_UNPINNED "$AITER_ALLOW_UNPINNED"
  [ -n "${KERNEL_OPT_BACKEND_ORDER:-}" ] && upsert_dotenv_var KERNEL_OPT_BACKEND_ORDER "$KERNEL_OPT_BACKEND_ORDER"
  [ -n "${HYPERLOOM_WHEEL_REPO:-}" ] && upsert_dotenv_var HYPERLOOM_WHEEL_REPO "$HYPERLOOM_WHEEL_REPO"
  [ -n "${HYPERLOOM_WHEEL_TAG:-}" ] && upsert_dotenv_var HYPERLOOM_WHEEL_TAG "$HYPERLOOM_WHEEL_TAG"
  [ -n "${HYPERLOOM_SKILL_PATH:-}" ] && upsert_dotenv_var HYPERLOOM_SKILL_PATH "$HYPERLOOM_SKILL_PATH"
  [ -n "${SGLANG_USE_AITER:-}" ] && upsert_dotenv_var SGLANG_USE_AITER "$SGLANG_USE_AITER"
  [ -n "${detected_framework}" ] && upsert_dotenv_var FRAMEWORK "$detected_framework"
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
  [ "$framework_hint" = "none" ] && framework_hint="sglang"
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
  # vLLM defaults to an isolated venv (its ROCm wheel pins a torch that would
  # clash with the shared host stack). Operators can still force shared with an
  # explicit $FRAMEWORK_ENV / --framework-env.
  if [ "$INSTALL_FRAMEWORK" = "vllm" ] && [ -z "$_FRAMEWORK_ENV_WAS_SET" ]; then
    FRAMEWORK_ENV="isolated"
    log "vLLM selected; defaulting to isolated framework env"
  fi
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
  user_data="${user_data:-/workspace/hyperloom}"
  export USER_DATA_PATH="$user_data"
  export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"

  if [ -n "$DEPS_ROOT_ARG" ]; then
    export HYPERLOOM_DEPS_ROOT="$DEPS_ROOT_ARG"
    export HYPERLOOM_OPEN_SOURCE_ROOT="$DEPS_ROOT_ARG"
  fi

  log "REPO_ROOT=${REPO_ROOT}"
  log "USER_DATA_PATH=${USER_DATA_PATH}"
  [ "$DRY_RUN" -eq 1 ] && log "mode: dry-run"
  [ "$CHECK_ONLY" -eq 1 ] && log "mode: check-only"

  local py_for_env
  if py_for_env="$(resolve_python 2>/dev/null)"; then
    export_virtualenv_for_python "$py_for_env"
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
