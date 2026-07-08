#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Install Hyperloom for bare-metal hosts (no-Docker / bare-host orchestrator).
#
# Runs Hyperloom directly on a host that already provides the ROCm framework
# base (ROCm runtime + a ROCm-built torch + a serving framework). For bare-metal
# installs, the script can optionally install SGLang or vLLM ROCm framework layers.
#
# Phase 0  base preflight  — ROCm / GPU arch / ROCm torch / serving framework
# Phase 1  framework       — optional bare-metal SGLang/vLLM install
# Phase 2  credentials     — resolve LLM gateway creds (single-gateway SAFE_API_KEY
#                            or split Anthropic/OpenAI keys) into .env
# Phase 3  dep checkouts   — src/hyperloom/inference_optimizer/assets/local_setup.sh
#                            (clone KernelForge/OOB, InferenceX, TraceLens)
# Phase 4  runtime install — src/hyperloom/inference_optimizer/assets/install.sh
#                            (io pkg, Magpie, InferenceX deps, forge-gemm-tune,
#                             + chained kernel-agent: Ray/GEAK/OOB/TraceLens/
#                             claude+codex+cursor CLIs; + framework-agent: fa)
# Phase 5  combined env    — write runtime/hyperloom.env.sh
# Phase 6  verify + print launch prompt
#
# Scope: core (native optimizer). PerfSkills/GEAK-e2e, live Langfuse, Quark,
# and gbrain KB are NOT installed here. It STOPS before launching.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_script_dir}/../../../.." && pwd)}"

LOCAL_SETUP_SH="${_script_dir}/local_setup.sh"
INSTALL_SH="${_script_dir}/install.sh"
ENV_TEMPLATE="${REPO_ROOT}/.env.template"
DOTENV="${REPO_ROOT}/.env"

DEFAULT_OPENAI_BASE_URL="https://global.primus-safe.amd.com/api/v1/llm-proxy/v1"
SAFE_API_KEY_PLACEHOLDER="ak-your-api-key-here"

FRAMEWORKS="sglang,vllm"
INSTALL_FRAMEWORK="none"
FRAMEWORK_ENV="${FRAMEWORK_ENV:-shared}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
# Framework versions track docs/QUICKSTART_LOCAL_MODE.md (SGLang v0.5.12,
# ROCm 7.2). vLLM uses the wheels.vllm.ai pip snapshot instead of the
# v0.21.0-rocm720 Docker image (no matching pip snapshot exists); 0.22.0+rocm722
# is the nearest published ROCm 7.2 wheel. AITER_REF pins ROCm/aiter to a
# released tag so source builds are reproducible (override to track another ref).
SGLANG_REF="${SGLANG_REF:-v0.5.12}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:-v0.1.16.post3}"
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

Install Hyperloom dependencies on a bare-metal host with ROCm + ROCm torch. Verifies
the base, optionally installs SGLang/vLLM, resolves credentials, then chains
local_setup.sh + install.sh. Stops BEFORE launching.

Options:
  --safe-api-key KEY     LLM gateway key (ak-...); overrides env / .env
  --openai-base-url URL  LLM gateway endpoint; overrides env / .env
  --user-data-path PATH  Writable artifact root (default: /workspace/hyperloom)
  --deps-root PATH       Directory for auto-cloned dependency checkouts
  --frameworks LIST      Comma list to verify in Phase 0 (default: sglang,vllm)
  --install-framework FW Install a missing bare-metal framework layer.
                         Supported: none, sglang, vllm. Default: none.
  --framework-env MODE   Install target for framework packages: shared or
                         isolated. Default: shared. Use isolated for vLLM to
                         avoid replacing the shared ROCm torch stack.
  --vllm-venv-root PATH  Isolated vLLM venv path (default:
                         /opt/hyperloom/vllm-venv).
  --require-frameworks   Treat a missing requested framework as fatal
  --skip-base-check      Skip Phase 0 base preflight
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
SGLANG_REPO, SGLANG_REF, SGLANG_ROOT, AITER_REPO, AITER_REF, AITER_ROOT, VLLM_VERSION,
VLLM_ROCM_VARIANT, VLLM_ROCM_INDEX, VLLM_VENV_ROOT.
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
  log "Phase 0: base preflight"

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
  log "Phase 0: base preflight OK"
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

# Ensure the ROCm Triton version required by AITER's gluon kernels is present.
ensure_rocm_triton_for_sglang() {
  local py="$1" current=""
  current="$(installed_dist_version "$py" triton 2>/dev/null || true)"
  case "$current" in
    3.6.0+rocm7.2.0.gitba5c1517)
      log "ROCm Triton ${current} already installed"
      return 0
      ;;
  esac
  log "installing ROCm Triton 3.6.0 for SGLang/AITER"
  "$py" -m pip install --force-reinstall --no-cache-dir \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.6.0%2Brocm7.2.0.gitba5c1517-cp312-cp312-linux_x86_64.whl"
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
  ensure_rocm_triton_for_sglang "$py"
  constraint_file="$(mktemp)"
  write_rocm_torch_constraints "$py" "$constraint_file"
  "$py" -m pip install --constraint "$constraint_file" -e "${sglang_root}/python[all_hip]"
  rm -f "$constraint_file"
}

# Install the AMD SGLang wheel only when its dependency set matches this Python.
install_sglang_from_wheel() {
  local py="$1"
  log "installing amd-sglang ROCm 7.2 wheel"
  "$py" -m pip uninstall -y sglang-kernel sgl-kernel sglang amd-sglang || true
  "$py" -m pip install \
    "amd-sglang[all-hip,rocm720]" \
    -i https://pypi.amd.com/rocm-7.2.0/simple \
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

  log "Phase 1: installing SGLang ROCm framework layer"
  log "framework python: ${py}"
  log "AITER_ROOT=${aiter_root}"
  log "AITER_REF=${AITER_REF}"

  if [ "$CHECK_ONLY" -eq 1 ]; then
    _py_has "$py" sglang && log "sglang import OK" || warn "sglang missing (check-only; would install amd-sglang[all-hip,rocm720])"
    _py_has "$py" aiter && log "aiter import OK" || warn "aiter missing (check-only; would clone/install ${AITER_REPO}@${AITER_REF})"
    _py_has "$py" sgl_kernel && log "sgl_kernel import OK" || warn "sgl_kernel missing (check-only; installed by amd-sglang)"
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$py_mm" = "3.10" ]; then
      log "would run: ${py} -m pip install 'amd-sglang[all-hip,rocm720]' -i https://pypi.amd.com/rocm-7.2.0/simple --extra-index-url https://pypi.org/simple"
    else
      log "would clone/build SGLang source ${SGLANG_REPO}@${SGLANG_REF} under ${SGLANG_ROOT:-${deps_root}/sglang}"
    fi
    log "would clone/update ${AITER_REPO}@${AITER_REF} at ${aiter_root} and install it with editable_mode=compat"
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
    mkdir -p "$(dirname "$aiter_root")"
    if [ ! -d "${aiter_root}/.git" ]; then
      git clone --recursive --branch "$AITER_REF" "$AITER_REPO" "$aiter_root"
    else
      git -C "$aiter_root" fetch --all --tags --prune
      git -C "$aiter_root" checkout "$AITER_REF"
      git -C "$aiter_root" submodule sync
      git -C "$aiter_root" submodule update --init --recursive
    fi
    "$py" -m pip install --config-settings editable_mode=compat -e "$aiter_root"
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

  log "Phase 1: installing vLLM ROCm framework layer"
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
    none) log "Phase 1: framework install skipped (--install-framework none)" ;;
    sglang) install_sglang_framework ;;
    vllm) install_vllm_framework ;;
  esac
}

read_dotenv_var() {
  local name="$1"
  [ -f "$DOTENV" ] || return 0
  grep -E "^[[:space:]]*(export[[:space:]]+)?${name}=" "$DOTENV" 2>/dev/null | tail -n 1 \
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

# Resolve LLM gateway credentials, accepting either the AMD single-gateway pair
# (SAFE_API_KEY + OPENAI_BASE_URL) or split Anthropic/OpenAI entrypoints. Mirrors
# inference_optimizer/cli.py::_validate_credentials: a usable endpoint needs at
# least one base URL and at least one key, so SAFE_API_KEY is no longer mandatory.
resolve_credentials() {
  log "Phase 2: credentials"
  local safe_key openai_key anthropic_key anthropic_token openai_url anthropic_url
  local dv_safe dv_openai_key dv_anthropic_key dv_anthropic_token dv_openai_url dv_anthropic_url
  local has_url=0 has_key=0

  if [ ! -f "$DOTENV" ] && [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    [ -f "$ENV_TEMPLATE" ] || die "no .env and no .env.template at ${ENV_TEMPLATE}"
    cp "$ENV_TEMPLATE" "$DOTENV"; chmod 600 "$DOTENV" 2>/dev/null || true
    log "created ${DOTENV} from .env.template"
  fi

  # .env fallbacks (used only for values missing from flags / process env).
  dv_safe="$(read_dotenv_var SAFE_API_KEY || true)"
  [ "$dv_safe" = "$SAFE_API_KEY_PLACEHOLDER" ] && dv_safe=""
  dv_openai_key="$(read_dotenv_var OPENAI_API_KEY || true)"
  dv_anthropic_key="$(read_dotenv_var ANTHROPIC_API_KEY || true)"
  dv_anthropic_token="$(read_dotenv_var ANTHROPIC_AUTH_TOKEN || true)"
  dv_openai_url="$(read_dotenv_var OPENAI_BASE_URL || true)"
  dv_anthropic_url="$(read_dotenv_var ANTHROPIC_BASE_URL || true)"

  # Precedence: flags > process env > .env (flags exist only for the single-gateway pair).
  safe_key="${SAFE_API_KEY_ARG:-${SAFE_API_KEY:-$dv_safe}}"
  openai_key="${OPENAI_API_KEY:-$dv_openai_key}"
  anthropic_key="${ANTHROPIC_API_KEY:-$dv_anthropic_key}"
  anthropic_token="${ANTHROPIC_AUTH_TOKEN:-$dv_anthropic_token}"
  openai_url="${OPENAI_BASE_URL_ARG:-${OPENAI_BASE_URL:-$dv_openai_url}}"
  anthropic_url="${ANTHROPIC_BASE_URL:-$dv_anthropic_url}"

  # Prompt for the single-gateway key only when no key of any kind is available.
  if [ -z "$safe_key" ] && [ -z "$openai_key" ] && [ -z "$anthropic_key" ] \
     && [ -z "$anthropic_token" ] && is_interactive; then
    read -rsp "[install-baremetal] Enter SAFE_API_KEY (ak-...) or leave blank if using ANTHROPIC/OPENAI keys: " safe_key; echo >&2
  fi

  # Single-gateway convenience: default the OpenAI endpoint only when no base URL is
  # set at all, so a pure Anthropic split config is not polluted with the AMD default.
  if [ -z "$openai_url" ] && [ -z "$anthropic_url" ]; then
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

  # Export resolved credentials for the chained local_setup.sh / install.sh.
  [ -n "$safe_key" ] && export SAFE_API_KEY="$safe_key"
  [ -n "$openai_key" ] && export OPENAI_API_KEY="$openai_key"
  [ -n "$anthropic_key" ] && export ANTHROPIC_API_KEY="$anthropic_key"
  [ -n "$anthropic_token" ] && export ANTHROPIC_AUTH_TOKEN="$anthropic_token"
  [ -n "$openai_url" ] && export OPENAI_BASE_URL="$openai_url"
  [ -n "$anthropic_url" ] && export ANTHROPIC_BASE_URL="$anthropic_url"

  # Persist resolved values to .env (skip on check-only / dry-run).
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    [ -n "$safe_key" ] && upsert_dotenv_var SAFE_API_KEY "$safe_key"
    [ -n "$openai_key" ] && upsert_dotenv_var OPENAI_API_KEY "$openai_key"
    [ -n "$anthropic_key" ] && upsert_dotenv_var ANTHROPIC_API_KEY "$anthropic_key"
    [ -n "$anthropic_token" ] && upsert_dotenv_var ANTHROPIC_AUTH_TOKEN "$anthropic_token"
    [ -n "$openai_url" ] && upsert_dotenv_var OPENAI_BASE_URL "$openai_url"
    [ -n "$anthropic_url" ] && upsert_dotenv_var ANTHROPIC_BASE_URL "$anthropic_url"
    log "credentials written to ${DOTENV}"
  fi
}

write_combined_env() {
  local combined="$1" local_env="$2" ka_env="$3"
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then log "would write combined env: ${combined}"; return 0; fi
  mkdir -p "$(dirname "$combined")"
  {
    echo '#!/bin/sh'
    echo '# Generated by src/hyperloom/inference_optimizer/assets/install_baremetal.sh'
    echo '# Source this single file before launching inference_optimizer.'
    printf 'export USER_DATA_PATH=%q\n' "$USER_DATA_PATH"
    [ -n "${SGLANG_USE_AITER:-}" ] && printf 'export SGLANG_USE_AITER=%q\n' "$SGLANG_USE_AITER"
    printf 'export HYPERLOOM_FRAMEWORK_ENV=%q\n' "$FRAMEWORK_ENV"
    if [ "$FRAMEWORK_ENV" = "isolated" ] && [ "$INSTALL_FRAMEWORK" = "vllm" ]; then
      printf 'export VLLM_VENV_ROOT=%q\n' "$VLLM_VENV_ROOT"
      printf 'export VLLM_PYTHON=%q\n' "${VLLM_VENV_ROOT}/bin/python"
      printf 'export PATH=%q:"$PATH"\n' "${VLLM_VENV_ROOT}/bin"
    fi
    printf '[ -f %q ] && . %q\n' "$local_env" "$local_env"
    printf '[ -f %q ] && . %q\n' "$ka_env" "$ka_env"
  } > "$combined"
  chmod 600 "$combined"
  log "wrote ${combined}"
}

print_next_steps() {
  local combined_env="$1" framework_hint
  framework_hint="$INSTALL_FRAMEWORK"
  [ "$framework_hint" = "none" ] && framework_hint="sglang"
  cat <<EOF

[install-baremetal] install complete.

Open this folder in Cursor as the workspace:
  ${REPO_ROOT}

Before launching, source the single combined env file:
  source '${combined_env}'

Then paste this into Cursor Chat and fill in your workload:

@src/hyperloom/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: ${framework_hint}
- GPU: ${DETECTED_GPU}
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Before launch, run exactly:
\`\`\`bash
source '${combined_env}'
\`\`\`

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
EOF
}

main() {
  [ -f "$LOCAL_SETUP_SH" ] || die "local_setup.sh not found at ${LOCAL_SETUP_SH}"
  [ -f "$INSTALL_SH" ] || die "install.sh not found at ${INSTALL_SH}"
  case "$FRAMEWORK_ENV" in
    shared|isolated) ;;
    *) die "FRAMEWORK_ENV must be one of: shared, isolated" ;;
  esac
  if [ "$FRAMEWORK_ENV" = "isolated" ] && [ "$INSTALL_FRAMEWORK" = "sglang" ]; then
    die "--framework-env isolated is currently supported for vLLM only"
  fi

  local user_data runtime_dir local_env ka_env combined_env
  user_data="${USER_DATA_PATH_ARG:-${USER_DATA_PATH:-/workspace/hyperloom}}"
  export USER_DATA_PATH="$user_data"
  # Honor the same override chain local_setup.sh / install.sh use so the
  # generated env files are located where those scripts actually write them.
  runtime_dir="${HYPERLOOM_RUNTIME_DIR:-${user_data}/runtime}"
  local_env="${LOCAL_SETUP_ENV:-${runtime_dir}/local-setup.env.sh}"
  ka_env="${KERNEL_AGENT_ENV:-${runtime_dir}/kernel-agent.env.sh}"
  combined_env="${runtime_dir}/hyperloom.env.sh"

  if [ -n "$DEPS_ROOT_ARG" ]; then
    export HYPERLOOM_DEPS_ROOT="$DEPS_ROOT_ARG"
    export HYPERLOOM_OPEN_SOURCE_ROOT="$DEPS_ROOT_ARG"
  fi

  log "REPO_ROOT=${REPO_ROOT}"
  log "USER_DATA_PATH=${USER_DATA_PATH}"
  [ "$DRY_RUN" -eq 1 ] && log "mode: dry-run"
  [ "$CHECK_ONLY" -eq 1 ] && log "mode: check-only"

  if [ "$SKIP_BASE_CHECK" -eq 1 ]; then
    warn "skipping Phase 0 base preflight (--skip-base-check)"
    DETECTED_GPU="$(detect_gpu_label "$(command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)")"
  else
    base_preflight
  fi

  install_requested_framework
  if [ "$INSTALL_FRAMEWORK" != "none" ] && [ "$SKIP_BASE_CHECK" -eq 0 ]; then
    base_preflight
  fi

  resolve_credentials

  # Phase 3: dependency checkouts.
  local ls_args=()
  [ "$DRY_RUN" -eq 1 ] && ls_args+=(--dry-run)
  [ "$CHECK_ONLY" -eq 1 ] && ls_args+=(--check-only)
  [ -n "$DEPS_ROOT_ARG" ] && ls_args+=(--deps-root "$DEPS_ROOT_ARG")
  log "Phase 3: local_setup.sh ${ls_args[*]}"
  # In preview modes (check-only / dry-run) a sub-script probe failure must not
  # abort the preview; only a real install aborts on local_setup.sh failure.
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    bash "$LOCAL_SETUP_SH" "${ls_args[@]}" || warn "local_setup.sh (preview) reported issues"
  else
    bash "$LOCAL_SETUP_SH" "${ls_args[@]}"
  fi

  if [ -f "$local_env" ]; then
    log "sourcing ${local_env}"
    # shellcheck disable=SC1090
    . "$local_env"
  elif [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    die "expected ${local_env} after local_setup.sh but it is missing"
  fi

  # Phase 4: runtime install (core scope — no --with-perfskills; Langfuse stays
  # off unless HYPERLOOM_LANGFUSE_ENABLE is already set in the environment/.env).
  local in_args=()
  [ "$DRY_RUN" -eq 1 ] && in_args+=(--dry-run)
  [ "$CHECK_ONLY" -eq 1 ] && in_args+=(--check-only)
  log "Phase 4: install.sh ${in_args[*]}"
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    bash "$INSTALL_SH" "${in_args[@]}" || warn "install.sh (preview) reported issues"
  else
    bash "$INSTALL_SH" "${in_args[@]}"
  fi
  if [ -f "$ka_env" ]; then
    log "sourcing ${ka_env}"
    # shellcheck disable=SC1090
    . "$ka_env"
  elif [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    warn "expected ${ka_env} after install.sh but it is missing"
  fi

  # Phase 5: combined env.
  write_combined_env "$combined_env" "$local_env" "$ka_env"

  # Phase 6: verification pass.
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    log "Phase 6: verifying (--check-only)"
    bash "$LOCAL_SETUP_SH" --check-only || warn "local_setup.sh --check-only reported issues"
    bash "$INSTALL_SH" --check-only || warn "install.sh --check-only reported issues"
  fi

  if [ "$DRY_RUN" -eq 1 ]; then log "done (dry-run: no changes made)"; return 0; fi
  if [ "$CHECK_ONLY" -eq 1 ]; then log "done (check-only: verification pass complete)"; return 0; fi

  print_next_steps "$combined_env"
}

main "$@"
