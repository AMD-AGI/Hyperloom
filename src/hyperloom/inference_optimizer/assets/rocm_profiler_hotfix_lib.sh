#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared ROCm profiler hotfix helpers for container (install.sh) and bare-metal
# (install_baremetal.sh) installers. torch.profiler loads libamdhip64 and
# libroctracer64 from torch's bundled lib/ directory, so updating /opt/rocm
# alone is not enough — the resolved libraries must also be copied into torch.

: "${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR:=/opt/rocm/lib}"
: "${ROCM_PROFILER_HOTFIX_ASSET:=${ROCM_PROFILER_HOTFIX_ASSET:-rocm-profiler-hotfix-libs.tar.gz}}"
: "${HYPERLOOM_WHEEL_REPO:=${HYPERLOOM_WHEEL_REPO:-AMD-AGI/Hyperloom}}"
: "${HYPERLOOM_WHEEL_TAG:=${HYPERLOOM_WHEEL_TAG:-v1.0.0b2}}"
: "${FRAMEWORKS:=${FRAMEWORKS:-sglang,vllm,atom}}"
: "${CHECK_ONLY:=0}"
: "${DRY_RUN:=0}"

if ! declare -F log >/dev/null 2>&1; then
  log() { echo "[rocm-profiler-hotfix] $*"; }
fi
if ! declare -F warn >/dev/null 2>&1; then
  warn() { echo "[rocm-profiler-hotfix WARN] $*" >&2; }
fi
if ! declare -F die >/dev/null 2>&1; then
  die() { echo "[rocm-profiler-hotfix ERROR] $*" >&2; exit 1; }
fi

if ! declare -F _py_has >/dev/null 2>&1; then
  _py_has() {
    "$1" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$2') else 1)" 2>/dev/null
  }
fi

if ! declare -F framework_probe_python >/dev/null 2>&1; then
  framework_probe_python() {
    local fw="$1" default_py="$2"
    if [ "$fw" = "vllm" ] && [ "${FRAMEWORK_ENV:-shared}" = "isolated" ] \
       && [ -x "${VLLM_VENV_ROOT:-}/bin/python" ]; then
      printf '%s' "${VLLM_VENV_ROOT}/bin/python"
    else
      printf '%s' "$default_py"
    fi
  }
fi

if ! declare -F resolve_python >/dev/null 2>&1; then
  resolve_python() {
    if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
      printf '%s\n' "$PYTHON"
      return 0
    fi
    if [ -x "/opt/venv/bin/python" ] && [ "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-0}" != "1" ]; then
      printf '%s\n' "/opt/venv/bin/python"
      return 0
    fi
    if [ -x "/venv/bin/python" ]; then
      printf '%s\n' "/venv/bin/python"
      return 0
    fi
    command -v python3 2>/dev/null || return 1
  }
fi

resolve_torch_lib_dir() {
  local py="$1"
  "$py" - <<'PY'
import pathlib
import torch

print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
}

sync_rocm_profiler_libs_to_torch_lib() {
  local py rocm_lib_dir="${1:-${ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR}}"
  local torch_lib_dir hip_src tracer_src

  py="$(resolve_python 2>/dev/null)" || {
    warn "cannot resolve Python; skipping ROCm profiler torch lib sync"
    return 0
  }
  [ -d "$rocm_lib_dir" ] || {
    warn "ROCm library directory not found (${rocm_lib_dir}); skipping torch lib sync"
    return 0
  }

  hip_src="$(readlink -f "${rocm_lib_dir}/libamdhip64.so" 2>/dev/null || true)"
  tracer_src="$(readlink -f "${rocm_lib_dir}/libroctracer64.so" 2>/dev/null || true)"
  [ -n "$hip_src" ] && [ -f "$hip_src" ] || {
    warn "libamdhip64.so not found under ${rocm_lib_dir}; skipping torch lib sync"
    return 0
  }
  [ -n "$tracer_src" ] && [ -f "$tracer_src" ] || {
    warn "libroctracer64.so not found under ${rocm_lib_dir}; skipping torch lib sync"
    return 0
  }

  torch_lib_dir="$(resolve_torch_lib_dir "$py" 2>/dev/null || true)"
  [ -n "$torch_lib_dir" ] && [ -d "$torch_lib_dir" ] || {
    warn "torch lib directory not found (${torch_lib_dir:-missing}); skipping torch lib sync"
    return 0
  }

  if [ "$CHECK_ONLY" -eq 1 ]; then
    log "check-only: would copy ${hip_src} -> ${torch_lib_dir}/libamdhip64.so"
    log "check-only: would copy ${tracer_src} -> ${torch_lib_dir}/libroctracer64.so"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would sync ROCm profiler libs from ${rocm_lib_dir} into ${torch_lib_dir}"
    return 0
  fi

  log "syncing ROCm profiler libs into torch lib (${torch_lib_dir})"
  cp -a "$hip_src" "${torch_lib_dir}/libamdhip64.so"
  cp -a "$tracer_src" "${torch_lib_dir}/libroctracer64.so"
  log "torch libamdhip64.so -> $(readlink -f "${torch_lib_dir}/libamdhip64.so" 2>/dev/null || echo copied)"
  log "torch libroctracer64.so -> $(readlink -f "${torch_lib_dir}/libroctracer64.so" 2>/dev/null || echo copied)"
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

  local found="" fw probe_py _hotfix_arr
  IFS=',' read -r -a _hotfix_arr <<< "$FRAMEWORKS"
  for fw in "${_hotfix_arr[@]}"; do
    fw="$(echo "$fw" | tr -d '[:space:]')"; [ -z "$fw" ] && continue
    probe_py="$(framework_probe_python "$fw" "$py")"
    _py_has "$probe_py" "$fw" && found="${found:+${found} }${fw}"
  done
  [ -n "$found" ] || { warn "no serving framework importable from '${FRAMEWORKS}'; skipping ROCm profiler hotfix"; return 1; }
  log "framework imports: ${found}"
}

download_rocm_profiler_hotfix_libs() {
  local tmp_dir archive url
  tmp_dir="$(mktemp -d)"
  command -v curl >/dev/null 2>&1 || {
    rm -rf "$tmp_dir"
    warn "curl not found; cannot download ROCm profiler hotfix asset"
    return 1
  }
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

  log "applying ROCm profiler hotfix"
  log "ROCM_PROFILER_HOTFIX_ASSET=${ROCM_PROFILER_HOTFIX_ASSET}"

  [ -d "$target_dir" ] || { warn "ROCm library directory not found (${target_dir}); skipping profiler hotfix"; return 0; }
  rocm_profiler_hotfix_compatible || return 0

  if [ "$CHECK_ONLY" -eq 1 ]; then
    log "check-only: ROCm profiler hotfix release asset will not be downloaded"
    log "current libamdhip64.so -> $(readlink -f "${target_dir}/libamdhip64.so" 2>/dev/null || echo missing)"
    log "current libroctracer64.so -> $(readlink -f "${target_dir}/libroctracer64.so" 2>/dev/null || echo missing)"
    sync_rocm_profiler_libs_to_torch_lib "$target_dir"
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "would download ${ROCM_PROFILER_HOTFIX_ASSET} from ${HYPERLOOM_WHEEL_REPO}@${HYPERLOOM_WHEEL_TAG}"
    log "would back up current ROCm libraries under ${target_dir}/.profiler_hotfix_backup_<timestamp>"
    log "would install hotfix libraries and update /opt/rocm libamdhip64/libroctracer64 symlinks"
    sync_rocm_profiler_libs_to_torch_lib "$target_dir"
    return 0
  fi

  extract_dir="$(download_rocm_profiler_hotfix_libs)" \
    || { warn "could not obtain ROCm profiler hotfix libraries; skipping"; return 0; }
  hip_lib="$(basename "$(find "$extract_dir" -maxdepth 1 -type f -name 'libamdhip64.so.*' | sort | tail -n 1)")"
  tracer_lib="$(basename "$(find "$extract_dir" -maxdepth 1 -type f -name 'libroctracer64.so.*' | sort | tail -n 1)")"
  if rocm_profiler_hotfix_applied "$target_dir" "$hip_lib" "$tracer_lib"; then
    log "ROCm profiler hotfix already applied (${hip_lib}, ${tracer_lib})"
    verify_rocm_profiler_hotfix "$target_dir" "$hip_lib" "$tracer_lib" \
      || warn "existing ROCm profiler hotfix verification reported issues"
    sync_rocm_profiler_libs_to_torch_lib "$target_dir"
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
  sync_rocm_profiler_libs_to_torch_lib "$target_dir"
}
