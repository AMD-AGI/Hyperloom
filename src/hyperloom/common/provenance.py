# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared provenance builder."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess  # nosec B404 - best-effort, guarded provenance probes only.
import sys
from importlib import metadata as _im
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.gpu_identity import gfx_arch_for_gpu_type

PROVENANCE_VERSION = 1
#: Tags a full shared provenance block apart from a placeholder stub.
PROVENANCE_SOURCE = "shared_v1"

# Env var priority per stack component (operator pins beat auto-detect).
_STACK_FINGERPRINT_ENVS: dict[str, tuple[str, ...]] = {
    "rocm": ("ROCM_VERSION", "HIP_VERSION"),
    "aiter": ("AITER_COMMIT", "AITER_VERSION"),
    "sglang": ("SGLANG_VERSION", "SGL_VERSION"),
    "vllm": ("VLLM_VERSION",),
}

# The interpreter preflight resolved for the serving framework. ``--framework-env
# isolated`` is the default for vLLM, whose ROCm wheel pins its own torch and so
# must not share the orchestrator's environment -- which also means
# ``importlib.metadata`` here cannot see it, and the version silently degraded to
# "unknown" on the default bare-metal vLLM path.
#
# Preflight already walks the candidate interpreters and locates the framework
# under one of them, so it describes the runtime this session resolved. The
# installer's ``$VLLM_VENV_ROOT`` is no longer read here directly: it is host
# state that is only ever written, never cleared, so on its own it cannot say
# whether the tree it names still holds the framework. It stays in play through
# preflight, which leads with it and probes it -- and that is the right answer
# either way, because ``_derive_runtime_paths`` also puts that root at the head
# of ``PATH`` and vLLM starts as a bare ``vllm serve``, so a root that still
# holds a working vLLM *is* what serves. A root that no longer does fails the
# probe and the scan moves on, where reading it directly would have recorded a
# version from an environment the run never touched.
#
# Distinct from ``HYPERLOOM_FRAMEWORK_PYTHON``, the per-attempt YAML key a
# framework-agent attempt uses to force its own launch interpreter: that one is
# a per-benchmark override, this one is the session-wide resolution.
#: Published by preflight; read here. Public so the two stay in one place.
RESOLVED_FRAMEWORK_PYTHON_ENV: str = "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON"

#: The framework that interpreter was resolved for. The scan answers for one
#: framework only, so the pair has to travel together: an SGLang session
#: publishes an interpreter that knows nothing about vLLM, and reading it as the
#: vLLM answer reports "unknown" for a vLLM this process can see.
RESOLVED_FRAMEWORK_ENV: str = "HYPERLOOM_RESOLVED_FRAMEWORK"

#: Runtime-arch overrides only. ``PYTORCH_ROCM_ARCH`` is deliberately absent:
#: it names the archs a wheel is *compiled* for, not the installed device, and
#: ``enablement/runtime/targeted_build.py`` sets it for exactly that purpose.
_GFX_ENVS = ("HYPERLOOM_GFX_ARCH", "GFX_ARCH")
_GRAPH_MODE_ENVS = ("HYPERLOOM_GRAPH_MODE", "GRAPH_MODE")
_SERVER_ARGS_ENVS = ("HYPERLOOM_SERVER_ARGS", "SERVER_ARGS")
_IMAGE_ENVS = ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE")
_CODE_REV_ENVS = ("HYPERLOOM_CODE_REVISION", "HYPERLOOM_GIT_SHA")

_GFX_RE = re.compile(r"gfx\d+[a-z0-9]*", re.IGNORECASE)


def _env_first(env: Mapping[str, str], *names: str) -> str | None:
    """Return the first set, non-empty (stripped) env value among ``names``."""
    for n in names:
        v = (env.get(n) or "").strip()
        if v:
            return v
    return None


def _arg_first(args: Any, *names: str) -> Any:
    """Return the first present, non-empty attribute of ``args`` among ``names``."""
    if args is None:
        return None
    for n in names:
        v = getattr(args, n, None)
        if v is not None and v != "":
            return v
    return None


def _int_or_none(value: Any) -> int | None:
    """Coerce to int, or ``None`` when unset/blank/non-numeric."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def detect_gfx_arch(env: Mapping[str, str], *, gpu_type: str | None = None, probe: bool = True) -> str | None:
    """Detect the ROCm gfx arch (e.g. ``gfx950``)."""
    raw = _env_first(env, *_GFX_ENVS)
    if raw:
        m = _GFX_RE.search(raw)
        return m.group(0).lower() if m else raw
    from_type = gfx_arch_for_gpu_type(gpu_type or _env_first(env, "GPU_TYPE"))
    if from_type:
        return from_type
    if not probe:
        return None
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=3)  # nosec B603 B607
        if out.returncode == 0:
            m = _GFX_RE.search(out.stdout or "")
            if m:
                return m.group(0).lower()
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        pass
    return None


def detect_graph_mode(env: Mapping[str, str]) -> str | None:
    """Return the graph-execution mode hint (``graph_capture``/``eager``/...)."""
    return _env_first(env, *_GRAPH_MODE_ENVS)


def _read_first_line(path: Path) -> str:
    """First non-empty stripped line of a file, or ``""`` when unreadable."""
    try:
        if not path.exists():
            return ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s:
                return s
    except OSError:
        return ""
    return ""


def detect_stack_fingerprint(env: Mapping[str, str], *, probe: bool = True) -> dict[str, str]:
    """Best-effort stack fingerprint: env -> rocm marker -> installed pkg."""
    out: dict[str, str] = {}
    for component, env_vars in _STACK_FINGERPRINT_ENVS.items():
        val = _env_first(env, *env_vars) or ""
        if not val and probe and component == "rocm":
            for marker in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-utils"):
                v = _read_first_line(Path(marker))
                if v:
                    val = v
                    break
        if not val and probe:
            val = _probe_pkg_version(component, _framework_site_packages(env, component))
        out[component] = val or "unknown"
    return out


def _framework_site_packages(env: Mapping[str, str], component: str) -> list[str] | None:
    """``site-packages`` dirs of the interpreter preflight resolved, if any."""
    resolved_for = _env_first(env, RESOLVED_FRAMEWORK_ENV)
    if not resolved_for or resolved_for.strip().lower() != component:
        return None
    python_exe = _env_first(env, RESOLVED_FRAMEWORK_PYTHON_ENV)
    if not python_exe or python_exe == sys.executable:
        # This process already answers for its own interpreter, without a derivation that only holds for venv-shaped
        # prefixes.
        return None
    # ``<venv>/bin/python`` -> ``<venv>/lib/python*/site-packages``.
    exe_path = Path(python_exe)
    if not exe_path.is_absolute():
        return None
    venv_root = exe_path.parent.parent
    try:
        hits = [str(p) for p in sorted(venv_root.glob("lib/python*/site-packages"))]
    except OSError:
        return None
    # No match means the prefix is not venv-shaped (a system prefix keeps its packages in ``dist-packages``), so the
    # derivation failed and this process is the better answer.
    return hits or None


def _probe_pkg_version(component: str, venv_path: list[str] | None = None) -> str:
    """Best-effort installed-package version for a stack component."""
    dist = {"sglang": "sglang", "vllm": "vllm", "aiter": "aiter"}.get(component)
    if not dist:
        return ""
    if venv_path is not None:
        try:
            for found in _im.distributions(path=list(venv_path)):
                name = (found.metadata["Name"] or "").strip().lower().replace("_", "-")
                if name == dist:
                    return (found.version or "").strip()
        except Exception:  # noqa: BLE001 — an unreadable venv is not a failure.
            return ""
        return ""
    try:
        return (_im.version(dist) or "").strip()
    except Exception:  # noqa: BLE001 — a missing package is normal.
        return ""


def detect_code_revision(env: Mapping[str, str], *, probe: bool = True) -> str:
    """Short git SHA of the repo containing this file, else a baked env rev."""
    if probe:
        try:
            here = Path(__file__).resolve().parent
            out = subprocess.run(  # nosec B603 B607
                ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            pass
    return _env_first(env, *_CODE_REV_ENVS) or ""


def detect_image(env: Mapping[str, str], *, probe: bool = True) -> str | None:
    """Container image from env vars or (when ``probe``) known marker files."""
    val = _env_first(env, *_IMAGE_ENVS)
    if val:
        return val
    if not probe:
        return None
    for marker in ("/etc/podinfo/image", "/etc/hyperloom-image"):
        v = _read_first_line(Path(marker))
        if v:
            return v
    return None


def _server_args_list(args: Any, env: Mapping[str, str]) -> list[str]:
    """Normalize server args (from args attr or env) into a list of tokens."""
    raw = _arg_first(args, "server_args", "extra_server_args")
    if raw is None:
        raw = _env_first(env, *_SERVER_ARGS_ENVS)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return str(raw).split()


def server_args_hash(server_args: list[str]) -> str:
    """Stable sha256 over the ordered server-arg tokens (``""`` when empty)."""
    if not server_args:
        return ""
    joined = "\n".join(str(x) for x in server_args)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_provenance(
    args: Any = None,
    *,
    env: Mapping[str, str] | None = None,
    probe: bool = True,
    source: str = PROVENANCE_SOURCE,
) -> dict[str, Any]:
    """Assemble the shared provenance block."""
    env = os.environ if env is None else env

    model_path = _arg_first(args, "model_path", "model")
    model_path = str(model_path) if model_path else None
    model_name = _arg_first(args, "model_display_name", "model_name")
    if not model_name and model_path:
        model_name = Path(model_path).name
    model_name = str(model_name) if model_name else None

    server_args = _server_args_list(args, env)

    return {
        "_provenance_source": source,
        "provenance_version": PROVENANCE_VERSION,
        # model identity
        "model_name": model_name,
        "model_path": model_path,
        "model_revision": _arg_first(args, "model_revision") or _env_first(env, "MODEL_REVISION"),
        # framework / stack
        "framework": (_arg_first(args, "framework") or _env_first(env, "FRAMEWORK")),
        "code_revision": detect_code_revision(env, probe=probe),
        "stack_fingerprint": detect_stack_fingerprint(env, probe=probe),
        "image": detect_image(env, probe=probe),
        # hardware / parallelism / graph
        "gpu_type": (_arg_first(args, "gpu_type") or _env_first(env, "GPU_TYPE")),
        "gfx_arch": detect_gfx_arch(env, gpu_type=_arg_first(args, "gpu_type"), probe=probe),
        "tp": _int_or_none(_arg_first(args, "tp") or _env_first(env, "TP")),
        "ep": _int_or_none(_arg_first(args, "ep") or _env_first(env, "EP")),
        "graph_mode": (_arg_first(args, "graph_mode") or detect_graph_mode(env)),
        # dtype / workload
        "dtype": (_arg_first(args, "precision", "dtype") or _env_first(env, "PRECISION")),
        "concurrency": _int_or_none(_arg_first(args, "conc", "concurrency") or _env_first(env, "CONC", "CONCURRENCY")),
        "isl": _int_or_none(_arg_first(args, "isl") or _env_first(env, "ISL")),
        "osl": _int_or_none(_arg_first(args, "osl") or _env_first(env, "OSL")),
        "max_model_len": _int_or_none(_arg_first(args, "max_model_len") or _env_first(env, "MAX_MODEL_LEN")),
        # full server args + fingerprint
        "server_args": server_args,
        "server_args_hash": server_args_hash(server_args),
    }


__all__ = [
    "PROVENANCE_VERSION",
    "PROVENANCE_SOURCE",
    "RESOLVED_FRAMEWORK_ENV",
    "RESOLVED_FRAMEWORK_PYTHON_ENV",
    "build_provenance",
    "server_args_hash",
    "detect_gfx_arch",
    "detect_graph_mode",
    "detect_stack_fingerprint",
    "detect_code_revision",
    "detect_image",
]
