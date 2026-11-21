# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a grid variant can run on this build: its config tables on disk and its env flags supported."""

from __future__ import annotations

import json
import os
import subprocess

from hyperloom_common.env import is_truthy
from hyperloom.common.env_safety import ungrounded_env_file_reason

from .._benchmark_interpreter import _resolve_probe_python
from ._grid_base import GridVariant


# Env-flag capability probe: a serving env flag can be defined in the build yet still crash the server at engine init
# because the code path it activates imports a module the build did not package.
_UNSET = object()

# Cached probe result keyed by framework.
_CAP_PROBE_CACHE: dict[str, str | None] = {}

# Subprocess probe: locate the installed vLLM package via ``find_spec`` without importing it, read the aiter
# shared-expert router source, extract the ``fused_moe.*`` modules it imports, and verify each resolves to a real file
# in THIS build.
_AITER_SHARED_EXPERT_PROBE_SCRIPT = (
    "import importlib.util as u, os, re, json\n"
    "def go():\n"
    "    try:\n"
    "        spec = u.find_spec('vllm')\n"
    "    except (ImportError, ValueError):\n"
    "        return {'status': 'unknown'}\n"
    "    if not spec or not spec.origin:\n"
    "        return {'status': 'unknown'}\n"
    "    vdir = os.path.dirname(spec.origin)\n"
    "    router = os.path.join(vdir, 'model_executor', 'layers', 'fused_moe',\n"
    "                          'router', 'aiter_shared_routed_fused_moe_router.py')\n"
    "    if not os.path.exists(router):\n"
    "        return {'status': 'unknown'}\n"
    "    try:\n"
    "        text = open(router, encoding='utf-8').read()\n"
    "    except (OSError, UnicodeDecodeError):\n"
    "        return {'status': 'unknown'}\n"
    "    mods = re.findall(r'from\\s+(vllm\\.model_executor\\.layers\\.fused_moe\\.[A-Za-z0-9_.]+)\\s+import', text)\n"
    "    missing = []\n"
    "    for m in sorted(set(mods)):\n"
    "        rel = m[len('vllm.'):].replace('.', os.sep)\n"
    "        if not (os.path.exists(os.path.join(vdir, rel + '.py'))\n"
    "                or os.path.exists(os.path.join(vdir, rel, '__init__.py'))):\n"
    "            missing.append(m)\n"
    "    return {'status': 'unsupported', 'missing': missing} if missing else {'status': 'ok'}\n"
    "print(json.dumps(go()))\n"
)


def _probe_vllm_aiter_shared_expert_unsupported() -> str | None:
    """Return a drop reason if the installed vLLM build can't honour the aiter shared-expert fusion flag, else ``None``."""
    cached = _CAP_PROBE_CACHE.get("vllm", _UNSET)
    if cached is not _UNSET:
        return cached  # type: ignore[return-value]
    try:
        proc = subprocess.run(
            [_resolve_probe_python(), "-c", _AITER_SHARED_EXPERT_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        payload = json.loads(lines[-1]) if lines else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    status = payload.get("status")
    if status == "ok":
        _CAP_PROBE_CACHE["vllm"] = None
        return None
    if status == "unsupported":
        missing = ", ".join(payload.get("missing") or []) or "(unknown module)"
        reason = (
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS enabled but the installed "
            "vLLM build's aiter shared-expert router imports missing module(s): "
            f"{missing}. Flag unusable on this build (server crashes at engine "
            "init); upgrade vLLM to a build that ships the module."
        )
        _CAP_PROBE_CACHE["vllm"] = reason
        return reason
    return None  # unknown => do not drop or cache


def unsupported_capability_reason(variant: "GridVariant") -> str | None:
    """Return why a variant cannot run (a config table not on disk, an env flag the build lacks), else ``None``."""
    envs = {str(k): str(v) for k, v in (getattr(variant, "extra_envs", None) or {}).items()}
    if missing_file := ungrounded_env_file_reason(envs):
        return missing_file
    if (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower() != "vllm":
        return None
    val = envs.get("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS")
    if val is None or not is_truthy(val, default=True):
        return None
    return _probe_vllm_aiter_shared_expert_unsupported()
