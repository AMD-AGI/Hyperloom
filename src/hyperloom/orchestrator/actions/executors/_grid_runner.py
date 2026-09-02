# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helper for the ``explore`` executor's grid runs.

Takes a base Magpie YAML + a list of (name, extra_server_args, extra_envs)
variants, runs Magpie once per variant, parses ``benchmark_report.json``,
returns the winners.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import (
    BLOCKED_CHILD_ENV_NAMES,
    BLOCKED_EXTERNAL_ENV_NAMES,
    _ENV_KEY_RE,
    is_python_package_root,
    redact_secret_values,
    scrub_benchmark_process_env,
)

from ...phases import machine_state as _phase_state
from ...trace.task_progress import heartbeat_while_output_flows, report_progress
from ..stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from ._accuracy_gate import materialized_run_eval_disabled
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_RETURNCODE,
    DETOKENIZER_STALL_RETURNCODE,
    EVAL_PROBE_UNPATCHABLE_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    OVERTIME_KILL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    run_with_session_kill,
    server_log_death_excerpt,
    session_deadline_to_remaining_sec,
)
from .benchmark_result import (
    estimate_killed_variant_throughput,
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
    select_run_workspace,
    snapshot_workspaces,
)
from .benchmark_backend import build_benchmark_command
from ._inferencex_patcher import (
    ensure_benchmark_lib_eval_start_patched,
    ensure_eval_probe_patched,
    eval_probe_targets_exist,
)
from ._launch_evidence import build_launch_evidence, persist_launch_evidence

# Re-exported from sibling modules to keep the module namespace intact.
from ._grid_base import (
    DEFAULT_VARIANT_TIMEOUT_SEC as DEFAULT_VARIANT_TIMEOUT_SEC,
    DEFAULT_KEEP_THRESHOLD_PCT as DEFAULT_KEEP_THRESHOLD_PCT,
    GridVariant as GridVariant,
    coerce_extra_envs as coerce_extra_envs,
    VariantResult as VariantResult,
    variant_fingerprint as variant_fingerprint,
)
from ._grid_server_args import (
    server_args_env_name as server_args_env_name,
    merge_server_args as merge_server_args,
    compose_server_args as compose_server_args,
    remove_server_args as remove_server_args,
    compact_json_server_args as compact_json_server_args,
    _SPACE_VALUE_FLAGS as _SPACE_VALUE_FLAGS,
    _MULTI_VALUE_FLAGS as _MULTI_VALUE_FLAGS,
    _VLLM_SINGLE_VALUE_FLAGS as _VLLM_SINGLE_VALUE_FLAGS,
    dedup_vllm_server_args as dedup_vllm_server_args,
    _shell_safe_dedupe as _shell_safe_dedupe,
    DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC as DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC,
    SGLANG_WATCHDOG_TIMEOUT_ENV as SGLANG_WATCHDOG_TIMEOUT_ENV,
    _SGLANG_WATCHDOG_FLAG as _SGLANG_WATCHDOG_FLAG,
    _SGLANG_WATCHDOG_RE as _SGLANG_WATCHDOG_RE,
    resolve_sglang_watchdog_timeout as resolve_sglang_watchdog_timeout,
    inject_sglang_watchdog_timeout as inject_sglang_watchdog_timeout,
    DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS as DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS,
    DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS as DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS,
    SGLANG_CONTEXT_HEADROOM_ENV as SGLANG_CONTEXT_HEADROOM_ENV,
    SGLANG_CONTEXT_FLOOR_ENV as SGLANG_CONTEXT_FLOOR_ENV,
    _SGLANG_CONTEXT_LENGTH_FLAG as _SGLANG_CONTEXT_LENGTH_FLAG,
    _SGLANG_CONTEXT_LENGTH_RE as _SGLANG_CONTEXT_LENGTH_RE,
    _SGLANG_ATTN_BACKEND_FLAG as _SGLANG_ATTN_BACKEND_FLAG,
    _SGLANG_ATTN_BACKEND_RE as _SGLANG_ATTN_BACKEND_RE,
    _SGLANG_DUAL_CHUNK_BACKEND as _SGLANG_DUAL_CHUNK_BACKEND,
    _resolve_nonneg_int_env as _resolve_nonneg_int_env,
    resolve_sglang_context_cap as resolve_sglang_context_cap,
    inject_sglang_context_length as inject_sglang_context_length,
    _resolve_dual_chunk_backend as _resolve_dual_chunk_backend,
    inject_sglang_attention_backend as inject_sglang_attention_backend,
    HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV as HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV,
    DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND as DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND,
    _SGLANG_MOE_RUNNER_BACKEND_FLAG as _SGLANG_MOE_RUNNER_BACKEND_FLAG,
    _SGLANG_MOE_RUNNER_BACKEND_RE as _SGLANG_MOE_RUNNER_BACKEND_RE,
    inject_sglang_moe_runner_backend as inject_sglang_moe_runner_backend,
    moe_runner_requires_aiter as moe_runner_requires_aiter,
    apply_runtime_benchmark_overrides as apply_runtime_benchmark_overrides,
)
from ._grid_variant_filter import (
    resolve_skip_spec as resolve_skip_spec,
    _parse_skip_spec as _parse_skip_spec,
    _RE_CUDA_GRAPH_MAX_BS as _RE_CUDA_GRAPH_MAX_BS,
    _MN_PARAMS_PRIORITY as _MN_PARAMS_PRIORITY,
    _MN_BACKENDS_PRIORITY as _MN_BACKENDS_PRIORITY,
    _mn_priority_index as _mn_priority_index,
    reorder_grid_for_multi_node as reorder_grid_for_multi_node,
    apply_multi_node_invalid_variants as apply_multi_node_invalid_variants,
    apply_aiter_moe_pin_filter as apply_aiter_moe_pin_filter,
    _COMPATIBILITY_FLAG_RULES as _COMPATIBILITY_FLAG_RULES,
    _XDIT_ENV_BLACKLIST as _XDIT_ENV_BLACKLIST,
    _XDIT_ENV_COMBO_BLACKLIST as _XDIT_ENV_COMBO_BLACKLIST,
    xdit_blacklist_reason as xdit_blacklist_reason,
    _HELP_TEXT_CACHE as _HELP_TEXT_CACHE,
    _HELP_PROBE_COMMANDS as _HELP_PROBE_COMMANDS,
    _probe_server_help_text as _probe_server_help_text,
    _detect_model_class as _detect_model_class,
    apply_compatibility_filter as apply_compatibility_filter,
    apply_user_skip_list as apply_user_skip_list,
)


log = logging.getLogger(__name__)


def _resolve_magpie_python() -> str:
    """Resolve the Python interpreter for Magpie subprocesses.

    Order: $MAGPIE_PYTHON (only when it can ``import Magpie``) > first PATH
    ``python3`` that can ``import Magpie`` > /opt/venv/bin/python when present
    > first PATH ``python3``. A stale ``$MAGPIE_PYTHON`` is validated and
    skipped to avoid ``ModuleNotFoundError`` at benchmark time.

    Returns:
        str: Path to a Python interpreter that can import Magpie, falling back
        to an existing interpreter that preflight can install Magpie into.
    """

    def _can_import_magpie(py: str) -> bool:
        """Whether an interpreter can import Magpie and its ``yaml`` dep.

        Probes both ``Magpie`` and ``yaml`` so interpreters that resolve
        Magpie via a ``.pth`` but lack PyYAML are rejected.

        Args:
            py: Path to the candidate Python interpreter.

        Returns:
            ``True`` if both imports succeed in the interpreter.
        """
        try:
            # Probe with ``importlib.util.find_spec`` rather than a bare
            # ``import`` so a missing module returns a non-zero exit code
            # WITHOUT the child emitting a ``ModuleNotFoundError`` traceback.
            # ``run_with_session_kill`` mirrors child stderr to the parent
            # stream, so a bare ``import Magpie`` on a candidate that lacks it
            # would leak an alarming traceback into the run log even though the
            # probe failing is an expected, benign step of interpreter
            # resolution.
            proc = run_with_session_kill(
                [
                    py,
                    "-c",
                    "import importlib.util as u, sys; "
                    "sys.exit(0 if u.find_spec('Magpie') and u.find_spec('yaml') else 1)",
                ],
                timeout=10,
            )
            return getattr(proc, "returncode", 1) == 0
        except Exception:
            return False

    env_val = os.environ.get("MAGPIE_PYTHON", "").strip()
    if env_val:
        if _can_import_magpie(env_val):
            return env_val
        log.warning(
            "MAGPIE_PYTHON=%s cannot import Magpie; ignoring it and "
            "auto-detecting an interpreter that can. (A stale value is often "
            "baked into kernel-agent.env.sh when install.sh resolved it "
            "before Magpie was pip-installed.)",
            env_val,
        )

    candidate = shutil.which("python3")
    if candidate and _can_import_magpie(candidate):
        return candidate

    opt_venv = "/opt/venv/bin/python"
    if Path(opt_venv).is_file():
        return opt_venv
    if candidate:
        return candidate
    return "python3"


def _validate_magpie_python_override(value: str) -> str:
    """Validate caller-supplied Magpie interpreter overrides before argv[0]."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if any(ch in raw for ch in ("\x00", "\n", "\r")):
        raise ValueError("magpie_python contains control characters")
    resolved = Path(raw)
    if not resolved.is_absolute():
        found = shutil.which(raw)
        if not found:
            raise ValueError(f"magpie_python is not executable on PATH: {raw}")
        resolved = Path(found)
    if not resolved.is_file():
        raise ValueError(f"magpie_python is not a file: {raw}")
    if not resolved.name.lower().startswith("python"):
        raise ValueError(f"magpie_python must point to a Python interpreter, got: {raw}")
    return str(resolved)


def _resolve_probe_python(framework: str = "vllm") -> str:
    """Resolve the interpreter a build-accuracy probe must use.

    A capability probe only produces a correct drop decision when it inspects
    the SAME framework install the benchmark server loads, so a bare ``python3``
    off ``$PATH`` is deliberately NOT a fallback.

    Only vLLM may live in its own venv, so that one leads for vLLM alone.

    Resolution order:
    1. ``$VLLM_VENV_ROOT/bin/python`` when probing vLLM and it is executable.
    2. ``_resolve_magpie_python()`` — the interpreter that runs the benchmark
       harness; on a single-venv install this is also the vLLM venv.
    3. The interpreter behind the ``vllm`` executable (``<venv>/bin/python``
       alongside ``shutil.which("vllm")``) when it exists on disk.
    4. ``_resolve_magpie_python()``'s canonical fallback via step 2.

    Args:
        framework (str): Framework whose install the probe must inspect.

    Returns:
        str: Path to the interpreter the probe should invoke.
    """
    if (framework or "").strip().lower() == "vllm":
        venv_root = os.environ.get("VLLM_VENV_ROOT", "").strip()
        if venv_root:
            venv_python = str(Path(venv_root) / "bin" / "python")
            if os.access(venv_python, os.X_OK):
                return venv_python
    magpie_python = _resolve_magpie_python()
    # Prefer the harness interpreter; on a single-venv box it already IS the
    # vLLM venv.
    if magpie_python and magpie_python != "/opt/venv/bin/python":
        return magpie_python
    # Fell through to the canonical default; pin the venv that backs ``vllm
    # serve`` so the probe hits the real server source.
    vllm_exe = shutil.which("vllm")
    if vllm_exe:
        vllm_python = os.path.join(os.path.dirname(vllm_exe), "python")
        if os.path.exists(vllm_python):
            return vllm_python
    return magpie_python


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root.

    Reads :func:`hyperloom.inference_optimizer.session.paths.session_dir` (honors
    ``$USER_DATA_PATH``, else ``/workspace/hyperloom``). Used by fallback
    paths when ``ctx.extra["workspace"]`` was not pre-mkdir'd.

    Returns:
        Path: The resolved active session directory.
    """
    from hyperloom.inference_optimizer.session.paths import session_dir as _sd

    return _sd()


# SKIP_VARIANTS: comma/whitespace patterns matched (exact or fnmatch) against
# ``GridVariant.name``. Order: params["skip_variants"] > $SKIP_VARIANTS > "".


# Env-flag capability probe: a serving env flag can be defined in the build yet
# still crash the server at engine init because the code path it activates
# imports a module the build did not package. Probe the installed build directly
# rather than maintaining a version→flag table.
_UNSET = object()

# Cached probe result keyed by framework. ``None`` (flag usable / n/a) or a
# reason string (flag would crash the server).
_CAP_PROBE_CACHE: dict[str, str | None] = {}

# Subprocess probe: locate the installed vLLM package via ``find_spec`` without
# importing it, read the aiter shared-expert router source, extract the
# ``fused_moe.*`` modules it imports, and verify each resolves to a real file in
# THIS build. Prints a single JSON line; any failure => ``status=unknown`` so
# the caller does NOT drop the variant.
_AITER_SHARED_EXPERT_PROBE_SCRIPT = (
    "import importlib.util as u, os, re, json\n"
    "def go():\n"
    "    try:\n"
    "        spec = u.find_spec('vllm')\n"
    "    except Exception:\n"
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
    "    except Exception:\n"
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
    """Return a drop reason if the installed vLLM build can't honour the aiter
    shared-expert fusion flag, else ``None``.

    Build-accurate: checks that every ``fused_moe.*`` module the installed
    aiter shared-expert router imports actually exists on disk. Best-effort —
    ``unknown`` results (vLLM absent / router absent / probe error) return
    ``None`` (do NOT drop) and are NOT cached so a transient failure re-probes.
    Definitive ``ok`` / ``unsupported`` results are cached for the session.

    Returns:
        str | None: A human-readable reason when the flag would crash the
        server on this build, else ``None``.
    """
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
    except Exception:  # noqa: BLE001
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
    """Return a drop reason if a variant sets an env flag the installed
    framework build cannot honour, else ``None``.

    Mirrors :func:`xdit_blacklist_reason`: pure per-variant inspection plus a
    cached build probe. Conservative — returns ``None`` (do NOT drop) whenever
    the probe cannot positively confirm the flag is broken.

    Args:
        variant (GridVariant): The candidate variant to inspect.

    Returns:
        str | None: A human-readable reason when the variant should be
        fast-failed before booting a server, else ``None``.
    """
    fw = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    if fw != "vllm":
        return None
    envs = {str(k): str(v) for k, v in (getattr(variant, "extra_envs", None) or {}).items()}
    val = envs.get("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS")
    if val is None or not is_truthy(val, default=True):
        return None
    return _probe_vllm_aiter_shared_expert_unsupported()


# Sanitization for LLM-supplied overrides (benchmark_script / result_dir):
# reject path separators / shell metacharacters, raising ``ValueError`` instead
# of running an unsafe subprocess.
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.sh$")
_RESULT_DIR_FORBID_RE = re.compile(r"[\s\"'`$;&|<>(){}\[\]\\*?!]")


def sanitize_script_name(value: Any) -> str | None:
    """Return ``value`` if it's a safe Magpie benchmark script file name.

    Must be a bare ``*.sh`` name (no slashes / ``..``). Empty/``None`` →
    ``None``; anything resembling shell injection raises ``ValueError``.

    Args:
        value (Any): Candidate script name; coerced to a stripped string.

    Returns:
        str | None: The validated bare ``*.sh`` name, or ``None`` for empty
        input.

    Raises:
        ValueError: When ``value`` is not a bare ``*.sh`` file name.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _SCRIPT_NAME_RE.match(text):
        raise ValueError(
            f"benchmark_script={text!r} rejected: must be a bare *.sh "
            "file name (no path separators, no shell metacharacters)"
        )
    return text


def sanitize_result_dir(value: Any) -> str | None:
    """Return ``value`` if it's a safe absolute (or workspace-relative) dir.

    Lands in a shell ``cd`` / ``mkdir`` via ``$RESULT_DIR``, so reject any
    character that could escape into a different shell word. Empty/``None`` →
    ``None``.

    Args:
        value (Any): Candidate directory; coerced to a stripped string.

    Returns:
        str | None: The validated path, or ``None`` for empty input.

    Raises:
        ValueError: When ``value`` contains whitespace or shell metacharacters.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _RESULT_DIR_FORBID_RE.search(text):
        raise ValueError(
            f"result_dir={text!r} rejected: contains whitespace or shell "
            "metacharacters; pass an absolute or workspace-relative path"
        )
    return text


def _is_safe_path_entry(entry: str) -> bool:
    """A single path entry is safe iff it has no ``:``, no ``..``, no control chars."""
    return (
        bool(entry)
        and ":" not in entry
        and ".." not in Path(entry).parts
        and not any(c in entry for c in ("\n", "\r", "\x00"))
    )


def _prepend_path_entry(envs: dict[str, str], var: str, entry: str) -> None:
    """Prepend a single already-validated entry onto a ``:``-joined env var."""
    parts = [p for p in str(envs.get(var, "") or "").split(":") if p]
    if entry not in parts:
        parts.insert(0, entry)
    envs[var] = ":".join(parts)


# Reserved keys carried by dedicated override fields; never accepted via runtime_env.
_RUNTIME_ENV_RESERVED: frozenset[str] = frozenset({"PATH", "PYTHONPATH", "LD_LIBRARY_PATH"})


def apply_runtime_override(envs: dict[str, str], override: dict[str, Any]) -> None:
    """Inject an attempt runtime override into the materialized YAML envs dict.

    Writes the base keys (path_prefix, pythonpath_prefix, framework_bin,
    framework_python, framework_venv_root) and the compiled-artifact keys
    (pythonpath_prefixes, ld_library_path_prefix, runtime_env, entrypoint_bin_dir)
    into benchmark.envs so the Magpie subprocess re-exports them to the server it
    boots. All writes land in the YAML layer; os.environ is never mutated.

    Path entries use the same containment checks as overlay_pythonpath (no colon
    separator, no traversal, no control chars); unsafe entries are dropped with a
    warning. runtime_env entries with an invalid/blocked/reserved key are dropped.
    An empty or all-missing override is a no-op.

    Args:
        envs: The benchmark.envs dict from the materialized YAML (mutated in place).
        override: Dict with optional keys path_prefix, pythonpath_prefix,
            framework_bin, framework_python, framework_venv_root,
            pythonpath_prefixes (list), ld_library_path_prefix (list),
            runtime_env (dict), entrypoint_bin_dir.
    """
    if not override:
        return
    path_prefix = str(override.get("path_prefix") or "").strip()
    if path_prefix:
        _prepend_path_entry(envs, "PATH", path_prefix)
    entrypoint_bin = str(override.get("entrypoint_bin_dir") or "").strip()
    if entrypoint_bin:
        if _is_safe_path_entry(entrypoint_bin):
            _prepend_path_entry(envs, "PATH", entrypoint_bin)
        else:
            log.warning("apply_runtime_override: dropping unsafe entrypoint_bin_dir %r", entrypoint_bin)
    pp_prefix = str(override.get("pythonpath_prefix") or "").strip()
    if pp_prefix:
        if _is_safe_path_entry(pp_prefix):
            _cur_pp = str(envs.get("PYTHONPATH", "") or "")
            envs["PYTHONPATH"] = f"{pp_prefix}:{_cur_pp}" if _cur_pp else pp_prefix
        else:
            log.warning("apply_runtime_override: dropping unsafe pythonpath_prefix %r", pp_prefix)
    # Multi-entry prefixes: prepend in order so the first entry wins, ahead of
    # any single-dir pythonpath_prefix and the inherited value.
    for entry in reversed(_coerce_prefix_list(override.get("pythonpath_prefixes"))):
        if _is_safe_path_entry(entry):
            _cur_pp = str(envs.get("PYTHONPATH", "") or "")
            envs["PYTHONPATH"] = f"{entry}:{_cur_pp}" if _cur_pp else entry
        else:
            log.warning("apply_runtime_override: dropping unsafe pythonpath entry %r", entry)
    # Native loader path: prepend attempt entries while preserving any inherited
    # entries (e.g. /opt/rocm/lib), which _prepend_path_entry never drops.
    for entry in reversed(_coerce_prefix_list(override.get("ld_library_path_prefix"))):
        if _is_safe_path_entry(entry):
            _prepend_path_entry(envs, "LD_LIBRARY_PATH", entry)
        else:
            log.warning("apply_runtime_override: dropping unsafe ld_library_path entry %r", entry)
    _runtime_env = override.get("runtime_env")
    if isinstance(_runtime_env, dict):
        for raw_k, raw_v in _runtime_env.items():
            key = str(raw_k)
            if not _ENV_KEY_RE.match(key) or key in BLOCKED_CHILD_ENV_NAMES or key in _RUNTIME_ENV_RESERVED:
                log.warning("apply_runtime_override: dropping unsafe runtime_env key %r", key)
                continue
            envs[key] = str(raw_v)
    for key in ("framework_bin", "framework_python", "framework_venv_root"):
        val = str(override.get(key) or "").strip()
        if val:
            envs[f"HYPERLOOM_{key.upper()}"] = val
    # runtime_python_exe takes priority over framework_python as the launch interpreter.
    rpe = str(override.get("runtime_python_exe") or "").strip()
    if rpe:
        envs["HYPERLOOM_FRAMEWORK_PYTHON"] = rpe


def _coerce_prefix_list(value: Any) -> list[str]:
    """Normalize a runtime-prefix field (list/tuple/str) to a list of entries."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _build_variant_yaml(
    base_yaml_path: Path,
    base_extra_args: str,
    variant: GridVariant,
    *,
    output_subdir: Path,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    server_lifecycle: dict[str, Any] | None = None,
    base_args_mode: str = "append",
) -> Path:
    """Materialize a per-variant Magpie YAML on disk.

    Injects the variant's flags into the framework's ``EXTRA_*_ARGS`` env,
    resolved by :func:`server_args_env_name` (``EXTRA_SGLANG_ARGS`` for sglang,
    ``EXTRA_VLLM_ARGS`` for vllm, etc.). ``model_path``
    overrides the legacy hardcoded ``benchmark.model``; ``gpu_type`` pins the
    generic ``{framework}_{gpu_type}.sh``; ``benchmark_script`` (pre-sanitized)
    force-pins a script, applied last so the operator pick wins.
    ``server_lifecycle`` (``{cleanup, pid_dir, port}``) enables Magpie's
    persistent-server reuse so a paired round can re-attach to a hot server.

    Args:
        base_yaml_path (Path): Path to the base Magpie YAML to template from.
        base_extra_args (str): Server args merged ahead of the variant's args.
        variant (GridVariant): The variant whose flags/envs are applied; its
            ``unset_envs`` may not remove a workload pin.
        output_subdir (Path): Directory the per-variant ``config.yaml`` is
            written into.
        model_path (str | None): Overrides ``benchmark.model`` when set.
        gpu_type (str | None): Pins the generic ``{framework}_{gpu_type}.sh``.
        benchmark_script (str | None): Pre-sanitized script name (applied last).
        server_lifecycle (dict[str, Any] | None): ``{cleanup, pid_dir, port}``
            enabling persistent-server reuse.

    Returns:
        Path: Path to the materialized per-variant ``config.yaml``.
    """
    with base_yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bench = cfg.setdefault("benchmark", {})
    envs = apply_runtime_benchmark_overrides(
        bench,
        model_path=model_path,
        gpu_type=gpu_type,
        benchmark_script=benchmark_script,
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    combined = compose_server_args(
        inherited_args="" if str(base_args_mode).strip().lower() == "replace" else str(envs.get(extra_args_env, "")),
        base_extra_args=base_extra_args,
        variant_extra_args=variant.extra_server_args,
        remove_args=getattr(variant, "remove_args", []),
        args_mode=getattr(variant, "args_mode", "append"),
    )
    # A grid variant never injects a MoE runner backend itself, but it does
    # inherit one -- from the baseline recipe it was seeded with, or from an
    # explicitly authored variant. On a model only the aiter runner can serve,
    # that inherited flag is a guaranteed crash on the first forward pass, and
    # the grid has no retry to salvage it, so drop it here.
    if combined and _SGLANG_MOE_RUNNER_BACKEND_RE.search(combined):
        from ._workload_envs import _remove_moe_runner_backend_arg

        if extra_args_env == "EXTRA_SGLANG_ARGS" and moe_runner_requires_aiter(combined, model_path):
            log.warning(
                "grid: dropping inherited --moe-runner-backend for variant %s: "
                "this checkpoint's MoE quant scheme is only implemented on the "
                "aiter runner and would crash on the first forward pass.",
                variant.name,
            )
            combined = _remove_moe_runner_backend_arg(combined)
    if combined:
        envs[extra_args_env] = _shell_safe_dedupe(combined)
    elif extra_args_env in envs:
        envs.pop(extra_args_env, None)
    for k in getattr(variant, "unset_envs", []) or []:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(k).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("grid: refusing to unset pinned env %s for variant %s", k, variant.name)
            continue
        envs.pop(str(k), None)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)
    # NOT re-derived per variant. The variant's own CONC only exists as of the
    # merge above, and it is tempting to re-scale AGENTX_WARMUP_GRACE_PERIOD from
    # it -- a conc sweep walks 256..2 while the session sits at one value, so a
    # high-CONC variant runs with a grace sized for the session's concurrency.
    #
    # Doing that here is WRONG while the two caps above it are session-scaled.
    # ``apply_agentx_switch`` sets ``bench["timeout_seconds"]`` and
    # ``agentx_variant_timeout_sec`` sets the subprocess cap, both from
    # os.environ's session CONC. Raising only the client's grace makes the round
    # wait inside a bound its own caps do not cover: at session CONC=8 with the
    # ladder at 256 the grace would become 57600s against a 10800s cap, so the
    # round is SIGKILLed at 10800s while warmup is still draining -- strictly
    # worse than leaving the grace alone, which is what shipped before.
    #
    # The real fix is to make the caps variant-aware too (thread the merged CONC
    # into ``agentx_variant_timeout_sec`` / ``_round_timeout_sec``); until then
    # all three numbers stay session-scaled and consistent with each other.
    # Authored-kernel overlay: prepend the built-kernel dir onto PYTHONPATH so
    # the relaunched server imports the overlay's kernels. Inert when
    # ``overlay_pythonpath`` is unset.
    _overlay = str(getattr(variant, "overlay_pythonpath", "") or "").strip()
    if _overlay:
        # Structural containment on the overlay dir before it is prepended to
        # PYTHONPATH: a legitimate authored-kernel overlay is a single existing
        # directory (never a ``:``-joined list, never a ``..`` traversal or
        # control char). Reject anything that would smuggle extra PYTHONPATH
        # entries or escape via traversal; a real overlay dir is unaffected.
        _overlay_ok = (
            ":" not in _overlay
            and ".." not in Path(_overlay).parts
            and not any(c in _overlay for c in ("\n", "\r", "\x00"))
            and Path(_overlay).is_dir()
        )
        if _overlay_ok:
            _cur_pp = str(envs.get("PYTHONPATH", "") or "")
            envs["PYTHONPATH"] = f"{_overlay}:{_cur_pp}" if _cur_pp else _overlay
        else:
            log.warning(
                "grid: dropping unsafe overlay_pythonpath %r (not a single "
                "existing directory / contains separator or traversal)",
                _overlay,
            )

    # Attempt runtime override: inject path_prefix/pythonpath_prefix/framework_bin
    # etc. into benchmark.envs so the server subprocess resolves the attempt
    # runtime.  All writes are YAML-layer; os.environ is never mutated.
    _rt_override = getattr(variant, "runtime_override", None) or {}
    if _rt_override:
        apply_runtime_override(envs, _rt_override)

    # PATH guard: the xdit wrapper needs both `/venv/bin` (the `xdit` console
    # script) and `/opt/rocm/bin` (`hipcc`); force-prepend both so an
    # LLM-supplied PATH can't drop one.
    if str(bench.get("framework", "")).strip().lower() == "xdit":
        _cur_path = str(envs.get("PATH", "") or "")
        _parts = [p for p in _cur_path.split(":") if p]
        for _essential in ("/opt/rocm/bin", "/venv/bin"):
            if _essential not in _parts:
                _parts.insert(0, _essential)
        envs["PATH"] = ":".join(_parts)

    if server_lifecycle is not None:
        from ._server_lifecycle import inject_lifecycle

        inject_lifecycle(
            bench,
            cleanup=bool(server_lifecycle.get("cleanup", True)),
            pid_dir=server_lifecycle["pid_dir"],
            port=int(server_lifecycle["port"]),
        )

    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    """Load ``benchmark_report.json`` from a benchmark workspace.

    Args:
        workspace (Path): Directory expected to contain
            ``benchmark_report.json``.

    Returns:
        dict[str, Any] | None: The parsed report dict, or ``None`` if the
        file is missing, unreadable, invalid JSON, or not a JSON object.
    """
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# How long to keep re-reading ``benchmark_report.json`` when the process exited
# cleanly but the report does not yet parse into a valid measurement.
#
# The report is written by the benchmark subprocess as it shuts down, and the
# reader runs the moment that subprocess is reaped — so on a loaded filesystem the
# two race. Measured on a live 24-hour session: six of thirteen variants aborted
# as ``benchmark_report_invalid_metric`` while every one of those reports, read
# afterwards, held valid throughput. Two of the six were authored patches worth
# +4.4% and +4.7% whose switch-off parity legs had already passed, so the race
# did not just cost measurements, it discarded accepted work.
#
# Only a clean exit is worth waiting on: a non-zero return code means the run
# genuinely failed and there is nothing to settle.
REPORT_SETTLE_SECONDS = 30.0
REPORT_SETTLE_POLL_SECONDS = 1.0


async def _settled_measurement(
    workspace: Path,
    *,
    subprocess_started_unix: float | None,
    settle_seconds: float = REPORT_SETTLE_SECONDS,
    poll_seconds: float = REPORT_SETTLE_POLL_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read the benchmark report, re-reading briefly while it is still settling.

    Args:
        workspace: Benchmark workspace holding ``benchmark_report.json``.
        subprocess_started_unix: Launch time, forwarded to the leak-salvage pass.
        settle_seconds: How long to keep retrying an invalid measurement. Pass
            ``0`` to read exactly once (the run already failed, so waiting only
            delays the verdict).
        poll_seconds: Delay between attempts.

    Returns:
        The parsed report (or ``None``) and the extracted measurement, from the
        first attempt that yielded a valid measurement, else from the last.
    """
    deadline = time.monotonic() + max(0.0, float(settle_seconds))
    attempts = 0
    while True:
        report = _parse_report(workspace)
        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=subprocess_started_unix,
        )
        attempts += 1
        if measurement.get("valid_measurement") or time.monotonic() >= deadline:
            if attempts > 1 and measurement.get("valid_measurement"):
                log.info(
                    "grid_runner: benchmark_report became valid after %d read(s); "
                    "the report was still being written when the subprocess was reaped",
                    attempts,
                )
            return report, measurement
        await asyncio.sleep(max(0.01, float(poll_seconds)))


def _run_grid_warmup_enabled() -> bool:
    """Whether ``run_grid`` should discard a cold warmup round when possible."""
    raw = os.environ.get("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP")
    if raw is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return (raw if raw is not None else "1").strip().lower() not in {"0", "false", "no", "off", ""}


def _read_pid_gpu_mask(pid: int) -> tuple[list[int], bool] | None:
    """Resolve ``pid``'s visible-GPU mask from its own environment.

    Mirrors :func:`hyperloom.orchestrator.bus.gpu_pool._visible_device_mask`
    (``ROCR_VISIBLE_DEVICES``, then ``HIP``/``CUDA``), but reads a foreign
    process's ``/proc/<pid>/environ`` instead of our own ``os.environ`` --
    used to scope :func:`_kill_stale_servers` to our own GPU allocation
    (AMD-AGI/Hyperloom#1354).

    Args:
        pid: Candidate process id to inspect.

    Returns:
        ``(ids, present)`` as in ``_visible_device_mask``, or ``None`` when
        ``/proc/<pid>/environ`` cannot be read (permission, already exited,
        etc.) -- callers must treat that as "unknown", not "no mask".
    """
    from ...bus.gpu_pool import _parse_gpu_list

    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            raw = fh.read()
    except (OSError, PermissionError):
        return None
    env = {}
    for entry in raw.split(b"\0"):
        if b"=" not in entry:
            continue
        key, _, value = entry.partition(b"=")
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    for env_name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if env_name in env:
            return _parse_gpu_list(env[env_name]), True
    return [], False


def _kill_stale_servers() -> None:
    """Deep-clean any lingering inference server processes + shared memory.

    Reaps vLLM::Worker / EngineCore children that escape Magpie's pgrp-leader
    cleanup. Called before every Magpie invocation; uses a /proc scan (not
    pgrep) to avoid clashing with test subprocess mocks. No-op in multi-node
    mode (servers live in RayJob pods).

    Scoped to our own GPU allocation when one is known: if our own process
    has a visible-GPU mask (``ROCR_VISIBLE_DEVICES`` et al, set whenever an
    operator carved us a subset of the machine's cards rather than handing us
    the whole box), a candidate process is only reaped when its own mask
    overlaps ours. A candidate whose mask cannot be read (permission,
    already exited) or that declares no mask at all is skipped rather than
    reaped -- unknown scope is treated as "not ours", never as "safe to
    kill" (AMD-AGI/Hyperloom#1354). With no mask on our own side (the whole
    machine is ours, or nothing is scoping either side), every match is
    reaped as before.

    Note:
        Side-effecting and best-effort: it sends signals to matching processes
        and unlinks stale shared-memory segments, swallowing errors. Returns
        nothing.
    """
    from ._multi_node_env import is_multi_node
    from ...bus.gpu_pool import _visible_device_mask

    if is_multi_node():
        return

    import signal
    import glob
    import time

    _KILL_PATTERNS = (
        "VLLM::Worker",
        "VLLM::EngineCore",
        "vllm.entrypoints",
        "vllm serve",
        "sglang.srt",
        "sglang.launch_server",
        "atom.entrypoints",
        "atom.entrypoints.openai_server",
    )

    # atom ModelRunner workers spawn with a generic ``--multiprocessing-fork``
    # cmdline (unmatchable by _KILL_PATTERNS) and can orphan holding VRAM;
    # identify survivors by atom/aiter JIT mmaps in their address space.
    _FORK_MARKERS = (b"--multiprocessing-fork", b"spawn_main")
    _ATOM_MAP_SIGNATURES = ("/ATOM/atom/", "/aiter/jit/", "/aiter-test/aiter/")

    my_pid = os.getpid()
    try:
        my_pgid = os.getpgrp()
    except OSError:
        my_pgid = -1
    my_gpu_ids, my_gpu_mask_present = _visible_device_mask()
    my_gpu_id_set = frozenset(my_gpu_ids)

    def _in_our_gpu_scope(pid: int) -> bool:
        """Whether ``pid`` overlaps our own visible-GPU mask.

        No-op (always True) when we have no mask ourselves -- nothing to
        scope against. Otherwise conservative: an unreadable or absent mask
        on the candidate's side is out of scope, not in it.
        """
        if not my_gpu_mask_present:
            return True
        candidate = _read_pid_gpu_mask(pid)
        if candidate is None:
            return False
        candidate_ids, candidate_present = candidate
        if not candidate_present:
            return False
        return not my_gpu_id_set.isdisjoint(candidate_ids)

    def _is_orphaned_atom_worker(pid: int, cmdline: bytes) -> bool:
        """Detect an orphaned atom ModelRunner worker by its memory maps.

        A spawned atom worker has a generic ``--multiprocessing-fork`` cmdline,
        so it is identified instead by atom/aiter signatures mmap'd into its
        address space. Workers belonging to this process group are excluded.

        Args:
            pid (int): Candidate process id.
            cmdline (bytes): The process's raw ``/proc/<pid>/cmdline``.

        Returns:
            bool: ``True`` iff ``cmdline`` carries a fork marker, the process
            is outside our process group, and its ``/proc/<pid>/maps`` shows
            an atom/aiter signature; ``False`` otherwise (including on any
            read/permission error).
        """
        if not any(m in cmdline for m in _FORK_MARKERS):
            return False
        # Never touch a worker that belongs to *our* process group.
        try:
            if my_pgid != -1 and os.getpgid(pid) == my_pgid:
                return False
        except (OSError, ProcessLookupError):
            return False
        try:
            with open(f"/proc/{pid}/maps", "r", errors="replace") as fh:
                maps = fh.read()
        except (OSError, PermissionError):
            return False
        return any(sig in maps for sig in _ATOM_MAP_SIGNATURES)

    killed_atom = False
    killed_any = False
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read()
        except (OSError, PermissionError):
            continue
        text = cmdline.replace(b"\0", b" ").decode("utf-8", "replace")
        is_atom_server = "atom.entrypoints" in text
        if not (any(pat in text for pat in _KILL_PATTERNS) or _is_orphaned_atom_worker(pid, cmdline)):
            continue
        if not _in_our_gpu_scope(pid):
            continue
        killed_any = True
        killed_atom = killed_atom or is_atom_server or b"--multiprocessing-fork" in cmdline
        # Kill the whole pgrp so atom children die with the leader.
        try:
            pgid = os.getpgid(pid)
            if pgid not in (my_pgid, 0):
                os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # Group gone or not ours; fall through to per-pid kill.
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Already exited or owned by another user.
            pass

    # Clear GPU runtime shared-memory segments that prevent re-binding. These
    # files carry no GPU/owner tag we can scope by, so -- unlike the per-pid
    # reap above -- they are only touched when we have no GPU mask of our own
    # (whole machine is ours): with a mask set, a co-tenant's otherwise-spared
    # server could still crash from having its shared-memory segments pulled
    # out from under it (AMD-AGI/Hyperloom#1354).
    if not my_gpu_mask_present:
        for pattern in (  # nosec B108 - intentionally targets known /dev/shm runtime prefixes.
            "/dev/shm/vllm*",
            "/dev/shm/nccl*",
            "/dev/shm/cuda*",
            "/dev/shm/torch*",
            "/dev/shm/atom*",
        ):
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except OSError:
                    # Already removed or held by another process.
                    pass

    # Pause for KFD async VRAM release; atom teardown lags past 2s. Skipped
    # entirely when nothing was actually killed this call -- this function now
    # runs at 4 call sites (was 1), so paying it unconditionally on the common
    # "GPU was already clean" case adds up fast (e.g. conc_sweep's own reap
    # immediately followed by baseline's).
    if killed_any:
        time.sleep(8 if killed_atom else 2)


def _prepend_magpie_pythonpath(magpie_dir: str, current_pythonpath: str) -> str:
    """Prepend Magpie's import root to PYTHONPATH, skipping package-root dirs.

    A ``site-packages`` MAGPIE_PATH is already importable; prepending it would
    shadow an isolated vLLM venv's torch. A checkout root is kept.
    """
    if not magpie_dir or is_python_package_root(magpie_dir):
        return current_pythonpath
    return f"{magpie_dir}:{current_pythonpath}" if current_pythonpath else magpie_dir


def _run_magpie(
    *,
    magpie_python: str,
    config_path: Path,
    output_dir: Path,
    timeout_sec: int,
    cwd: str,
    result_dir: str | None = None,
    soft_deadline_sec: float | None = None,
    preclean: bool = True,
    server_already_ready: bool = False,
    serving_lease: Any = None,
    on_output: Callable[[], None] | None = None,
    session_deadline_sec: float | None = None,
) -> tuple[int, str, str]:
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr).

    ``result_dir`` (pre-sanitized via :func:`sanitize_result_dir`) overrides
    ``$RESULT_DIR``, which is always set (default ``output_dir``) so results
    land in the per-task workspace, not ``/workspace/``. ``soft_deadline_sec``
    is the Fix-E overtime cap: the tree is reaped and a sentinel
    ``OVERTIME_KILL_RETURNCODE`` returned instead of raising ``TimeoutExpired``.
    ``server_already_ready`` is forwarded to :func:`run_with_session_kill` so
    warm reuse rounds (client-only, no server boot) use the from-spawn soft clock
    instead of the from-ready clock (which would never arm on an empty log).

    Args:
        magpie_python (str): Python interpreter used to launch Magpie.
        config_path (Path): Path to the per-variant Magpie config YAML.
        output_dir (Path): Per-task output/workspace directory.
        timeout_sec (int): Hard subprocess timeout in seconds.
        cwd (str): Working directory for the Magpie subprocess.
        result_dir (str | None): Pre-sanitized ``$RESULT_DIR`` override;
            defaults to ``output_dir``.
        soft_deadline_sec (float | None): Overtime soft deadline; reaps the tree
            and returns ``OVERTIME_KILL_RETURNCODE``.
        preclean (bool): Whether to pre-clean stale servers before launch.
        server_already_ready (bool): Pass ``True`` for warm reuse rounds so the
            soft-deadline clock runs from process spawn, not the ready marker.
        serving_lease: When set (Ray-managed GPU execution, §12 T1), the round
            runs inside the lease's actor — which holds ``num_gpus`` across every
            round sharing this server — instead of a local subprocess. ``None``
            keeps the existing local ``run_with_session_kill`` path unchanged.
        on_output: Liveness callback invoked from the reader thread on each
            line the benchmark emits, so the caller's heartbeat can keep
            reporting across a run that blocks for hours. Ignored on the
            ``serving_lease`` path — see the note there.
        session_deadline_sec (float | None): Absolute ``time.monotonic()`` instant
            at which the session budget expires. Reaps the tree and returns
            ``SESSION_TIME_EXHAUSTED_RETURNCODE``. Enforced in every phase,
            including the accuracy eval that retires ``soft_deadline_sec``. The
            lease path sends it across as a remaining-seconds duration, the only
            form that means the same thing in the actor's process.

    Returns:
        tuple[int, str, str]: ``(returncode, stdout, stderr)``.
    """
    # Pre-clean lingering servers + shared memory (skip under pytest, and for
    # lifecycle re-attach rounds that would kill the warm server).
    if preclean and not os.environ.get("PYTEST_CURRENT_TEST"):
        _kill_stale_servers()

    env = scrub_benchmark_process_env(os.environ.copy())
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_PATH") or ""
    if magpie_dir:
        env["PYTHONPATH"] = _prepend_magpie_pythonpath(magpie_dir, env.get("PYTHONPATH", ""))

    # Multi-node: tell Magpie to skip its local-server launch and point
    # benchmark_serving at the head pod's ClusterIP.
    from ._multi_node_env import magpie_remote_env

    env.update(magpie_remote_env())

    # Pin Magpie's InferenceX resolution to ``$INFERENCEX_PATH`` (its
    # highest-precedence rung) so it loads the patched checkout, not a stale copy.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Baseline patches its own checkout, but explore / sweep never pass
        # through that hook: re-assert here so a resumed session or a re-cloned
        # checkout still emits the eval-start marker. Idempotent.
        ensure_benchmark_lib_eval_start_patched(Path(inferencex_path))

    # The generation bounds + pathology probe are asserted whether or not
    # ``$INFERENCEX_PATH`` is set: unset falls back to the same env discovery the
    # baseline arm uses ($MAGPIE_PATH/InferenceX). Gating this on that variable
    # is what allowed a bounded baseline to be compared against an unbounded
    # candidate -- the differential accuracy gate subtracts the two arms, so it
    # only means something when both truncate at the same place. A target that is
    # present but unpatchable therefore fails this variant with the same
    # ``eval_probe_unpatchable`` class the baseline arm raises, instead of
    # silently benching without bounds. A target that is absent entirely stays a
    # warning: that is an unrecognized layout, not a broken contract, and the
    # baseline arm makes the same distinction.
    probe_root = Path(inferencex_path) if inferencex_path else None
    if not ensure_eval_probe_patched(probe_root) and not materialized_run_eval_disabled(config_path):
        eval_bounds_msg = (
            "eval generation bounds + pathology probe are not installed "
            "(utils/evals/patches/lm_eval_sitecustomize.py, inferencex="
            f"{inferencex_path or '<unset>'}, INFERENCEX_PATH="
            f"{os.environ.get('INFERENCEX_PATH', '') or '<unset>'}); this "
            "variant runs eval, so it would be scored against a baseline that "
            "truncated at a different point"
        )
        if eval_probe_targets_exist(probe_root):
            log.error("grid_runner: %s; failing this benchmark", eval_bounds_msg)
            return (EVAL_PROBE_UNPATCHABLE_RETURNCODE, "", eval_bounds_msg)
        log.warning("grid_runner: %s", eval_bounds_msg)
    # AgentX: deploy the aiperf client into InferenceX ``benchmarks/`` + preflight
    # aiperf right before Magpie runs it, via the shared helper (also used by the
    # baseline/profile shell-out). No-op under pytest / when AgentX is off (the
    # helper keeps the agentx package import lazy for the OFF path, A2). A failed
    # preflight becomes a structured nonzero rc so the grid records a failed
    # benchmark instead of crashing.
    from ._workload_envs import prepare_agentx_runtime

    _agx_err = prepare_agentx_runtime(env=env, inferencex_path=inferencex_path, config_path=config_path)
    if _agx_err:
        log.error("%s; failing this benchmark", _agx_err)
        return (AGENTX_PREFLIGHT_RETURNCODE, "", _agx_err)
    # RESULT_DIR default; leaks are picked up by the salvage path.
    env["RESULT_DIR"] = result_dir or str(output_dir)
    # InferenceX ``run_lm_eval`` cleans ``$EVAL_RESULT_DIR`` after processing
    # lm-eval output. Keep it under the task slot but separate from Magpie traces.
    env["EVAL_RESULT_DIR"] = str(Path(env["RESULT_DIR"]) / "eval_output")
    # Pin SERVER_LOG / GPU_METRICS_CSV per-task so logs land alongside
    # ``benchmark_report.json``. Always overwrite so a stale parent value can't
    # redirect into a prior run's slot.
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")

    # Ray-managed GPU execution (§12 T1): route the round through the serving
    # lease's actor, which holds ``num_gpus`` across every round sharing this
    # server. The lease is owned by the caller (conc_sweep arm / baseline /
    # sweep / explore), so one lease spans boot + all reuse rounds — no detached
    # GPU process outlives its Ray lease (§4.2). Ray sets ``*_VISIBLE_DEVICES``
    # in the worker, so the YAML's device list is stripped first (T2) to stop
    # Magpie re-exporting it and overriding Ray's card assignment.
    if serving_lease is not None:
        from ._ray_backend import strip_visible_devices_from_config

        ray_config_path = strip_visible_devices_from_config(config_path)
        cmd = build_benchmark_command(
            python_exe=magpie_python,
            config_path=ray_config_path,
            output_dir=output_dir,
        )
        # ``on_output`` cannot follow the round here: the benchmark runs inside a
        # Ray actor in another process (potentially on another node) and only its
        # final ``(rc, stdout, stderr)`` comes back, so there is nothing local to
        # call per line. A Ray-backed variant therefore reports on entry and then
        # goes quiet until it returns — a known gap, not an oversight.
        return serving_lease.run_session_kill(
            cmd,
            env=env,
            cwd=cwd,
            timeout=timeout_sec,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=str(output_dir / "server.log"),
            server_already_ready=server_already_ready,
            # Converted here, at the last moment before the process boundary:
            # the actor cannot read this process's monotonic clock.
            session_remaining_sec=session_deadline_to_remaining_sec(session_deadline_sec),
        )

    cmd = build_benchmark_command(
        python_exe=magpie_python,
        config_path=config_path,
        output_dir=output_dir,
    )
    # run_with_session_kill launches Magpie in its own POSIX session and tears
    # down the whole descendant tree on every exit path.
    proc = run_with_session_kill(
        cmd,
        env=env,
        cwd=cwd,
        timeout=timeout_sec,
        soft_deadline_sec=soft_deadline_sec,
        server_log_path=str(output_dir / "server.log"),
        server_already_ready=server_already_ready,
        on_output=on_output,
        session_deadline_sec=session_deadline_sec,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _num_gpus_for_config(config_path: Path) -> float:
    """Read the tensor-parallel size (``TP``) from a materialized benchmark YAML.

    Used as the Ray ``num_gpus`` for a benchmark lease so Ray leases exactly the
    cards the server needs. Falls back to 1 on any read/parse error.

    Args:
        config_path: Path to the materialized benchmark config YAML.

    Returns:
        The GPU count (``TP``) as a float for Ray's ``num_gpus``.
    """
    try:
        with Path(config_path).open(encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
        envs = (cfg.get("benchmark") or {}).get("envs") or {}
        return float(int(envs.get("TP", 1) or 1))
    except Exception:  # noqa: BLE001 — best-effort; default to 1 GPU
        return 1.0


def _mn_restart_env(
    reference_envs: dict[str, str],
    variant: Any,
    unset_envs: list[str],
) -> dict[str, str]:
    """Merge the reference server envs under one variant's own overrides.

    Priority reference < variant, matching how the args side layers reference <
    operator < per-task. Keys the variant unsets are dropped rather than
    re-added, so an unset still means unset once the reference carries the key.

    Args:
        reference_envs: Reference-recipe server envs for the whole grid.
        variant: The grid variant supplying ``extra_envs``.
        unset_envs: Env names this variant removes.

    Returns:
        The env mapping to forward to the multi-node server restart.
    """
    merged = {k: v for k, v in reference_envs.items() if k not in unset_envs}
    merged.update({str(k): str(v) for k, v in (variant.extra_envs or {}).items()})
    return merged


def _resolve_mn_effective_server_args(
    cfg_path: Path,
    base_yaml_path: Path,
    variant: Any,
    *,
    base_extra_args: str,
    base_args_mode: str,
) -> str:
    """Resolve the multi-node server args for a variant restart; prefer the
    materialized variant YAML, falling back to a recompose from the base YAML."""
    try:
        with cfg_path.open(encoding="utf-8") as _f:
            _variant_cfg = yaml.safe_load(_f) or {}
        _variant_bench = _variant_cfg.get("benchmark") or {}
        _variant_envs = _variant_bench.get("envs") or {}
        _variant_framework_env = server_args_env_name(_variant_bench.get("framework"))
        return str(_variant_envs.get(_variant_framework_env) or "")
    except Exception:  # noqa: BLE001 - restart path still reports validation errors
        log.debug(
            "grid_runner: failed to read materialized variant args from %s",
            cfg_path,
            exc_info=True,
        )
        try:
            with base_yaml_path.open(encoding="utf-8") as _f:
                _base_cfg = yaml.safe_load(_f) or {}
            _base_bench = _base_cfg.get("benchmark") or {}
            _base_envs = _base_bench.get("envs") or {}
            _base_framework_env = server_args_env_name(_base_bench.get("framework"))
            _fallback_inherited_args = str(_base_envs.get(_base_framework_env) or "")
        except Exception:  # noqa: BLE001 - best-effort parity fallback
            _fallback_inherited_args = ""
        return _shell_safe_dedupe(
            compose_server_args(
                inherited_args="" if str(base_args_mode).strip().lower() == "replace" else _fallback_inherited_args,
                base_extra_args=base_extra_args,
                variant_extra_args=variant.extra_server_args,
                remove_args=getattr(variant, "remove_args", []),
                args_mode=getattr(variant, "args_mode", "append"),
            )
        )


def _variant_progress_note(
    grid: list[GridVariant],
    results: list[VariantResult],
    idx: int,
) -> dict[str, Any]:
    """Build the progress note for the variant at ``idx`` from that variant's own row.

    The row is located by index and never taken from the tail of ``results``: a
    stop cause that ends the batch records the round it stopped and then a
    not-run row for every later variant, so the tail is the last variant in the
    grid rather than the one that just ran. Each variant contributes exactly one
    row, in order, before it reports, which is what makes ``idx`` where its row
    is. The log line beside this note derives from ``idx`` already; the note is
    the artefact the heartbeat exists to make honest, and the one a stall signal
    reads, so it cannot be the one that names the wrong variant.

    Args:
        grid (list[GridVariant]): The variants being run.
        results (list[VariantResult]): Rows recorded so far.
        idx (int): Zero-based index of the variant being reported.

    Returns:
        dict[str, Any]: Keyword note for :func:`report_progress`.
    """
    landed = results[idx] if idx < len(results) else None
    return {
        "unit": "variant",
        "label": grid[idx].name,
        "index": idx + 1,
        "total": len(grid),
        "status": getattr(landed, "status", None),
        "output_throughput": getattr(landed, "output_throughput", None),
    }


# Seconds by which a round's hard cap is allowed to sit past the session deadline,
# so the in-process session watchdog (which attributes the kill correctly) trips
# before the hard cap (which the ledger reads as a variant timeout). Small enough
# that it cannot meaningfully eat the close-window reserve.
_SESSION_KILL_GRACE_SEC: int = 15

# The returncode side of :mod:`...stop_attribution`: the same two causes, keyed
# by the sentinel a subprocess comes back with. The classes themselves live in
# that leaf because the ledgers downstream of here carry the class, not the
# returncode.
_STOPPED_BY_THE_RUN: dict[int, StoppedByTheRun] = {
    SESSION_TIME_EXHAUSTED_RETURNCODE: STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
    ORCHESTRATOR_CANCELLED_RETURNCODE: STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS],
}


def stopped_by_the_run(returncode: int | None) -> StoppedByTheRun | None:
    """Return how to record a round the run itself stopped, if it did.

    Args:
        returncode: The round's returncode.

    Returns:
        StoppedByTheRun | None: How to record it, or ``None`` when the
            returncode says something about the round rather than about the run.
    """
    if returncode is None:
        return None
    return _STOPPED_BY_THE_RUN.get(int(returncode))


def agentx_variant_timeout_sec(cap: int, *, shared_state: Any = None) -> int:
    """Raise a variant's hard cap to what an AgentX round actually needs.

    Every variant cap in the tree is sized for the synthetic 1024/1024 shape --
    7800s for integrate, 2400s for explore, 1800s for the conc sweep. A
    canonical AgentX warmup is 10 requests per lane against real agentic
    traces, which on a 700B-class model runs well past two hours before the
    measured round even begins, so those caps kill the round mid-warmup.
    Measured on GLM-5.2: a variant launched 09:47:41 was killed at
    11:47:41.575, twenty-plus connections dropping in the same millisecond
    while the server was still prefilling with 55 requests running. Downstream
    that reads as a warmup failure, because aiperf treats a cancelled root
    warmup credit as terminal -- so the real cause (a subprocess kill) is
    invisible in the abort reason.

    ``baseline`` already derives an AgentX-aware cap; only that one path got
    it. This reuses the same derivation rather than introducing a second number
    to keep in sync, and never lowers a cap, so an operator who asked for
    longer keeps it.

    AgentX is an opt-in benchmark branch: with it disabled this returns ``cap``
    untouched and the default path is unaffected.

    ``shared_state`` is what makes the check survive a lost env var. The
    original report is exactly that case: a session resumed into a shell without
    HYPERLOOM_AGENTX, or a variant round driven from a subprocess that did not
    inherit it, reads as synthetic here and the round is killed by the synthetic
    cap mid-warmup -- the failure this function exists to prevent, reached by the
    one route it did not cover. ``benchmark_mode`` is stamped at seed for
    precisely this, so a caller holding the session state should pass it.

    Known gap: ``run_grid``'s own call sites do not pass it yet. Threading state
    through nine call sites in six files is a change of its own, and
    ``agentx_active(None)`` is exactly today's behaviour, so those paths are no
    worse than before while the sweep -- which already holds the state -- gets
    the durable signal.

    Args:
        cap: The declared hard timeout for the round, in seconds.
        shared_state: Session state, when the caller has one. Consulted only
            when the env var is absent.

    Returns:
        int: ``cap``, or the AgentX-derived cap when that is larger.
    """
    # Local import: baseline imports from this module, and the rest of the file
    # already resolves _workload_envs this way.
    from ._workload_envs import agentx_active

    if not agentx_active(shared_state):
        return cap
    from .baseline import agentx_baseline_timeout_sec

    return max(cap, agentx_baseline_timeout_sec())


def session_clamped_timeout_sec(
    cap: int,
    session_deadline_sec: float | None,
    *,
    reserve_sec: float = 0.0,
) -> int:
    """Reduce a hard timeout to what the session budget can still pay for.

    ``session_deadline_sec`` already excludes the close-window reserve, so no
    further margin is taken for the session itself. Never returns less than 1 --
    a non-positive cap reads as "no timeout" to the subprocess layer, which is
    the opposite of what an exhausted budget means. Whether the round should
    start at all is the caller's decision, not this one's.

    A small grace is added past the session deadline so the in-process session
    watchdog trips first: both fire at the same instant, and the hard cap raises
    ``TimeoutExpired``, which the ledger records as a timeout of the thing being
    measured. The watchdog's sentinel says the run ran out of time, which is what
    actually happened.

    A non-zero ``reserve_sec`` is the only thing that can move the returned cap
    earlier than the session deadline: with no reserve the grace puts the cap at
    least ``_SESSION_KILL_GRACE_SEC`` past it, so the watchdog always gets there
    first. Every caller that reserves therefore re-opens the spurious-timeout
    window the grace exists to close, and owes a reason its round cannot be reaped
    by its own cap: :func:`run_grid` refuses a variant whose rounds do not fit,
    both before the per-variant server restart and again after it, so what it
    reserves for is a round it has just re-checked there is room for.

    Args:
        cap: The timeout the caller would grant with an unbounded budget.
        session_deadline_sec: Monotonic-clock session deadline, or ``None`` when
            the budget is unbounded, which leaves ``cap`` untouched.
        reserve_sec: Seconds held back for rounds that must still follow this
            one, so an early round cannot spend the budget a later one needs.

    Returns:
        int: The hard timeout to grant this round, in seconds.
    """
    if session_deadline_sec is None:
        return int(cap)
    usable = int(session_deadline_sec - time.monotonic() - max(0.0, reserve_sec)) + _SESSION_KILL_GRACE_SEC
    return int(cap) if usable >= int(cap) else max(1, usable)


def session_grid_bounds(shared_state: Any) -> tuple[float | None, float | None]:
    """Resolve ``(session_deadline_sec, variant_expected_sec)`` for a :func:`run_grid` call.

    Every arm that benches on the GPU needs the same two numbers, and they have
    to agree: a deadline derived one way in one executor and another way in the
    next produces arms that abandon different amounts of the tail budget. Both
    are read here so there is one definition.

    ``variant_expected_sec`` is what a normally-behaving variant needs -- its own
    server boot and then its benchmark
    (:func:`~...phases.machine_state.one_more_measurement_sec`) -- and is
    deliberately not the declared timeout, which is a catastrophic-hang backstop
    roughly twice as large. Admitting on the backstop would refuse to start a
    20-minute round with 30 minutes left.

    Nor is it the baseline's cold wall-clock, which is the fallback only for a
    session whose baseline reported no boot/benchmark split. That figure also
    carries the first request's kernel compile on a cold JIT cache, which a
    variant on the now-populated cache does not pay again, so admitting on it
    abandons the tail of the budget to variants that would have finished.

    Args:
        shared_state: The session ``SharedState``, or ``None`` when the caller
            has no session context (direct executor invocation, tests).

    Returns:
        tuple[float | None, float | None]: The monotonic-clock session deadline
            and the expected per-variant runtime. Either is ``None`` when
            unknown, which leaves the corresponding check disabled rather than
            guessing a bound.
    """
    if shared_state is None:
        return (None, None)
    deadline_fn = getattr(shared_state, "grid_session_deadline_sec", None)
    deadline = deadline_fn() if callable(deadline_fn) else None
    variant_sec = _phase_state.one_more_measurement_sec(shared_state)
    if variant_sec is None:
        variant_sec = _phase_state.measured_seconds(shared_state, "baseline_runtime_sec")
    return (deadline, variant_sec)


async def run_grid(
    *,
    base_yaml_path: Path,
    base_extra_args: str,
    grid: list[GridVariant],
    output_root: Path,
    magpie_python: str | None = None,
    variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
    keep_going_on_failure: bool = True,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    result_dir: str | None = None,
    soft_deadline_sec: float | None = None,
    server_lifecycle: dict[str, Any] | None = None,
    base_args_mode: str = "append",
    warmup_before_measure: bool | None = None,
    preclean_before_run: bool = True,
    server_already_ready: bool = False,
    serving_lease: Any = None,
    session_deadline_sec: float | None = None,
    variant_expected_sec: float | None = None,
) -> list[VariantResult]:
    """Execute each grid variant and return all per-variant results.

    ``serving_lease`` (Ray-managed GPU execution, §12 T1) is a caller-owned
    :class:`~._ray_serving.ServingLease` that every round in this grid runs
    through, so one Ray GPU lease spans the variant's warmup + measure rounds.
    ``None`` keeps the local-subprocess path. The lease's lifecycle (create /
    close) is owned by the caller so it can also span multiple ``run_grid``
    calls that reuse one persistent server (conc_sweep arm, explore warmup +
    decision).

    ``session_deadline_sec`` is a ``time.monotonic()`` deadline for the whole
    session budget, already excluding the close-window reserve. It does two
    things, deliberately split:

    * **Whether to start.** A variant is skipped, and every variant after it,
      once the remaining budget cannot fit ``variant_expected_sec`` -- what a
      normally-behaving variant needs. Judging by ``variant_timeout_sec``
      instead means judging by the catastrophic backstop (``baseline x 2`` for
      explore), which abandons the tail of the budget to variants that would
      have finished comfortably. ``None`` falls back to ``variant_timeout_sec``,
      preserving the stricter behaviour for callers that cannot estimate.
    * **How long it may run.** The cap handed to each round is clamped to what
      the session can still pay for, recomputed per round because a warmup
      consumes budget too. Without this a variant admitted near the end can
      outlive the whole session: explore derives caps up to 4h from the measured
      baseline and never consulted the budget, so a 3h run could grant one
      variant more time than the run was given.

    Every pass runs with ``output_root`` as its working directory, the way the
    baseline arm anchors Magpie to its own output dir. That is what marks the
    benchmark subtree as this session's on a shared node: the robustness reactor
    only believes a load generator that it can tie to the session, and a grid
    launched from the system temp directory carried no such tie, so a server
    dying mid-variant read as the idle gap between two variants.
    """
    if not magpie_python:
        # Backend-aware: bypass uses a plain python3, not Magpie's venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        magpie_python = resolve_benchmark_interpreter()
    else:
        magpie_python = _validate_magpie_python_override(magpie_python)
    if warmup_before_measure is None:
        warmup_before_measure = _run_grid_warmup_enabled()
    auto_warmup_requested = bool(warmup_before_measure and server_lifecycle is None)
    results: list[VariantResult] = []
    # This function names the working directory, so it creates it: callers and
    # the per-variant config writer both happen to create it first today, and
    # neither is a contract. The old system-temp default never needed one.
    output_root.mkdir(parents=True, exist_ok=True)
    cwd = str(output_root)

    # Reap orphaned aiter JIT build locks before booting any server. A prior GPU
    # process killed mid-``hipcc`` (e.g. an OOM'd co-scheduled server, or a
    # specialist reaped mid-compile) leaves a zero-byte ``FileBaton`` lock, and
    # aiter's ``FileBaton.wait()`` spins forever with no timeout — hanging every
    # later cold server boot (observed as repeated conc_sweep boot timeouts).
    # ``baseline.py`` sweeps on its own cold path; this covers the grid serving
    # paths (explore / sweep / conc_sweep / integrate_patch / framework bench).
    # Gated on no live compiler so a genuine in-flight build is never disturbed.
    try:
        from ._aiter_jit import sweep_stale_aiter_locks_if_dead

        _lock_sweep = sweep_stale_aiter_locks_if_dead()
        if _lock_sweep.get("deleted"):
            log.warning(
                "grid_runner: reaped %d orphaned aiter JIT lock(s) under %s before server launch (compiler_alive=%s)",
                _lock_sweep.get("deleted"),
                _lock_sweep.get("dir"),
                _lock_sweep.get("compiler_alive"),
            )
    except Exception as exc:  # noqa: BLE001 — best-effort hygiene, never blocks the grid
        log.debug("grid_runner: aiter lock sweep swallowed: %r", exc)

    # Multi-node: the reference recipe's server envs do not reach the variant
    # restart through restart_server_for_round, which never reads them. The
    # baseline forwards them explicitly, so without this every variant runs on a
    # server missing the envs the baseline was measured with and the gain is
    # misattributed — the env twin of the skewed-baseline bug the operator-args
    # folding fixed.
    from ._multi_node_env import is_multi_node as _mn_is_multi_node

    _mn_ref_envs: dict[str, str] = {}
    if _mn_is_multi_node():
        from ._workload_envs import resolve_reference_base

        _, _mn_ref_envs = resolve_reference_base()

    # Reported on entry, not on completion: ``_report_finished_variant`` only runs once a
    # result has been appended, so a first variant that hangs — or a branch that
    # raises before reaching it — would emit nothing at all, which is exactly
    # the silence the heartbeat exists to break.
    async def _unit_started(idx: int, label: str) -> None:
        """Report that a unit of variant ``idx`` is about to start.

        Args:
            idx (int): Zero-based index of the variant the unit belongs to.
            label (str): Unit name (``"variant"``, ``"warmup"``, ...).
        """
        await report_progress(
            unit="variant_step",
            label=f"{grid[idx].name}:{label}",
            index=idx + 1,
            total=len(grid),
            status="started",
        )

    async def _reported_magpie(idx: int, label: str, **kwargs: Any) -> tuple[int, str, str]:
        """Run one Magpie pass, announced on entry and kept alive by its output.

        The entry note covers the wait before the child says anything; the
        heartbeat covers the hours after it does. Without the second half a
        benchmark could hold the row silent for a whole variant timeout against
        a suppression window three orders of magnitude shorter.

        Args:
            idx (int): Zero-based index of the variant this pass belongs to.
            label (str): Unit name (``"warmup"``, ``"mn_warmup"``,
                ``"benchmark"``).
            **kwargs (Any): Forwarded to :func:`_run_magpie`.

        Returns:
            tuple[int, str, str]: ``(returncode, stdout, stderr)``.
        """
        await _unit_started(idx, label)
        async with heartbeat_while_output_flows(
            unit="variant_step",
            label=f"{grid[idx].name}:{label}",
            index=idx + 1,
            total=len(grid),
        ) as activity:
            return await asyncio.to_thread(_run_magpie, on_output=activity.note, **kwargs)

    # Variant boundary: a progress heartbeat so a grid that runs for hours is
    # distinguishable from one that hung on its first variant. Stall detection
    # reads this note as the counter-evidence that withholds an accusation
    # against the agent the work is attributed to.
    async def _report_finished_variant(idx: int) -> None:
        """Report the variant that just landed.

        Called once the variant's result has been appended, so the note carries
        what actually landed rather than what was about to run.

        Args:
            idx (int): Zero-based index of the just-finished variant.
        """
        await report_progress(**_variant_progress_note(grid, results, idx))

    def _round_timeout_sec(idx: int, name: str, *, round_label: str, reserve_sec: float = 0.0) -> int:
        """``variant_timeout_sec`` capped at what the session can still pay for.

        Recomputed per round rather than once per variant: a warmup round spends
        budget the measure round would otherwise still count on.

        ``reserve_sec`` holds back what the variant's remaining rounds still
        need, so a slow warmup cannot consume the budget belonging to the
        measured round. The measured round is the only one that yields a usable
        data point, so it is the round the budget is kept for; a discarded
        warmup that overruns is the cheaper thing to cut short.

        The clamp itself is :func:`session_clamped_timeout_sec`; this adds the
        grid's own log line.

        Args:
            idx (int): Zero-based variant index, for the log line.
            name (str): Variant name, for the log line.
            round_label (str): Which round is being capped (warmup / measure).
            reserve_sec (float): Seconds held back for this variant's later
                rounds. Zero for the last round.

        Returns:
            int: The hard timeout to grant this round, in seconds.
        """
        declared = int(variant_timeout_sec)
        cap = agentx_variant_timeout_sec(declared)
        if cap != declared:
            log.info(
                "grid_runner: variant %d/%d name=%s %s cap raised %ds -> %ds "
                "(AgentX: AGENTX_DURATION + overhead; the synthetic default "
                "cannot cover a canonical agentic warmup)",
                idx + 1,
                len(grid),
                name,
                round_label,
                declared,
                cap,
            )
        clamped = session_clamped_timeout_sec(cap, session_deadline_sec, reserve_sec=reserve_sec)
        if clamped == cap:
            return cap
        log.info(
            "grid_runner: variant %d/%d name=%s %s cap clamped %ds -> %ds by the session budget",
            idx + 1,
            len(grid),
            name,
            round_label,
            cap,
            clamped,
        )
        return clamped

    def _record_round_stop(
        stopped: StoppedByTheRun,
        *,
        idx: int,
        variant: GridVariant,
        slot: Path,
        round_label: str,
        returncode: int | None,
        started_unix: float,
        server_log: Path,
    ) -> bool:
        """Record a round the run stopped and say whether the grid is over.

        Every round a variant runs is a full benchmark pass -- the discarded
        warmup and the multi-node client warmup as much as the measured one -- so
        any of them can be reaped by the session deadline or by an orchestrator
        cancel, and none of them says anything about the variant when it is.
        Hence one place decides what such a round means, called after each launch
        and before any grading: the row is ``skipped``, exactly like a variant the
        budget refused to start, because nothing was measured and there is no
        verdict to record. Grading it as a failure -- or worse as
        ``killed_overtime``, which asserts the variant is abnormally slow -- puts
        a conclusion the run never reached into the ledger and the KB.

        A cause that ``ends_the_batch`` also fills in every variant after this
        one: nothing new may start under it, and their rows have to say why they
        were never tested rather than be absent.

        Args:
            stopped (StoppedByTheRun): How to record the cause.
            idx (int): Zero-based index of the variant whose round was stopped.
            variant (GridVariant): That variant.
            slot (Path): Its slot directory, where the abort marker is written.
            round_label (str): Which round was stopped, for the log line.
            returncode (int | None): The round's returncode, kept on the row.
            started_unix (float): ``time.time()`` when the round was launched.
            server_log (Path): The stopped round's ``server.log``, if it wrote one.

        Returns:
            bool: ``True`` when the caller must stop testing variants.
        """
        runtime_sec = round(max(0.0, time.time() - started_unix), 2)
        log.warning(
            "grid_runner: variant %d/%d name=%s %s round reaped after %.1fs: %s; recorded as skipped, not failed",
            idx + 1,
            len(grid),
            variant.name,
            round_label,
            runtime_sec,
            stopped.interrupted,
        )
        _write_variant_abort_marker(
            slot,
            variant_name=variant.name,
            error_class=stopped.error_class,
            error_summary=f"{stopped.interrupted}; tree reaped",
            extra_args=variant.extra_server_args,
        )
        results.append(
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="skipped",
                returncode=returncode,
                runtime_sec=runtime_sec,
                error=stopped.interrupted,
                error_class=stopped.error_class,
                server_log_path=_measurement_server_log_path(server_log, slot=slot),
                note=variant.note,
            )
        )
        if stopped.ends_the_batch:
            results.extend(_not_run_skip_result(rest, stopped) for rest in grid[idx + 1 :])
            return True
        return not keep_going_on_failure

    # How many full benchmark passes one variant costs. Both warmups run the same
    # workload as the measured pass -- neither is a reduced one -- so a variant
    # that warms up costs twice what its measured round does. Admitting on a
    # single round's estimate would systematically let in variants that then get
    # their measured round clamped to nothing, turning a budget shortfall into a
    # ledger full of spurious timeouts. Both flags are run-level, so this is known
    # before the loop; the per-variant ``auto_warmup`` is only ever narrower.
    _mn_warmup_rounds = 0
    if variant_expected_sec is not None:
        from ._multi_node_env import (
            is_multi_node as _mn_is_multi_node,
            mn_bench_warmup_enabled as _mn_bench_warmup_enabled,
        )

        _mn_warmup_rounds = 1 if (_mn_is_multi_node() and _mn_bench_warmup_enabled()) else 0
    variant_rounds = 1 + (1 if auto_warmup_requested else 0) + _mn_warmup_rounds

    def _skip_rest_for_budget(idx: int, *, spent_on: str, rounds_left: int = variant_rounds) -> bool:
        """Skip variant ``idx`` and every one after it, or say the budget still fits.

        Checked twice per variant, because the two things that spend the budget
        between the check and the launch are not under any cap: the per-variant
        multi-node server restart, and the variant before this one overrunning.
        Once a variant does not fit, none after it does either -- the clock only
        shrinks -- so this ends the batch rather than trying the next name.

        ``rounds_left`` is what makes the second check a different question from
        the first rather than the same one asked with less clock. A pass this
        variant has already run is paid for, and charging for it again refuses a
        variant that fits and throws away the GPU time already spent on it.

        Args:
            idx: Zero-based index of the variant about to run.
            spent_on: What has been spent since the last check, for the log line.
            rounds_left: Passes of this variant still to launch. Defaults to all
                of them, which is right for the check before any has run.

        Returns:
            bool: ``True`` when the batch is over and nothing more was launched.
        """
        if session_deadline_sec is None:
            return False
        remaining_sec = session_deadline_sec - time.monotonic()
        # Falls back to a single ``variant_timeout_sec`` when no estimate was
        # given, which is what callers that cannot estimate already got. Must
        # be the AgentX-raised cap, not the declared one: the declared cap
        # understates what the round will actually be granted, so this check
        # would admit a variant it cannot fit, which then gets its timeout
        # clamped down to the (too-small) remaining budget by
        # ``session_clamped_timeout_sec`` -- reproducing the mid-warmup kill
        # this module's AgentX cap-raise exists to prevent.
        # Under AgentX the estimate branch has to clear the raised cap too, not
        # just the no-estimate one. An estimate is only as good as the shape it
        # was made on, and every estimate in this tree is sized for the
        # synthetic 1024/1024 round -- the very thing the cap-raise exists to
        # correct. Admitting on a synthetic estimate (say 1500s against 2000s
        # remaining) grants the raised cap, then lets
        # ``session_clamped_timeout_sec`` squeeze it back to the 2000s that are
        # actually left, and the round dies mid-warmup anyway. Taking the max
        # only ever skips EARLIER: losing a variant costs one data point, while
        # admitting one that gets killed costs the same data point plus the GPU
        # time spent on it.
        #
        # Gated on the cap having actually been RAISED, which only happens with
        # AgentX on -- ``agentx_variant_timeout_sec`` returns its argument
        # untouched otherwise. Without that gate the default path would start
        # charging every variant its full backstop instead of its expected cost,
        # skipping variants that fit comfortably. The estimate stays the
        # admission price on the default path, exactly as before.
        _raised = float(agentx_variant_timeout_sec(variant_timeout_sec))
        _cap_was_raised = _raised > float(variant_timeout_sec)
        if variant_expected_sec is None:
            required_sec = _raised
        else:
            estimated_sec = float(variant_expected_sec) * rounds_left
            required_sec = max(estimated_sec, _raised) if _cap_was_raised else estimated_sec
        if remaining_sec >= required_sec:
            return False
        log.warning(
            "grid_runner: %.0fs left cannot fit this variant's %d remaining round(s) "
            "of %.0fs (spent on: %s); skipping %d remaining variant(s) rather than "
            "launching a pass whose measured round cannot follow it",
            max(0.0, remaining_sec),
            rounds_left,
            required_sec,
            spent_on,
            len(grid) - idx,
        )
        for skipped_variant in grid[idx:]:
            results.append(
                _not_run_skip_result(
                    skipped_variant,
                    _STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_RETURNCODE],
                )
            )
        return True

    for i, variant in enumerate(grid):
        # Session-budget stop: skip the remaining variants once the wall-clock
        # deadline is reached or the remaining budget cannot fit another variant,
        # so a timeout halts the grid instead of draining it (and the last variant
        # cannot overrun the close window).
        if _skip_rest_for_budget(i, spent_on="the variants before this one"):
            break
        await _unit_started(i, "variant")
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
        server_log = slot / "server.log"
        # Capability fast-fail: drop a variant whose env flag the build cannot
        # honour before booting a doomed server, still recording the failure so
        # the LLM learns not to re-pick it.
        cap_reason = unsupported_capability_reason(variant)
        if cap_reason:
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: capability_unsupported: %s",
                i + 1,
                len(grid),
                variant.name,
                cap_reason,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="capability_unsupported",
                error_summary=cap_reason,
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    error=cap_reason,
                    error_class="capability_unsupported",
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue
        lifecycle: dict[str, Any] = {"eligible": False}
        auto_warmup = False
        try:
            cfg_path = _build_variant_yaml(
                base_yaml_path,
                base_extra_args,
                variant,
                output_subdir=slot,
                model_path=model_path,
                gpu_type=gpu_type,
                benchmark_script=benchmark_script,
                server_lifecycle=server_lifecycle,
                base_args_mode=base_args_mode,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: yaml_build_error: %r",
                i + 1,
                len(grid),
                variant.name,
                exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="yaml_build_error",
                error_summary=repr(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    error=f"yaml_build_error: {exc!r}",
                    error_class="yaml_build_error",
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        _mn_effective_args = _resolve_mn_effective_server_args(
            cfg_path,
            base_yaml_path,
            variant,
            base_extra_args=base_extra_args,
            base_args_mode=base_args_mode,
        )

        if auto_warmup_requested:
            try:
                from ._server_lifecycle import resolve_lifecycle_params

                lifecycle = resolve_lifecycle_params(cfg_path)
                auto_warmup = bool(lifecycle.get("eligible"))
                if auto_warmup:
                    cfg_path = _build_variant_yaml(
                        base_yaml_path,
                        base_extra_args,
                        variant,
                        output_subdir=slot,
                        model_path=model_path,
                        gpu_type=gpu_type,
                        benchmark_script=benchmark_script,
                        server_lifecycle={
                            "cleanup": True,
                            "pid_dir": str(slot),
                            "port": int(lifecycle.get("port") or 0),
                        },
                        base_args_mode=base_args_mode,
                    )
                else:
                    log.info(
                        "grid_runner: warmup-before-measure not eligible (%s); running single measured round.",
                        lifecycle.get("reason") or "unknown",
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "grid_runner: warmup-before-measure eligibility/materialization failed (%r); running single measured round.",
                    exc,
                )
                auto_warmup = False

        warmup_tput: float | None = None
        if auto_warmup:
            warmup_slot = slot / "warmup_round"
            warmup_server_log = warmup_slot / "server.log"
            warmup_lifecycle = {
                "cleanup": False,
                "pid_dir": str(slot),
                "port": int(lifecycle.get("port") or 0),
            }
            try:
                warmup_cfg_path = _build_variant_yaml(
                    base_yaml_path,
                    base_extra_args,
                    variant,
                    output_subdir=warmup_slot,
                    model_path=model_path,
                    gpu_type=gpu_type,
                    benchmark_script=benchmark_script,
                    server_lifecycle=warmup_lifecycle,
                    base_args_mode=base_args_mode,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "grid_runner: variant %d/%d name=%s aborted: warmup_yaml_build_error: %r",
                    i + 1,
                    len(grid),
                    variant.name,
                    exc,
                )
                _write_variant_abort_marker(
                    slot,
                    variant_name=variant.name,
                    error_class="warmup_yaml_build_error",
                    error_summary=repr(exc),
                    extra_args=variant.extra_server_args,
                )
                results.append(
                    VariantResult(
                        name=variant.name,
                        extra_server_args=variant.extra_server_args,
                        extra_envs=dict(variant.extra_envs),
                        status="failed",
                        error=f"warmup_yaml_build_error: {exc!r}",
                        error_class="warmup_yaml_build_error",
                        note=variant.note,
                    )
                )
                await _report_finished_variant(i)
                if not keep_going_on_failure:
                    break
                continue

            warmup_workspaces_before = snapshot_workspaces(warmup_slot)
            warmup_started_unix = time.time()
            # Held in a local because the abort line below has to name the cap the
            # round was actually granted: the declared one is a hang backstop, and
            # a round killed at the reserved cap logged as a two-hour timeout reads
            # as a variant that hangs rather than a budget that ran out. The
            # admission gate above is what keeps this reserve from starving the
            # round -- it refuses a variant whose passes do not fit before any of
            # them is reserved for.
            warmup_cap_sec = _round_timeout_sec(
                i,
                variant.name,
                round_label="warmup",
                reserve_sec=float(variant_expected_sec or 0.0) * (1 + _mn_warmup_rounds),
            )
            try:
                warmup_rc, warmup_stdout, warmup_stderr = await _reported_magpie(
                    i,
                    "warmup",
                    magpie_python=magpie_python,
                    config_path=warmup_cfg_path,
                    output_dir=warmup_slot,
                    timeout_sec=warmup_cap_sec,
                    cwd=cwd,
                    result_dir=result_dir,
                    soft_deadline_sec=None,
                    preclean=True,
                    serving_lease=serving_lease,
                    session_deadline_sec=session_deadline_sec,
                )
            except subprocess.TimeoutExpired as exc:
                _teardown_variant_server(slot, lifecycle)
                log.warning(
                    "grid_runner: variant %d/%d name=%s aborted: warmup timeout (timeout_sec=%d): %s",
                    i + 1,
                    len(grid),
                    variant.name,
                    warmup_cap_sec,
                    exc,
                )
                _write_variant_abort_marker(
                    slot,
                    variant_name=variant.name,
                    error_class="warmup_magpie_timeout",
                    error_summary=str(exc),
                    extra_args=variant.extra_server_args,
                )
                results.append(
                    VariantResult(
                        name=variant.name,
                        extra_server_args=variant.extra_server_args,
                        extra_envs=dict(variant.extra_envs),
                        status="failed",
                        error=f"warmup_timeout: {exc}",
                        error_class="warmup_magpie_timeout",
                        server_log_path=_existing_log_path(warmup_server_log),
                        note=variant.note,
                        runtime_sec=round(max(0.0, time.time() - warmup_started_unix), 2),
                        nonfatal_warnings=["run_grid_warmup_round_failed"],
                    )
                )
                await _report_finished_variant(i)
                if not keep_going_on_failure:
                    break
                continue

            warmup_stopped = stopped_by_the_run(warmup_rc)
            if warmup_stopped is not None:
                _teardown_variant_server(slot, lifecycle)
                grid_is_over = _record_round_stop(
                    warmup_stopped,
                    idx=i,
                    variant=variant,
                    slot=slot,
                    round_label="warmup",
                    returncode=warmup_rc,
                    started_unix=warmup_started_unix,
                    server_log=warmup_server_log,
                )
                await _report_finished_variant(i)
                if grid_is_over:
                    break
                continue

            warmup_run_ws = select_run_workspace(warmup_slot, known_before=warmup_workspaces_before)
            warmup_workspace = warmup_run_ws if warmup_run_ws is not None else warmup_slot
            warmup_harvested = harvest_leaked_artifacts(
                warmup_workspace,
                subprocess_started_unix=warmup_started_unix,
            )
            if warmup_run_ws is not None:
                _, warmup_measurement = await _settled_measurement(
                    warmup_workspace,
                    subprocess_started_unix=warmup_started_unix,
                    settle_seconds=REPORT_SETTLE_SECONDS if warmup_rc == 0 else 0.0,
                )
            else:
                warmup_measurement = extract_benchmark_measurement(
                    None,
                    workspace=warmup_workspace,
                    subprocess_started_unix=warmup_started_unix,
                )
            if warmup_rc != 0 or not warmup_measurement.get("valid_measurement"):
                _teardown_variant_server(slot, lifecycle)
                warmup_error = (
                    server_log_death_excerpt(str(warmup_server_log))
                    or redact_secret_values((warmup_stderr or warmup_stdout)[-2000:])
                    if warmup_rc != 0
                    else "warmup benchmark_report missing valid throughput/completed requests"
                )
                log.warning(
                    "grid_runner: variant %d/%d name=%s aborted: warmup_round_failed (rc=%s): %s",
                    i + 1,
                    len(grid),
                    variant.name,
                    warmup_rc,
                    warmup_error[:200],
                )
                _write_variant_abort_marker(
                    slot,
                    variant_name=variant.name,
                    error_class="warmup_round_failed",
                    error_summary=warmup_error,
                    extra_args=variant.extra_server_args,
                )
                results.append(
                    VariantResult(
                        name=variant.name,
                        extra_server_args=variant.extra_server_args,
                        extra_envs=dict(variant.extra_envs),
                        status="failed",
                        workspace=str(warmup_workspace) if warmup_run_ws is not None else None,
                        report_path=(
                            str(warmup_workspace / "benchmark_report.json")
                            if (warmup_workspace / "benchmark_report.json").exists()
                            else None
                        ),
                        raw_result_path=warmup_measurement.get("raw_result_path"),
                        reported_success=warmup_measurement.get("reported_success"),
                        returncode=warmup_rc,
                        error=warmup_error,
                        error_class="warmup_round_failed",
                        server_log_path=_existing_log_path(warmup_server_log),
                        note=variant.note,
                        runtime_sec=round(max(0.0, time.time() - warmup_started_unix), 2),
                        nonfatal_warnings=[
                            "run_grid_warmup_round_failed",
                            *[f"harvested_leaked_artifact:{src}" for src, _ in warmup_harvested],
                        ],
                    )
                )
                await _report_finished_variant(i)
                if not keep_going_on_failure:
                    break
                continue
            warmup_tput = warmup_measurement.get("output_throughput")
            log.info(
                "grid_runner: variant %s warmup tput=%.1f tok/s discarded; measuring hot round next",
                variant.name,
                warmup_tput or 0.0,
            )

        from ._multi_node_env import log_mn_banner

        log_mn_banner(
            "grid_runner",
            log,
            variant=f"{i + 1}/{len(grid)}:{variant.name}",
        )
        log.info(
            "grid_runner: variant %d/%d name=%s args=%s",
            i + 1,
            len(grid),
            variant.name,
            variant.extra_server_args,
        )

        # Multi-node only: restart sglang/vllm with this variant's flags so each
        # row runs against a fresh server. No-op in single-node mode.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )

        try:
            # PD knobs auto-resolved from $PD_* env; PD config stays constant
            # across variants within one run.
            _variant_unset = [str(k) for k in getattr(variant, "unset_envs", []) or [] if str(k).strip()]
            _restart_env = _mn_restart_env(_mn_ref_envs, variant, _variant_unset)
            await restart_server_for_round(
                extra_server_args=_mn_effective_args,
                # Reference recipe envs under this variant's own overrides (e.g.
                # MORI_* MoE-dispatch tuning) so server-side env knobs proposed
                # by specialists take effect on the restarted sglang without
                # dropping the envs the baseline was measured with.
                extra_env=_restart_env,
                unset_env=_variant_unset,
                model_path=model_path,
                ep=int(os.environ.get("EP") or 0) or None,
            )
        except ServerRestartFailed as exc:
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: mn_server_restart_failed: %s",
                i + 1,
                len(grid),
                variant.name,
                exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="mn_server_restart_failed",
                error_summary=str(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    error=f"mn_server_restart_failed: {exc}",
                    error_class="mn_server_restart_failed",
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # The restart above is a launch of its own and is under no cap, so the
        # budget admitted at the top of the loop may no longer be there. Booting
        # a large model can take longer than a benchmark pass. Only the passes
        # still ahead are charged for: the auto warmup above this point has
        # already run, and re-charging it here would end the batch over time
        # that is spent either way.
        if _skip_rest_for_budget(
            i,
            spent_on="this variant's server restart",
            rounds_left=1 + _mn_warmup_rounds,
        ):
            if auto_warmup:
                _teardown_variant_server(slot, lifecycle)
            break

        # Multi-node client warmup: one discarded benchmark pass against the
        # just-restarted, persistent remote server to warm JIT / steady-state
        # before the measured pass. No lifecycle / no restart between; best-
        # effort (a warmup failure never fails the variant). Default ON
        # (INFERENCE_OPTIMIZER_MN_BENCH_WARMUP=0 disables).
        from ._multi_node_env import (
            is_multi_node as _mn_imn,
            mn_bench_warmup_enabled as _mn_warm,
        )

        if _mn_imn() and _mn_warm():
            _mn_warm_slot = slot / "mn_warmup"
            _mn_warm_started_unix = time.time()
            # The measurement is discarded, but the returncode is not: a warmup
            # the run stopped is the same stop as one in the measured round, and
            # discarding it launches the measured round after the cancel.
            _mn_warm_rc: int | None = None
            try:
                _mn_warm_rc, _, _ = await _reported_magpie(
                    i,
                    "mn_warmup",
                    magpie_python=magpie_python,
                    config_path=cfg_path,
                    output_dir=_mn_warm_slot,
                    timeout_sec=_round_timeout_sec(
                        i,
                        variant.name,
                        round_label="mn_warmup",
                        reserve_sec=float(variant_expected_sec or 0.0),
                    ),
                    cwd=cwd,
                    result_dir=None,
                    soft_deadline_sec=None,
                    preclean=False,
                    serving_lease=serving_lease,
                    session_deadline_sec=session_deadline_sec,
                )
                log.info(
                    "grid_runner: MN warmup pass done (discarded) %d/%d name=%s rc=%s",
                    i + 1,
                    len(grid),
                    variant.name,
                    _mn_warm_rc,
                )
            except Exception as exc:  # noqa: BLE001 - warmup is best-effort
                log.warning(
                    "grid_runner: MN warmup pass failed (ignored) name=%s: %r",
                    variant.name,
                    exc,
                )
            _mn_warm_stopped = stopped_by_the_run(_mn_warm_rc)
            if _mn_warm_stopped is not None:
                grid_is_over = _record_round_stop(
                    _mn_warm_stopped,
                    idx=i,
                    variant=variant,
                    slot=slot,
                    round_label="mn_warmup",
                    returncode=_mn_warm_rc,
                    started_unix=_mn_warm_started_unix,
                    server_log=_mn_warm_slot / "server.log",
                )
                await _report_finished_variant(i)
                if grid_is_over:
                    break
                continue

        # Snapshot wall-clock before launch so the salvage path can mtime-gate
        # leak destinations per-variant.
        slot_workspaces_before = snapshot_workspaces(slot)
        variant_started_unix = time.time()
        measure_cap_sec = _round_timeout_sec(i, variant.name, round_label="measure")
        try:
            rc, stdout, stderr = await _reported_magpie(
                i,
                "benchmark",
                magpie_python=magpie_python,
                config_path=cfg_path,
                output_dir=slot,
                timeout_sec=measure_cap_sec,
                cwd=cwd,
                result_dir=result_dir,
                soft_deadline_sec=soft_deadline_sec,
                preclean=(False if auto_warmup else preclean_before_run),
                server_already_ready=(server_already_ready or auto_warmup),
                serving_lease=serving_lease,
                session_deadline_sec=session_deadline_sec,
            )
        except subprocess.TimeoutExpired as exc:
            # Harvest pre-timeout leaks.
            to_destination = select_run_workspace(slot, known_before=slot_workspaces_before) or slot
            to_harvested = harvest_leaked_artifacts(
                to_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: magpie timeout (timeout_sec=%d): %s",
                i + 1,
                len(grid),
                variant.name,
                measure_cap_sec,
                exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="magpie_timeout",
                error_summary=str(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    error=f"timeout: {exc}",
                    error_class="magpie_timeout",
                    server_log_path=_existing_log_path(server_log),
                    note=variant.note,
                    runtime_sec=round(
                        max(0.0, time.time() - variant_started_unix),
                        2,
                    ),
                    nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in to_harvested],
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue
        finally:
            if auto_warmup:
                _teardown_variant_server(slot, lifecycle)

        # Eval bounds could not be installed for a variant that runs eval, so
        # nothing launched. Labelled rather than left to the generic path, which
        # would call it ``no_benchmark_workspace`` and hide an eval-contract gap
        # behind a missing-directory message. ``keep_going_on_failure`` is
        # honoured: the install is flock-serialized, so a transient loss can
        # succeed on the next variant.
        if rc == EVAL_PROBE_UNPATCHABLE_RETURNCODE:
            variant_runtime_sec = round(max(0.0, time.time() - variant_started_unix), 2)
            log.error(
                "grid_runner: variant %d/%d name=%s aborted: eval_probe_unpatchable: %s",
                i + 1,
                len(grid),
                variant.name,
                stderr,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="eval_probe_unpatchable",
                error_summary=stderr,
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    returncode=rc,
                    runtime_sec=variant_runtime_sec,
                    error=stderr,
                    error_class="eval_probe_unpatchable",
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Server-liveness watchdog fired: engine/worker bootstrap died but the
        # parent hung. Record a fast failure so the round proceeds; harvest the
        # crash server.log.
        if rc == SERVER_DEAD_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix),
                2,
            )
            sd_destination = select_run_workspace(slot, known_before=slot_workspaces_before) or slot
            sd_harvested = harvest_leaked_artifacts(
                sd_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "server_init_dead (engine/worker bootstrap failed; parent "
                "hung) after %.1fs",
                i + 1,
                len(grid),
                variant.name,
                variant_runtime_sec,
            )
            death_excerpt = server_log_death_excerpt(str(server_log)) or (
                "server engine/worker init failed; parent process hung and was reaped by the liveness watchdog"
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="server_init_dead",
                error_summary=death_excerpt,
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    returncode=rc,
                    runtime_sec=variant_runtime_sec,
                    error=death_excerpt,
                    error_class="server_init_dead",
                    server_log_path=_existing_log_path(server_log),
                    note=variant.note,
                    nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in sd_harvested],
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Detokenizer-stall watchdog fired: the server came up healthy but then
        # went silent (hung engine / wedged detokenizer). Fast-prune with a
        # distinct ``error_class`` and harvest leaks for RCA.
        if rc == DETOKENIZER_STALL_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix),
                2,
            )
            ds_destination = select_run_workspace(slot, known_before=slot_workspaces_before) or slot
            ds_harvested = harvest_leaked_artifacts(
                ds_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: detokenizer_stall "
                "(server ready but log went silent) after %.1fs",
                i + 1,
                len(grid),
                variant.name,
                variant_runtime_sec,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="detokenizer_stall",
                error_summary=(
                    "server reported ready but emitted no log output (hung "
                    "engine / detokenizer stall); reaped by the "
                    "detokenizer-stall watchdog"
                ),
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    returncode=rc,
                    runtime_sec=variant_runtime_sec,
                    error="detokenizer_stall: server ready but log went silent",
                    error_class="detokenizer_stall",
                    server_log_path=_existing_log_path(server_log),
                    note=variant.note,
                    nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in ds_harvested],
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        stopped = stopped_by_the_run(rc)
        if stopped is not None:
            grid_is_over = _record_round_stop(
                stopped,
                idx=i,
                variant=variant,
                slot=slot,
                round_label="measured",
                returncode=rc,
                started_unix=variant_started_unix,
                server_log=server_log,
            )
            await _report_finished_variant(i)
            if grid_is_over:
                break
            continue

        # Soft overtime gate fired: record a ``killed_overtime=True`` result with
        # no tput and still harvest leaks for post-mortem.
        if rc == OVERTIME_KILL_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix),
                2,
            )
            ok_destination = select_run_workspace(slot, known_before=slot_workspaces_before) or slot
            ok_harvested = harvest_leaked_artifacts(
                ok_destination,
                subprocess_started_unix=variant_started_unix,
            )
            # Best-effort rough tput from server.log throughput logs;
            # informational only — the variant stays failed and never feeds
            # winner selection.
            ok_warnings = [f"harvested_leaked_artifact:{src}" for src, _ in ok_harvested]
            ok_estimate = estimate_killed_variant_throughput(slot)
            estimated_tput = ok_estimate.get("output_throughput") if ok_estimate else None
            if estimated_tput is not None:
                ok_warnings.append(
                    "estimated_output_throughput_from_server_log:"
                    f"{estimated_tput:.1f}tok/s"
                    f"(n={ok_estimate.get('num_samples')})"
                )
            overtime_error = (
                f"killed_overtime: wall-clock {variant_runtime_sec:.1f}s "
                f"exceeded soft_deadline_sec={float(soft_deadline_sec or 0.0):.1f}s"
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="killed_overtime",
                error_summary=overtime_error,
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    returncode=rc,
                    killed_overtime=True,
                    runtime_sec=variant_runtime_sec,
                    estimated_output_throughput=estimated_tput,
                    error=overtime_error,
                    error_class="killed_overtime",
                    server_log_path=_existing_log_path(server_log),
                    note=variant.note,
                    nonfatal_warnings=ok_warnings,
                )
            )
            log.info(
                "_grid_runner: variant %s killed_overtime (runtime=%.1fs deadline=%.1fs est_output_tput=%s tok/s)",
                variant.name,
                variant_runtime_sec,
                float(soft_deadline_sec or 0.0),
                f"{estimated_tput:.1f}" if estimated_tput is not None else "n/a",
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        workspace = select_run_workspace(slot, known_before=slot_workspaces_before)
        # Always-on artifact harvest so each slot keeps its server.log /
        # gpu_metrics / profile relay for Robustness RCA.
        harvest_destination = workspace if workspace is not None else slot
        harvested = harvest_leaked_artifacts(
            harvest_destination,
            subprocess_started_unix=variant_started_unix,
        )
        if harvested:
            log.info(
                "_grid_runner: variant=%s harvested %d leaked artifact(s): %s",
                variant.name,
                len(harvested),
                ", ".join(src.name for src, _ in harvested),
            )
        if workspace is None:
            harvest_tags = [f"harvested_leaked_artifact:{src}" for src, _ in harvested]
            no_ws_error_summary = server_log_death_excerpt(str(server_log)) or (
                redact_secret_values((stderr or stdout)[-2000:]) if rc != 0 else "no benchmark_* workspace produced"
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: no_benchmark_workspace (rc=%s)",
                i + 1,
                len(grid),
                variant.name,
                rc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="no_benchmark_workspace",
                error_summary=no_ws_error_summary,
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    returncode=rc,
                    error=no_ws_error_summary,
                    error_class="no_benchmark_workspace",
                    server_log_path=_existing_log_path(server_log),
                    nonfatal_warnings=harvest_tags,
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue
        report_path = workspace / "benchmark_report.json"
        # A clean exit is worth waiting on: the report is written during shutdown
        # and the reader runs the moment the subprocess is reaped.
        report, measurement = await _settled_measurement(
            workspace,
            subprocess_started_unix=variant_started_unix,
            settle_seconds=REPORT_SETTLE_SECONDS if rc == 0 else 0.0,
        )
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")
        if warmup_tput is not None:
            warnings.append("run_grid_warmup_discarded_first")
            warnings.append(f"warmup_round_tput:{float(warmup_tput):.1f}")

        if not measurement.get("valid_measurement"):
            death_excerpt = server_log_death_excerpt(str(server_log))
            if rc != 0:
                error = death_excerpt or redact_secret_values((stderr or stdout)[-2000:])
                # The bypass/scriptable path runs the customer body in a child
                # whose stderr is redirected to benchmark_stderr.log, not the
                # parent pipe — so `stderr`/`stdout` here are often empty and the
                # real diagnostic (e.g. argparse "unrecognized arguments") is
                # only on disk. Fall back to it so abort_reason.json carries the
                # actual failure instead of a blank `error`.
                if not error.strip():
                    error = redact_secret_values(_on_disk_stderr_tail(workspace, slot))
                # Last resort: report.errors when the pipe and on-disk logs are
                # all empty. Scriptable pre-spawn miss normally hits the log
                # fallback (run_scriptable writes scriptable_stderr.log, then
                # write_log_aliases copies it to benchmark_stderr.log). This
                # rung covers log-write OSError and other backends that named
                # the failure only in the report.
                if not error.strip():
                    error = redact_secret_values(_report_errors_summary(report))
                invalid_class = "magpie_nonzero_invalid_measurement"
            elif not report:
                error = death_excerpt or "benchmark_report missing"
                invalid_class = "benchmark_report_missing"
            else:
                error = death_excerpt or "benchmark_report missing valid throughput/completed requests"
                invalid_class = "benchmark_report_invalid_metric"
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: %s (rc=%s): %s",
                i + 1,
                len(grid),
                variant.name,
                invalid_class,
                rc,
                error[:200],
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class=invalid_class,
                error_summary=error,
                extra_args=variant.extra_server_args,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    workspace=str(workspace),
                    report_path=str(report_path) if report_path.exists() else None,
                    raw_result_path=measurement.get("raw_result_path"),
                    reported_success=measurement.get("reported_success"),
                    returncode=rc,
                    nonfatal_warnings=warnings,
                    error=error,
                    error_class=invalid_class,
                    server_log_path=_existing_log_path(server_log),
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue

        if rc != 0:
            nonzero_error = redact_secret_values((stderr or stdout)[-2000:])
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="magpie_nonzero_after_valid_measurement",
                error_summary=nonzero_error,
                extra_args=variant.extra_server_args,
            )
            log.warning(
                "grid_runner: variant %s aborted: magpie_nonzero_after_valid_measurement (rc=%d)",
                variant.name,
                rc,
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    workspace=str(workspace),
                    report_path=str(report_path) if report_path.exists() else None,
                    raw_result_path=measurement.get("raw_result_path"),
                    reported_success=measurement.get("reported_success"),
                    returncode=rc,
                    nonfatal_warnings=warnings,
                    error=nonzero_error,
                    error_class="magpie_nonzero_after_valid_measurement",
                    server_log_path=None,
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if not keep_going_on_failure:
                break
            continue

        results.append(
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="succeeded",
                output_throughput=measurement.get("output_throughput"),
                request_throughput=measurement.get("request_throughput"),
                total_token_throughput=measurement.get("total_token_throughput"),
                completed_requests=measurement.get("completed_requests"),
                duration_seconds=measurement.get("duration_seconds"),
                ttft_mean_ms=measurement.get("ttft_mean_ms"),
                e2el_mean_ms=measurement.get("e2el_mean_ms"),
                tpot_mean_ms=measurement.get("tpot_mean_ms"),
                input_throughput=measurement.get("input_throughput"),
                total_throughput=measurement.get("total_token_throughput"),
                tpot_p90_ms=measurement.get("tpot_p90_ms"),
                intvty_p90=measurement.get("intvty_p90"),
                workspace=str(workspace),
                report_path=str(report_path) if report_path.exists() else None,
                raw_result_path=measurement.get("raw_result_path"),
                reported_success=measurement.get("reported_success"),
                returncode=rc,
                nonfatal_warnings=warnings,
                server_log_path=None,
                note=variant.note,
                runtime_sec=round(
                    max(0.0, time.time() - variant_started_unix),
                    2,
                ),
            )
        )
        log.info(
            "grid_runner: variant %s tput=%.1f tok/s",
            variant.name,
            results[-1].output_throughput or 0.0,
        )
        await _report_finished_variant(i)
    _attach_grid_launch_evidence(
        results,
        grid=grid,
        output_root=output_root,
        caller_reused_ready_server=server_already_ready,
    )
    return results


def _teardown_variant_server(slot: Path, lifecycle: dict[str, Any]) -> None:
    """Stop the server this variant booted for its own rounds.

    Every path out of a variant that booted one owes this call, including the
    ones that leave over the budget rather than over a result: the process
    outlives the loop iteration that started it, and the next variant boots its
    own. Kept in one function so a new exit path is one line rather than a
    four-line block someone can leave out.

    Args:
        slot: The variant's workspace, which is also its pid directory.
        lifecycle: The variant's resolved lifecycle, carrying framework and port.
    """
    from ._server_lifecycle import teardown_lifecycle_server

    teardown_lifecycle_server(
        pid_dir=slot,
        framework=str(lifecycle.get("framework") or ""),
        port=int(lifecycle.get("port") or 0),
    )


def _not_run_skip_result(variant: GridVariant, stopped: StoppedByTheRun) -> VariantResult:
    """Build the ``skipped`` result for a variant the run never got to.

    Args:
        variant: The variant that was dropped.
        stopped: Why the run stopped, which is the whole content of the result:
            nothing was measured, so there is nothing else to report.

    Returns:
        VariantResult: A synthetic ``skipped`` result carrying that cause.
    """
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        error=stopped.never_started,
        error_class=stopped.error_class,
        note=variant.note,
    )


def _existing_log_path(path: Path) -> str | None:
    """Return ``path`` as a string when it exists, else ``None``.

    Args:
        path (Path): Candidate ``server.log`` path.

    Returns:
        str | None: The stringified path, or ``None`` when absent.
    """
    return str(path) if path.exists() else None


def _measurement_server_log_path(
    server_log: Path,
    workspace: Path | None = None,
    *,
    slot: Path | None = None,
) -> str | None:
    """Return the server log attributable to this variant's measurement.

    This is deliberately shared by success and failure paths. A ready-server
    reuse leaves the measured workspace without a local log, so the only
    fallback is this variant's own ``warmup_round`` subtree. It never searches
    older session history, which could attach a different launch's argv.
    """
    owning_slot = slot or server_log.parent
    direct = _existing_log_path(server_log)
    if not direct and workspace is not None:
        direct = _existing_log_path(workspace / "server.log")
    if direct:
        return direct
    warmup_dir = owning_slot / "warmup_round"
    try:
        candidates = [path for path in warmup_dir.glob("*/server.log") if path.is_file()]
    except OSError:
        candidates = []
    if not candidates:
        return None
    try:
        return str(max(candidates, key=lambda path: path.stat().st_mtime))
    except OSError:
        return None


def _attach_grid_launch_evidence(
    results: list[VariantResult],
    *,
    grid: list[GridVariant],
    output_root: Path,
    caller_reused_ready_server: bool,
) -> None:
    """Persist declared and observed launch evidence for each grid result.

    Results stay backward-compatible: the new fields are additive, and a
    skipped pre-materialization result produces no evidence. Evidence lives in
    the variant slot, never in an external rescue directory.
    """
    for idx, result in enumerate(results):
        if idx >= len(grid):
            break
        # A skipped variant never launched a measurement. Do not materialize a
        # synthetic evidence directory whose warm-reuse metadata could later be
        # mistaken for an observed launch.
        if result.status == "skipped" or result.error_class in {
            "capability_unsupported",
            "yaml_build_error",
            "warmup_yaml_build_error",
        }:
            continue
        slot = output_root / f"variant_{idx:02d}_{_safe(grid[idx].name)}"
        config_path = slot / "config.yaml"
        workspace = Path(result.workspace) if result.workspace else None
        primary_log = Path(result.server_log_path) if result.server_log_path else slot / "server.log"
        actual_log = _measurement_server_log_path(primary_log, workspace, slot=slot)
        result.server_log_path = actual_log
        evidence = build_launch_evidence(
            config_path=config_path,
            actual_server_log=actual_log,
            framework=os.environ.get("FRAMEWORK", ""),
            slot=slot,
            caller_reused_ready_server=caller_reused_ready_server,
        )
        result.launch_evidence = evidence
        result.launch_evidence_path = persist_launch_evidence(evidence, slot=slot)


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names.

    Args:
        name (str): The variant name to slugify.

    Returns:
        str: ``name`` with every character that is not alphanumeric or in
        ``-_.`` replaced by ``_``, truncated to 60 characters.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


def _write_variant_abort_marker(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str = "",
) -> None:
    """Write ``abort_reason.json`` into the variant slot directory.

    Lets final-report / post-mortem tools distinguish "tested-but-failed" from
    "untested" and find an explicit reason. Failure to write it is non-fatal.
    """
    _write_variant_abort_marker_impl(
        slot,
        variant_name=variant_name,
        error_class=error_class,
        error_summary=error_summary,
        extra_args=extra_args,
    )


def _report_errors_summary(report: dict[str, Any] | None, limit: int = 2000) -> str:
    """Join ``benchmark_report.json`` ``errors`` into a single diagnostic.

    Last-resort fallback after the parent pipe and on-disk logs: the report
    may still name the failure when log writes were swallowed (``OSError``)
    or another backend never opened a log. Empty or non-list ``errors``
    yield ``""`` so callers can chain this after ``_on_disk_stderr_tail``.

    Args:
        report: Parsed ``benchmark_report.json`` mapping, or None.
        limit: Max characters returned (tail).

    Returns:
        Joined error strings, or ``""``.
    """
    if not isinstance(report, dict):
        return ""
    errors = report.get("errors")
    if not isinstance(errors, list):
        return ""
    text = "; ".join(str(item).strip() for item in errors if str(item).strip())
    return text[-limit:] if text else ""


def _on_disk_stderr_tail(*dirs: Path, limit: int = 2000) -> str:
    """Return the tail of the first non-empty on-disk benchmark log.

    The piped ``stderr``/``stdout`` from :func:`run_with_session_kill` is empty
    for the bypass/scriptable path — the customer body (torchrun/bench_fps.py)
    writes its own stderr (e.g. an argparse ``unrecognized arguments`` error)
    to ``benchmark_stderr.log`` in the workspace, NOT the parent's pipe. Without
    this, ``magpie_nonzero_invalid_measurement`` aborts land in
    ``abort_reason.json`` with an empty ``error`` and the real diagnostic is
    only discoverable by hand. Scans the given dirs for
    ``benchmark_stderr.log`` then ``benchmark_stdout.log`` and returns the tail
    of the first with content.

    Args:
        *dirs (Path): Directories to search (workspace, slot), in order.
        limit (int): Max characters returned (tail).

    Returns:
        str: The log tail, or ``""`` if none found / all empty.
    """
    for d in dirs:
        if not d:
            continue
        for name in ("benchmark_stderr.log", "benchmark_stdout.log"):
            try:
                p = d / name
                if not p.is_file():
                    continue
                text = p.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return f"[{name}] {text[-limit:]}"
            except OSError:
                continue
    return ""


def _write_variant_abort_marker_impl(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str,
) -> None:
    """Implementation body for :func:`_write_variant_abort_marker`.

    Args:
        slot (Path): Variant slot directory the marker is written into.
        variant_name (str): Name of the aborted variant.
        error_class (str): Short failure-classification tag.
        error_summary (str): Error detail (truncated to 2000 chars).
        extra_args (str): The variant's server args, recorded for context.
    """
    try:
        slot.mkdir(parents=True, exist_ok=True)
        marker = {
            "variant": variant_name,
            "error_class": error_class,
            "error": (error_summary or "")[:2000],
            "extra_args": extra_args,
            "aborted_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        (slot / "abort_reason.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "_grid_runner: failed to write abort_reason.json at %s: %s",
            slot,
            exc,
        )


__all__ = [
    "DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC",
    "GridVariant",
    "ORCHESTRATOR_CANCELLED_CLASS",
    "SESSION_TIME_EXHAUSTED_CLASS",
    "StoppedByTheRun",
    "SGLANG_WATCHDOG_TIMEOUT_ENV",
    "VariantResult",
    "apply_multi_node_invalid_variants",
    "apply_aiter_moe_pin_filter",
    "apply_runtime_benchmark_overrides",
    "reorder_grid_for_multi_node",
    "inject_sglang_attention_backend",
    "inject_sglang_context_length",
    "inject_sglang_watchdog_timeout",
    "merge_server_args",
    "remove_server_args",
    "resolve_sglang_watchdog_timeout",
    "run_grid",
    "sanitize_result_dir",
    "sanitize_script_name",
    "server_args_env_name",
    "session_clamped_timeout_sec",
    "session_grid_bounds",
    "stopped_by_the_run",
    # Re-exported from the sibling modules to keep the namespace intact.
    "coerce_extra_envs",
    "compact_json_server_args",
    "_SPACE_VALUE_FLAGS",
    "_MULTI_VALUE_FLAGS",
    "_VLLM_SINGLE_VALUE_FLAGS",
    "dedup_vllm_server_args",
    "_SGLANG_WATCHDOG_FLAG",
    "_SGLANG_WATCHDOG_RE",
    "DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS",
    "DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS",
    "SGLANG_CONTEXT_HEADROOM_ENV",
    "SGLANG_CONTEXT_FLOOR_ENV",
    "_SGLANG_CONTEXT_LENGTH_FLAG",
    "_SGLANG_CONTEXT_LENGTH_RE",
    "_SGLANG_ATTN_BACKEND_FLAG",
    "_SGLANG_ATTN_BACKEND_RE",
    "_SGLANG_DUAL_CHUNK_BACKEND",
    "_resolve_nonneg_int_env",
    "resolve_sglang_context_cap",
    "_resolve_dual_chunk_backend",
    "HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV",
    "DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND",
    "_SGLANG_MOE_RUNNER_BACKEND_FLAG",
    "_SGLANG_MOE_RUNNER_BACKEND_RE",
    "moe_runner_requires_aiter",
    "inject_sglang_moe_runner_backend",
    "resolve_skip_spec",
    "_parse_skip_spec",
    "_RE_CUDA_GRAPH_MAX_BS",
    "_MN_PARAMS_PRIORITY",
    "_MN_BACKENDS_PRIORITY",
    "_mn_priority_index",
    "_COMPATIBILITY_FLAG_RULES",
    "_XDIT_ENV_BLACKLIST",
    "_XDIT_ENV_COMBO_BLACKLIST",
    "xdit_blacklist_reason",
    "_HELP_TEXT_CACHE",
    "_HELP_PROBE_COMMANDS",
    "_probe_server_help_text",
    "_detect_model_class",
    "apply_compatibility_filter",
    "apply_user_skip_list",
]
