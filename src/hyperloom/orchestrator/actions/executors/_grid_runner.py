# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helper for the ``explore`` executor's grid runs."""

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

from hyperloom.common.coerce import to_str_list
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
    AGENTX_PREFLIGHT_ERROR_CLASS,
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
from ._server_argv import seal_server_argv

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
    _SGLANG_MOE_RUNNER_BACKEND_FLAG as _SGLANG_MOE_RUNNER_BACKEND_FLAG,
    _SGLANG_MOE_RUNNER_BACKEND_RE as _SGLANG_MOE_RUNNER_BACKEND_RE,
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
    """Resolve the Python interpreter for Magpie subprocesses."""

    def _can_import_magpie(py: str) -> bool:
        """Whether an interpreter can import Magpie and its ``yaml`` dep."""
        try:
            # Probe with ``importlib.util.find_spec`` rather than a bare
            # ``import`` so a missing module returns a non-zero exit code
            # WITHOUT the child emitting a ``ModuleNotFoundError`` traceback.
            # The launch mirrors child stderr to the parent stream, so a bare
            # ``import Magpie`` on a candidate that lacks it would leak an
            # alarming traceback into the run log even though the probe failing
            # is an expected, benign step of interpreter resolution.
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
    """Resolve the interpreter a build-accuracy probe must use."""
    if (framework or "").strip().lower() == "vllm":
        venv_root = os.environ.get("VLLM_VENV_ROOT", "").strip()
        if venv_root:
            venv_python = str(Path(venv_root) / "bin" / "python")
            if os.access(venv_python, os.X_OK):
                return venv_python
    magpie_python = _resolve_magpie_python()
    # Prefer the harness interpreter; on a single-venv box it already IS the vLLM venv.
    if magpie_python and magpie_python != "/opt/venv/bin/python":
        return magpie_python
    # Fell through to the canonical default; pin the venv that backs ``vllm serve`` so the probe hits the real server
    # source.
    vllm_exe = shutil.which("vllm")
    if vllm_exe:
        vllm_python = os.path.join(os.path.dirname(vllm_exe), "python")
        if os.path.exists(vllm_python):
            return vllm_python
    return magpie_python


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root."""
    from hyperloom.inference_optimizer.session.paths import session_dir as _sd

    return _sd()


# SKIP_VARIANTS: comma/whitespace patterns matched (exact or fnmatch) against ``GridVariant.name``.


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
    """Return a drop reason if a variant sets an env flag the installed framework build cannot honour, else ``None``."""
    fw = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    if fw != "vllm":
        return None
    envs = {str(k): str(v) for k, v in (getattr(variant, "extra_envs", None) or {}).items()}
    val = envs.get("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS")
    if val is None or not is_truthy(val, default=True):
        return None
    return _probe_vllm_aiter_shared_expert_unsupported()


# Sanitization for LLM-supplied overrides (benchmark_script / result_dir): reject path separators / shell
# metacharacters, raising ``ValueError`` instead of running an unsafe subprocess.
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.sh$")
_RESULT_DIR_FORBID_RE = re.compile(r"[\s\"'`$;&|<>(){}\[\]\\*?!]")


def sanitize_script_name(value: Any) -> str | None:
    """Return ``value`` if it's a safe Magpie benchmark script file name."""
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
    """Return ``value`` if it's a safe absolute (or workspace-relative) dir."""
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
    """Inject an attempt runtime override into the materialized YAML envs dict."""
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
    # Multi-entry prefixes: prepend in order so the first entry wins, ahead of any single-dir pythonpath_prefix and
    # the inherited value.
    for entry in reversed(_coerce_prefix_list(override.get("pythonpath_prefixes"))):
        if _is_safe_path_entry(entry):
            _cur_pp = str(envs.get("PYTHONPATH", "") or "")
            envs["PYTHONPATH"] = f"{entry}:{_cur_pp}" if _cur_pp else entry
        else:
            log.warning("apply_runtime_override: dropping unsafe pythonpath entry %r", entry)
    # Native loader path: prepend attempt entries while preserving any inherited entries (e.g. /opt/rocm/lib), which
    # _prepend_path_entry never drops.
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
    base_extra_envs: dict[str, str] | None = None,
    base_remove_args: list[str] | None = None,
    base_unset_envs: list[str] | None = None,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk."""
    with base_yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bench = cfg.setdefault("benchmark", {})
    envs = apply_runtime_benchmark_overrides(
        bench,
        model_path=model_path,
        gpu_type=gpu_type,
        benchmark_script=benchmark_script,
        conc=variant_conc(variant),
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    replacing = str(base_args_mode).strip().lower() == "replace"
    variant_remove = to_str_list(getattr(variant, "remove_args", []))
    # A replacing base drops the inherited string wholesale, so only the
    # variant's own removals still name flags that survive to be stripped.
    effective_remove = (
        variant_remove if replacing else list(dict.fromkeys(to_str_list(base_remove_args) + variant_remove))
    )
    combined = compose_server_args(
        inherited_args="" if replacing else str(envs.get(extra_args_env, "")),
        base_extra_args=base_extra_args,
        variant_extra_args=variant.extra_server_args,
        remove_args=effective_remove,
        args_mode=getattr(variant, "args_mode", "append"),
    )
    # A grid variant never injects a MoE runner backend itself, but it does inherit one -- from the baseline recipe it
    # was seeded with, or from an explicitly authored variant.
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
    # Composed base-then-variant, so a variant unsetting a key the base sets
    # removes it: the last layer to name a key is the one that decides it.
    for k in to_str_list(base_unset_envs):
        if k.strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            continue
        envs.pop(k, None)
    for k, v in (base_extra_envs or {}).items():
        envs[str(k)] = str(v)
    for k in getattr(variant, "unset_envs", []) or []:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(k).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("grid: refusing to unset pinned env %s for variant %s", k, variant.name)
            continue
        envs.pop(str(k), None)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)
    # The three AgentX bounds took this rung's CONC through ``variant_conc`` above, not through this merge: raising
    # the client's grace alone would make the round wait inside a cap that did not move with it.
    _overlay = str(getattr(variant, "overlay_pythonpath", "") or "").strip()
    if _overlay:
        # Structural containment on the overlay dir before it is prepended to PYTHONPATH: a legitimate authored-kernel
        # overlay is a single existing directory (never a ``:``-joined list, never a ``..`` traversal or control
        # char).
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

    # Attempt runtime override: inject path_prefix/pythonpath_prefix/framework_bin etc. into benchmark.envs so the
    # server subprocess resolves the attempt runtime.
    _rt_override = getattr(variant, "runtime_override", None) or {}
    if _rt_override:
        apply_runtime_override(envs, _rt_override)

    # PATH guard: the xdit wrapper needs both `/venv/bin` (the `xdit` console script) and `/opt/rocm/bin` (`hipcc`);
    # force-prepend both so an LLM-supplied PATH can't drop one.
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

    # The final write to the argument env; nothing below may touch it.
    seal_server_argv(envs, bench.get("framework"))
    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    """Load ``benchmark_report.json`` from a benchmark workspace."""
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# How long to keep re-reading ``benchmark_report.json`` when the process exited cleanly but the report does not yet
# parse into a valid measurement.
REPORT_SETTLE_SECONDS = 30.0
REPORT_SETTLE_POLL_SECONDS = 1.0


async def _settled_measurement(
    workspace: Path,
    *,
    subprocess_started_unix: float | None,
    settle_seconds: float = REPORT_SETTLE_SECONDS,
    poll_seconds: float = REPORT_SETTLE_POLL_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read the benchmark report, re-reading briefly while it is still settling."""
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
    """Resolve ``pid``'s visible-GPU mask from its own environment."""
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
    """Deep-clean any lingering inference server processes + shared memory."""
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

    # atom ModelRunner workers spawn with a generic ``--multiprocessing-fork`` cmdline (unmatchable by _KILL_PATTERNS)
    # and can orphan holding VRAM; identify survivors by atom/aiter JIT mmaps in their address space.
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
        """Whether ``pid`` overlaps our own visible-GPU mask."""
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
        """Detect an orphaned atom ModelRunner worker by its memory maps."""
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

    # Clear GPU runtime shared-memory segments that prevent re-binding.
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

    # Pause for KFD async VRAM release; atom teardown lags past 2s.
    if killed_any:
        time.sleep(8 if killed_atom else 2)


def _prepend_magpie_pythonpath(magpie_dir: str, current_pythonpath: str) -> str:
    """Prepend Magpie's import root to PYTHONPATH, skipping package-root dirs."""
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
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr)."""
    # Pre-clean lingering servers + shared memory (skip under pytest, and for lifecycle re-attach rounds that would
    # kill the warm server).
    if preclean and not os.environ.get("PYTEST_CURRENT_TEST"):
        _kill_stale_servers()

    env = scrub_benchmark_process_env(os.environ.copy())
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_PATH") or ""
    if magpie_dir:
        env["PYTHONPATH"] = _prepend_magpie_pythonpath(magpie_dir, env.get("PYTHONPATH", ""))

    # Multi-node: tell Magpie to skip its local-server launch and point benchmark_serving at the head pod's ClusterIP.
    from ._multi_node_env import magpie_remote_env

    env.update(magpie_remote_env())

    # Pin Magpie's InferenceX resolution to ``$INFERENCEX_PATH`` (its highest-precedence rung) so it loads the patched
    # checkout, not a stale copy.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Baseline patches its own checkout, but explore / sweep never pass through that hook: re-assert here so a
        # resumed session or a re-cloned checkout still emits the eval-start marker.
        ensure_benchmark_lib_eval_start_patched(Path(inferencex_path))

    # The generation bounds + pathology probe are asserted whether or not ``$INFERENCEX_PATH`` is set: unset falls
    # back to the same env discovery the baseline arm uses ($MAGPIE_PATH/InferenceX).
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
    # AgentX: deploy the aiperf client into InferenceX ``benchmarks/`` + preflight aiperf right before Magpie runs it,
    # via the shared helper (also used by the baseline/profile shell-out).
    from ._workload_envs import prepare_agentx_runtime

    _agx_err = prepare_agentx_runtime(env=env, inferencex_path=inferencex_path, config_path=config_path)
    if _agx_err:
        log.error("%s; failing this benchmark", _agx_err)
        return (AGENTX_PREFLIGHT_RETURNCODE, "", _agx_err)
    # RESULT_DIR default; leaks are picked up by the salvage path.
    env["RESULT_DIR"] = result_dir or str(output_dir)
    # InferenceX ``run_lm_eval`` cleans ``$EVAL_RESULT_DIR`` after processing lm-eval output.
    env["EVAL_RESULT_DIR"] = str(Path(env["RESULT_DIR"]) / "eval_output")
    # Pin SERVER_LOG / GPU_METRICS_CSV per-task so logs land alongside ``benchmark_report.json``.
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")

    # Ray-managed GPU execution (§12 T1): route the round through the serving lease's actor, which holds ``num_gpus``
    # across every round sharing this server.
    if serving_lease is not None:
        from ._ray_backend import strip_visible_devices_from_config

        ray_config_path = strip_visible_devices_from_config(config_path)
        cmd = build_benchmark_command(
            python_exe=magpie_python,
            config_path=ray_config_path,
            output_dir=output_dir,
        )
        # ``on_output`` cannot follow the round here: the benchmark runs inside a Ray actor in another process
        # (potentially on another node) and only its final ``(rc, stdout, stderr)`` comes back, so there is nothing
        # local to call per line.
        return serving_lease.run_session_kill(
            cmd,
            env=env,
            cwd=cwd,
            timeout=timeout_sec,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=str(output_dir / "server.log"),
            server_already_ready=server_already_ready,
            # Converted here, at the last moment before the process boundary: the actor cannot read this process's
            # monotonic clock.
            session_remaining_sec=session_deadline_to_remaining_sec(session_deadline_sec),
        )

    cmd = build_benchmark_command(
        python_exe=magpie_python,
        config_path=config_path,
        output_dir=output_dir,
    )
    # The launch puts Magpie in its own POSIX session and tears down the whole
    # descendant tree on every exit path.
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
    """Read the tensor-parallel size (``TP``) from a materialized benchmark YAML."""
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
    """Merge the reference server envs under one variant's own overrides."""
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
    """Resolve the multi-node server args for a variant restart; prefer the materialized variant YAML, falling back to a recompose from the base YAML."""
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
    """Build the progress note for the variant at ``idx`` from that variant's own row."""
    landed = results[idx] if idx < len(results) else None
    return {
        "unit": "variant",
        "label": grid[idx].name,
        "index": idx + 1,
        "total": len(grid),
        "status": getattr(landed, "status", None),
        "output_throughput": getattr(landed, "output_throughput", None),
    }


# Seconds by which a round's hard cap is allowed to sit past the session deadline, so the in-process session watchdog
# (which attributes the kill correctly) trips before the hard cap (which the ledger reads as a variant timeout).
_SESSION_KILL_GRACE_SEC: int = 15

# The returncode side of :mod:`...stop_attribution`: the same two causes, keyed by the sentinel a subprocess comes
# back with.
_STOPPED_BY_THE_RUN: dict[int, StoppedByTheRun] = {
    SESSION_TIME_EXHAUSTED_RETURNCODE: STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS],
    ORCHESTRATOR_CANCELLED_RETURNCODE: STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS],
}


def stopped_by_the_run(returncode: int | None) -> StoppedByTheRun | None:
    """Return how to record a round the run itself stopped, if it did."""
    if returncode is None:
        return None
    return _STOPPED_BY_THE_RUN.get(int(returncode))


def variant_conc(variant: Any) -> int | None:
    """The concurrency a variant will run at, or ``None`` when it names none."""
    raw = (getattr(variant, "extra_envs", None) or {}).get("CONC")
    try:
        conc = int(raw)
    except (TypeError, ValueError):
        return None
    return conc if conc > 0 else None


def agentx_variant_timeout_sec(cap: int, *, shared_state: Any = None, conc: int | None = None) -> int:
    """Raise a variant's hard cap to what an AgentX round actually needs."""
    # Local import: baseline imports from this module, and the rest of the file already resolves _workload_envs this
    # way.
    from ._workload_envs import agentx_active, agentx_env_for_conc

    if not agentx_active(shared_state):
        return cap
    from .baseline import agentx_baseline_timeout_sec

    return max(cap, agentx_baseline_timeout_sec(agentx_env_for_conc(conc)))


def session_clamped_timeout_sec(
    cap: int,
    session_deadline_sec: float | None,
    *,
    reserve_sec: float = 0.0,
) -> int:
    """Reduce a hard timeout to what the session budget can still pay for."""
    if session_deadline_sec is None:
        return int(cap)
    usable = int(session_deadline_sec - time.monotonic() - max(0.0, reserve_sec)) + _SESSION_KILL_GRACE_SEC
    return int(cap) if usable >= int(cap) else max(1, usable)


def session_grid_bounds(shared_state: Any) -> tuple[float | None, float | None]:
    """Resolve ``(session_deadline_sec, variant_expected_sec)`` for a :func:`run_grid` call."""
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
    base_extra_envs: dict[str, str] | None = None,
    base_remove_args: list[str] | None = None,
    base_unset_envs: list[str] | None = None,
    warmup_before_measure: bool | None = None,
    preclean_before_run: bool = True,
    server_already_ready: bool = False,
    serving_lease: Any = None,
    session_deadline_sec: float | None = None,
    variant_expected_sec: float | None = None,
) -> list[VariantResult]:
    """Execute each grid variant and return all per-variant results."""
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
    # This function names the working directory, so it creates it: callers and the per-variant config writer both
    # happen to create it first today, and neither is a contract.
    output_root.mkdir(parents=True, exist_ok=True)
    cwd = str(output_root)

    # Reap orphaned aiter JIT build locks before booting any server.
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

    # Multi-node: the reference recipe's server envs do not reach the variant restart through
    # restart_server_for_round, which never reads them.
    from ._multi_node_env import is_multi_node as _mn_is_multi_node

    _mn_ref_envs: dict[str, str] = {}
    if _mn_is_multi_node():
        from ._workload_envs import resolve_reference_base

        _, _mn_ref_envs = resolve_reference_base()

    # Reported on entry, not on completion: ``_report_finished_variant`` only runs once a result has been appended, so
    # a first variant that hangs — or a branch that raises before reaching it — would emit nothing at all, which is
    # exactly the silence the heartbeat exists to break.
    async def _unit_started(idx: int, label: str) -> None:
        """Report that a unit of variant ``idx`` is about to start."""
        await report_progress(
            unit="variant_step",
            label=f"{grid[idx].name}:{label}",
            index=idx + 1,
            total=len(grid),
            status="started",
        )

    async def _reported_magpie(idx: int, label: str, **kwargs: Any) -> tuple[int, str, str]:
        """Run one Magpie pass, announced on entry and kept alive by its output."""
        await _unit_started(idx, label)
        async with heartbeat_while_output_flows(
            unit="variant_step",
            label=f"{grid[idx].name}:{label}",
            index=idx + 1,
            total=len(grid),
        ) as activity:
            return await asyncio.to_thread(_run_magpie, on_output=activity.note, **kwargs)

    # Variant boundary: a progress heartbeat so a grid that runs for hours is distinguishable from one that hung on
    # its first variant.
    async def _report_finished_variant(idx: int) -> None:
        """Report the variant that just landed."""
        await report_progress(**_variant_progress_note(grid, results, idx))

    def _round_timeout_sec(idx: int, name: str, *, round_label: str, reserve_sec: float = 0.0) -> int:
        """``variant_timeout_sec`` capped at what the session can still pay for."""
        declared = int(variant_timeout_sec)
        cap = agentx_variant_timeout_sec(declared, conc=variant_conc(grid[idx]))
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
        """Record a round the run stopped and say whether the grid is over."""
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

    # How many full benchmark passes one variant costs.
    _mn_warmup_rounds = 0
    if variant_expected_sec is not None:
        from ._multi_node_env import (
            is_multi_node as _mn_is_multi_node,
            mn_bench_warmup_enabled as _mn_bench_warmup_enabled,
        )

        _mn_warmup_rounds = 1 if (_mn_is_multi_node() and _mn_bench_warmup_enabled()) else 0
    variant_rounds = 1 + (1 if auto_warmup_requested else 0) + _mn_warmup_rounds

    def _skip_rest_for_budget(idx: int, *, spent_on: str, rounds_left: int = variant_rounds) -> bool:
        """Skip variant ``idx`` and every one after it, or say the budget still fits."""
        if session_deadline_sec is None:
            return False
        remaining_sec = session_deadline_sec - time.monotonic()
        # Falls back to a single ``variant_timeout_sec`` when no estimate was given, which is what callers that cannot
        # estimate already got.
        _raised = float(agentx_variant_timeout_sec(variant_timeout_sec, conc=variant_conc(grid[idx])))
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
        # Session-budget stop: skip the remaining variants once the wall-clock deadline is reached or the remaining
        # budget cannot fit another variant, so a timeout halts the grid instead of draining it (and the last variant
        # cannot overrun the close window).
        if _skip_rest_for_budget(i, spent_on="the variants before this one"):
            break
        await _unit_started(i, "variant")
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
        server_log = slot / "server.log"
        # Capability fast-fail: drop a variant whose env flag the build cannot honour before booting a doomed server,
        # still recording the failure so the LLM learns not to re-pick it.
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
                base_extra_envs=base_extra_envs,
                base_remove_args=base_remove_args,
                base_unset_envs=base_unset_envs,
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
                        base_extra_envs=base_extra_envs,
                        base_remove_args=base_remove_args,
                        base_unset_envs=base_unset_envs,
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
                    base_extra_envs=base_extra_envs,
                    base_remove_args=base_remove_args,
                    base_unset_envs=base_unset_envs,
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
            # Held in a local because the abort line below has to name the cap the round was actually granted: the
            # declared one is a hang backstop, and a round killed at the reserved cap logged as a two-hour timeout
            # reads as a variant that hangs rather than a budget that ran out.
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

        # Multi-node only: restart sglang/vllm with this variant's flags so each row runs against a fresh server.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )

        try:
            # PD knobs auto-resolved from $PD_* env; PD config stays constant across variants within one run.
            _variant_unset = [str(k) for k in getattr(variant, "unset_envs", []) or [] if str(k).strip()]
            _restart_env = _mn_restart_env(_mn_ref_envs, variant, _variant_unset)
            await restart_server_for_round(
                extra_server_args=_mn_effective_args,
                # Reference recipe envs under this variant's own overrides (e.g. MORI_* MoE-dispatch tuning) so
                # server-side env knobs proposed by specialists take effect on the restarted sglang without dropping
                # the envs the baseline was measured with.
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

        # The restart above is a launch of its own and is under no cap, so the budget admitted at the top of the loop
        # may no longer be there.
        if _skip_rest_for_budget(
            i,
            spent_on="this variant's server restart",
            rounds_left=1 + _mn_warmup_rounds,
        ):
            if auto_warmup:
                _teardown_variant_server(slot, lifecycle)
            break

        # Multi-node client warmup: one discarded benchmark pass against the just-restarted, persistent remote server
        # to warm JIT / steady-state before the measured pass.
        from ._multi_node_env import (
            is_multi_node as _mn_imn,
            mn_bench_warmup_enabled as _mn_warm,
        )

        if _mn_imn() and _mn_warm():
            _mn_warm_slot = slot / "mn_warmup"
            _mn_warm_started_unix = time.time()
            # The measurement is discarded, but the returncode is not: a warmup the run stopped is the same stop as
            # one in the measured round, and discarding it launches the measured round after the cancel.
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

        # Snapshot wall-clock before launch so the salvage path can mtime-gate leak destinations per-variant.
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

        # Eval bounds could not be installed for a variant that runs eval, so nothing launched.
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

        # Server-liveness watchdog fired: engine/worker bootstrap died but the parent hung.
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

        # Detokenizer-stall watchdog fired: the server came up healthy but then went silent (hung engine / wedged
        # detokenizer).
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

        # Soft overtime gate fired: record a ``killed_overtime=True`` result with no tput and still harvest leaks for
        # post-mortem.
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
            # Best-effort rough tput from server.log throughput logs; informational only — the variant stays failed
            # and never feeds winner selection.
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
        # Always-on artifact harvest so each slot keeps its server.log / gpu_metrics / profile relay for Robustness
        # RCA.
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
            # An AgentX preflight abort never reaches Magpie, so of course no workspace exists -- but calling it
            # ``no_benchmark_workspace`` erases the one fact that decides what to do next.
            agentx_preflight_abort = rc == AGENTX_PREFLIGHT_RETURNCODE
            if agentx_preflight_abort:
                # Same class the baseline path uses for the same abort, so both routes classify it identically.
                no_ws_error_class = AGENTX_PREFLIGHT_ERROR_CLASS
                no_ws_error_summary = (
                    redact_secret_values((stderr or "").strip()[-2000:]) or "AgentX preflight aborted the round"
                )
            else:
                no_ws_error_class = "no_benchmark_workspace"
                no_ws_error_summary = server_log_death_excerpt(str(server_log)) or (
                    redact_secret_values((stderr or stdout)[-2000:]) if rc != 0 else "no benchmark_* workspace produced"
                )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: %s (rc=%s)",
                i + 1,
                len(grid),
                variant.name,
                no_ws_error_class,
                rc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class=no_ws_error_class,
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
                    error_class=no_ws_error_class,
                    server_log_path=_existing_log_path(server_log),
                    nonfatal_warnings=harvest_tags,
                    note=variant.note,
                )
            )
            await _report_finished_variant(i)
            if agentx_preflight_abort:
                # Environment, not variant.
                log.error(
                    "grid_runner: variant %d/%d name=%s aborted on the AgentX preflight; "
                    "abandoning the remaining %d point(s) -- the client is missing for the "
                    "whole grid, not for this variant: %s",
                    i + 1,
                    len(grid),
                    variant.name,
                    len(grid) - (i + 1),
                    no_ws_error_summary,
                )
                # Recorded rather than dropped: a grid that just returns fewer points than it declared reads as a grid
                # that was that size.
                results.extend(
                    VariantResult(
                        name=rest.name,
                        extra_server_args=rest.extra_server_args,
                        extra_envs=dict(rest.extra_envs),
                        status="skipped",
                        error="AgentX preflight aborted the grid before this variant ran",
                        error_class=AGENTX_PREFLIGHT_ERROR_CLASS,
                        note=rest.note,
                    )
                    for rest in grid[i + 1 :]
                )
                break
            if rc != 0 and not keep_going_on_failure:
                break
            continue
        report_path = workspace / "benchmark_report.json"
        # A clean exit is worth waiting on: the report is written during shutdown and the reader runs the moment the
        # subprocess is reaped.
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
                # The bypass/scriptable path runs the customer body in a child whose stderr is redirected to
                # benchmark_stderr.log, not the parent pipe — so `stderr`/`stdout` here are often empty and the real
                # diagnostic (e.g. argparse "unrecognized arguments") is only on disk.
                if not error.strip():
                    error = redact_secret_values(_on_disk_stderr_tail(workspace, slot))
                # Last resort: report.errors when the pipe and on-disk logs are all empty.
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
                tpot_p90_ms=measurement.get("tpot_p90_ms"),
                intvty_p90=measurement.get("e2e_norm_intvty_p90"),
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
    """Stop the server this variant booted for its own rounds."""
    from ._server_lifecycle import teardown_lifecycle_server

    teardown_lifecycle_server(
        pid_dir=slot,
        framework=str(lifecycle.get("framework") or ""),
        port=int(lifecycle.get("port") or 0),
    )


def _not_run_skip_result(variant: GridVariant, stopped: StoppedByTheRun) -> VariantResult:
    """Build the ``skipped`` result for a variant the run never got to."""
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
    """Return ``path`` as a string when it exists, else ``None``."""
    return str(path) if path.exists() else None


def _measurement_server_log_path(
    server_log: Path,
    workspace: Path | None = None,
    *,
    slot: Path | None = None,
) -> str | None:
    """Return the server log attributable to this variant's measurement."""
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
    """Persist declared and observed launch evidence for each grid result."""
    for idx, result in enumerate(results):
        if idx >= len(grid):
            break
        # A skipped variant never launched a measurement.
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
    """Filesystem-safe slug for variant directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


def _write_variant_abort_marker(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str = "",
) -> None:
    """Write ``abort_reason.json`` into the variant slot directory."""
    _write_variant_abort_marker_impl(
        slot,
        variant_name=variant_name,
        error_class=error_class,
        error_summary=error_summary,
        extra_args=extra_args,
    )


def _report_errors_summary(report: dict[str, Any] | None, limit: int = 2000) -> str:
    """Join ``benchmark_report.json`` ``errors`` into a single diagnostic."""
    if not isinstance(report, dict):
        return ""
    errors = report.get("errors")
    if not isinstance(errors, list):
        return ""
    text = "; ".join(str(item).strip() for item in errors if str(item).strip())
    return text[-limit:] if text else ""


def _on_disk_stderr_tail(*dirs: Path, limit: int = 2000) -> str:
    """Return the tail of the first non-empty on-disk benchmark log."""
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
    """Implementation body for :func:`_write_variant_abort_marker`."""
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
    "_SGLANG_MOE_RUNNER_BACKEND_FLAG",
    "_SGLANG_MOE_RUNNER_BACKEND_RE",
    "moe_runner_requires_aiter",
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
