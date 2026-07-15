# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
from typing import Any

import yaml

from hyperloom.common.env import is_truthy

from ...roles.robustness_pulse import pulse as _robustness_pulse
from ._subprocess_kill import (
    DETOKENIZER_STALL_RETURNCODE,
    OVERTIME_KILL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    run_with_session_kill,
)
from .benchmark_result import (
    estimate_killed_variant_throughput,
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
)
from .benchmark_backend import build_benchmark_command

# Re-exported from sibling modules to keep the module namespace intact.
from ._grid_base import (
    _MAGPIE_CWD_DEFAULT as _MAGPIE_CWD_DEFAULT,
    _VARIANT_TIMEOUT_SEC_DEFAULT as _VARIANT_TIMEOUT_SEC_DEFAULT,
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
        # Probe Magpie AND ``yaml`` so an interpreter that resolves Magpie via a
        # .pth but lacks PyYAML is skipped in favour of the canonical /opt/venv.
        try:
            # run_with_session_kill captures output internally and rejects
            # capture_output.
            proc = run_with_session_kill(
                [py, "-c", "import Magpie, yaml"],
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


def _resolve_probe_python() -> str:
    """Resolve the interpreter a build-accuracy probe must use.

    A capability probe only produces a correct drop decision when it inspects
    the SAME framework install the benchmark server loads, so a bare ``python3``
    off ``$PATH`` is deliberately NOT a fallback.

    Resolution order:
    1. ``_resolve_magpie_python()`` — the interpreter that runs the benchmark
       harness; on a single-venv install this is also the vLLM venv.
    2. The interpreter behind the ``vllm`` executable (``<venv>/bin/python``
       alongside ``shutil.which("vllm")``) when it exists on disk.
    3. ``_resolve_magpie_python()``'s canonical fallback via step 1.

    Returns:
        str: Path to the interpreter the probe should invoke.
    """
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

    Injects the variant's flags via ``EXTRA_SGLANG_ARGS``. ``model_path``
    overrides the legacy hardcoded ``benchmark.model``; ``gpu_type`` pins the
    generic ``{framework}_{gpu_type}.sh``; ``benchmark_script`` (pre-sanitized)
    force-pins a script, applied last so the operator pick wins.
    ``server_lifecycle`` (``{cleanup, pid_dir, port}``) enables Magpie's
    persistent-server reuse so a paired round can re-attach to a hot server.

    Args:
        base_yaml_path (Path): Path to the base Magpie YAML to template from.
        base_extra_args (str): Server args merged ahead of the variant's args.
        variant (GridVariant): The variant whose flags/envs are applied.
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
    if combined:
        envs[extra_args_env] = _shell_safe_dedupe(combined)
    elif extra_args_env in envs:
        envs.pop(extra_args_env, None)
    for k in getattr(variant, "unset_envs", []) or []:
        envs.pop(str(k), None)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)
    # Authored-kernel overlay: prepend the built-kernel dir onto PYTHONPATH so
    # the relaunched server imports the overlay's kernels. Inert when
    # ``overlay_pythonpath`` is unset.
    _overlay = str(getattr(variant, "overlay_pythonpath", "") or "").strip()
    if _overlay:
        _cur_pp = str(envs.get("PYTHONPATH", "") or "")
        envs["PYTHONPATH"] = f"{_overlay}:{_cur_pp}" if _cur_pp else _overlay

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


def _run_grid_warmup_enabled() -> bool:
    """Whether ``run_grid`` should discard a cold warmup round when possible."""
    raw = os.environ.get("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP")
    if raw is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return (raw if raw is not None else "1").strip().lower() not in {"0", "false", "no", "off", ""}


def _kill_stale_servers() -> None:
    """Deep-clean any lingering inference server processes + shared memory.

    Reaps vLLM::Worker / EngineCore children that escape Magpie's pgrp-leader
    cleanup. Called before every Magpie invocation; uses a /proc scan (not
    pgrep) to avoid clashing with test subprocess mocks. No-op in multi-node
    mode (servers live in RayJob pods).

    Note:
        Side-effecting and best-effort: it sends signals to matching processes
        and unlinks stale shared-memory segments, swallowing errors. Returns
        nothing.
    """
    from ._multi_node_env import is_multi_node

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
        if any(pat in text for pat in _KILL_PATTERNS) or _is_orphaned_atom_worker(pid, cmdline):
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

    # Clear /dev/shm segments that prevent re-binding.
    for pattern in ("/dev/shm/vllm*", "/dev/shm/nccl*", "/dev/shm/cuda*", "/dev/shm/torch*", "/dev/shm/atom*"):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                # Already removed or held by another process.
                pass

    # Pause for KFD async VRAM release; atom teardown lags past 2s.
    time.sleep(8 if killed_atom else 2)


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

    Returns:
        tuple[int, str, str]: ``(returncode, stdout, stderr)``.
    """
    # Pre-clean lingering servers + shared memory (skip under pytest, and for
    # lifecycle re-attach rounds that would kill the warm server).
    if preclean and not os.environ.get("PYTEST_CURRENT_TEST"):
        _kill_stale_servers()

    env = os.environ.copy()
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_PATH") or ""
    if magpie_dir:
        env["PYTHONPATH"] = f"{magpie_dir}:{env.get('PYTHONPATH', '')}"

    # Multi-node: tell Magpie to skip its local-server launch and point
    # benchmark_serving at the head pod's ClusterIP.
    from ._multi_node_env import magpie_remote_env

    env.update(magpie_remote_env())

    # Pin Magpie's InferenceX resolution to ``$INFERENCEX_PATH`` (its
    # highest-precedence rung) so it loads the patched checkout, not a stale copy.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
    # RESULT_DIR default; leaks are picked up by the salvage path.
    env["RESULT_DIR"] = result_dir or str(output_dir)
    # Pin SERVER_LOG / GPU_METRICS_CSV per-task so logs land alongside
    # ``benchmark_report.json``. Always overwrite so a stale parent value can't
    # redirect into a prior run's slot.
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")
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
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


async def run_grid(
    *,
    base_yaml_path: Path,
    base_extra_args: str,
    grid: list[GridVariant],
    output_root: Path,
    magpie_python: str | None = None,
    cwd: str = _MAGPIE_CWD_DEFAULT,
    variant_timeout_sec: int = _VARIANT_TIMEOUT_SEC_DEFAULT,
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
) -> list[VariantResult]:
    """Execute each grid variant and return all per-variant results."""
    if not magpie_python:
        # Backend-aware: bypass uses a plain python3, not Magpie's venv.
        from .benchmark_backend import resolve_benchmark_interpreter

        magpie_python = resolve_benchmark_interpreter()
    if warmup_before_measure is None:
        warmup_before_measure = _run_grid_warmup_enabled()
    auto_warmup_requested = bool(warmup_before_measure and server_lifecycle is None)
    results: list[VariantResult] = []

    # Variant-boundary robustness pulse: a bounded tick after every variant so
    # a mid-grid leak/crash surfaces between variants. Best-effort.
    async def _pulse_after_variant(idx: int) -> None:
        """Run a best-effort robustness pulse after a variant completes.

        Exceptions from the pulse are swallowed (logged at debug) so a pulse
        failure never aborts the grid.

        Args:
            idx (int): Zero-based index of the just-finished variant, passed
                through as the pulse ``tick_index``.
        """
        try:
            await _robustness_pulse(tick_index=idx)
        except Exception as exc:  # noqa: BLE001
            log.debug("robustness pulse swallowed: %r", exc)

    for i, variant in enumerate(grid):
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
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
            await _pulse_after_variant(i)
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
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        try:
            with cfg_path.open(encoding="utf-8") as _f:
                _variant_cfg = yaml.safe_load(_f) or {}
            _variant_bench = _variant_cfg.get("benchmark") or {}
            _variant_envs = _variant_bench.get("envs") or {}
            _variant_framework_env = server_args_env_name(_variant_bench.get("framework"))
            _mn_effective_args = str(_variant_envs.get(_variant_framework_env) or "")
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
            _mn_effective_args = _shell_safe_dedupe(
                compose_server_args(
                    inherited_args="" if str(base_args_mode).strip().lower() == "replace" else _fallback_inherited_args,
                    base_extra_args=base_extra_args,
                    variant_extra_args=variant.extra_server_args,
                    remove_args=getattr(variant, "remove_args", []),
                    args_mode=getattr(variant, "args_mode", "append"),
                )
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
                await _pulse_after_variant(i)
                if not keep_going_on_failure:
                    break
                continue

            warmup_started_unix = time.time()
            try:
                warmup_rc, warmup_stdout, warmup_stderr = await asyncio.to_thread(
                    _run_magpie,
                    magpie_python=magpie_python,
                    config_path=warmup_cfg_path,
                    output_dir=warmup_slot,
                    timeout_sec=variant_timeout_sec,
                    cwd=cwd,
                    result_dir=result_dir,
                    soft_deadline_sec=None,
                    preclean=True,
                )
            except subprocess.TimeoutExpired as exc:
                from ._server_lifecycle import teardown_lifecycle_server

                teardown_lifecycle_server(
                    pid_dir=slot,
                    framework=str(lifecycle.get("framework") or ""),
                    port=int(lifecycle.get("port") or 0),
                )
                log.warning(
                    "grid_runner: variant %d/%d name=%s aborted: warmup timeout (timeout_sec=%d): %s",
                    i + 1,
                    len(grid),
                    variant.name,
                    variant_timeout_sec,
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
                        note=variant.note,
                        runtime_sec=round(max(0.0, time.time() - warmup_started_unix), 2),
                        nonfatal_warnings=["run_grid_warmup_round_failed"],
                    )
                )
                await _pulse_after_variant(i)
                if not keep_going_on_failure:
                    break
                continue

            warmup_candidates = sorted(warmup_slot.glob("benchmark_*"))
            warmup_workspace = warmup_candidates[-1] if warmup_candidates else warmup_slot
            warmup_harvested = harvest_leaked_artifacts(
                warmup_workspace,
                subprocess_started_unix=warmup_started_unix,
            )
            warmup_report = _parse_report(warmup_workspace) if warmup_candidates else None
            warmup_measurement = extract_benchmark_measurement(
                warmup_report,
                workspace=warmup_workspace,
                subprocess_started_unix=warmup_started_unix,
            )
            if warmup_rc != 0 or not warmup_measurement.get("valid_measurement"):
                from ._server_lifecycle import teardown_lifecycle_server

                teardown_lifecycle_server(
                    pid_dir=slot,
                    framework=str(lifecycle.get("framework") or ""),
                    port=int(lifecycle.get("port") or 0),
                )
                warmup_error = (
                    (warmup_stderr or warmup_stdout)[-2000:]
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
                        workspace=str(warmup_workspace) if warmup_candidates else None,
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
                        note=variant.note,
                        runtime_sec=round(max(0.0, time.time() - warmup_started_unix), 2),
                        nonfatal_warnings=[
                            "run_grid_warmup_round_failed",
                            *[f"harvested_leaked_artifact:{src}" for src, _ in warmup_harvested],
                        ],
                    )
                )
                await _pulse_after_variant(i)
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
            await restart_server_for_round(
                extra_server_args=_mn_effective_args,
                # Per-variant env overrides (e.g. MORI_* MoE-dispatch
                # tuning) so server-side env knobs proposed by specialists
                # actually take effect on the restarted sglang. Empty dict
                # for arg-only variants → forwarded as a no-op.
                extra_env=dict(variant.extra_envs),
                unset_env=[str(k) for k in getattr(variant, "unset_envs", []) or [] if str(k).strip()],
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
            if not keep_going_on_failure:
                break
            continue

        # Snapshot wall-clock before launch so the salvage path can mtime-gate
        # leak destinations per-variant.
        variant_started_unix = time.time()
        try:
            rc, stdout, stderr = await asyncio.to_thread(
                _run_magpie,
                magpie_python=magpie_python,
                config_path=cfg_path,
                output_dir=slot,
                timeout_sec=variant_timeout_sec,
                cwd=cwd,
                result_dir=result_dir,
                soft_deadline_sec=soft_deadline_sec,
                preclean=(False if auto_warmup else preclean_before_run),
                server_already_ready=(server_already_ready or auto_warmup),
            )
        except subprocess.TimeoutExpired as exc:
            # Harvest pre-timeout leaks.
            to_candidates = sorted(slot.glob("benchmark_*"))
            to_destination = to_candidates[-1] if to_candidates else slot
            to_harvested = harvest_leaked_artifacts(
                to_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: magpie timeout (timeout_sec=%d): %s",
                i + 1,
                len(grid),
                variant.name,
                variant_timeout_sec,
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
                    note=variant.note,
                    runtime_sec=round(
                        max(0.0, time.time() - variant_started_unix),
                        2,
                    ),
                    nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in to_harvested],
                )
            )
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue
        finally:
            if auto_warmup:
                from ._server_lifecycle import teardown_lifecycle_server

                teardown_lifecycle_server(
                    pid_dir=slot,
                    framework=str(lifecycle.get("framework") or ""),
                    port=int(lifecycle.get("port") or 0),
                )

        # Server-liveness watchdog fired: engine/worker bootstrap died but the
        # parent hung. Record a fast failure so the round proceeds; harvest the
        # crash server.log.
        if rc == SERVER_DEAD_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix),
                2,
            )
            sd_candidates = sorted(slot.glob("benchmark_*"))
            sd_destination = sd_candidates[-1] if sd_candidates else slot
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
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="server_init_dead",
                error_summary=(
                    "server engine/worker init failed; parent process hung and was reaped by the liveness watchdog"
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
                    error="server_init_dead: engine/worker bootstrap failed",
                    error_class="server_init_dead",
                    note=variant.note,
                    nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in sd_harvested],
                )
            )
            await _pulse_after_variant(i)
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
            ds_candidates = sorted(slot.glob("benchmark_*"))
            ds_destination = ds_candidates[-1] if ds_candidates else slot
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
                    note=variant.note,
                    nonfatal_warnings=[f"harvested_leaked_artifact:{src}" for src, _ in ds_harvested],
                )
            )
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Soft overtime gate fired: record a ``killed_overtime=True`` result with
        # no tput and still harvest leaks for post-mortem.
        if rc == OVERTIME_KILL_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix),
                2,
            )
            ok_candidates = sorted(slot.glob("benchmark_*"))
            ok_destination = ok_candidates[-1] if ok_candidates else slot
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
                    error=(
                        f"killed_overtime: wall-clock {variant_runtime_sec:.1f}s "
                        f"exceeded soft_deadline_sec={float(soft_deadline_sec or 0.0):.1f}s"
                    ),
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
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Locate workspace inside slot.
        candidates = sorted(slot.glob("benchmark_*"))
        # Always-on artifact harvest so each slot keeps its server.log /
        # gpu_metrics / profile relay for Robustness RCA.
        harvest_destination = candidates[-1] if candidates else slot
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
        if not candidates:
            harvest_tags = [f"harvested_leaked_artifact:{src}" for src, _ in harvested]
            no_ws_error_summary = (stderr or stdout)[-2000:] if rc != 0 else "no benchmark_* workspace produced"
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
                    nonfatal_warnings=harvest_tags,
                    note=variant.note,
                )
            )
            await _pulse_after_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue
        workspace = candidates[-1]
        report = _parse_report(workspace)
        report_path = workspace / "benchmark_report.json"
        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=variant_started_unix,
        )
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        if rc != 0:
            warnings.append("magpie_nonzero_after_valid_measurement")
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")
        if warmup_tput is not None:
            warnings.append("run_grid_warmup_discarded_first")
            warnings.append(f"warmup_round_tput:{float(warmup_tput):.1f}")

        if not measurement.get("valid_measurement"):
            if rc != 0:
                error = (stderr or stdout)[-2000:]
                invalid_class = "magpie_nonzero_invalid_measurement"
            elif not report:
                error = "benchmark_report missing"
                invalid_class = "benchmark_report_missing"
            else:
                error = "benchmark_report missing valid throughput/completed requests"
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
                    note=variant.note,
                )
            )
            await _pulse_after_variant(i)
            if rc != 0 and not keep_going_on_failure:
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
                workspace=str(workspace),
                report_path=str(report_path) if report_path.exists() else None,
                raw_result_path=measurement.get("raw_result_path"),
                reported_success=measurement.get("reported_success"),
                returncode=rc,
                nonfatal_warnings=warnings,
                error=(stderr or stdout)[-2000:] if rc != 0 else None,
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
        await _pulse_after_variant(i)
    return results


SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT = 1.0
MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT = 2.0


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
    "MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT",
    "SGLANG_WATCHDOG_TIMEOUT_ENV",
    "SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT",
    "VariantResult",
    "apply_multi_node_invalid_variants",
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
