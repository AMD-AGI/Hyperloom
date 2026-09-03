# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared single-node ``server_lifecycle`` helpers.

Magpie's ``server_lifecycle`` reuse protocol lets two benchmark rounds share
one persistent server: round 1 boots it (``cleanup=false``) and round 2
re-attaches as a client-only run (``cleanup=true``). Used by the baseline cold-start
double-run guard. Explore reuses the server the same way across its warmup and
decision rounds but keeps ``cleanup=false`` on both, so its teardown runs
through :func:`teardown_lifecycle_server` rather than Magpie's own leg.

Also hosts :func:`reap_orphaned_servers`, which scans this session's
``runs/*.pid`` files at startup/resume and reaps serving processes orphaned by
a prior monitor-process death.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import yaml

from ._subprocess_kill import TERM_GRACE_SECONDS, _process_group_alive, _signal_group


log = logging.getLogger(__name__)

# Substrings that identify a Hyperloom-spawned serving process in
# ``/proc/<pid>/cmdline``. A pidfile is only acted on when the live pid's
# cmdline matches one of these, so a recycled pid running an unrelated program
# is never signalled.
_SERVER_CMDLINE_MARKERS: tuple[str, ...] = (
    "sglang.launch_server",
    "sglang serve",
    "vllm.entrypoints",
    "vllm serve",
    "launch_server",
)


# Magpie built-in benchmark scripts that support the server_lifecycle reuse
# protocol. Mirrors Magpie's ``benchmarker.MAGPIE_BUILTIN_SCRIPTS`` (duplicated
# to avoid an import-time Magpie dependency).
MAGPIE_BUILTIN_SCRIPTS = frozenset(
    {
        "vllm_mi300x.sh",
        "vllm_mi355x.sh",
        "sglang_mi300x.sh",
        "sglang_mi355x.sh",
        "atom_mi300x.sh",
        "atom_mi355x.sh",
    }
)

# Default HTTP port for the persistent server when ``benchmark.envs.PORT`` is
# unset; pinned into the per-round YAML so Magpie's reuse keying and our
# teardown agree.
REUSE_PORT_DEFAULT = 8888

# Server-boot budget for the persistent server phase. Sized for a TB-scale
# checkpoint, whose weight read alone outlasts a mid-size model's whole boot; a
# hung server is still bounded by the phase and session budgets.
# Override via ``INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC``.
SERVER_READY_TIMEOUT_SEC = 7200


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port.

    Binds to port 0 (ephemeral) and reads back the kernel-chosen port. There
    is a small TOCTOU window before the server actually binds, acceptable for
    the high ephemeral range. Raises ``OSError`` if no port can be obtained.

    Returns:
        int: A currently-free TCP port.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _assign_free_port(current_port: int) -> int:
    """Return a per-session free ephemeral port for the persistent server.

    Falls back to ``current_port`` if no free port can be bound. Used on the
    common resolution path so every backend (Magpie and bypass) gets the same
    stale/co-tenant port-collision protection.

    Args:
        current_port: Port to keep if free-port allocation fails.

    Returns:
        int: A free port, or ``current_port`` on ``OSError``.
    """
    try:
        return _pick_free_port()
    except OSError as exc:
        log.warning(
            "server_lifecycle: free-port pick failed (%s); keeping port %d",
            exc,
            current_port,
        )
        return current_port


def resolve_lifecycle_params(materialized_config_path: Path) -> dict[str, Any]:
    """Inspect the materialized YAML for server_lifecycle eligibility.

    Returns a dict with ``eligible`` (bool), ``framework`` (str),
    ``port`` (int) and ``reason`` (str, populated when ineligible).
    Reuse is single-node only, requires a Magpie built-in script, and is
    incompatible with torch_profiler.

    Args:
        materialized_config_path: Path to the materialized benchmark YAML to
            inspect.

    Returns:
        A dict with ``eligible`` (bool), ``framework`` (str), ``port`` (int)
        and ``reason`` (str, populated when ineligible).
    """
    info: dict[str, Any] = {
        "eligible": False,
        "framework": "",
        "port": REUSE_PORT_DEFAULT,
        "reason": "",
    }
    try:
        with Path(materialized_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        info["reason"] = f"could not read materialized config: {exc}"
        return info
    bench = cfg.get("benchmark") or {}
    info["framework"] = str(bench.get("framework") or "").lower()
    envs = bench.get("envs") or {}
    try:
        info["port"] = int(envs.get("PORT", REUSE_PORT_DEFAULT))
    except (TypeError, ValueError):
        info["port"] = REUSE_PORT_DEFAULT

    # Backend delegation: a non-Magpie backend (e.g. bypass) decides its
    # own server_lifecycle eligibility. Magpie returns None here so the
    # historical script-name-based path below runs unchanged.
    from .benchmark_backend import resolve_backend

    backend_verdict = resolve_backend().lifecycle_eligibility(bench)
    if backend_verdict is not None:
        # Free-port assignment lives on this common path so a non-Magpie backend
        # (e.g. bypass) gets the same stale/co-tenant collision protection
        # instead of falling back to the fixed default port.
        if backend_verdict.get("eligible"):
            backend_verdict["port"] = _assign_free_port(int(backend_verdict.get("port", REUSE_PORT_DEFAULT)))
        return backend_verdict

    # Server-less (scriptable) frameworks — e.g. xDiT diffusion — never boot a
    # persistent server, so the reuse protocol does not apply. Bail out before
    # the Magpie built-in-script check (whose script names are serving-only).
    from hyperloom.inference_optimizer import framework_registry

    if str(
        bench.get("workload_kind") or ""
    ).strip().lower() == framework_registry.SCRIPTABLE or framework_registry.is_scriptable(info["framework"]):
        info["reason"] = "scriptable framework (server-less; no server_lifecycle)"
        return info

    from ._multi_node_env import is_multi_node

    if is_multi_node():
        info["reason"] = "multi-node (server_lifecycle is local-only)"
        return info

    script_name = Path(str(bench.get("benchmark_script") or "")).name
    if script_name not in MAGPIE_BUILTIN_SCRIPTS:
        info["reason"] = f"benchmark_script={script_name!r} is not a Magpie built-in ({sorted(MAGPIE_BUILTIN_SCRIPTS)})"
        return info

    profiler_on = bool((bench.get("profiler") or {}).get("torch_profiler", {}).get("enabled"))
    if profiler_on:
        info["reason"] = "torch_profiler enabled (incompatible with reuse)"
        return info

    # Use a per-session free ephemeral port for the persistent server instead of
    # a fixed default: a stale/leaked server from a prior baseline attempt
    # (round 1 uses cleanup=false and leaves the server running) or a co-tenant
    # job holding the fixed port makes Magpie abort every warmup with "Reuse
    # metadata mismatch ... server on PORT=... is incompatible", so the baseline
    # never records an accuracy and the run wrongly stops with
    # baseline_accuracy_failed. Resolved once here; both double-run rounds share
    # it via inject_lifecycle -> envs.PORT (server bind, Magpie reuse keying,
    # lm-eval base_url and teardown all agree).
    info["port"] = _assign_free_port(info["port"])

    info["eligible"] = True
    return info


def inject_lifecycle(
    bench: dict[str, Any],
    *,
    cleanup: bool,
    pid_dir: Path | str,
    port: int,
) -> None:
    """Mutate ``bench`` in place to enable the server_lifecycle protocol.

    Both rounds share ``pid_dir`` + ``port`` so round 2 re-attaches; only
    ``cleanup`` differs (round 1 persists, round 2 tears down).

    Args:
        bench: The benchmark config dict to mutate in place.
        cleanup: Whether this round tears the server down (``False`` for round
            1, ``True`` for round 2).
        pid_dir: Shared directory for the server pid/meta files.
        port: HTTP port pinned for the persistent server.
    """
    ready_timeout = int(
        os.environ.get(
            "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC",
            SERVER_READY_TIMEOUT_SEC,
        )
    )
    bench["server_lifecycle"] = {
        "enabled": True,
        "cleanup": bool(cleanup),
        "force_reuse": False,
        "pid_dir": str(pid_dir),
        "server_ready_timeout_s": ready_timeout,
    }
    # Pin PORT so Magpie's reuse keying and our teardown agree.
    envs = bench.setdefault("envs", {})
    envs["PORT"] = int(port)


def teardown_lifecycle_server(
    *,
    pid_dir: Path | str,
    framework: str,
    port: int,
) -> None:
    """Best-effort teardown of a persistent server left by a lifecycle round.

    Idempotent and never raises (safe in ``finally``); a no-op on the happy
    path, real work only on abnormal paths. The recorded pid is signalled only
    when its cmdline still names a Hyperloom server, so a recycled pid is not
    killed; the pid/meta files are removed either way.

    Args:
        pid_dir: Directory holding the server pid/meta files.
        framework: Framework name used to key the pid/meta filenames.
        port: HTTP port used to key the pid/meta filenames.
    """
    base = Path(pid_dir)
    tag = f"{framework}_{port}"
    pid_file = base / f"{tag}.pid"
    meta_file = base / f"{tag}.json"
    server_pid: int | None = None
    server_pgid: int | None = None
    try:
        if pid_file.exists():
            parts = pid_file.read_text(encoding="utf-8").split()
            if parts:
                server_pid = int(parts[0])
            if len(parts) > 1:
                server_pgid = int(parts[1])
    except (OSError, ValueError):
        # Best-effort: proceed with whatever was parsed; never raise.
        pass

    if server_pid is not None and os.name == "posix":
        if not _looks_like_server_process(server_pid):
            log.warning(
                "server_lifecycle teardown — pid %d is not a Hyperloom server (pid reuse); not signalling",
                server_pid,
            )
        else:
            # Server is setsid'd, so pgid == pid unless the pid file gave one.
            pgid = server_pgid if server_pgid is not None else server_pid
            _signal_group(pgid, signal.SIGTERM)
            deadline = time.monotonic() + TERM_GRACE_SECONDS
            while time.monotonic() < deadline:
                if not _process_group_alive(pgid):
                    break
                time.sleep(0.1)
            if _process_group_alive(pgid):
                _signal_group(pgid, signal.SIGKILL)
            log.info(
                "server_lifecycle teardown — reaped persistent server pgid=%d (%s:%d)",
                pgid,
                framework,
                port,
            )
    for p in (pid_file, meta_file):
        try:
            p.unlink()
        except OSError:
            # Already gone or unremovable; teardown must not raise.
            pass


def _pid_cmdline(pid: int) -> str:
    """Return ``/proc/<pid>/cmdline`` as a space-joined string, or ``""``.

    Args:
        pid: The process id to read.

    Returns:
        The process command line with NULs turned into spaces, or ``""`` when
        the process is gone or ``/proc`` is unreadable.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _looks_like_server_process(pid: int) -> bool:
    """Return whether ``pid``'s cmdline matches a Hyperloom serving process.

    Guards against pid reuse: only a live pid whose cmdline contains a known
    serving marker is treated as a reapable orphan.

    Args:
        pid: The candidate process id.

    Returns:
        ``True`` when the pid's cmdline names a Hyperloom-spawned server.
    """
    cmdline = _pid_cmdline(pid)
    if not cmdline:
        return False
    return any(marker in cmdline for marker in _SERVER_CMDLINE_MARKERS)


def _process_group_looks_like_server(pgid: int) -> bool:
    """Return whether any process in ``pgid`` still looks like a serving process."""
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat = (entry / "stat").read_text(encoding="utf-8")
            after_comm = stat.rsplit(")", 1)[1].split()
            entry_pgid = int(after_comm[2])
        except (IndexError, OSError, ValueError):
            continue
        if entry_pgid == pgid and _looks_like_server_process(pid):
            return True
    return False


def reap_orphaned_servers(session_dir: Path | str) -> list[int]:
    """Reap serving processes orphaned by a prior monitor-process death.

    A crash that takes the optimizer down (e.g. a raylet death) leaves the
    setsid'd SGLang/vLLM server tree alive with its ``{framework}_{port}.pid``
    file still on disk, polluting the next benchmark on the shared port. On
    startup/resume this scans **only the current session's** ``runs/`` pidfiles
    and reaps each pid whose cmdline still matches a serving process (SIGTERM →
    grace → SIGKILL on the whole group). Scoped strictly to this session's own
    pidfiles and gated on a cmdline match, so a co-located session's server and
    a recycled pid are both left untouched. Best-effort and never raises (safe
    to call unconditionally at boot); a no-op for a fresh session.

    A pidfile pointing at a dead pid is removed only after its process group is
    also gone; if the leader exited but server children still occupy the group,
    the group is reaped. A pidfile pointing at a live pid whose cmdline does NOT
    match is left in place so a later pass can re-evaluate it.

    Args:
        session_dir: The current session directory whose ``runs/`` subtree is
            scanned for orphaned server pidfiles.

    Returns:
        The list of pids that were signalled for reaping.
    """
    if os.name != "posix":
        return []
    runs_dir = Path(session_dir) / "runs"
    if not runs_dir.is_dir():
        return []

    reaped: list[int] = []
    for pid_file in sorted(runs_dir.rglob("*.pid")):
        try:
            parts = pid_file.read_text(encoding="utf-8").split()
        except OSError:
            continue
        if not parts:
            _unlink_quietly(pid_file)
            continue
        try:
            server_pid = int(parts[0])
            recorded_pgid = int(parts[1]) if len(parts) > 1 else server_pid
        except ValueError:
            _unlink_quietly(pid_file)
            continue

        pid_alive = _pid_alive_simple(server_pid)
        if pid_alive:
            try:
                server_pgid = os.getpgid(server_pid)
            except OSError:
                server_pgid = recorded_pgid
        else:
            server_pgid = recorded_pgid

        if pid_alive and not _looks_like_server_process(server_pid):
            # Live pid but not one of our servers (pid reuse): do not touch the
            # process; leave the pidfile for a later re-evaluation.
            log.info(
                "orphan-reaper: pid=%d from %s no longer looks like a server (cmdline mismatch); leaving it untouched",
                server_pid,
                pid_file,
            )
            continue
        if not pid_alive and not _process_group_alive(server_pgid):
            # Stale pidfile from a fully-exited server tree; just clean it up.
            _unlink_quietly(pid_file)
            _unlink_quietly(pid_file.with_suffix(".json"))
            continue
        if not pid_alive and not _process_group_looks_like_server(server_pgid):
            # The original leader is gone, and the remaining/reused pgid has no
            # server-looking member; do not risk signalling an unrelated group.
            log.info(
                "orphan-reaper: pid=%d from %s is gone and pgid=%d has no server-looking member; leaving it untouched",
                server_pid,
                pid_file,
                server_pgid,
            )
            continue

        _signal_group(server_pgid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not _process_group_alive(server_pgid):
                break
            time.sleep(0.1)
        if _process_group_alive(server_pgid):
            _signal_group(server_pgid, signal.SIGKILL)
        reaped.append(server_pid)
        log.warning(
            "orphan-reaper: reaped leftover server pid=%d pgid=%d from %s "
            "(likely orphaned by a prior monitor-process crash)",
            server_pid,
            server_pgid,
            pid_file,
        )
        _unlink_quietly(pid_file)
        _unlink_quietly(pid_file.with_suffix(".json"))

    return reaped


def _pid_alive_simple(pid: int) -> bool:
    """Return whether ``pid`` currently exists (``kill(pid, 0)`` probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _unlink_quietly(path: Path) -> None:
    """Best-effort ``unlink`` that never raises."""
    try:
        path.unlink()
    except OSError:
        pass
