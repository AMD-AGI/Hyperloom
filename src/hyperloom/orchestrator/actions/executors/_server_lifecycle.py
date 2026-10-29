# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared single-node ``server_lifecycle`` helpers."""

from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common import rocm_smi
from hyperloom.common.agentx_workload import AGENTX_SERVER_PIDFILE
from hyperloom.common.env_safety import redact_secret_values
from common.proctree import TERM_GRACE_SEC, group_members, running, signal_group
from hyperloom.inference_optimizer.gpu_types import magpie_reuse_scripts

from .bypass_engine import lifecycle_meta_file, lifecycle_pid_file


log = logging.getLogger(__name__)

# Substrings that identify a Hyperloom-spawned serving process in ``/proc/<pid>/cmdline``.
_SERVER_CMDLINE_MARKERS: tuple[str, ...] = (
    "sglang.launch_server",
    "sglang serve",
    "sglang.srt",
    "vllm.entrypoints",
    "vllm serve",
    "atom.entrypoints",
    # The launcher's own cmdline is what the pidfile records, but the processes
    # still holding VRAM once that launcher exits are the engine/worker ranks,
    # and they carry none of the names above -- so the dead-leader-live-group
    # branch below needs them too.
    "EngineCore",
    "VLLM::Worker",
    "sglang::",
    "launch_server",
)

#: How long a SIGKILL'd group is given to disappear. A rank that held VRAM can sit in uninterruptible sleep while the
#: driver releases its memory.
_KILL_CONFIRM_SECONDS = 30.0

#: How often a hand-off re-reads VRAM while it waits for the GPUs to drain.
_GPU_FREE_POLL_SECONDS = 5.0

#: How many still-running serving process groups a busy hand-off names.
_HOLDERS_REPORTED = 8


# Magpie built-in benchmark scripts that support the server_lifecycle reuse protocol.
MAGPIE_BUILTIN_SCRIPTS = magpie_reuse_scripts()

# Default HTTP port for the persistent server when ``benchmark.envs.PORT`` is unset; pinned into the per-round YAML so
# Magpie's reuse keying and our teardown agree.
REUSE_PORT_DEFAULT = 8888

# Server-boot budget for the persistent server phase.
SERVER_READY_TIMEOUT_SEC = 7200


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _assign_free_port(current_port: int) -> int:
    """Return a per-session free ephemeral port for the persistent server."""
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
    """Inspect the materialized YAML for server_lifecycle eligibility."""
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

    from .benchmark_backend import resolve_backend

    refusal = resolve_backend().lifecycle_refusal(bench, info["framework"])
    if refusal is not None:
        info["reason"] = refusal
        return info

    from ._multi_node_env import is_multi_node

    if is_multi_node():
        info["reason"] = "multi-node (server_lifecycle is local-only)"
        return info

    profiler_on = bool((bench.get("profiler") or {}).get("torch_profiler", {}).get("enabled"))
    if profiler_on:
        info["reason"] = "torch_profiler enabled (incompatible with reuse)"
        return info

    # Use a per-session free ephemeral port for the persistent server instead of a fixed default: a stale/leaked
    # server from a prior baseline attempt (round 1 uses cleanup=false and leaves the server running) or a co-tenant
    # job holding the fixed port makes Magpie abort every warmup with "Reuse metadata mismatch ... server on PORT=...
    # is incompatible", so the baseline never records an accuracy and the run wrongly stops with
    # baseline_accuracy_failed.
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
    """Mutate ``bench`` in place to enable the server_lifecycle protocol."""
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
    """Best-effort teardown of a persistent server left by a lifecycle round."""
    pid_file = lifecycle_pid_file(pid_dir, framework, port)
    meta_file = lifecycle_meta_file(pid_dir, framework, port)
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
            _terminate_group(pgid)
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


def _terminate_group(pgid: int) -> bool:
    """SIGTERM a process group, then SIGKILL it if any member outlives the grace; return whether it is gone."""
    signal_group(pgid, signal.SIGTERM, what="_server_lifecycle")
    if _group_gone_within(pgid, TERM_GRACE_SEC):
        return True
    signal_group(pgid, signal.SIGKILL, what="_server_lifecycle")
    return _group_gone_within(pgid, _KILL_CONFIRM_SECONDS)


def _group_has_live_member(pgid: int) -> bool:
    """Whether ``pgid`` still has a member that is not a zombie.

    A server re-parented to an init that never waits stays a zombie once it exits. A zombie holds no memory and takes
    no signal, so it is not a survivor.
    """
    return any(state != "Z" for _pid, _start, state in group_members(pgid))


def _group_gone_within(pgid: int, window_sec: float) -> bool:
    """Wait up to ``window_sec`` for ``pgid`` to have no live member; return whether it got there."""
    deadline = time.monotonic() + window_sec
    while _group_has_live_member(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _pid_cmdline(pid: int) -> str:
    """Return ``/proc/<pid>/cmdline`` as a space-joined string, or ``\"\"``."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _looks_like_server_process(pid: int) -> bool:
    """Return whether ``pid``'s cmdline matches a Hyperloom serving process."""
    cmdline = _pid_cmdline(pid)
    if not cmdline:
        return False
    return any(marker in cmdline for marker in _SERVER_CMDLINE_MARKERS)


def _process_group_looks_like_server(pgid: int) -> bool:
    """Return whether any process in ``pgid`` still looks like a serving process."""
    return any(_looks_like_server_process(pid) for pid, _start, _state in group_members(pgid))


class ServerReap(NamedTuple):
    """What reaping the servers a set of pidfiles names did."""

    #: Recorded pids whose process group was signalled.
    reaped: list[int]
    #: The subset of ``reaped`` whose group still had a live member after SIGKILL.
    survivors: list[int]


def reap_orphaned_servers(session_dir: Path | str) -> list[int]:
    """Reap serving processes orphaned by a prior monitor-process death, or by a run that simply ended with a server up.

    Scans only the current session's ``runs/`` pidfiles and reaps each pid whose cmdline still matches a serving
    process (SIGTERM -> grace -> SIGKILL on the group), so a co-located session's server and a recycled pid are never
    touched.
    """
    return reap_recorded_servers(Path(session_dir) / "runs").reaped


def stop_agentx_servers(root: Path | str) -> list[int]:
    """Stop any server an AgentX client under ``root`` booted and left running; return the pids alive after SIGKILL.

    Its :data:`AGENTX_SERVER_PIDFILE` is the only handle on that server once the round's process tree is gone.
    """
    return reap_recorded_servers(root, AGENTX_SERVER_PIDFILE).survivors


def reap_recorded_servers(root: Path | str, pidfile_glob: str = "*.pid") -> ServerReap:
    """Reap every serving process still up that a pidfile matching ``pidfile_glob`` under ``root`` names.

    Nothing is signalled on the recorded pid alone: a live pid must still look like a serving process, and a gone
    leader's group is signalled only while a live member of it does, so a recycled pid or pgid is left untouched. A
    pidfile is removed once nothing it names is left; a survivor's stays for the next reap.
    """
    reap = ServerReap(reaped=[], survivors=[])
    root = Path(root)
    if os.name != "posix" or not root.is_dir():
        return reap

    own_pgid = os.getpgrp()
    for pid_file in sorted(root.rglob(pidfile_glob)):
        recorded = _read_pidfile(pid_file)
        if recorded is None:
            continue
        server_pid, recorded_pgid = recorded

        # A leader that exited but was never waited on is a zombie: gone for this purpose, while its ranks may not be.
        pid_alive = running(server_pid)
        server_pgid = recorded_pgid
        if pid_alive:
            with suppress(OSError):
                server_pgid = os.getpgid(server_pid)

        if pid_alive and not _looks_like_server_process(server_pid):
            # Live pid but not one of our servers (pid reuse): do not touch the process; leave the pidfile for a later
            # re-evaluation.
            log.info(
                "orphan-reaper: pid=%d from %s no longer looks like a server (cmdline mismatch); leaving it untouched",
                server_pid,
                pid_file,
            )
            continue
        if not pid_alive and not _group_has_live_member(server_pgid):
            # Stale pidfile from a fully-exited server tree; just clean it up.
            _unlink_quietly(pid_file)
            _unlink_quietly(pid_file.with_suffix(".json"))
            continue
        if not pid_alive and not _process_group_looks_like_server(server_pgid):
            # The original leader is gone, and the remaining/reused pgid has no server-looking member; do not risk
            # signalling an unrelated group.
            log.info(
                "orphan-reaper: pid=%d from %s is gone and pgid=%d has no server-looking member; leaving it untouched",
                server_pid,
                pid_file,
                server_pgid,
            )
            continue
        if server_pgid == own_pgid:
            log.error(
                "orphan-reaper: pid=%d from %s is in this process's own group pgid=%d; refusing to signal it",
                server_pid,
                pid_file,
                server_pgid,
            )
            continue

        reap.reaped.append(server_pid)
        if not _terminate_group(server_pgid):
            reap.survivors.append(server_pid)
            log.error(
                "orphan-reaper: server pid=%d pgid=%d from %s is still alive after SIGKILL; keeping its pidfile",
                server_pid,
                server_pgid,
                pid_file,
            )
            continue
        log.warning(
            "orphan-reaper: reaped leftover server pid=%d pgid=%d from %s",
            server_pid,
            server_pgid,
            pid_file,
        )
        _unlink_quietly(pid_file)
        _unlink_quietly(pid_file.with_suffix(".json"))

    return reap


def _read_pidfile(pid_file: Path) -> tuple[int, int] | None:
    """Return the ``(pid, pgid)`` a server pidfile records, or ``None``; an empty or malformed pidfile is removed."""
    try:
        parts = pid_file.read_text(encoding="utf-8").split()
    except OSError:
        return None
    if not parts:
        _unlink_quietly(pid_file)
        return None
    try:
        server_pid = int(parts[0])
        recorded_pgid = int(parts[1]) if len(parts) > 1 else server_pid
    except ValueError:
        _unlink_quietly(pid_file)
        return None
    return server_pid, recorded_pgid


def _unlink_quietly(path: Path) -> None:
    """Best-effort ``unlink`` that never raises."""
    try:
        path.unlink()
    except OSError:
        pass


class GpuHandoff(NamedTuple):
    """Whether the GPUs a hand-off names are free, and what freeing them took."""

    #: This session's leftover servers that were stopped first.
    reaped: list[int]
    #: ``(card, used_mib, total_mib)`` for every named card still holding VRAM at the deadline.
    busy: list[tuple[int, float, float]]
    #: The serving process groups still up when ``busy`` is non-empty, one line each, to say who holds the cards.
    holders: list[str]
    #: Why the cards could not be checked at all; empty when they were.
    unverified: str


def free_gpus_for_handoff(
    session_dir: Path | str,
    cards: Sequence[int] | None,
    *,
    timeout_sec: float,
    poll_sec: float = _GPU_FREE_POLL_SECONDS,
) -> GpuHandoff:
    """Stop this session's leftover servers, then wait up to ``timeout_sec`` for ``cards`` to hold no VRAM.

    For a caller holding every serving lane: no round of this session can be serving then, so a server its pidfiles
    still name was left behind by a teardown that never reached it. The reap runs whatever VRAM reads, since a
    leftover still loading weights holds none yet.

    Args:
        session_dir: The session whose ``runs/`` pidfiles name the leftovers.
        cards: Physical card indices as rocm-smi numbers them; ``None`` when the caller could not resolve them, which
            still reaps but verifies nothing.
        timeout_sec: How long the cards are given to drain.
        poll_sec: How often VRAM is re-read meanwhile.

    Returns:
        What was reaped, and either which cards were still busy at the deadline and who holds them, or why the cards
        could not be checked.
    """
    reaped = reap_orphaned_servers(session_dir)
    if cards is None:
        return GpuHandoff(reaped, [], [], "the cards could not be resolved to physical GPU indices")
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        usage = rocm_smi.gpu_vram_usage()
        if usage is None:
            return GpuHandoff(reaped, [], [], "rocm-smi could not report VRAM")
        if any(not 0 <= card < len(usage) for card in cards):
            return GpuHandoff(
                reaped, [], [], f"rocm-smi reports {len(usage)} GPU(s), which does not cover cards {list(cards)}"
            )
        busy = [
            (card, gpu.used_mib, gpu.total_mib)
            for card, gpu in enumerate(usage)
            if card in cards and rocm_smi.vram_busy(gpu)
        ]
        if not busy:
            return GpuHandoff(reaped, [], [], "")
        if time.monotonic() >= deadline:
            return GpuHandoff(reaped, busy, _serving_process_groups(), "")
        time.sleep(poll_sec)


def _serving_process_groups() -> list[str]:
    """Name up to :data:`_HOLDERS_REPORTED` live process groups that look like serving processes, one line each.

    A line names the program, not the flags it was started with: those can carry a credential.
    """
    try:
        pids = sorted(int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit())
    except OSError:
        return []
    groups: dict[int, str] = {}
    for pid in pids:
        if not _looks_like_server_process(pid):
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue
        if pgid not in groups:
            groups[pgid] = redact_secret_values(f"pgid={pgid} {_program_of(pid)[:160]}")
        if len(groups) >= _HOLDERS_REPORTED:
            break
    return list(groups.values())


def _program_of(pid: int) -> str:
    """``pid``'s command line up to its first ``--`` flag, e.g. ``python3 -m sglang.launch_server``."""
    words = _pid_cmdline(pid).split()
    first_flag = next((i for i, word in enumerate(words) if i and word.startswith("--")), len(words))
    return " ".join(words[:first_flag])
