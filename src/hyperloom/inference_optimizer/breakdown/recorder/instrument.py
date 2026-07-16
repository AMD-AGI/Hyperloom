# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time instrumentation for ``session_breakdown.json``.

These helpers are called from the producing code (the Coordinator's
``SharedState``) to record breakdown facts where they are born, instead of
having the exporter re-walk artifacts later.

Every helper is best-effort: all failures are swallowed (logged at debug).
Payloads are shaped to the matching ``schema.py`` TypedDict.

Coverage in this module (state-owned sections; single owner = Coordinator):

* ``session`` / ``workload`` / ``final`` / ``explore_search`` / ``sweep``
  -- singletons snapshotted from in-memory state at each persist.
* ``optimization_stack`` / ``roofline`` -- event items keyed by a stable id
  (idempotent).
* ``phase_timeline`` -- one event per recorded action attempt.

File-born sections are produced by other processes/executors and are
instrumented at those sites separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float
from hyperloom.common.jsonio import read_json
from hyperloom.common.timeutil import now_iso

log = logging.getLogger(__name__)

PRODUCER_COORDINATOR = "coordinator"
PRODUCER_KERNEL_AGENT = "kernel-agent"

# kernel-agent backend -> invocation section.
_GEAK_BACKENDS = frozenset({"geak"})
_FORGE_BACKENDS = frozenset({"forge"})

_FAILED_STATUSES = frozenset({"failed", "error", "crashed", "timeout"})


def _now_iso_safe() -> str:
    """Return the current UTC time as an ISO-8601 string (``""`` on failure).

    Returns:
        The current UTC time as a microsecond-precision ISO-8601 string, or
        ``""`` if the clock read fails.
    """
    try:
        return now_iso(timespec="microseconds")
    except Exception:  # noqa: BLE001
        return ""


def _recorder(session_dir: Path | str, producer: str):
    """Return the process-cached recorder for ``session_dir`` and ``producer``.

    Args:
        session_dir (Path | str): the session directory backing the recorder.
        producer (str): the breakdown producer label owning the fragments.

    Returns:
        The process-cached recorder for the ``(session_dir, producer)`` pair.
    """
    from .recorder import get_recorder

    return get_recorder(session_dir, producer=producer)


def _rel(path: Path, session_dir: Path | str) -> str:
    """Render ``path`` relative to ``session_dir`` (falls back to str).

    Args:
        path (Path): the path to render.
        session_dir (Path | str): the session directory to relativize against.

    Returns:
        str: ``path`` relative to ``session_dir``, or the plain string form when
            it is not under the session dir.
    """
    try:
        return str(Path(path).relative_to(Path(session_dir)))
    except (ValueError, TypeError):
        return str(path)


def record_phase_event(
    session_dir: Path | str | None,
    *,
    action: str,
    entry: dict[str, Any],
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one ``phase_timeline`` event from a ``record_action_attempt`` entry.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        action (str): the action name the event is keyed by.
        entry (dict[str, Any]): the ``record_action_attempt`` entry to project
            into a phase_timeline payload.
        producer (str): the breakdown producer label (defaults to the
            Coordinator).
    """
    if not session_dir or not isinstance(entry, dict):
        return
    try:
        task_id = str(entry.get("task_id") or "")
        payload = {
            "ts": str(entry.get("ts") or ""),
            "action": str(action or ""),
            "task_id": task_id,
            "status": str(entry.get("status") or ""),
            "decision": str(entry.get("decision") or ""),
            "key_metric": to_float(entry.get("key_metric")),
            "key_metric_kind": entry.get("key_metric_kind"),
            "workspace": entry.get("workspace"),
            "error_class": entry.get("error_class"),
            "extras": dict(entry.get("extras") or {}),
        }
        # Stable key per (action, task) so a re-recorded attempt overwrites.
        key = f"{action}-{task_id}" if task_id else None
        _recorder(session_dir, producer).record_item(
            "phase_timeline",
            payload,
            key=key,
        )
    except Exception:  # noqa: BLE001
        log.debug("record_phase_event failed", exc_info=True)


def snapshot_state_sections(
    session_dir: Path | str | None,
    state: Any,
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Snapshot every state-owned breakdown section from a live ``SharedState``.

    Singletons overwrite the producer's own file; event-stream items are keyed
    by a stable id so repeated snapshots are idempotent. Best-effort per
    section: one failing section never blocks the others.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        state (Any): the live ``SharedState`` snapshotted into each section.
        producer (str): the breakdown producer label (defaults to the
            Coordinator).
    """
    if not session_dir or state is None:
        return
    rec = None
    try:
        rec = _recorder(session_dir, producer)
    except Exception:  # noqa: BLE001
        log.debug("recorder unavailable", exc_info=True)
        return

    for name, fn in (
        ("session", _snapshot_session),
        ("workload", _snapshot_workload),
        ("final", _snapshot_final),
        ("explore_search", _snapshot_explore_search),
        ("sweep", _snapshot_sweep),
        ("optimization_stack", _snapshot_optimization_stack),
        ("roofline", _snapshot_roofline),
    ):
        try:
            fn(rec, state)
        except Exception:  # noqa: BLE001
            log.debug("snapshot section %s failed", name, exc_info=True)


def _snapshot_session(rec, st: Any) -> None:
    """Snapshot the ``session`` singleton from ``st`` (no-op without a session id).

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    session_id = str(getattr(st, "session_id", "") or "")
    if not session_id:
        return
    rec.record_singleton(
        "session",
        {
            "session_id": session_id,
            "claw_session_id": getattr(st, "claw_session_id", "") or "",
            "sandbox_user_id": getattr(st, "sandbox_user_id", "") or "",
            "start_ts": str(getattr(st, "start_ts", "") or ""),
            "stop_reason": str(getattr(st, "stop_reason", "") or ""),
            "max_minutes": int(getattr(st, "max_minutes", 0) or 0),
            "tick_count": int(getattr(st, "tick", 0) or 0),
            "phase": str(getattr(st, "phase", "") or ""),
        },
    )


def _snapshot_workload(rec, st: Any) -> None:
    """Snapshot the ``workload`` singleton from ``st``.

    A no-op when neither a framework nor a model is set.

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    framework = str(getattr(st, "framework", "") or "")
    model = str(getattr(st, "model_name", "") or getattr(st, "model_path", "") or "")
    if not framework and not model:
        return
    rec.record_singleton(
        "workload",
        {
            "framework": framework,
            "model_name": str(getattr(st, "model_name", "") or ""),
            "model_path": str(getattr(st, "model_path", "") or ""),
            "model_class": str(getattr(st, "model_class", "") or ""),
            "gpu_type": str(getattr(st, "gpu_type", "") or ""),
            "tp": int(getattr(st, "tp", 0) or 0),
            "ep": int(getattr(st, "ep", 0) or 0),
            "precision": str(getattr(st, "precision", "") or ""),
            "conc": int(getattr(st, "conc", 0) or 0),
            "isl": int(getattr(st, "isl", 0) or 0),
            "osl": int(getattr(st, "osl", 0) or 0),
            "max_model_len": int(getattr(st, "max_model_len", 0) or 0),
        },
    )


def _snapshot_final(rec, st: Any) -> None:
    """Snapshot the ``final`` singleton (current best + cumulative gains) from ``st``.

    A no-op when there is neither a current best nor an optimization stack.

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    cb = getattr(st, "current_best", None) or {}
    stack = getattr(st, "optimization_stack", None) or []
    if not cb and not stack:
        return
    from ... import framework_registry

    framework = str(getattr(st, "framework", "") or "")
    tput = to_float(cb.get("tput"))
    # Latency is the primary result for scriptable/diffusion (xDiT) image models
    # (throughput_tok_s_per_gpu is only ``1 / latency`` there and misleading as a
    # headline). Emit e2el/ttft alongside the throughput-unit + primary-metric
    # markers so consumers pick the right result field per framework. e2el falls
    # back to the tput-derived per-image latency when no measured value exists.
    e2el = to_float(cb.get("e2el_mean_ms"))
    if e2el is None and framework_registry.is_scriptable(framework) and tput is not None and tput > 0:
        derived = framework_registry.primary_metric_value(framework, tput)
        e2el = round(float(derived), 4) if derived is not None and derived > 0 else None
    rec.record_singleton(
        "final",
        {
            "current_best_action": str(cb.get("action") or ""),
            "throughput_tok_s_per_gpu": tput,
            "throughput_unit": framework_registry.throughput_unit(framework),
            "primary_metric": framework_registry.primary_metric_name(framework),
            "e2el_mean_ms": e2el,
            "ttft_mean_ms": to_float(cb.get("ttft_mean_ms")),
            "cumulative_gain_pct_validated": to_float(getattr(st, "cumulative_gain_validated", 0.0)) or 0.0,
            "cumulative_gain_pct_per_round_sum": to_float(getattr(st, "cumulative_gain", 0.0)) or 0.0,
            "validated_ts": str(getattr(st, "cumulative_gain_validated_ts", "") or ""),
            "stack_len": len(stack),
            "extra_server_args": str(cb.get("extra_server_args") or ""),
            "extra_envs": dict(cb.get("extra_envs") or {}),
        },
    )


def _snapshot_explore_search(rec, st: Any) -> None:
    """Snapshot the ``explore_search`` singleton from ``st`` (no-op when empty).

    Augments the base search dict with winner history, no-promote streak,
    discovered flags, and synergy/backend-winner history pulled from ``st``.

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    search = dict(getattr(st, "explore_search", None) or {})
    if not search:
        return
    search["winner_history"] = list(getattr(st, "params_winner_history", None) or [])
    search["no_promote_streak"] = int(getattr(st, "params_no_promote_streak", 0) or 0)
    search["discovered_flags"] = dict(getattr(st, "discovered_flags", None) or {})
    search["synergy_attempted"] = list(getattr(st, "synergy_attempted", None) or [])
    search["backend_winners_history"] = list(getattr(st, "backend_winners_history", None) or [])
    rec.record_singleton("explore_search", search)


def _snapshot_sweep(rec, st: Any) -> None:
    """Snapshot the ``sweep`` singleton from ``st.last_sweep`` (no-op when empty).

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    last_sweep = dict(getattr(st, "last_sweep", None) or {})
    if not last_sweep:
        return
    rec.record_singleton("sweep", last_sweep)


def _snapshot_optimization_stack(rec, st: Any) -> None:
    """Snapshot each ``optimization_stack`` entry from ``st`` as a keyed item.

    Backfills a missing per-entry ``gain_pct`` from ``st.gain_per_stack_entry``
    when available; each item is keyed by its stack index for idempotency.

    Args:
        rec: the recorder used to write the items.
        st (Any): the live ``SharedState`` to snapshot.
    """
    stack = getattr(st, "optimization_stack", None) or []
    gains = getattr(st, "gain_per_stack_entry", None) or []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        payload = dict(entry)
        if payload.get("gain_pct") is None and i < len(gains):
            payload["gain_pct"] = to_float(gains[i])
        rec.record_item("optimization_stack", payload, key=str(i))


def _snapshot_roofline(rec, st: Any) -> None:
    """Snapshot each ``roofline`` snapshot from ``st`` as a keyed item.

    Each item is keyed by its snapshot id (falling back to the list index) for
    idempotency.

    Args:
        rec: the recorder used to write the items.
        st (Any): the live ``SharedState`` to snapshot.
    """
    snapshots = getattr(st, "roofline_snapshots", None) or []
    for idx, snap in enumerate(snapshots):
        if not isinstance(snap, dict):
            continue
        sid = str(snap.get("snapshot_id") or snap.get("id") or idx)
        rec.record_item("roofline", snap, key=sid)


def _best_attempt_id(
    attempts: list[Any],
    verification: dict[str, Any],
) -> str:
    """Pick the adopted attempt id: verification hint, else highest speedup.

    Mirrors the collector's selection so the kernel-level decision lands on the
    same attempt the breakdown would attribute it to.

    Args:
        attempts (list[Any]): the per-backend attempt rows.
        verification (dict[str, Any]): the verification block carrying the
            ``best_attempt_id`` / ``best_backend`` hints.

    Returns:
        str: the adopted attempt id (verification hint, else highest speedup),
            or ``""`` when there are no attempt rows.
    """
    rows = [a for a in attempts if isinstance(a, dict)]
    if not rows:
        return ""
    want_id = str(verification.get("best_attempt_id") or "")
    if want_id:
        return want_id
    want_backend = str(verification.get("best_backend") or "").lower()
    candidates = rows
    if want_backend:
        backend_rows = [a for a in rows if str(a.get("backend") or "").lower() == want_backend]
        if backend_rows:
            candidates = backend_rows

    def _spd(a: dict[str, Any]) -> float:
        """Return an attempt's micro/plain speedup (``-inf`` when absent).

        Args:
            a: An attempt record mapping.

        Returns:
            The attempt's ``micro_speedup`` (or ``speedup``) as a float, or
            ``-inf`` when neither is present.
        """
        v = to_float(a.get("micro_speedup") or a.get("speedup"))
        return v if v is not None else float("-inf")

    best = max(candidates, key=_spd)
    return str(best.get("attempt_id") or best.get("id") or "")


def _invocation_section(backend: str) -> str | None:
    """Map a kernel-agent backend to its invocation section name.

    Args:
        backend (str): the backend name (geak / forge / ...).

    Returns:
        str | None: the matching invocation section, or ``None`` when the backend
            has no invocation lane.
    """
    b = str(backend or "").lower()
    if b in _GEAK_BACKENDS:
        return "geak_invocations"
    if b in _FORGE_BACKENDS:
        return "forge_invocations"
    return None


def record_kernel_invocations(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record kernel backend invocations from an in-process kernel-agent result.

    Reads ``result['attempts']`` (per-backend ladder) so backend-level failures
    are captured even when the kernel-agent crashed before persisting the
    on-disk source. When the whole invocation failed before any backend ran
    (pre-dispatch gating), a single ``FAILED`` marker is recorded so the failure
    stays visible in the invocation view.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        result (dict[str, Any]): the in-process kernel-agent result carrying the
            per-backend ``attempts`` ladder, verification, and proposal.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not isinstance(result, dict):
        return
    try:
        rec = _recorder(session_dir, producer)
        kid = str(result.get("kernel_id") or "")
        run_id = str(result.get("run_id") or result.get("session_id") or "")
        attempts = result.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        kernel_decision = str(proposal.get("decision") or "").upper()
        best_attempt_id = _best_attempt_id(attempts, verification)

        recorded_any = False
        for att in attempts:
            if not isinstance(att, dict):
                continue
            backend = str(att.get("backend") or "").lower()
            section = _invocation_section(backend)
            if section is None:
                continue
            status = str(att.get("status") or "").lower()
            decision = str(att.get("decision") or "").upper()
            if not decision and status in _FAILED_STATUSES:
                decision = "FAILED"
            attempt_id = str(att.get("attempt_id") or att.get("id") or "")
            # Stamp the kernel-level decision onto the adopted (best) attempt.
            if kernel_decision and attempt_id and attempt_id == best_attempt_id:
                decision = kernel_decision
            optimized = att.get("optimized_path") or att.get("optimized_file")
            payload = {
                "kernel_id": kid,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "ts": str(att.get("ts") or att.get("started_at") or ""),
                "backend": backend,
                "decision": decision,
                "status": status,
                "micro_speedup": to_float(att.get("micro_speedup") or att.get("speedup")),
                "optimized_files": [str(optimized)] if optimized else [],
                "error": att.get("error") or att.get("error_message"),
            }
            key = attempt_id or f"{kid}-{backend}"
            rec.record_item(section, payload, key=key)
            recorded_any = True

        if recorded_any:
            return

        # No per-backend attempts: capture a pre-dispatch / infra failure so
        # the invocation view still shows it (root cause of invisible failures).
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        decision = str((result.get("proposal") or {}).get("decision") or "").upper()
        failed = status in _FAILED_STATUSES or (decision == "REVERT" and bool(err_class))
        if not failed:
            return
        backend = str(result.get("backend") or "").lower()
        section = _invocation_section(backend)
        if section is None:
            # The backend could not be determined; do not fabricate a GEAK
            # invocation. The failure stays visible via the kernel_dispatch /
            # kernel_backend_result journey lanes.
            return
        payload = {
            "kernel_id": kid,
            "attempt_id": "",
            "run_id": run_id,
            "backend": backend,
            "decision": "FAILED",
            "status": status or "failed",
            "error": result.get("error") or err_class or None,
            "error_class": err_class or None,
            # Distinguishes a pre-dispatch gating failure from a backend that ran and failed.
            "pre_dispatch_failure": True,
        }
        rec.record_item(section, payload, key=f"{kid}-predispatch" if kid else None)
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_invocations failed", exc_info=True)


def _to_bool(value: Any) -> bool | None:
    """Coerce a loosely-typed truthy/falsy value to ``bool``.

    Args:
        value (Any): the value to interpret (a bool, or a string like
            ``"true"`` / ``"failed"`` / ``"ok"``).

    Returns:
        bool | None: the interpreted boolean, or ``None`` when ``value`` is
            None or not a recognized truthy/falsy token.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "pass", "passed", "ok"):
        return True
    if s in ("false", "0", "no", "fail", "failed"):
        return False
    return None


# Cache of resolved tool metadata, keyed by ``tool:root_dir`` (one-shot probe per key).
_TOOL_META_CACHE: dict[str, dict[str, Any]] = {}

# Per-tool "authoritative version" recipe. ``root_env`` holds the install root
# (used for the commit probe and git-based version strategies). ``version``
# picks how the human version is derived:
#   * "git_describe" -> ``git describe --tags --always --dirty`` of the root
#   * "git_short"    -> ``git rev-parse --short HEAD`` of the root (== commit)
#   * ("cmd", argv)  -> first line of ``argv --version`` style CLI output
#   * ("dist", names)-> importlib.metadata version of the first matching dist
_TOOL_PROVENANCE: dict[str, dict[str, Any]] = {
    "tracelens": {"root_env": "TRACELENS_ROOT", "version": "git_describe"},
    # The whole-pipeline GEAK e2e optimizer. Its checkout lives under $GEAK_ROOT
    # and its version is that repo's git SHA.
    "geak": {"root_env": "GEAK_ROOT", "version": "git_short"},
        # forge (Kernel-Forge autonomous loop) locates its repo via $FORGE_PATH.
    "forge": {"root_env": "FORGE_PATH", "version": "git_short"},
    "claude": {"root_env": "", "version": ("cmd", ("claude", "--version"))},
    "codex": {"root_env": "", "version": ("cmd", ("codex", "--version"))},
    "inferencex": {"root_env": "INFERENCEX_PATH", "version": "git_short"},
    "kernel_agent": {"root_env": "HYPERLOOM_KERNEL_AGENT_ROOT", "version": "git_short"},
}


def _run_first_line(argv: list[str]) -> str:
    """Run ``argv`` and return the trimmed first output line (never raises).

    Args:
        argv (list[str]): the command argv to run.

    Returns:
        str: the trimmed first line of output (capped at 120 chars), or ``""``
            on failure / non-zero exit.
    """
    import subprocess  # local: keep module import cost off the common path

    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return ""
    if out.returncode != 0:
        return ""
    text = (out.stdout or "").strip() or (out.stderr or "").strip()
    return text.splitlines()[0].strip()[:120] if text else ""


def _git_short_commit(root: Path) -> str:
    """Best-effort ``git rev-parse --short HEAD`` for ``root`` (never raises).

    Args:
        root (Path): the repo root to inspect.

    Returns:
        str: the short commit hash, or ``""`` when it cannot be resolved.
    """
    return _run_first_line(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
    )


def _git_describe(root: Path) -> str:
    """Best-effort ``git describe --tags --always --dirty`` (never raises).

    Args:
        root (Path): the repo root to inspect.

    Returns:
        str: the ``git describe`` output, or ``""`` when it cannot be resolved.
    """
    return _run_first_line(
        ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
    )


def _dist_version(names: tuple[str, ...]) -> str:
    """First resolvable ``importlib.metadata`` version among ``names`` ("" if none).

    Args:
        names (tuple[str, ...]): candidate distribution names to resolve in
            order.

    Returns:
        str: the first resolvable distribution version (rejecting a stale
            ``0.0.0``), or ``""`` when none resolve.
    """
    try:
        from importlib.metadata import version as _dist_ver
    except Exception:  # noqa: BLE001
        return ""
    for name in names:
        try:
            v = str(_dist_ver(name) or "").strip()
        except Exception:  # noqa: BLE001
            continue
        # Reject a stale 0.0.0 masquerade.
        if v and v != "0.0.0":
            return v
    return ""


def _probe_tool_version(strategy: Any, root_dir: str) -> str:
    """Resolve a tool's human version per its provenance ``strategy``.

    Args:
        strategy (Any): the provenance strategy (``"git_describe"`` /
            ``"git_short"`` / a ``("cmd", argv)`` or ``("dist", names)`` tuple).
        root_dir (str): the tool install root for git-based strategies.

    Returns:
        str: the resolved version string, or ``""`` when it cannot be derived.
    """
    try:
        if strategy == "git_describe":
            return _git_describe(Path(root_dir)) if root_dir else ""
        if strategy == "git_short":
            return _git_short_commit(Path(root_dir)) if root_dir else ""
        if isinstance(strategy, tuple) and len(strategy) == 2:
            kind, arg = strategy
            if kind == "cmd":
                return _run_first_line(list(arg))
            if kind == "dist":
                return _dist_version(tuple(arg))
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _tool_metadata(
    tool: str,
    *,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Resolve ``{tool, root_dir, commit, version}`` for an external tool.

    Root resolution: explicit ``root`` > caller ``root_env`` > the tool's
    registered ``root_env``. ``commit`` is a cached ``git rev-parse`` of the
    root. ``version`` is the caller-supplied value, else a cached per-tool probe
    following ``_TOOL_PROVENANCE``. Best-effort: never raises into the optimizer.

    Args:
        tool (str): the external tool name (keys into ``_TOOL_PROVENANCE``).
        root (str | None): an explicit install root, highest precedence.
        root_env (str | None): a caller-supplied env var naming the root.
        version (str | None): a caller-supplied version, preferred over the
            probe.

    Returns:
        dict[str, Any]: the resolved ``{tool, root_dir, commit, version}``
            metadata.
    """
    import os

    key = str(tool or "").lower()
    hint = _TOOL_PROVENANCE.get(key, {})
    root_dir = str(
        root or os.environ.get(root_env or "", "") or os.environ.get(str(hint.get("root_env") or ""), "")
    ).strip()
    cache_key = f"{key}:{root_dir}"
    cached = _TOOL_META_CACHE.get(cache_key)
    if cached is None:
        commit = ""
        if root_dir:
            try:
                if Path(root_dir).is_dir():
                    commit = _git_short_commit(Path(root_dir))
            except Exception:  # noqa: BLE001
                commit = ""
        probed = _probe_tool_version(hint.get("version"), root_dir) if hint else ""
        cached = {
            "tool": tool,
            "root_dir": root_dir,
            "commit": commit,
            "_probed_version": probed,
        }
        _TOOL_META_CACHE[cache_key] = cached
    meta = {
        "tool": cached["tool"],
        "root_dir": cached["root_dir"],
        "commit": cached["commit"],
    }
    meta["version"] = str(version or "") or str(cached.get("_probed_version") or "")
    return meta


def _normalize_hot_kernel(k: dict[str, Any]) -> dict[str, Any]:
    """Project a raw hot-kernel candidate onto the discovery shape.

    Args:
        k (dict[str, Any]): the raw hot-kernel candidate dict.

    Returns:
        dict[str, Any]: the candidate projected onto the normalized discovery
            shape.
    """
    return {
        "kernel_id": str(k.get("kernel_id") or k.get("id") or ""),
        "name": str(k.get("name") or k.get("kernel_name") or ""),
        "gpu_pct": to_float(k.get("gpu_pct") or k.get("gpu_percent")),
        "time_ms": to_float(k.get("time_ms") or k.get("duration_ms")),
        "bound_type": str(k.get("bound_type") or k.get("bottleneck") or ""),
        "arithmetic_intensity": to_float(k.get("arithmetic_intensity")),
        "flops_per_byte": to_float(k.get("flops_per_byte")),
        "efficiency_percent": to_float(k.get("efficiency_percent")),
        "reusable_native_kernel": bool(k.get("reusable_native_kernel") or False),
        "source_file": k.get("source_file"),
        "recommended_backends": list(k.get("recommended_backends") or []),
        "selected_for_optimization": bool(k.get("selected_for_optimization") or False),
    }


def record_kernel_discovery(
    session_dir: Path | str | None,
    *,
    source: str,
    status: str,
    hot_kernels: list[Any] | None = None,
    scan: dict[str, Any] | None = None,
    tool: str | None = None,
    tool_root: str | None = None,
    tool_root_env: str | None = None,
    tool_version: str | None = None,
    duration_sec: Any = None,
    error: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record one hot-kernel discovery run (stage 1 of ``kernel_journey``).

    One item per discovery invocation, keyed by the candidates/report path so a
    re-run with the same artifact overwrites rather than duplicates. Carries the
    full hot-kernel list the run surfaced.

    ``source`` is the discovery *route* label the dashboard groups by. ``tool``
    is the underlying tool whose authoritative version lands in the top-level
    ``versions`` map; it defaults to ``source`` but is decoupled because routes
    can share one toolchain (e.g. ``bypass`` reuses the TraceLens toolchain).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        source (str): the discovery route label the dashboard groups by.
        status (str): the discovery run status.
        hot_kernels (list[Any] | None): the hot-kernel candidates the run
            surfaced.
        scan (dict[str, Any] | None): scan metadata (carries the
            candidates/report path used as the idempotency key).
        tool (str | None): the underlying tool whose version is recorded
            (defaults to ``source``).
        tool_root (str | None): an explicit tool install root.
        tool_root_env (str | None): an env var naming the tool root.
        tool_version (str | None): a caller-supplied tool version.
        duration_sec (Any): the run duration in seconds.
        error (str | None): an error string when the run failed.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir:
        return
    try:
        kernels = [_normalize_hot_kernel(k) for k in (hot_kernels or []) if isinstance(k, dict)]
        scan = dict(scan or {})
        payload = {
            "source": str(source or ""),
            "status": str(status or ""),
            "ts": _now_iso_safe(),
            "duration_sec": to_float(duration_sec),
            "scan": scan,
            "hot_kernel_count": len(kernels),
            "hot_kernels": kernels,
            "error": error,
        }
        key = str(scan.get("candidates_path") or scan.get("trace_report_path") or "") or None
        _recorder(session_dir, producer).record_item(
            "kernel_discovery",
            payload,
            key=key,
        )
        # The discovery tool's authoritative version lands in the top-level
        # ``versions`` map, following the underlying ``tool``.
        record_tool_version(
            session_dir,
            tool=(tool or source),
            root=tool_root,
            root_env=tool_root_env,
            version=tool_version,
            producer=producer,
        )
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_discovery failed", exc_info=True)


def record_tool_version(
    session_dir: Path | str | None,
    *,
    tool: str,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record one external tool's authoritative version into ``versions``.

    Idempotent per tool name (last write wins). Resolves ``{tool, root_dir,
    commit, version}`` via the tool provenance registry and spools it as one
    ``versions`` item; the assembler folds the substream into the top-level
    ``versions`` map. Best-effort: never raises into the optimizer.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        tool (str): the external tool name; a falsy value is a no-op.
        root (str | None): an explicit tool install root.
        root_env (str | None): an env var naming the tool root.
        version (str | None): a caller-supplied version, preferred over the
            probe.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not tool:
        return
    try:
        meta = _tool_metadata(
            tool,
            root=root,
            root_env=root_env,
            version=version,
        )
        _recorder(session_dir, producer).record_item(
            "versions",
            meta,
            key=str(tool).lower(),
        )
    except Exception:  # noqa: BLE001
        log.debug("record_tool_version failed", exc_info=True)


def record_kernel_dispatch(
    session_dir: Path | str | None,
    *,
    kernel_id: str,
    dispatched: bool,
    backends: list[str] | None = None,
    skip_reason: str = "",
    orchestration_commit: str = "",
    task_group: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record the dispatch decision for one kernel (stage 2 of ``kernel_journey``).

    Idempotent per ``kernel_id`` (last decision wins). ``dispatched`` is False
    for kernels gated out before any backend ran, with ``skip_reason`` holding
    the gate (non_reusable_kernel / missing_source / budget_exhausted / ...).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        kernel_id (str): the kernel id the decision is keyed by; a falsy value
            is a no-op.
        dispatched (bool): whether the kernel was dispatched to a backend.
        backends (list[str] | None): the backends the kernel was dispatched to.
        skip_reason (str): the gate that blocked dispatch when ``dispatched`` is
            False.
        orchestration_commit (str): the orchestration commit at dispatch time.
        task_group (str | None): the task group label.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not kernel_id:
        return
    try:
        payload = {
            "kernel_id": str(kernel_id),
            "dispatched": bool(dispatched),
            "backends": [str(b) for b in (backends or [])],
            "skip_reason": str(skip_reason or ""),
            "orchestration_commit": str(orchestration_commit or ""),
            "task_group": task_group,
            "ts": _now_iso_safe(),
        }
        _recorder(session_dir, producer).record_item(
            "kernel_dispatch",
            payload,
            key=str(kernel_id),
        )
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_dispatch failed", exc_info=True)


def record_kernel_backend_result(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record per-backend attempts for one kernel (stage 3 of ``kernel_journey``).

    One item per attempt, keyed by ``attempt_id`` (falls back to
    ``run_id-backend``) so retries across runs are preserved rather than
    collapsed. Mirrors the attempt ladder in ``result['attempts']`` and carries
    the per-attempt timing + tool metadata when the kernel-agent surfaced them.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        result (dict[str, Any]): the kernel-agent result carrying the
            per-backend ``attempts`` ladder, verification, and metadata.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not isinstance(result, dict):
        return
    try:
        rec = _recorder(session_dir, producer)
        kid = str(result.get("kernel_id") or "")
        run_id = str(result.get("run_id") or result.get("session_id") or "")
        attempts = result.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        result_meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        # The kernel-level micro_speedup (best across attempts) lives in
        # ``verification``; stamp it onto the adopted (best) attempt.
        verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
        best_attempt_id = _best_attempt_id(attempts, verification)
        kernel_micro_speedup = to_float(verification.get("micro_speedup"))
        recorded_any = False
        for att in attempts:
            if not isinstance(att, dict):
                continue
            backend = str(att.get("backend") or "").lower()
            attempt_id = str(att.get("attempt_id") or att.get("id") or "")
            optimized = att.get("optimized_path") or att.get("optimized_file")
            att_meta = att.get("metadata") if isinstance(att.get("metadata"), dict) else {}
            micro_speedup = to_float(att.get("micro_speedup") or att.get("speedup"))
            if (
                micro_speedup is None
                and kernel_micro_speedup is not None
                and attempt_id
                and attempt_id == best_attempt_id
            ):
                micro_speedup = kernel_micro_speedup
            payload = {
                "kernel_id": kid,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "backend": backend,
                "model": att.get("model"),
                "ts": str(att.get("ts") or att.get("started_at") or att.get("created_at") or ""),
                "status": str(att.get("status") or "").lower(),
                "decision": str(att.get("decision") or "").upper(),
                "micro_speedup": micro_speedup,
                "compile_passed": _to_bool(att.get("compile_passed")),
                "correctness_passed": _to_bool(att.get("correctness_passed")),
                "optimized_files": [str(optimized)] if optimized else [],
                "error": att.get("error") or att.get("error_message"),
                "error_class": str(att.get("error_type") or "") or None,
                "duration_sec": to_float(att.get("duration_sec") or att.get("elapsed_sec") or att.get("elapsed_s")),
            }
            key = attempt_id or (f"{run_id}-{backend}" if run_id else None)
            rec.record_item("kernel_backend_result", payload, key=key)
            recorded_any = True
            # The backend's authoritative version lands in the top-level ``versions`` map.
            if backend:
                record_tool_version(
                    session_dir,
                    tool=backend,
                    root=str(att_meta.get("root_dir") or result_meta.get("root_dir") or "") or None,
                    version=str(att_meta.get("version") or result_meta.get("version") or "") or None,
                    producer=producer,
                )

        if recorded_any or not kid:
            return

        # No per-backend attempts: capture a pre-dispatch / infra failure as a
        # synthetic FAILED attempt so kernel_journey shows the failure too.
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        decision = str((result.get("proposal") or {}).get("decision") or "").upper()
        failed = status in _FAILED_STATUSES or (decision == "REVERT" and bool(err_class))
        if not failed:
            return
        # Never default an unattributable failure to GEAK; record it as
        # "unknown" so GEAK's failure count is not inflated.
        backend = str(result.get("backend") or "").lower() or "unknown"
        payload = {
            "kernel_id": kid,
            "attempt_id": "",
            "run_id": run_id,
            "backend": backend,
            "model": None,
            "ts": _now_iso_safe(),
            "status": status or "failed",
            "decision": "FAILED",
            "micro_speedup": None,
            "compile_passed": None,
            "correctness_passed": None,
            "optimized_files": [],
            "error": result.get("error") or err_class or None,
            "error_class": err_class or None,
            "duration_sec": None,
            # Distinguishes a pre-dispatch gating failure from a backend that ran and failed.
            "pre_dispatch_failure": True,
        }
        rec.record_item(
            "kernel_backend_result",
            payload,
            key=f"{kid}-predispatch",
        )
        if backend != "unknown":
            record_tool_version(
                session_dir,
                tool=backend,
                root=str(result_meta.get("root_dir") or "") or None,
                version=str(result_meta.get("version") or "") or None,
                producer=producer,
            )
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_backend_result failed", exc_info=True)


def record_kernel_e2e(
    session_dir: Path | str | None,
    *,
    kernel_id: str,
    integrated: bool,
    e2e_gain_pct: Any = None,
    validated: bool | None = None,
    decision: str = "",
    patch_path: str | None = None,
    target_file: str | None = None,
    extra_server_args: str = "",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record the end-to-end integrate outcome for one kernel (stage 4).

    Idempotent per ``kernel_id``. ``e2e_gain_pct`` is the validated end-to-end
    gain at integrate (negative => regressed and reverted).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        kernel_id (str): the kernel id the outcome is keyed by; a falsy value is
            a no-op.
        integrated (bool): whether the kernel change was integrated.
        e2e_gain_pct (Any): the validated end-to-end gain percent at integrate.
        validated (bool | None): whether the gain was validated.
        decision (str): the integrate decision (KEEP / REVERT / ...).
        patch_path (str | None): the applied patch path.
        target_file (str | None): the integrated target file.
        extra_server_args (str): extra server args carried by the change.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not kernel_id:
        return
    try:
        payload = {
            "kernel_id": str(kernel_id),
            "integrated": bool(integrated),
            "e2e_gain_pct": to_float(e2e_gain_pct),
            "validated": bool(validated) if validated is not None else None,
            "decision": str(decision or "").upper(),
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_server_args": str(extra_server_args or ""),
            "ts": _now_iso_safe(),
        }
        _recorder(session_dir, producer).record_item(
            "kernel_e2e",
            payload,
            key=str(kernel_id),
        )
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_e2e failed", exc_info=True)


def record_specialist_round(
    session_dir: Path | str | None,
    entry: dict[str, Any],
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one ``specialist_runs`` round (idempotent by ``round_id``).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        entry (dict[str, Any]): the specialist round entry (keyed by
            ``round_id``); an empty/non-dict value is a no-op.
        producer (str): the breakdown producer label (defaults to the
            Coordinator).
    """
    if not session_dir or not isinstance(entry, dict) or not entry:
        return
    try:
        key = str(entry.get("round_id") or "") or None
        _recorder(session_dir, producer).record_item(
            "specialist_runs",
            dict(entry),
            key=key,
        )
    except Exception:  # noqa: BLE001
        log.debug("record_specialist_round failed", exc_info=True)


def record_critic_iteration(
    session_dir: Path | str | None,
    *,
    iter_n: int,
    review: dict[str, Any] | None,
    emit: dict[str, Any] | None,
    workdir: Path | str | None,
    kb_assess: dict[str, Any] | None = None,
    kb_priors: dict[str, Any] | None = None,
    producer: str = "critic",
) -> None:
    """Record one ``critic_robustness.critic_iterations`` item.

    Recorded per-iteration (idempotent on ``iter_n``) so the critic backend's
    workdir pruning never erases history; payload mirrors
    ``collectors.collect_critic_robustness``.

    ``kb_assess`` / ``kb_priors`` (when provided) carry the per-iteration KB
    integration trace: whether the substrate assess / historical priors were
    used, the request, the response, and whether the final verdict referenced
    them. Omitted from the payload when empty so historical items are unchanged.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        iter_n (int): the critic iteration number (idempotency key).
        review (dict[str, Any] | None): the critic review payload.
        emit (dict[str, Any] | None): the critic emit payload.
        workdir (Path | str | None): the critic backend workdir holding the
            per-iteration artifact files.
        kb_assess (dict[str, Any] | None): the per-iteration substrate KB assess
            trace; omitted when empty.
        kb_priors (dict[str, Any] | None): the per-iteration historical KB
            priors trace; omitted when empty.
        producer (str): the breakdown producer label (defaults to ``critic``).
    """
    if not session_dir:
        return
    try:
        review = review if isinstance(review, dict) else {}
        emit = emit if isinstance(emit, dict) else {}
        wd = Path(workdir) if workdir else None
        payload = {
            "iter": int(iter_n),
            "ts": str(emit.get("ts") or review.get("ts") or ""),
            "topic": str(emit.get("topic") or review.get("topic") or ""),
            "verdict": str(review.get("verdict") or emit.get("verdict") or ""),
            "summary": str(review.get("summary") or emit.get("summary") or "")[:500],
            "request_path": _rel(wd / "request.json", session_dir) if wd else None,
            "judge_bundle_path": _rel(wd / "judge_bundle.json", session_dir) if wd else None,
            "emit_path": _rel(wd / "emit.json", session_dir) if wd else None,
            "review_path": _rel(wd / "review.json", session_dir) if wd else None,
        }
        if isinstance(kb_assess, dict) and kb_assess:
            payload["kb_assess"] = kb_assess
        if isinstance(kb_priors, dict) and kb_priors:
            payload["kb_priors"] = kb_priors
        _recorder(session_dir, producer).record_item(
            "critic_iterations",
            payload,
            key=str(iter_n),
        )
    except Exception:  # noqa: BLE001
        log.debug("record_critic_iteration failed", exc_info=True)


def record_robustness_signal(
    session_dir: Path | str | None,
    *,
    workdir: Path | str | None,
    producer: str = "robustness",
) -> None:
    """Record one ``critic_robustness.robustness_signals`` item.

    Reads ``signal.json`` / ``action.json`` from the just-written ``workdir``
    (idempotent on the workdir name) so the signal is captured before the
    robustness backend prunes old workdirs; payload mirrors the collector.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        workdir (Path | str | None): the just-written robustness workdir holding
            ``signal.json`` / ``action.json`` (idempotency key); a falsy value
            is a no-op.
        producer (str): the breakdown producer label (defaults to
            ``robustness``).
    """
    if not session_dir or not workdir:
        return
    try:
        wd = Path(workdir)
        signal_data = read_json(wd / "signal.json", default={}, require_dict=True)
        action_data = read_json(wd / "action.json", default={}, require_dict=True)
        payload = {
            "ts": str(signal_data.get("ts") or action_data.get("ts") or ""),
            "signal": str(signal_data.get("signal") or signal_data.get("kind") or ""),
            "action": str(action_data.get("action") or action_data.get("kind") or ""),
            "workdir": _rel(wd, session_dir),
        }
        _recorder(session_dir, producer).record_item(
            "robustness_signals",
            payload,
            key=wd.name,
        )
    except Exception:  # noqa: BLE001
        log.debug("record_robustness_signal failed", exc_info=True)


def record_singleton_section(
    session_dir: Path | str | None,
    section: str,
    payload: dict[str, Any],
    *,
    producer: str,
) -> None:
    """Record a producer-owned singleton section (report summaries, etc.).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        section (str): the singleton section name to record.
        payload (dict[str, Any]): the section payload; an empty/non-dict value
            is a no-op.
        producer (str): the breakdown producer label that owns the section.
    """
    if not session_dir or not isinstance(payload, dict) or not payload:
        return
    try:
        _recorder(session_dir, producer).record_singleton(section, payload)
    except Exception:  # noqa: BLE001
        log.debug("record_singleton_section %s failed", section, exc_info=True)


__all__ = [
    "PRODUCER_COORDINATOR",
    "PRODUCER_KERNEL_AGENT",
    "record_critic_iteration",
    "record_kernel_backend_result",
    "record_kernel_discovery",
    "record_kernel_dispatch",
    "record_kernel_e2e",
    "record_kernel_invocations",
    "record_phase_event",
    "record_robustness_signal",
    "record_singleton_section",
    "record_specialist_round",
    "record_tool_version",
    "snapshot_state_sections",
]
