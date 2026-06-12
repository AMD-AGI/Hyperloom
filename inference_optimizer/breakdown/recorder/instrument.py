# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Author-time instrumentation for ``session_breakdown.json``.

These helpers are called from the producing code (the Coordinator's
``SharedState``) to record breakdown facts where they are born, via the
recorder toolkit, instead of having the exporter re-walk artifacts later.

Every helper is best-effort: instrumentation must never break the optimizer,
so all failures are swallowed (logged at debug). Payloads are shaped to the
matching ``schema.py`` TypedDict so assembly stays structure-preserving.

Coverage in this module (state-owned sections; single owner = Coordinator):

* ``session`` / ``workload`` / ``final`` / ``explore_search`` / ``sweep``
  -- singletons snapshotted from in-memory state at each persist.
* ``optimization_stack`` / ``roofline`` -- event items keyed by a stable id
  (idempotent: re-recording overwrites rather than duplicates).
* ``phase_timeline`` -- one event per recorded action attempt.

File-born sections (geak/oob invocations, kernel/conc-sweep report summaries,
critic_robustness, specialist_runs, telemetry, kb_provenance) are produced by
other processes/executors and are instrumented at those sites separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRODUCER_COORDINATOR = "coordinator"
PRODUCER_KERNEL_AGENT = "kernel-agent"

# kernel-agent backend -> invocation section. geak is its own lane; the
# out-of-band LLM backends (claude/codex) share the oob lane.
_GEAK_BACKENDS = frozenset({"geak"})
_OOB_BACKENDS = frozenset({"claude", "codex"})

_FAILED_STATUSES = frozenset({"failed", "error", "crashed", "timeout"})


def _now_iso_safe() -> str:
    from datetime import datetime, timezone

    try:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")
    except Exception:  # noqa: BLE001
        return ""


def _to_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _recorder(session_dir: Path | str, producer: str):
    from .recorder import get_recorder

    return get_recorder(session_dir, producer=producer)


def _rel(path: Path, session_dir: Path | str) -> str:
    """Render ``path`` relative to ``session_dir`` (falls back to str)."""
    try:
        return str(Path(path).relative_to(Path(session_dir)))
    except (ValueError, TypeError):
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record_phase_event(
    session_dir: Path | str | None,
    *,
    action: str,
    entry: dict[str, Any],
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one ``phase_timeline`` event from a ``record_action_attempt`` entry."""
    if not session_dir or not isinstance(entry, dict):
        return
    try:
        task_id = str(entry.get("task_id") or "")
        payload = {
            "ts":              str(entry.get("ts") or ""),
            "action":          str(action or ""),
            "task_id":         task_id,
            "status":          str(entry.get("status") or ""),
            "decision":        str(entry.get("decision") or ""),
            "key_metric":      _to_float(entry.get("key_metric")),
            "key_metric_kind": entry.get("key_metric_kind"),
            "workspace":       entry.get("workspace"),
            "error_class":     entry.get("error_class"),
            "extras":          dict(entry.get("extras") or {}),
        }
        # Stable key per (action, task) so a re-recorded attempt overwrites.
        key = f"{action}-{task_id}" if task_id else None
        _recorder(session_dir, producer).record_item(
            "phase_timeline", payload, key=key,
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
        ("session",            _snapshot_session),
        ("workload",           _snapshot_workload),
        ("final",              _snapshot_final),
        ("explore_search",     _snapshot_explore_search),
        ("sweep",              _snapshot_sweep),
        ("optimization_stack", _snapshot_optimization_stack),
        ("roofline",           _snapshot_roofline),
    ):
        try:
            fn(rec, state)
        except Exception:  # noqa: BLE001
            log.debug("snapshot section %s failed", name, exc_info=True)


def _snapshot_session(rec, st: Any) -> None:
    session_id = str(getattr(st, "session_id", "") or "")
    if not session_id:
        return
    rec.record_singleton("session", {
        "session_id":      session_id,
        "claw_session_id": getattr(st, "claw_session_id", "") or "",
        "sandbox_user_id": getattr(st, "sandbox_user_id", "") or "",
        "start_ts":        str(getattr(st, "start_ts", "") or ""),
        "stop_reason":     str(getattr(st, "stop_reason", "") or ""),
        "max_minutes":     int(getattr(st, "max_minutes", 0) or 0),
        "tick_count":      int(getattr(st, "tick", 0) or 0),
        "phase":           str(getattr(st, "phase", "") or ""),
    })


def _snapshot_workload(rec, st: Any) -> None:
    framework = str(getattr(st, "framework", "") or "")
    model = str(getattr(st, "model_name", "") or getattr(st, "model_path", "") or "")
    if not framework and not model:
        return
    rec.record_singleton("workload", {
        "framework":     framework,
        "model_name":    str(getattr(st, "model_name", "") or ""),
        "model_path":    str(getattr(st, "model_path", "") or ""),
        "model_class":   str(getattr(st, "model_class", "") or ""),
        "gpu_type":      str(getattr(st, "gpu_type", "") or ""),
        "tp":            int(getattr(st, "tp", 0) or 0),
        "ep":            int(getattr(st, "ep", 0) or 0),
        "precision":     str(getattr(st, "precision", "") or ""),
        "conc":          int(getattr(st, "conc", 0) or 0),
        "isl":           int(getattr(st, "isl", 0) or 0),
        "osl":           int(getattr(st, "osl", 0) or 0),
        "max_model_len": int(getattr(st, "max_model_len", 0) or 0),
    })


def _snapshot_final(rec, st: Any) -> None:
    cb = getattr(st, "current_best", None) or {}
    stack = getattr(st, "optimization_stack", None) or []
    if not cb and not stack:
        return
    rec.record_singleton("final", {
        "current_best_action":                str(cb.get("action") or ""),
        "throughput_tok_s_per_gpu":           _to_float(cb.get("tput")),
        "cumulative_gain_pct_validated":      _to_float(
            getattr(st, "cumulative_gain_validated", 0.0)) or 0.0,
        "cumulative_gain_pct_per_round_sum":  _to_float(
            getattr(st, "cumulative_gain", 0.0)) or 0.0,
        "validated_ts":                       str(
            getattr(st, "cumulative_gain_validated_ts", "") or ""),
        "stack_len":                          len(stack),
        "extra_server_args":                  str(cb.get("extra_server_args") or ""),
        "extra_envs":                         dict(cb.get("extra_envs") or {}),
    })


def _snapshot_explore_search(rec, st: Any) -> None:
    search = dict(getattr(st, "explore_search", None) or {})
    if not search:
        return
    search["winner_history"] = list(getattr(st, "params_winner_history", None) or [])
    search["no_promote_streak"] = int(
        getattr(st, "params_no_promote_streak", 0) or 0)
    search["discovered_flags"] = dict(getattr(st, "discovered_flags", None) or {})
    search["synergy_attempted"] = list(getattr(st, "synergy_attempted", None) or [])
    search["backend_winners_history"] = list(
        getattr(st, "backend_winners_history", None) or [])
    rec.record_singleton("explore_search", search)


def _snapshot_sweep(rec, st: Any) -> None:
    last_sweep = dict(getattr(st, "last_sweep", None) or {})
    if not last_sweep:
        return
    rec.record_singleton("sweep", last_sweep)


def _snapshot_optimization_stack(rec, st: Any) -> None:
    stack = getattr(st, "optimization_stack", None) or []
    gains = getattr(st, "gain_per_stack_entry", None) or []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        payload = dict(entry)
        if payload.get("gain_pct") is None and i < len(gains):
            payload["gain_pct"] = _to_float(gains[i])
        rec.record_item("optimization_stack", payload, key=str(i))


def _snapshot_roofline(rec, st: Any) -> None:
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
        backend_rows = [
            a for a in rows if str(a.get("backend") or "").lower() == want_backend
        ]
        if backend_rows:
            candidates = backend_rows

    def _spd(a: dict[str, Any]) -> float:
        v = _to_float(a.get("micro_speedup") or a.get("speedup"))
        return v if v is not None else float("-inf")

    best = max(candidates, key=_spd)
    return str(best.get("attempt_id") or best.get("id") or "")


def _invocation_section(backend: str) -> str | None:
    b = str(backend or "").lower()
    if b in _GEAK_BACKENDS:
        return "geak_invocations"
    if b in _OOB_BACKENDS:
        return "oob_invocations"
    return None


def record_kernel_invocations(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record geak/oob invocations from an in-process kernel-agent result.

    Reads ``result['attempts']`` (per-backend ladder) so backend-level
    failures are captured even when the kernel-agent crashed before persisting
    ``optimization_attempts.jsonl`` (the on-disk source the collector reads).
    When the whole invocation failed before any backend ran (pre-dispatch
    gating: non_reusable_kernel / missing_source / kernel_agent_root_missing /
    ...), a single ``FAILED`` marker is recorded so the failure is never
    invisible in the geak/oob view.
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
            # Stamp the kernel-level KEEP/PARTIAL/REVERT onto the adopted (best)
            # attempt, mirroring the collector's _stamp_kernel_level_decisions.
            if kernel_decision and attempt_id and attempt_id == best_attempt_id:
                decision = kernel_decision
            optimized = att.get("optimized_path") or att.get("optimized_file")
            payload = {
                "kernel_id":       kid,
                "attempt_id":      attempt_id,
                "run_id":          run_id,
                "ts":              str(att.get("ts") or att.get("started_at") or ""),
                "backend":         backend,
                "decision":        decision,
                "status":          status,
                "micro_speedup":   _to_float(
                    att.get("micro_speedup") or att.get("speedup")),
                "optimized_files": [str(optimized)] if optimized else [],
                "error":           att.get("error") or att.get("error_message"),
            }
            key = attempt_id or f"{kid}-{backend}"
            rec.record_item(section, payload, key=key)
            recorded_any = True

        if recorded_any:
            return

        # No per-backend attempts: capture a pre-dispatch / infra failure so
        # the geak/oob view still shows it (root cause of invisible failures).
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        decision = str(
            (result.get("proposal") or {}).get("decision") or "").upper()
        failed = status in _FAILED_STATUSES or (
            decision == "REVERT" and bool(err_class)
        )
        if not failed:
            return
        backend = str(result.get("backend") or "").lower()
        section = _invocation_section(backend) or "geak_invocations"
        payload = {
            "kernel_id":            kid,
            "attempt_id":           "",
            "run_id":               run_id,
            "backend":              backend or "geak",
            "decision":             "FAILED",
            "status":               status or "failed",
            "error":                result.get("error") or err_class or None,
            "error_class":          err_class or None,
            # Distinguishes a pre-dispatch gating failure (no backend ran) from
            # a backend that ran and failed.
            "pre_dispatch_failure": True,
        }
        rec.record_item(section, payload, key=f"{kid}-predispatch" if kid else None)
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_invocations failed", exc_info=True)


def _to_bool(value: Any) -> bool | None:
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


# Cache of resolved tool metadata, keyed by the resolved root dir. The git
# probe is a one-shot per root: it never re-runs in the hot path.
_TOOL_META_CACHE: dict[str, dict[str, Any]] = {}


def _git_short_commit(root: Path) -> str:
    """Best-effort ``git rev-parse --short HEAD`` for ``root`` (never raises)."""
    import subprocess  # local: keep module import cost off the common path

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _tool_metadata(
    tool: str,
    *,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Resolve ``{tool, root_dir, commit, version}`` for an external tool.

    ``root`` (explicit) wins over ``root_env`` (an env var holding the path).
    The commit is a cached ``git rev-parse`` of the root; ``version`` is passed
    through verbatim when the tool already surfaced its own. All best-effort:
    a missing root just yields empty strings.
    """
    import os

    root_dir = str(root or (os.environ.get(root_env or "") or "")).strip()
    cache_key = f"{tool}:{root_dir}"
    cached = _TOOL_META_CACHE.get(cache_key)
    if cached is None:
        commit = ""
        if root_dir:
            try:
                if Path(root_dir).is_dir():
                    commit = _git_short_commit(Path(root_dir))
            except Exception:  # noqa: BLE001
                commit = ""
        cached = {"tool": tool, "root_dir": root_dir, "commit": commit}
        _TOOL_META_CACHE[cache_key] = cached
    meta = dict(cached)
    meta["version"] = str(version or "")
    return meta


def _normalize_hot_kernel(k: dict[str, Any]) -> dict[str, Any]:
    """Project a raw hot-kernel candidate onto the discovery shape."""
    return {
        "kernel_id":               str(k.get("kernel_id") or k.get("id") or ""),
        "name":                    str(k.get("name") or k.get("kernel_name") or ""),
        "gpu_pct":                 _to_float(k.get("gpu_pct") or k.get("gpu_percent")),
        "time_ms":                 _to_float(k.get("time_ms") or k.get("duration_ms")),
        "bound_type":              str(k.get("bound_type") or k.get("bottleneck") or ""),
        "arithmetic_intensity":    _to_float(k.get("arithmetic_intensity")),
        "flops_per_byte":          _to_float(k.get("flops_per_byte")),
        "efficiency_percent":      _to_float(k.get("efficiency_percent")),
        "reusable_native_kernel":  bool(k.get("reusable_native_kernel") or False),
        "source_file":             k.get("source_file"),
        "recommended_backends":    list(k.get("recommended_backends") or []),
        "selected_for_optimization": bool(k.get("selected_for_optimization") or False),
    }


def record_kernel_discovery(
    session_dir: Path | str | None,
    *,
    source: str,
    status: str,
    hot_kernels: list[Any] | None = None,
    scan: dict[str, Any] | None = None,
    tool_root: str | None = None,
    tool_root_env: str | None = None,
    tool_version: str | None = None,
    duration_sec: Any = None,
    error: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record one hot-kernel discovery run (stage 1 of ``kernel_journey``).

    One item per discovery invocation (tracelens / roofline scan), keyed by the
    candidates/report path so a re-run with the same artifact overwrites rather
    than duplicates. Carries the tool metadata (root + commit + version) and the
    full hot-kernel list the run surfaced.
    """
    if not session_dir:
        return
    try:
        kernels = [
            _normalize_hot_kernel(k) for k in (hot_kernels or [])
            if isinstance(k, dict)
        ]
        scan = dict(scan or {})
        meta = _tool_metadata(
            source, root=tool_root, root_env=tool_root_env, version=tool_version,
        )
        payload = {
            "source":           str(source or ""),
            "status":           str(status or ""),
            "ts":               _now_iso_safe(),
            "duration_sec":     _to_float(duration_sec),
            "tool":             meta,
            "scan":             scan,
            "hot_kernel_count": len(kernels),
            "hot_kernels":      kernels,
            "error":            error,
        }
        key = (
            str(scan.get("candidates_path") or scan.get("trace_report_path") or "")
            or None
        )
        _recorder(session_dir, producer).record_item(
            "kernel_discovery", payload, key=key,
        )
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_discovery failed", exc_info=True)


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
    """
    if not session_dir or not kernel_id:
        return
    try:
        payload = {
            "kernel_id":            str(kernel_id),
            "dispatched":           bool(dispatched),
            "backends":             [str(b) for b in (backends or [])],
            "skip_reason":          str(skip_reason or ""),
            "orchestration_commit": str(orchestration_commit or ""),
            "task_group":           task_group,
            "ts":                   _now_iso_safe(),
        }
        _recorder(session_dir, producer).record_item(
            "kernel_dispatch", payload, key=str(kernel_id),
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
    """
    if not session_dir or not isinstance(result, dict):
        return
    try:
        rec = _recorder(session_dir, producer)
        kid = str(result.get("kernel_id") or "")
        run_id = str(result.get("run_id") or result.get("session_id") or "")
        attempts = result.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        result_meta = result.get("metadata") if isinstance(
            result.get("metadata"), dict) else {}
        # The kernel-level micro_speedup is derived (best across attempts) and
        # lives in ``verification`` -- the raw per-attempt dict carries none. We
        # stamp it onto the adopted (best) attempt so the journey can correlate
        # achieved speedup with the e2e gain.
        verification = result.get("verification") if isinstance(
            result.get("verification"), dict) else {}
        best_attempt_id = _best_attempt_id(attempts, verification)
        kernel_micro_speedup = _to_float(verification.get("micro_speedup"))
        recorded_any = False
        for att in attempts:
            if not isinstance(att, dict):
                continue
            backend = str(att.get("backend") or "").lower()
            attempt_id = str(att.get("attempt_id") or att.get("id") or "")
            optimized = att.get("optimized_path") or att.get("optimized_file")
            att_meta = att.get("metadata") if isinstance(
                att.get("metadata"), dict) else {}
            micro_speedup = _to_float(
                att.get("micro_speedup") or att.get("speedup"))
            if (micro_speedup is None and kernel_micro_speedup is not None
                    and attempt_id and attempt_id == best_attempt_id):
                micro_speedup = kernel_micro_speedup
            payload = {
                "kernel_id":          kid,
                "attempt_id":         attempt_id,
                "run_id":             run_id,
                "backend":            backend,
                "model":              att.get("model"),
                "ts":                 str(att.get("ts") or att.get("started_at")
                                          or att.get("created_at") or ""),
                "status":             str(att.get("status") or "").lower(),
                "decision":           str(att.get("decision") or "").upper(),
                "micro_speedup":      micro_speedup,
                "compile_passed":     _to_bool(att.get("compile_passed")),
                "correctness_passed": _to_bool(att.get("correctness_passed")),
                "optimized_files":    [str(optimized)] if optimized else [],
                "error":              att.get("error") or att.get("error_message"),
                "error_class":        str(att.get("error_type") or "") or None,
                "duration_sec":       _to_float(
                    att.get("duration_sec") or att.get("elapsed_sec")
                    or att.get("elapsed_s")),
                "tool": _tool_metadata(
                    backend or "kernel_agent",
                    root=str(att_meta.get("root_dir") or result_meta.get("root_dir") or "")
                    or None,
                    root_env="HYPERLOOM_KERNEL_AGENT_ROOT",
                    version=str(att_meta.get("version") or result_meta.get("version") or ""),
                ),
            }
            key = attempt_id or (f"{run_id}-{backend}" if run_id else None)
            rec.record_item("kernel_backend_result", payload, key=key)
            recorded_any = True

        if recorded_any or not kid:
            return

        # No per-backend attempts: capture a pre-dispatch / infra failure as a
        # synthetic FAILED attempt so kernel_journey shows the failure too
        # (mirrors record_kernel_invocations' pre-dispatch marker; without this
        # the kernel looks merely "dispatched" with an empty attempt ladder).
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        decision = str(
            (result.get("proposal") or {}).get("decision") or "").upper()
        failed = status in _FAILED_STATUSES or (
            decision == "REVERT" and bool(err_class)
        )
        if not failed:
            return
        backend = str(result.get("backend") or "").lower() or "geak"
        payload = {
            "kernel_id":            kid,
            "attempt_id":           "",
            "run_id":               run_id,
            "backend":              backend,
            "model":                None,
            "ts":                   _now_iso_safe(),
            "status":               status or "failed",
            "decision":             "FAILED",
            "micro_speedup":        None,
            "compile_passed":       None,
            "correctness_passed":   None,
            "optimized_files":      [],
            "error":                result.get("error") or err_class or None,
            "error_class":          err_class or None,
            "duration_sec":         None,
            # Distinguishes a pre-dispatch gating failure (no backend ran) from
            # a backend that ran and failed.
            "pre_dispatch_failure": True,
            "tool": _tool_metadata(
                backend,
                root=str(result_meta.get("root_dir") or "") or None,
                root_env="HYPERLOOM_KERNEL_AGENT_ROOT",
                version=str(result_meta.get("version") or ""),
            ),
        }
        rec.record_item(
            "kernel_backend_result", payload,
            key=f"{kid}-predispatch",
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
    """
    if not session_dir or not kernel_id:
        return
    try:
        payload = {
            "kernel_id":         str(kernel_id),
            "integrated":        bool(integrated),
            "e2e_gain_pct":      _to_float(e2e_gain_pct),
            "validated":         bool(validated) if validated is not None else None,
            "decision":          str(decision or "").upper(),
            "patch_path":        patch_path,
            "target_file":       target_file,
            "extra_server_args": str(extra_server_args or ""),
            "ts":                _now_iso_safe(),
        }
        _recorder(session_dir, producer).record_item(
            "kernel_e2e", payload, key=str(kernel_id),
        )
    except Exception:  # noqa: BLE001
        log.debug("record_kernel_e2e failed", exc_info=True)


def record_specialist_round(
    session_dir: Path | str | None,
    entry: dict[str, Any],
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one ``specialist_runs`` round (idempotent by ``round_id``)."""
    if not session_dir or not isinstance(entry, dict) or not entry:
        return
    try:
        key = str(entry.get("round_id") or "") or None
        _recorder(session_dir, producer).record_item(
            "specialist_runs", dict(entry), key=key,
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
    producer: str = "critic",
) -> None:
    """Record one ``critic_robustness.critic_iterations`` item.

    Recorded per-iteration (idempotent on ``iter_n``) so the critic backend's
    workdir pruning never erases history; payload mirrors
    ``collectors.collect_critic_robustness``.
    """
    if not session_dir:
        return
    try:
        review = review if isinstance(review, dict) else {}
        emit = emit if isinstance(emit, dict) else {}
        wd = Path(workdir) if workdir else None
        payload = {
            "iter":    int(iter_n),
            "ts":      str(emit.get("ts") or review.get("ts") or ""),
            "topic":   str(emit.get("topic") or review.get("topic") or ""),
            "verdict": str(review.get("verdict") or emit.get("verdict") or ""),
            "summary": str(review.get("summary") or emit.get("summary") or "")[:500],
            "request_path":      _rel(wd / "request.json", session_dir) if wd else None,
            "judge_bundle_path": _rel(wd / "judge_bundle.json", session_dir) if wd else None,
            "emit_path":         _rel(wd / "emit.json", session_dir) if wd else None,
            "review_path":       _rel(wd / "review.json", session_dir) if wd else None,
        }
        _recorder(session_dir, producer).record_item(
            "critic_iterations", payload, key=str(iter_n),
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
    """
    if not session_dir or not workdir:
        return
    try:
        wd = Path(workdir)
        signal_data = _read_json(wd / "signal.json")
        action_data = _read_json(wd / "action.json")
        payload = {
            "ts":      str(signal_data.get("ts") or action_data.get("ts") or ""),
            "signal":  str(signal_data.get("signal") or signal_data.get("kind") or ""),
            "action":  str(action_data.get("action") or action_data.get("kind") or ""),
            "workdir": _rel(wd, session_dir),
        }
        _recorder(session_dir, producer).record_item(
            "robustness_signals", payload, key=wd.name,
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
    """Record a producer-owned singleton section (report summaries, etc.)."""
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
    "snapshot_state_sections",
]
