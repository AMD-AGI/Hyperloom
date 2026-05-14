"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function: it reads only from
``session_dir``, ``state`` (a :class:`SharedState`-shaped dict) and
``manifest`` (a manifest.json dict), and returns the matching schema
section. Collectors NEVER mutate state, NEVER fabricate values, and
NEVER raise — failures are caught, recorded in ``warnings``, and the
section returns a best-effort partial.

The schema each collector matches is defined in :mod:`.schema`.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _load_json_safe(path: Path | None, warnings: list[str]) -> Any | None:
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse {path}: {exc!r}")
        return None


def _load_jsonl_safe(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"malformed jsonl line in {path}: {exc!r}")
                continue
            if isinstance(entry, dict):
                out.append(entry)
    except OSError as exc:
        warnings.append(f"failed to read {path}: {exc!r}")
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() == "SKIPPED":
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path | None, session_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(session_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _find_benchmark_report(workspace: Path | None) -> Path | None:
    """Locate the ``benchmark_*/benchmark_report.json`` under a task workspace.

    Returns the most recent one (by mtime) if multiple are present (e.g.
    retries within the same task), else ``None``.
    """
    if workspace is None or not workspace.exists():
        return None
    candidates: list[Path] = []
    for p in workspace.glob("benchmark_*/benchmark_report.json"):
        candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


# ---------------------------------------------------------------------------
# §1 Session metadata
# ---------------------------------------------------------------------------
def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Top-level session identification + lifecycle."""
    start_ts = str(state.get("start_ts") or manifest.get("created_at_utc") or "")
    stop_reason = str(state.get("stop_reason") or "")
    elapsed_min: float | None = None
    if start_ts:
        try:
            start = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
            elapsed_min = (datetime.now(timezone.utc) - start).total_seconds() / 60.0
        except (ValueError, TypeError):
            pass
    return {
        "session_id":       str(state.get("session_id") or manifest.get("session_id") or ""),
        "claw_session_id":  manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id":  manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc":   manifest.get("created_at_utc") or start_ts,
        "ended_at_utc":     _utc_now_iso() if stop_reason else "",
        "stop_reason":      stop_reason,
        "max_minutes":      int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes":  round(elapsed_min, 2) if elapsed_min is not None else 0.0,
        "host":             str(manifest.get("host") or ""),
        "code_revision":    str(manifest.get("code_revision") or ""),
        "pid":              int(manifest.get("pid") or 0),
        "session_dir":      str(session_dir),
        "tick_count":       int(state.get("tick") or 0),
    }


# ---------------------------------------------------------------------------
# §2 Workload
# ---------------------------------------------------------------------------
def collect_workload(
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    wl = manifest.get("workload") or {}
    return {
        "framework":         str(state.get("framework") or manifest.get("framework") or ""),
        "framework_version": str(manifest.get("framework_version") or ""),
        "model_name":        str(state.get("model_name") or manifest.get("model_name") or ""),
        "model_path":        str(state.get("model_path") or manifest.get("model_path") or ""),
        "model_class":       str(state.get("model_class") or ""),
        "gpu_type":          str(state.get("gpu_type") or manifest.get("gpu_type") or ""),
        "tp":                _to_int(manifest.get("tp")),
        "conc":              _to_int(wl.get("conc")),
        "isl":               _to_int(wl.get("isl")),
        "osl":               _to_int(wl.get("osl")),
        "max_model_len":     _to_int(wl.get("max_model_len")),
        "precision":         str(wl.get("precision") or ""),
        "objective":         dict(manifest.get("objective") or {"kind": "time_only", "value": None}),
    }


# ---------------------------------------------------------------------------
# §3 Baseline
# ---------------------------------------------------------------------------
def collect_baseline(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    last_b = state.get("last_baseline") or {}
    workspace_str = last_b.get("workspace") or ""
    workspace = Path(workspace_str) if workspace_str else None
    report_path = _find_benchmark_report(workspace) if workspace else None
    report = _load_json_safe(report_path, warnings) if report_path else None

    ttft: float | None = None
    e2el: float | None = None
    if isinstance(report, dict):
        ttft = _to_float(_safe_get(report, "mean_ttft_ms") or _safe_get(report, "result", "mean_ttft_ms"))
        e2el = _to_float(_safe_get(report, "mean_e2el_ms") or _safe_get(report, "result", "mean_e2el_ms"))

    attempts = state.get("baseline_attempts") or []
    history: list[dict[str, Any]] = []
    for a in attempts:
        if not isinstance(a, dict):
            continue
        history.append({
            "ts":            a.get("ts"),
            "task_id":       a.get("task_id"),
            "status":        a.get("status"),
            "decision":      a.get("decision"),
            "key_metric":    _to_float(a.get("key_metric")),
            "workspace":     a.get("workspace"),
            "error_class":   a.get("error_class"),
        })

    return {
        "throughput_tok_s_per_gpu": _to_float(state.get("baseline_tput")) or 0.0,
        "accuracy":                 _to_float(state.get("baseline_accuracy")) or 0.0,
        "ttft_mean_ms":             ttft,
        "e2el_mean_ms":             e2el,
        "config_path":              state.get("baseline_config_path") or None,
        "benchmark_report_path":    _rel(report_path, session_dir) if report_path else None,
        "attempts_history":         history,
        "failure_streak":           int(state.get("baseline_failure_streak") or 0),
    }


# ---------------------------------------------------------------------------
# §4 Final (validated)
# ---------------------------------------------------------------------------
def collect_final(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    cb = state.get("current_best") or {}
    stack = state.get("optimization_stack") or []
    stack_len = len(stack) if isinstance(stack, list) else 0
    val_stack_len = int(state.get("cumulative_gain_validated_stack_len") or 0)

    action_path: list[str] = []
    for entry in stack if isinstance(stack, list) else []:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action") or "")
        variant = str(entry.get("variant_name") or "")
        action_path.append(f"{action}:{variant}" if variant else action)

    return {
        "throughput_tok_s_per_gpu":          _to_float(cb.get("tput")),
        "cumulative_gain_pct_validated":     _to_float(state.get("cumulative_gain_validated")) or 0.0,
        "cumulative_gain_pct_per_round_sum": _to_float(state.get("cumulative_gain")) or 0.0,
        "validated_at_stack_len":            val_stack_len,
        "validated_ts":                      str(state.get("cumulative_gain_validated_ts") or ""),
        "stack_changed_after_validation":    stack_len > val_stack_len > 0,
        "extra_sglang_args":                 str(cb.get("extra_sglang_args") or ""),
        "extra_envs":                        dict(cb.get("extra_envs") or {}),
        "action_path":                       action_path,
        "ttft_mean_ms":                      _to_float(cb.get("ttft_mean_ms")),
        "e2el_mean_ms":                      _to_float(cb.get("e2el_mean_ms")),
    }


# ---------------------------------------------------------------------------
# §5 Phase timeline
# ---------------------------------------------------------------------------
_AUDIT_ACTIONS = (
    "baseline", "profile", "backends", "params", "sweep", "validate_stack",
)


def collect_phase_timeline(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for action in _AUDIT_ACTIONS:
        attempts = state.get(f"{action}_attempts") or []
        if not isinstance(attempts, list):
            continue
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            events.append({
                "ts":             entry.get("ts") or "",
                "action":         action,
                "task_id":        str(entry.get("task_id") or ""),
                "kernel_id":      None,
                "status":         str(entry.get("status") or ""),
                "decision":       str(entry.get("decision") or ""),
                "key_metric":     _to_float(entry.get("key_metric")),
                "key_metric_kind": entry.get("key_metric_kind"),
                "workspace":      entry.get("workspace"),
                "error_class":    entry.get("error_class"),
                "extras":         dict(entry.get("extras") or {}),
            })

    # Kernel opt attempts (per-kernel history -> flatten to per-attempt rows)
    kernel_opt = state.get("kernel_opt_attempts") or {}
    if isinstance(kernel_opt, dict):
        for kid, ent in kernel_opt.items():
            if not isinstance(ent, dict):
                continue
            for h in ent.get("history") or []:
                if not isinstance(h, dict):
                    continue
                events.append({
                    "ts":          h.get("ts") or ent.get("last_ts") or "",
                    "action":      "kernel_opt",
                    "task_id":     "",
                    "kernel_id":   str(kid),
                    "status":      "",
                    "decision":    str(h.get("decision") or ""),
                    "key_metric":  None,
                    "workspace":   None,
                    "error_class": None,
                    "extras":      {},
                })

    # Integrate attempts (decision history per patch key)
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for key, ent in integ.items():
            if not isinstance(ent, dict):
                continue
            for a in ent.get("attempts") or []:
                if not isinstance(a, dict):
                    continue
                events.append({
                    "ts":          a.get("ts") or "",
                    "action":      "integrate",
                    "task_id":     "",
                    "kernel_id":   str(ent.get("kernel_id") or ""),
                    "status":      str(a.get("status") or ""),
                    "decision":    str(a.get("decision") or ""),
                    "key_metric":  _to_float(a.get("gain_pct")),
                    "key_metric_kind": "gain_pct",
                    "workspace":   a.get("workspace"),
                    "error_class": None,
                    "extras":      {"patch_path": ent.get("patch_path"),
                                    "report_path": a.get("report_path")},
                })

    events.sort(key=lambda e: e.get("ts") or "")
    return events


# ---------------------------------------------------------------------------
# §6 Capability summary
# ---------------------------------------------------------------------------
def _capability_for_action(
    state: dict[str, Any], action: str,
) -> dict[str, Any]:
    attempts_list = state.get(f"{action}_attempts") or []
    n_attempts = len(attempts_list) if isinstance(attempts_list, list) else 0
    n_keeps = sum(
        1 for a in attempts_list
        if isinstance(a, dict) and a.get("decision") in ("promoted", "salvaged")
    ) if isinstance(attempts_list, list) else 0
    status = (
        "kept"      if n_keeps > 0 else
        "tried"     if n_attempts > 0 else
        "not_attempted"
    )
    return {
        "status":   status,
        "attempts": n_attempts,
        "keeps":    n_keeps,
    }


def collect_capability_summary(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    oob_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    # GEAK / OOB driven by actual invocations on disk (most reliable)
    def _from_invocations(invs: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = len(invs)
        keeps = sum(1 for v in invs if v.get("decision") == "KEEP")
        status = (
            "kept"          if keeps > 0 else
            "attempted"     if attempts > 0 else
            "not_attempted"
        )
        return {"status": status, "attempts": attempts, "keeps": keeps}

    geak_cap = _from_invocations(geak_invocations)
    oob_cap = _from_invocations(oob_invocations)

    backends = _capability_for_action(state, "backends")
    backends_search = state.get("backends_search") or {}
    if isinstance(backends_search, dict):
        backends["tested"] = len(backends_search.get("tested") or {})
        if backends_search.get("accepted"):
            backends["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0
                 for v in backends_search["accepted"]
                 if isinstance(v, dict)),
                default=None,
            )

    params = _capability_for_action(state, "params")
    params_search = state.get("params_search") or {}
    if isinstance(params_search, dict):
        params["tested"] = len(params_search.get("tested") or {})
        if params_search.get("accepted"):
            params["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0
                 for v in params_search["accepted"]
                 if isinstance(v, dict)),
                default=None,
            )

    sweep_cap = _capability_for_action(state, "sweep")
    last_sweep = state.get("last_sweep") or {}
    if isinstance(last_sweep, dict):
        sweep_cap["grid_size"] = _to_int(last_sweep.get("grid_size"))
        bo = last_sweep.get("best_overall")
        if isinstance(bo, dict):
            sweep_cap["best_throughput"] = _to_float(
                bo.get("output_throughput") or bo.get("tput")
            )
        if sweep_cap.get("attempts", 0) > 0:
            sweep_cap["status"] = "completed"

    validate = _capability_for_action(state, "validate_stack")
    validate["last_validated_gain_pct"] = _to_float(
        state.get("cumulative_gain_validated")
    )

    return {
        "geak":           geak_cap,
        "oob":            oob_cap,
        "backends":       backends,
        "params":         params,
        "sweep":          sweep_cap,
        "validate_stack": validate,
    }


# ---------------------------------------------------------------------------
# §7 / §8 GEAK / OOB invocations
# ---------------------------------------------------------------------------
def _kernel_agent_run_dirs(session_dir: Path) -> list[Path]:
    """All ``$SD/kernel-agent-workspace/<*>/kernel-agent/runs/<sid>/`` dirs.

    Hyperloom v2 spawns ``kernel_optimization.py --workspace-path
    $SD/kernel-agent-workspace``; the script then creates
    ``<workspace>/kernel-agent/runs/<session_id>/...``. We scan both the
    canonical single-level form and the legacy nested form for safety.
    """
    candidates: list[Path] = []
    root = session_dir / "kernel-agent-workspace"
    if not root.exists():
        return candidates
    # Canonical: $SD/kernel-agent-workspace/kernel-agent/runs/<sid>/
    for sub in (root / "kernel-agent" / "runs").glob("*"):
        if sub.is_dir():
            candidates.append(sub)
    # Per-kernel: $SD/kernel-agent-workspace/<kid>/...  (older layout)
    for kid_dir in root.glob("*/kernel-agent/runs/*"):
        if kid_dir.is_dir() and kid_dir not in candidates:
            candidates.append(kid_dir)
    return candidates


def _parse_invocation_attempt(
    attempt: dict[str, Any],
    run_dir: Path,
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Parse one ``optimization_attempts.jsonl`` row into an Invocation.

    Per-attempt fields are read from ``attempt`` directly. Kernel-level
    artifacts (verification / result) are referenced by path only — the
    KEEP/PARTIAL/REVERT decision is NOT stamped here because
    ``results/<kid>.json`` and ``verification/<kid>.json`` describe the
    kernel's BEST attempt, not every attempt. Stamping happens in
    :func:`_stamp_kernel_level_decisions` after all attempts are parsed.
    """
    kid = str(attempt.get("kernel_id") or "")
    backend = str(attempt.get("backend") or "").lower()
    attempt_id = str(
        attempt.get("attempt_id")
        or attempt.get("id")
        or attempt.get("run_id")
        or "",
    )

    # Resolve auxiliary paths
    prompt_path: Path | None = None
    for p in (run_dir / "prompts").glob(f"{attempt_id}*") if attempt_id else []:
        prompt_path = p
        break

    optimized_files: list[str] = []
    if attempt_id:
        for p in sorted((run_dir / "optimized").glob(f"{attempt_id}*")):
            optimized_files.append(_rel(p, session_dir) or str(p))

    verification_path = run_dir / "verification" / f"{kid}.json" if kid else None
    result_path = run_dir / "results" / f"{kid}.json" if kid else None

    # Per-attempt decision: derived ONLY from this attempt's own fields.
    decision = str(attempt.get("decision") or "").upper()
    if not decision:
        status = str(attempt.get("status") or "").lower()
        if status in ("failed", "error", "crashed"):
            decision = "FAILED"
        # otherwise leave empty; kernel-level decision is stamped later

    # Per-attempt speedup (preferred) — falls back to None if not recorded.
    micro_speedup = _to_float(
        attempt.get("speedup")
        or attempt.get("micro_speedup")
    )

    return {
        "kernel_id":       kid,
        "attempt_id":      attempt_id,
        "run_id":          str(attempt.get("run_id") or run_dir.name),
        "ts":              str(attempt.get("ts") or attempt.get("started_at") or ""),
        "backend":         backend,
        "model":           attempt.get("model"),
        "kernel_metadata": _shape_kernel_metadata({}, attempt),
        "prompt_path":     _rel(prompt_path, session_dir) if prompt_path else None,
        "optimized_files": optimized_files,
        "result_path":     _rel(result_path, session_dir) if result_path and result_path.exists() else None,
        "verification_path": _rel(verification_path, session_dir) if verification_path and verification_path.exists() else None,
        "decision":        decision,
        "micro_speedup":   micro_speedup,
        # compile_passed / correctness_passed are kernel-level (in verification.json);
        # stamped later if this attempt is the BEST one for the kernel.
        "compile_passed":  None,
        "correctness_passed": None,
        "best_artifact_path": None,
        "error":           attempt.get("error"),
        "cli_log_path":    None,
    }


def _stamp_kernel_level_decisions(
    invocations: list[dict[str, Any]],
    run_dirs: list[Path],
    session_dir: Path,
    warnings: list[str],
) -> None:
    """Stamp the kernel-level KEEP/PARTIAL/REVERT decision onto the
    single BEST attempt for each kernel.

    For each ``(run_dir, kernel_id)`` group, find ``results/<kid>.json``
    and ``verification/<kid>.json`` (kernel-level, one file per kernel,
    written at the END of the kernel-agent run). Pick the attempt with
    the highest ``micro_speedup`` (ties: latest ts), and stamp the
    kernel-level decision + verification fields onto only that attempt.
    All other attempts for the same kernel keep their per-attempt
    ``decision`` (empty string for non-failed in-progress steps).
    """
    # Group attempts by (run_id, kernel_id) — same kid in different
    # run_dirs is treated as separate sessions.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for inv in invocations:
        key = (inv.get("run_id") or "", inv.get("kernel_id") or "")
        if not key[1]:
            continue
        groups.setdefault(key, []).append(inv)

    # Map run_id -> Path for kernel-level json lookup.
    run_by_id: dict[str, Path] = {rd.name: rd for rd in run_dirs}

    for (run_id, kid), atts in groups.items():
        run_dir = run_by_id.get(run_id)
        if run_dir is None:
            continue
        result_path = run_dir / "results" / f"{kid}.json"
        verification_path = run_dir / "verification" / f"{kid}.json"
        result = _load_json_safe(result_path if result_path.exists() else None, warnings) or {}
        verification = _load_json_safe(verification_path if verification_path.exists() else None, warnings) or {}

        decision = ""
        proposal = result.get("proposal") if isinstance(result, dict) else None
        if isinstance(proposal, dict):
            decision = str(proposal.get("decision") or "").upper()

        if not decision and not verification:
            continue  # nothing to stamp

        # Pick the BEST attempt: highest micro_speedup, ties broken by latest ts.
        def _attempt_key(a: dict[str, Any]) -> tuple[float, str]:
            spd = a.get("micro_speedup")
            return (
                float(spd) if isinstance(spd, (int, float)) else float("-inf"),
                str(a.get("ts") or ""),
            )

        best = max(atts, key=_attempt_key)

        if decision:
            best["decision"] = decision
        if isinstance(verification, dict):
            if best.get("micro_speedup") is None and verification.get("micro_speedup") is not None:
                best["micro_speedup"] = _to_float(verification.get("micro_speedup"))
            best["compile_passed"] = verification.get("compile_passed")
            best["correctness_passed"] = verification.get("correctness_passed")
            best["best_artifact_path"] = (
                verification.get("best_artifact_path")
                or (result.get("best_artifact_path") if isinstance(result, dict) else None)
            )
        if isinstance(result, dict) and result.get("cli_log_path"):
            best["cli_log_path"] = result["cli_log_path"]
        # Refresh kernel metadata from the (richer) result file
        best["kernel_metadata"] = _shape_kernel_metadata(result, {
            "name": best.get("kernel_metadata", {}).get("name"),
            "source_file": best.get("kernel_metadata", {}).get("source_file"),
        })


def _shape_kernel_metadata(
    result: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort kernel metadata for the UI.

    Prefers fields explicit in ``result['kernel_metadata']`` (newer
    kernel_optimization.py writes that); otherwise falls back to the
    attempt's own metadata.
    """
    meta = result.get("kernel_metadata") if isinstance(result, dict) else None
    if isinstance(meta, dict) and meta:
        return {
            "name":        meta.get("name") or "",
            "source_file": meta.get("source_file") or result.get("source_file") or "",
            "shapes":      list(meta.get("shapes") or []),
            "gpu_pct":     _to_float(meta.get("gpu_pct")),
            "arithmetic_intensity": _to_float(meta.get("arithmetic_intensity")),
        }
    return {
        "name":        str(attempt.get("name") or ""),
        "source_file": str(attempt.get("source_file") or ""),
        "shapes":      [],
        "gpu_pct":     None,
        "arithmetic_intensity": None,
    }


def collect_kernel_invocations(
    session_dir: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(geak_invocations, oob_invocations)``.

    Reads ``optimization_attempts.jsonl`` from every kernel-agent run dir,
    splits attempts by ``backend`` field, and stamps the kernel-level
    KEEP/PARTIAL/REVERT decision onto only the BEST attempt for each
    kernel (via :func:`_stamp_kernel_level_decisions`).
    """
    all_invocations: list[dict[str, Any]] = []
    run_dirs = _kernel_agent_run_dirs(session_dir)
    for run_dir in run_dirs:
        attempts = _load_jsonl_safe(run_dir / "optimization_attempts.jsonl", warnings)
        for att in attempts:
            inv = _parse_invocation_attempt(att, run_dir, session_dir, warnings)
            backend = inv.get("backend") or ""
            if not backend:
                inv["backend"] = "unknown"
            all_invocations.append(inv)

    # Stamp kernel-level KEEP/PARTIAL/REVERT onto the BEST attempt per kernel
    # (kernel-agent writes one verification/<kid>.json + one results/<kid>.json
    # per kernel, not per attempt).
    _stamp_kernel_level_decisions(all_invocations, run_dirs, session_dir, warnings)

    geak: list[dict[str, Any]] = []
    oob: list[dict[str, Any]] = []
    for inv in all_invocations:
        backend = inv.get("backend") or ""
        if backend == "geak":
            geak.append(inv)
        elif backend in ("claude", "codex"):
            oob.append(inv)
        else:
            oob.append(inv)
    geak.sort(key=lambda e: (e.get("kernel_id") or "", e.get("ts") or ""))
    oob.sort(key=lambda e: (e.get("kernel_id") or "", e.get("ts") or ""))
    return geak, oob


# ---------------------------------------------------------------------------
# §9 Kernel lifecycle
# ---------------------------------------------------------------------------
def _scan_profile_reports(session_dir: Path) -> list[tuple[Path, Path]]:
    """Return list of ``(task_dir, benchmark_report.json)`` under runs/profile/."""
    out: list[tuple[Path, Path]] = []
    root = session_dir / "runs" / "profile"
    if not root.exists():
        return out
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        report = _find_benchmark_report(task_dir)
        if report is not None:
            out.append((task_dir, report))
    return out


def _collect_detected_kernels(
    session_dir: Path, warnings: list[str], cap: int = 50,
) -> list[dict[str, Any]]:
    detected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task_dir, report_path in _scan_profile_reports(session_dir):
        report = _load_json_safe(report_path, warnings)
        if not isinstance(report, dict):
            continue
        kernel_summary = report.get("kernel_summary") or []
        bottlenecks = report.get("top_bottlenecks") or []
        bottleneck_by_kid = {
            b.get("kernel_id"): b
            for b in (bottlenecks if isinstance(bottlenecks, list) else [])
            if isinstance(b, dict)
        }
        for k in kernel_summary if isinstance(kernel_summary, list) else []:
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or k.get("name") or "")
            if not kid or kid in seen:
                continue
            seen.add(kid)
            bn = bottleneck_by_kid.get(kid) or {}
            detected.append({
                "kernel_id":               kid,
                "name":                    str(k.get("name") or ""),
                "gpu_pct":                 _to_float(k.get("gpu_pct")),
                "time_ms":                 _to_float(k.get("time_ms")),
                "bottleneck":              str(bn.get("bottleneck") or k.get("bottleneck") or ""),
                "arithmetic_intensity":    _to_float(k.get("arithmetic_intensity")),
                "reusable_native_kernel":  bool(k.get("reusable_native_kernel")),
                "source_file":             k.get("source_file"),
                "detected_from_task":      task_dir.name,
                "benchmark_report_path":   _rel(report_path, session_dir) or str(report_path),
            })
            if len(detected) >= cap:
                return detected
    return detected


def _collect_recommended_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    sk = state.get("last_select_kernels") or {}
    if not isinstance(sk, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in sk.get("hot_kernels_top15") or []:
        if not isinstance(entry, dict):
            continue
        out.append({
            "kernel_id":            str(entry.get("kernel_id") or ""),
            "name":                 str(entry.get("name") or ""),
            "gpu_pct":              _to_float(entry.get("gpu_pct")),
            "recommended_backends": list(entry.get("recommended_backends") or []),
            "recommended_actions":  list(entry.get("recommended_actions") or []),
            "bottleneck":           str(entry.get("bottleneck") or ""),
            "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
        })
    return out


def _collect_optimized_kernels(
    geak: list[dict[str, Any]],
    oob: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fold per-attempt invocations into per-kernel summaries."""
    by_kid: dict[str, dict[str, Any]] = {}
    for invs in (geak, oob):
        for inv in invs:
            kid = inv.get("kernel_id") or ""
            if not kid:
                continue
            entry = by_kid.setdefault(kid, {
                "kernel_id": kid,
                "backend": inv.get("backend") or "",
                "total_attempts": 0,
                "successful_attempts": 0,
                "best_micro_speedup": None,
                "last_decision": "",
                "best_artifact_path": None,
                "attempts_summary": [],
            })
            entry["total_attempts"] += 1
            spd = inv.get("micro_speedup")
            cur_best = entry["best_micro_speedup"]
            if isinstance(spd, (int, float)):
                if cur_best is None or spd > cur_best:
                    entry["best_micro_speedup"] = float(spd)
                    entry["best_artifact_path"] = inv.get("best_artifact_path") or entry["best_artifact_path"]
            if inv.get("decision") in ("KEEP", "PARTIAL"):
                entry["successful_attempts"] += 1
            entry["last_decision"] = inv.get("decision") or entry["last_decision"]
            entry["attempts_summary"].append({
                "attempt_id":   inv.get("attempt_id"),
                "backend":      inv.get("backend"),
                "decision":     inv.get("decision"),
                "micro_speedup": spd,
                "ts":           inv.get("ts"),
            })
    # Cross-reference with state.kernel_opt_attempts (the orchestrator's own
    # record of decisions — covers cases where the on-disk verification was
    # rotated away but the orchestrator decision is still in state).
    ko_attempts = state.get("kernel_opt_attempts") or {}
    if isinstance(ko_attempts, dict):
        for kid, ent in ko_attempts.items():
            if not isinstance(ent, dict):
                continue
            entry = by_kid.setdefault(kid, {
                "kernel_id": kid,
                "backend": "",
                "total_attempts": 0,
                "successful_attempts": 0,
                "best_micro_speedup": None,
                "last_decision": "",
                "best_artifact_path": None,
                "attempts_summary": [],
            })
            entry["total_attempts"] = max(entry["total_attempts"], int(ent.get("attempts", 0)))
            entry["last_decision"] = entry["last_decision"] or str(ent.get("last_decision") or "")
    return sorted(by_kid.values(), key=lambda e: e.get("kernel_id") or "")


def _collect_adopted_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    """KEEP-promoted kernel entries (from state.kernel_integrate_attempts + optimization_stack)."""
    out: list[dict[str, Any]] = []
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for key, ent in integ.items():
            if not isinstance(ent, dict):
                continue
            if ent.get("last_decision") != "KEEP":
                continue
            out.append({
                "kernel_id":         str(ent.get("kernel_id") or ""),
                "patch_path":        str(ent.get("patch_path") or ""),
                "target_file":       str(ent.get("target_file") or ""),
                "extra_sglang_args": str(ent.get("extra_sglang_args") or ""),
                "e2e_gain_pct":      _to_float(ent.get("best_gain_pct")),
                "validated":         True,
                "last_status":       str(ent.get("last_status") or ""),
                "adopted_at":        str(ent.get("updated_at") or ""),
                "attempt_count":     int(ent.get("attempt_count") or 0),
            })
    return out


def _collect_rejected_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rejected = state.get("rejected_kernel_patches") or []
    if isinstance(rejected, list):
        for r in rejected:
            if not isinstance(r, dict):
                continue
            out.append({
                "kernel_id":      str(r.get("kernel_id") or ""),
                "reason":         str(r.get("reason") or ""),
                "patch_path":     r.get("patch_path"),
                "target_file":    r.get("target_file"),
                "attempt_count":  int(r.get("attempt_count") or 0),
                "best_gain_pct":  _to_float(r.get("best_gain_pct")),
                "ts":             str(r.get("ts") or ""),
            })
    # also surface rejected_kernel_ids that didn't make it into rejected_kernel_patches
    seen_ids = {entry["kernel_id"] for entry in out if entry.get("kernel_id")}
    for kid in state.get("rejected_kernel_ids") or []:
        kid_s = str(kid or "")
        if not kid_s or kid_s in seen_ids:
            continue
        out.append({
            "kernel_id":  kid_s,
            "reason":     "retired",
            "patch_path": None,
            "target_file": None,
            "attempt_count": 0,
            "best_gain_pct": None,
            "ts": "",
        })
    return out


def collect_kernel_lifecycle(
    session_dir: Path,
    state: dict[str, Any],
    geak: list[dict[str, Any]],
    oob: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "detected":    _collect_detected_kernels(session_dir, warnings),
        "recommended": _collect_recommended_kernels(state),
        "optimized":   _collect_optimized_kernels(geak, oob, state),
        "adopted":     _collect_adopted_kernels(state),
        "rejected":    _collect_rejected_kernels(state),
    }


# ---------------------------------------------------------------------------
# §10 Param / backends search
# ---------------------------------------------------------------------------
def _shape_ledger(
    ledger: dict[str, Any] | None,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        return {
            "schema_version": 0, "tested_count": 0,
            "accepted": [], "rejected": [], "top_by_gain": [],
        }

    def _shape_entry(e: Any) -> dict[str, Any]:
        if not isinstance(e, dict):
            return {}
        return {
            "name":              str(e.get("name") or ""),
            "fingerprint":       str(e.get("fingerprint") or ""),
            "extra_sglang_args": str(e.get("extra_sglang_args") or ""),
            "extra_envs":        dict(e.get("extra_envs") or {}),
            "output_throughput": _to_float(e.get("output_throughput") or e.get("tput")),
            "gain_pct":          _to_float(e.get("gain_pct")),
            "ts":                str(e.get("ts") or ""),
        }

    accepted = [_shape_entry(e) for e in ledger.get("accepted") or []]
    rejected = [_shape_entry(e) for e in ledger.get("rejected") or []]
    tested = list((ledger.get("tested") or {}).values()) if isinstance(ledger.get("tested"), dict) else []
    tested_shaped = [_shape_entry(e) for e in tested]
    top_by_gain = sorted(
        (e for e in tested_shaped if e.get("gain_pct") is not None),
        key=lambda e: e.get("gain_pct") or 0.0,
        reverse=True,
    )[:top_n]
    return {
        "schema_version":  int(ledger.get("schema_version") or 0),
        "tested_count":    len(tested),
        "accepted":        accepted,
        "rejected":        rejected,
        "top_by_gain":     top_by_gain,
    }


def collect_param_search(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    params_ledger = _shape_ledger(state.get("params_search"))
    params_ledger["winner_history"] = list(state.get("params_winner_history") or [])
    params_ledger["no_promote_streak"] = int(state.get("params_no_promote_streak") or 0)

    backends_ledger = _shape_ledger(state.get("backends_search"))

    return {
        "params":                  params_ledger,
        "backends":                backends_ledger,
        "synergy_attempted":       list(state.get("synergy_attempted") or []),
        "discovered_flags":        dict(state.get("discovered_flags") or {}),
        "backend_winners_history": list(state.get("backend_winners_history") or []),
    }


# ---------------------------------------------------------------------------
# §11 Sweep
# ---------------------------------------------------------------------------
_VARIANT_NAME_RE = re.compile(r"variant_(\d+)_conc(\d+)_isl(\d+)_osl(\d+)", re.IGNORECASE)


def _scan_sweep_variants(session_dir: Path) -> list[Path]:
    """Return list of variant directories under runs/sweep/<task>/variant_*/."""
    root = session_dir / "runs" / "sweep"
    if not root.exists():
        return []
    out: list[Path] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for v in sorted(task_dir.glob("variant_*")):
            if v.is_dir():
                out.append(v)
    return out


def _shape_sweep_point(
    variant_dir: Path,
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    name = variant_dir.name
    m = _VARIANT_NAME_RE.search(name)
    conc: int | None = None
    isl_: int | None = None
    osl_: int | None = None
    if m:
        try:
            conc = int(m.group(2))
            isl_ = int(m.group(3))
            osl_ = int(m.group(4))
        except ValueError:
            pass
    report = _find_benchmark_report(variant_dir)
    report_data = _load_json_safe(report, warnings) if report else None
    status = "ok"
    out_tput = ttft = tpot = e2el = None
    if isinstance(report_data, dict):
        out_tput = _to_float(
            _safe_get(report_data, "output_throughput_tok_s")
            or _safe_get(report_data, "result", "output_throughput_tok_s")
            or _safe_get(report_data, "output_throughput")
        )
        ttft = _to_float(
            _safe_get(report_data, "mean_ttft_ms")
            or _safe_get(report_data, "result", "mean_ttft_ms")
        )
        tpot = _to_float(
            _safe_get(report_data, "mean_tpot_ms")
            or _safe_get(report_data, "result", "mean_tpot_ms")
        )
        e2el = _to_float(
            _safe_get(report_data, "mean_e2el_ms")
            or _safe_get(report_data, "result", "mean_e2el_ms")
        )
        if report_data.get("success") is False:
            status = "failed"
    elif report is None:
        status = "skipped"
    return {
        "variant_name":            name,
        "conc":                    conc,
        "isl":                     isl_,
        "osl":                     osl_,
        "output_throughput_tok_s": out_tput,
        "ttft_mean_ms":            ttft,
        "tpot_mean_ms":            tpot,
        "e2el_mean_ms":            e2el,
        "status":                  status,
        "benchmark_report_path":   _rel(report, session_dir) if report else None,
    }


def collect_sweep(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    ls = state.get("last_sweep") or {}
    if not isinstance(ls, dict):
        ls = {}

    variants_on_disk = [
        _shape_sweep_point(v, session_dir, warnings)
        for v in _scan_sweep_variants(session_dir)
    ]
    return {
        "grid_size":          _to_int(ls.get("grid_size")) or len(variants_on_disk),
        "best_overall":       dict(ls.get("best_overall") or {}),
        "best_for_each_conc": list(ls.get("best_for_each_conc") or []),
        "pareto_front":       list(ls.get("pareto_front") or []),
        "all_variants":       variants_on_disk,
        "config_path":        ls.get("config_path"),
    }


# ---------------------------------------------------------------------------
# §12 Critic / Robustness
# ---------------------------------------------------------------------------
def collect_critic_robustness(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    critic_iters: list[dict[str, Any]] = []
    critic_root = session_dir / "critic-workdir"
    if critic_root.exists():
        for iter_dir in sorted(critic_root.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            try:
                iter_n = int(iter_dir.name)
            except ValueError:
                iter_n = -1
            review = _load_json_safe(iter_dir / "review.json", warnings) or {}
            emit = _load_json_safe(iter_dir / "emit.json", warnings) or {}
            critic_iters.append({
                "iter":               iter_n,
                "ts":                 str(emit.get("ts") or review.get("ts") or ""),
                "topic":              str(emit.get("topic") or review.get("topic") or ""),
                "verdict":            str(review.get("verdict") or emit.get("verdict") or ""),
                "summary":            str(
                    review.get("summary") or emit.get("summary") or ""
                )[:500],
                "request_path":       _rel(iter_dir / "request.json", session_dir),
                "judge_bundle_path":  _rel(iter_dir / "judge_bundle.json", session_dir),
                "emit_path":          _rel(iter_dir / "emit.json", session_dir),
                "review_path":        _rel(iter_dir / "review.json", session_dir),
            })

    robustness_signals: list[dict[str, Any]] = []
    rob_root = session_dir / "robustness-workdir"
    if rob_root.exists():
        for iter_dir in sorted(rob_root.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            signal_data = _load_json_safe(iter_dir / "signal.json", warnings) or {}
            action_data = _load_json_safe(iter_dir / "action.json", warnings) or {}
            robustness_signals.append({
                "ts":      str(signal_data.get("ts") or action_data.get("ts") or ""),
                "signal":  str(signal_data.get("signal") or signal_data.get("kind") or ""),
                "action":  str(action_data.get("action") or action_data.get("kind") or ""),
                "workdir": _rel(iter_dir, session_dir) or str(iter_dir),
            })

    return {
        "critic_iterations":  critic_iters,
        "robustness_signals": robustness_signals,
    }


# ---------------------------------------------------------------------------
# §13 Telemetry
# ---------------------------------------------------------------------------
def _scan_all_benchmark_reports(session_dir: Path) -> Iterable[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return ()
    return sorted(runs.rglob("benchmark_*/benchmark_report.json"))


def _scan_torch_traces(session_dir: Path) -> list[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(p for p in runs.rglob("torch_trace*") if p.is_dir())


def _scan_system_profiles(session_dir: Path) -> list[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(p for p in runs.rglob("system_profile*") if p.is_dir())


def _scan_server_logs(session_dir: Path) -> list[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(runs.rglob("server*.log"))


def _aggregate_gpu_monitor(
    reports: list[Path],
    warnings: list[str],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for r in reports:
        d = _load_json_safe(r, warnings)
        if not isinstance(d, dict):
            continue
        gm = d.get("gpu_monitor")
        if isinstance(gm, list):
            for s in gm:
                if isinstance(s, dict):
                    samples.append(s)
        elif isinstance(gm, dict):
            samples.append(gm)
    if not samples:
        return {}

    def _avg(key: str) -> float:
        vals = [_to_float(s.get(key)) for s in samples]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _max(key: str) -> float:
        vals = [_to_float(s.get(key)) for s in samples]
        vals = [v for v in vals if v is not None]
        return round(max(vals), 2) if vals else 0.0

    return {
        "samples":       len(samples),
        "avg_power_w":   _avg("power_w") or _avg("power"),
        "max_power_w":   _max("power_w") or _max("power"),
        "avg_temp_c":    _avg("temperature_c") or _avg("temperature"),
        "max_temp_c":    _max("temperature_c") or _max("temperature"),
        "avg_clock_mhz": _avg("clock_mhz") or _avg("sclk_mhz"),
    }


def collect_telemetry(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    baseline_report: Path | None = None
    last_b = state.get("last_baseline") or {}
    if isinstance(last_b, dict) and last_b.get("workspace"):
        baseline_report = _find_benchmark_report(Path(last_b["workspace"]))

    profile_reports = [
        report for _task, report in _scan_profile_reports(session_dir)
    ]
    all_reports = list(_scan_all_benchmark_reports(session_dir))

    return {
        "baseline_report_path": _rel(baseline_report, session_dir) if baseline_report else None,
        "profile_report_paths": [_rel(p, session_dir) or str(p) for p in profile_reports],
        "torch_trace_paths":    [_rel(p, session_dir) or str(p) for p in _scan_torch_traces(session_dir)],
        "system_profile_paths": [_rel(p, session_dir) or str(p) for p in _scan_system_profiles(session_dir)],
        "server_log_paths":     [_rel(p, session_dir) or str(p) for p in _scan_server_logs(session_dir)],
        "gpu_monitor_aggregate": _aggregate_gpu_monitor(all_reports, warnings),
    }


# ---------------------------------------------------------------------------
# §14 Attribution
# ---------------------------------------------------------------------------
def _action_family(action: str) -> str:
    """Map an action label to a family for source_breakdown bucketing."""
    s = (action or "").lower()
    if s.startswith("kernel_opt") or s == "integrate":
        return "kernel"
    if s == "backends":
        return "backends"
    if s == "params":
        return "params"
    if s == "sweep":
        return "sweep"
    if s == "validate_stack":
        return "validate"
    return "other"


def collect_attribution(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    oob_invocations: list[dict[str, Any]],
    adopted_kernels: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    # Prefer authoritative per-stack ledger if Coordinator wrote it
    # (new state field `gain_per_stack_entry`). Otherwise reconstruct a
    # best-effort approximation from optimization_stack alone.
    entries = state.get("gain_per_stack_entry")
    if not isinstance(entries, list):
        entries = _reconstruct_gain_ledger(state, warnings)

    # Bucket entries by family for source_breakdown. Honor validated
    # total as the ground-truth denominator.
    validated_total = _to_float(state.get("cumulative_gain_validated")) or 0.0
    family_totals: dict[str, float] = {
        "kernel": 0.0, "backends": 0.0, "params": 0.0,
        "sweep": 0.0, "validate": 0.0, "other": 0.0,
    }
    for e in entries:
        if not isinstance(e, dict):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None:
            continue
        fam = _action_family(str(e.get("action") or ""))
        family_totals[fam] = family_totals.get(fam, 0.0) + max(delta, 0.0)

    # Split "kernel" between GEAK / OOB based on adopted KEEP entries' backend
    geak_kept_kids = {inv.get("kernel_id") for inv in geak_invocations if inv.get("decision") == "KEEP"}
    oob_kept_kids  = {inv.get("kernel_id") for inv in oob_invocations  if inv.get("decision") == "KEEP"}
    kernel_total = family_totals.get("kernel", 0.0)
    geak_total = 0.0
    oob_total = 0.0
    for k in adopted_kernels:
        kid = k.get("kernel_id")
        gain = _to_float(k.get("e2e_gain_pct")) or 0.0
        if kid in geak_kept_kids:
            geak_total += gain
        elif kid in oob_kept_kids:
            oob_total += gain
    if geak_total + oob_total == 0.0 and kernel_total > 0.0:
        # All-OOB by default when no KEEP'd adopt entry is on disk
        oob_total = kernel_total

    notes: list[str] = []
    if not isinstance(state.get("gain_per_stack_entry"), list):
        notes.append(
            "gain_per_stack_entry not written by Coordinator; "
            "attribution reconstructed best-effort from optimization_stack."
        )

    return {
        "gain_per_stack_entry": entries,
        "source_breakdown": {
            "geak_pct_of_total":     round(geak_total, 2),
            "oob_pct_of_total":      round(oob_total, 2),
            "backends_pct_of_total": round(family_totals.get("backends", 0.0), 2),
            "params_pct_of_total":   round(family_totals.get("params", 0.0), 2),
            "sweep_pct_of_total":    round(family_totals.get("sweep", 0.0), 2),
            "validated_total_pct":   round(validated_total, 2),
        },
        "notes": notes,
    }


def _reconstruct_gain_ledger(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Approximate per-stack contribution when Coordinator hasn't recorded it.

    optimization_stack is an ordered list. We treat each entry's
    ``gain_pct`` field as that step's delta. This is best-effort only;
    consumers should check ``attribution.notes`` for the caveat.
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list):
        return []
    cum_before = 0.0
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        delta = _to_float(entry.get("gain_pct"))
        cum_after = cum_before + (delta or 0.0)
        out.append({
            "ts":                str(entry.get("ts") or ""),
            "stack_len_before":  i,
            "stack_len_after":   i + 1,
            "action":            str(entry.get("action") or ""),
            "variant_name":      str(entry.get("variant_name") or ""),
            "cum_gain_before":   round(cum_before, 4),
            "cum_gain_after":    round(cum_after, 4),
            "delta_pct":         delta,
            "extra_sglang_args": str(entry.get("extra_sglang_args") or ""),
        })
        cum_before = cum_after
    return out


# ---------------------------------------------------------------------------
# source_files map
# ---------------------------------------------------------------------------
def collect_source_files(
    session_dir: Path,
    baseline_path: str | None,
    profile_reports: list[str],
    sweep_reports: list[str],
) -> dict[str, Any]:
    kernel_attempts = [
        _rel(run_dir / "optimization_attempts.jsonl", session_dir) or str(run_dir / "optimization_attempts.jsonl")
        for run_dir in _kernel_agent_run_dirs(session_dir)
        if (run_dir / "optimization_attempts.jsonl").exists()
    ]
    critic = session_dir / "critic-workdir"
    rob = session_dir / "robustness-workdir"
    return {
        "manifest":           "manifest.json",
        "state":              "state.json",
        "baseline_report":    baseline_path,
        "profile_reports":    profile_reports,
        "sweep_reports":      sweep_reports,
        "kernel_attempts":    kernel_attempts,
        "critic_workdir":     "critic-workdir" if critic.exists() else None,
        "robustness_workdir": "robustness-workdir" if rob.exists() else None,
    }


__all__ = [
    "collect_attribution",
    "collect_baseline",
    "collect_capability_summary",
    "collect_critic_robustness",
    "collect_final",
    "collect_kernel_invocations",
    "collect_kernel_lifecycle",
    "collect_param_search",
    "collect_phase_timeline",
    "collect_session",
    "collect_source_files",
    "collect_sweep",
    "collect_telemetry",
    "collect_workload",
]
