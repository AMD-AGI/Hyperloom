# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.common.io import safe_mtime

from ._common import (
    _load_json_safe,
    _load_jsonl_safe,
    _rel,
    _scan_profile_reports,
    _to_float,
)


# Kernel backend invocations
def _kernel_agent_run_dirs(session_dir: Path) -> list[Path]:
    """All ``<sd>/kernel-agent/runs/<sid>/`` dirs plus the two legacy layouts.

    Args:
        session_dir (Path): Absolute session root.

    Returns:
        list[Path]: Every kernel-agent run directory across the canonical and
        two legacy layouts. Empty when none exist.
    """
    candidates: list[Path] = []
    # Canonical: <sd>/kernel-agent/runs/<sid>/
    new_root = session_dir / "kernel-agent" / "runs"
    if new_root.is_dir():
        for sub in new_root.glob("*"):
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)
    # Legacy double-nested layout.
    legacy_root = session_dir / "kernel-agent-workspace"
    if legacy_root.is_dir():
        for sub in (legacy_root / "kernel-agent" / "runs").glob("*"):
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)
        # Even older per-kernel form.
        for kid_dir in legacy_root.glob("*/kernel-agent/runs/*"):
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

    Args:
        attempt (dict[str, Any]): One row from ``optimization_attempts.jsonl``.
        run_dir (Path): The kernel-agent run directory the attempt belongs to.
        session_dir (Path): Absolute session root (used to relativize paths).
        warnings (list[str]): Shared warnings list (kept for signature
            symmetry; not mutated here).

    Returns:
        dict[str, Any]: An Invocation dict for this single attempt, with
        kernel-level decision / verification fields left unset for later
        stamping.
    """
    kid = str(attempt.get("kernel_id") or "")
    backend = str(attempt.get("backend") or "").lower()
    attempt_id = str(
        attempt.get("attempt_id") or attempt.get("id") or attempt.get("run_id") or "",
    )

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

    micro_speedup = _to_float(attempt.get("speedup") or attempt.get("micro_speedup"))

    return {
        "kernel_id": kid,
        "attempt_id": attempt_id,
        "run_id": str(attempt.get("run_id") or run_dir.name),
        "ts": str(attempt.get("ts") or attempt.get("started_at") or ""),
        "backend": backend,
        "model": attempt.get("model"),
        "kernel_metadata": _shape_kernel_metadata({}, attempt),
        "prompt_path": _rel(prompt_path, session_dir) if prompt_path else None,
        "optimized_files": optimized_files,
        "result_path": _rel(result_path, session_dir) if result_path and result_path.exists() else None,
        "verification_path": _rel(verification_path, session_dir)
        if verification_path and verification_path.exists()
        else None,
        "decision": decision,
        "micro_speedup": micro_speedup,
        # compile/correctness are kernel-level; stamped later onto the BEST attempt.
        "compile_passed": None,
        "correctness_passed": None,
        "best_artifact_path": None,
        "error": attempt.get("error"),
        "cli_log_path": None,
    }


def _stamp_kernel_level_decisions(
    invocations: list[dict[str, Any]],
    run_dirs: list[Path],
    session_dir: Path,
    warnings: list[str],
) -> None:
    """Stamp the kernel-level KEEP/PARTIAL/REVERT decision onto the single best attempt per kernel.

    Reads kernel-level ``results/<kid>.json`` + ``verification/<kid>.json``;
    the best attempt is chosen by backend hint, else highest micro_speedup
    (ties: latest ts). Other attempts keep their per-attempt decision.

    Args:
        invocations (list[dict[str, Any]]): All parsed per-attempt invocations
            (mutated in place — the best attempt per kernel is stamped).
        run_dirs (list[Path]): The kernel-agent run directories the
            invocations came from.
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place).
    """
    # Group by (run_id, kernel_id); same kid in different run_dirs is separate.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for inv in invocations:
        key = (inv.get("run_id") or "", inv.get("kernel_id") or "")
        if not key[1]:
            continue
        groups.setdefault(key, []).append(inv)

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

        def _attempt_key(a: dict[str, Any]) -> tuple[float, str]:
            """Sort key selecting the best attempt for a kernel.

            Args:
                a (dict[str, Any]): One attempt invocation.

            Returns:
                tuple[float, str]: ``(micro_speedup, ts)`` with a missing
                speedup treated as ``-inf`` so it sorts last.
            """
            spd = a.get("micro_speedup")
            return (
                float(spd) if isinstance(spd, (int, float)) else float("-inf"),
                str(a.get("ts") or ""),
            )

        # Attribute the KEEP to the adopted backend via
        # ``verification.best_attempt_id`` then ``best_backend``; the micro/ts
        # heuristic is a last resort (it can pick a FAILED lane).
        best = None
        if isinstance(verification, dict):
            want_id = str(verification.get("best_attempt_id") or "")
            want_backend = str(verification.get("best_backend") or "").lower()
            if want_id:
                best = next(
                    (a for a in atts if str(a.get("attempt_id") or "") == want_id),
                    None,
                )
            if best is None and want_backend:
                cands = [a for a in atts if str(a.get("backend") or "").lower() == want_backend]
                if cands:
                    best = max(cands, key=_attempt_key)
        if best is None:
            best = max(atts, key=_attempt_key)

        if decision:
            best["decision"] = decision
        if isinstance(verification, dict):
            if best.get("micro_speedup") is None and verification.get("micro_speedup") is not None:
                best["micro_speedup"] = _to_float(verification.get("micro_speedup"))
            best["compile_passed"] = verification.get("compile_passed")
            best["correctness_passed"] = verification.get("correctness_passed")
            best["best_artifact_path"] = verification.get("best_artifact_path") or (
                result.get("best_artifact_path") if isinstance(result, dict) else None
            )
        if isinstance(result, dict) and result.get("cli_log_path"):
            best["cli_log_path"] = result["cli_log_path"]
        best["kernel_metadata"] = _shape_kernel_metadata(
            result,
            {
                "name": best.get("kernel_metadata", {}).get("name"),
                "source_file": best.get("kernel_metadata", {}).get("source_file"),
            },
        )


def _shape_kernel_metadata(
    result: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort kernel metadata, preferring ``result['kernel_metadata']`` over the attempt's own.

    Args:
        result (dict[str, Any]): The kernel-level ``results/<kid>.json`` dict
            (may be empty).
        attempt (dict[str, Any]): The per-attempt record used as a fallback
            source.

    Returns:
        dict[str, Any]: Shaped kernel metadata (name, source file, shapes,
        gpu_pct, arithmetic_intensity).
    """
    meta = result.get("kernel_metadata") if isinstance(result, dict) else None
    if isinstance(meta, dict) and meta:
        return {
            "name": meta.get("name") or "",
            "source_file": meta.get("source_file") or result.get("source_file") or "",
            "shapes": list(meta.get("shapes") or []),
            "gpu_pct": _to_float(meta.get("gpu_pct")),
            "arithmetic_intensity": _to_float(meta.get("arithmetic_intensity")),
        }
    return {
        "name": str(attempt.get("name") or ""),
        "source_file": str(attempt.get("source_file") or ""),
        "shapes": [],
        "gpu_pct": None,
        "arithmetic_intensity": None,
    }


def _infer_run_dir_kernel_id(run_dir: Path) -> str:
    """Recover the kernel id for a run dir whose attempts omit ``kernel_id``, only when the dir holds a single kid.

    Args:
        run_dir (Path): A kernel-agent run directory.

    Returns:
        str: The single kernel id inferred from ``results`` / ``verification``
        filenames, or ``""`` when zero or more than one is present.
    """
    kids: set[str] = set()
    for sub in ("results", "verification"):
        d = run_dir / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            if p.stem:
                kids.add(p.stem)
    return next(iter(kids)) if len(kids) == 1 else ""


def collect_kernel_invocations(
    session_dir: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(geak_invocations, forge_invocations)`` from optimization attempts.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        The ``(geak, forge)`` invocation lanes, each sorted by
        ``(kernel_id, ts)``.
    """
    all_invocations: list[dict[str, Any]] = []
    run_dirs = _kernel_agent_run_dirs(session_dir)
    for run_dir in run_dirs:
        attempts = _load_jsonl_safe(run_dir / "optimization_attempts.jsonl", warnings)
        parsed = [_parse_invocation_attempt(att, run_dir, session_dir, warnings) for att in attempts]
        # Backfill kernel_id when the jsonl row omitted it.
        if any(not (inv.get("kernel_id") or "") for inv in parsed):
            inferred = _infer_run_dir_kernel_id(run_dir)
            if inferred:
                for inv in parsed:
                    if inv.get("kernel_id"):
                        continue
                    inv["kernel_id"] = inferred
                    rp = run_dir / "results" / f"{inferred}.json"
                    vp = run_dir / "verification" / f"{inferred}.json"
                    if inv.get("result_path") is None and rp.exists():
                        inv["result_path"] = _rel(rp, session_dir) or str(rp)
                    if inv.get("verification_path") is None and vp.exists():
                        inv["verification_path"] = _rel(vp, session_dir) or str(vp)
        for inv in parsed:
            backend = inv.get("backend") or ""
            if not backend:
                inv["backend"] = "unknown"
            all_invocations.append(inv)

    # Stamp the kernel-level KEEP/PARTIAL/REVERT onto the BEST attempt per kernel.
    _stamp_kernel_level_decisions(all_invocations, run_dirs, session_dir, warnings)

    geak: list[dict[str, Any]] = []
    forge: list[dict[str, Any]] = []
    for inv in all_invocations:
        backend = inv.get("backend") or ""
        if backend == "geak":
            geak.append(inv)
        elif backend == "forge":
            forge.append(inv)
    for lane in (geak, forge):
        lane.sort(key=lambda e: (e.get("kernel_id") or "", e.get("ts") or ""))
    return geak, forge


def _read_kernel_candidates(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Return the ``hot_kernels`` array from ``kernel_candidates.json``.

    Resolves via the orchestrator-recorded path, then the new and legacy
    on-disk layouts (glob fallbacks), then ``last_trace_analyze.hot_kernels_top15``.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        list[dict[str, Any]]: The resolved ``hot_kernels`` list, or ``[]`` when
        no source yields a non-empty list.
    """
    sk = state.get("last_trace_analyze") or {}
    raw_path = sk.get("candidates_path") if isinstance(sk, dict) else None
    candidate_paths: list[Path] = []
    if raw_path:
        # Re-root the (usually container) path under session_dir via the
        # kernel-agent[-workspace] anchors before glob.
        p = Path(str(raw_path))
        candidate_paths.append(p)
        for anchor in ("kernel-agent-workspace", "kernel-agent"):
            try:
                idx = p.parts.index(anchor)
                candidate_paths.append(session_dir.joinpath(*p.parts[idx:]))
                break
            except ValueError:
                continue
    # New layout: <sd>/kernel-agent/runs/<session_id>/.
    candidate_paths.extend(sorted((session_dir / "kernel-agent").rglob("kernel_candidates.json")))
    # Legacy double-nested layout, kept so historical sessions rehydrate.
    candidate_paths.append(
        session_dir / "kernel-agent-workspace" / "kernel-agent" / "runs" / "hyperloom" / "kernel_candidates.json"
    )
    candidate_paths.extend(sorted((session_dir / "kernel-agent-workspace").rglob("kernel_candidates.json")))
    for path in candidate_paths:
        if not path or not path.exists():
            continue
        data = _load_json_safe(path, warnings)
        if isinstance(data, dict):
            hk = data.get("hot_kernels")
            # Only accept a non-empty list, else an on-disk ``hot_kernels: []``
            # would wrongly short-circuit the state fallback below.
            if isinstance(hk, list) and hk:
                return hk
    # Final fallback: state.last_trace_analyze.hot_kernels_top15 (the
    # orchestrator's truncated copy), used when the on-disk file is missing.
    inline = sk.get("hot_kernels_top15") if isinstance(sk, dict) else None
    if isinstance(inline, list):
        return inline
    return []


def _index_invocations_by_kernel(
    invs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fold per-attempt invocations into a per-kernel summary.

    Args:
        invs (list[dict[str, Any]]): Per-attempt invocation records.

    Returns:
        dict[str, dict[str, Any]]: Per-kernel summary keyed by ``kernel_id``.
    """
    out: dict[str, dict[str, Any]] = {}
    for inv in invs:
        kid = str(inv.get("kernel_id") or "")
        if not kid:
            continue
        ent = out.setdefault(
            kid,
            {
                "attempts": 0,
                "best_speedup": None,
                "decision": "",
                "last_status": "",
            },
        )
        ent["attempts"] += 1
        spd = inv.get("micro_speedup")
        if isinstance(spd, (int, float)):
            cur = ent["best_speedup"]
            if cur is None or float(spd) > cur:
                ent["best_speedup"] = float(spd)
        # KEEP/PARTIAL outrank everything else; FAILED never overrides KEEP.
        dec = str(inv.get("decision") or "")
        if dec in ("KEEP", "PARTIAL") or not ent["decision"]:
            ent["decision"] = dec or ent["decision"]
        ent["last_status"] = str(inv.get("status") or ent["last_status"])
    return out


def _collect_detected_kernels(
    session_dir: Path,
    state: dict[str, Any],
    geak: list[dict[str, Any]],
    warnings: list[str],
    *,
    forge: list[dict[str, Any]] | None = None,
    cap: int | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical per-kernel lifecycle row, keyed by ``kernel_id``.

    Merges static profile fields (from ``kernel_candidates.json``,
    preferred, else ``benchmark_report.kernel_summary``),
    ``selected_for_optimization``, per-lane ``geak`` / ``forge``
    summaries, ``adopted_by`` (from integrate KEEPs), and ``final_decision``.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        geak (list[dict[str, Any]]): GEAK-lane invocations.
        warnings (list[str]): Shared warnings list (mutated in place).
        forge (list[dict[str, Any]] | None): Forge-lane invocations. Defaults
            to ``None`` (treated as empty).
        cap (int | None): Optional maximum number of rows to return (highest
            GPU share first). Defaults to ``None`` (no cap).

    Returns:
        list[dict[str, Any]]: Per-kernel lifecycle rows sorted by descending
        GPU share, truncated to ``cap`` when given.
    """
    forge = forge or []
    by_kid: dict[str, dict[str, Any]] = {}

    # 1) candidates.json (preferred: has call_count / duration_us).
    for k in _read_kernel_candidates(session_dir, state, warnings):
        if not isinstance(k, dict):
            continue
        kid = str(k.get("kernel_id") or k.get("name") or "")
        if not kid:
            continue
        by_kid[kid] = {
            "kernel_id": kid,
            "name": str(k.get("name") or ""),
            "gpu_pct": _to_float(k.get("gpu_pct")),
            "duration_us": _to_float(k.get("duration_us")),
            "call_count": int(k.get("call_count") or 0) or None,
            "bandwidth_util_pct": _to_float(k.get("bandwidth_utilization_pct")),
            "compute_util_pct": _to_float(k.get("compute_utilization_pct")),
            "kernel_category": str(k.get("kernel_category") or ""),
            "bottleneck": str(k.get("bottleneck") or ""),
            "arithmetic_intensity": _to_float(k.get("arithmetic_intensity")),
            "reusable_native_kernel": bool(k.get("reusable_native_kernel")),
            "source_file": k.get("source_file") or "",
            "recommended_actions": list(k.get("recommended_actions") or []),
            "recommended_backends": list(k.get("recommended_backends") or []),
            "optimization_notes": str(k.get("optimization_notes") or ""),
        }

    # 2) benchmark_report.kernel_summary fallback for the tail of trace kernels
    #    that didn't make the top-N candidates. Dedupe by name against the
    #    candidates entries; new fallback entries get a short ``rNNN`` alias.
    name_to_kid = {e["name"]: kid for kid, e in by_kid.items() if e.get("name")}
    residual_counter = 0
    for task_dir, report_path in _scan_profile_reports(session_dir):
        report = _load_json_safe(report_path, warnings)
        if not isinstance(report, dict):
            continue
        kernel_summary = report.get("kernel_summary") or []
        bottlenecks = report.get("top_bottlenecks") or []
        bottleneck_by_kid = {
            b.get("kernel_id"): b for b in (bottlenecks if isinstance(bottlenecks, list) else []) if isinstance(b, dict)
        }
        for k in kernel_summary if isinstance(kernel_summary, list) else []:
            if not isinstance(k, dict):
                continue
            name_str = str(k.get("name") or "")
            if not name_str:
                continue
            existing_kid = name_to_kid.get(name_str)
            if existing_kid is not None:
                # Merge missing fields into the candidates-side entry.
                entry = by_kid[existing_kid]
                if entry.get("gpu_pct") is None:
                    entry["gpu_pct"] = _to_float(k.get("gpu_pct"))
                if entry.get("duration_us") is None:
                    t_ms = _to_float(k.get("time_ms"))
                    entry["duration_us"] = (t_ms * 1000.0) if t_ms is not None else None
                if not entry.get("bottleneck"):
                    bn = bottleneck_by_kid.get(k.get("kernel_id")) or {}
                    entry["bottleneck"] = str(bn.get("bottleneck") or k.get("bottleneck") or "")
                if entry.get("arithmetic_intensity") is None:
                    entry["arithmetic_intensity"] = _to_float(k.get("arithmetic_intensity"))
                continue

            input_kid = str(k.get("kernel_id") or "")
            # Keep the input kernel_id when it's already a short alias (e.g.
            # ``k002``); mangled C++ symbols get a generated ``rNNN`` alias.
            is_short_alias = input_kid and input_kid != name_str and len(input_kid) <= 8 and input_kid not in by_kid
            if is_short_alias:
                alias = input_kid
            else:
                residual_counter += 1
                alias = f"r{residual_counter:03d}"
            bn = bottleneck_by_kid.get(k.get("kernel_id")) or {}
            t_ms = _to_float(k.get("time_ms"))
            by_kid[alias] = {
                "kernel_id": alias,
                "name": name_str,
                "gpu_pct": _to_float(k.get("gpu_pct")),
                "duration_us": (t_ms * 1000.0) if t_ms is not None else None,
                "call_count": None,
                "bandwidth_util_pct": None,
                "compute_util_pct": None,
                "kernel_category": "",
                "bottleneck": str(bn.get("bottleneck") or k.get("bottleneck") or ""),
                "arithmetic_intensity": _to_float(k.get("arithmetic_intensity")),
                "reusable_native_kernel": bool(k.get("reusable_native_kernel")),
                "source_file": k.get("source_file") or "",
                "recommended_actions": [],
                "recommended_backends": [],
                "optimization_notes": "",
                "detected_from_task": task_dir.name,
                "benchmark_report_path": _rel(report_path, session_dir) or str(report_path),
            }
            name_to_kid[name_str] = alias

    # 3) lifecycle stamps (selected / geak / forge / adopted_by / final_decision)
    selected_ids = {
        str(e.get("kernel_id") or "")
        for e in ((state.get("last_trace_analyze") or {}).get("hot_kernels_top15") or [])
        if isinstance(e, dict)
    }
    geak_idx = _index_invocations_by_kernel(geak)
    forge_idx = _index_invocations_by_kernel(forge)

    integ = state.get("kernel_integrate_attempts") or {}
    adopted_kids: set[str] = set()
    reverted_kids: set[str] = set()
    integ_gain_by_kid: dict[str, float | None] = {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
            integ_gain_by_kid[kid] = _to_float(ent.get("best_gain_pct"))
            dec = ent.get("last_decision")
            if dec == "KEEP":
                adopted_kids.add(kid)
            elif dec in ("REVERT", "REJECT"):
                reverted_kids.add(kid)

    rejected_kids = {str(k or "") for k in (state.get("rejected_kernel_ids") or [])} - adopted_kids

    for kid, entry in by_kid.items():
        entry["selected_for_optimization"] = kid in selected_ids
        entry["geak"] = geak_idx.get(kid)  # None if lane never touched this kid
        entry["forge"] = forge_idx.get(kid)
        # e2e (integrate) gain so the table shows why a micro-KEPT kernel reverted.
        if kid in integ_gain_by_kid:
            entry["integrate_gain_pct"] = integ_gain_by_kid[kid]
        if kid in adopted_kids:
            # Disambiguate which lane's patch was kept. Pick the KEPT lane
            # with the highest micro-speedup; fall back to 'kernel_agent' when
            # integrate KEPT but no single lane shows a KEEP.
            kept_lanes: list[tuple[str, float]] = []
            for lane in ("geak", "forge"):
                row = entry.get(lane)
                if row and row.get("decision") in ("KEEP", "PARTIAL"):
                    kept_lanes.append((lane, row.get("best_speedup") or 0.0))
            if kept_lanes:
                entry["adopted_by"] = max(kept_lanes, key=lambda t: t[1])[0]
            else:
                # Integrate KEEP but no lane KEPT: record 'kernel_agent', don't guess.
                entry["adopted_by"] = "kernel_agent"
            entry["final_decision"] = "kept"
        elif kid in reverted_kids:
            entry["adopted_by"] = None
            entry["final_decision"] = "reverted"
        elif kid in rejected_kids:
            entry["adopted_by"] = None
            entry["final_decision"] = "rejected"
        elif entry["geak"] or entry["forge"]:
            entry["adopted_by"] = None
            entry["final_decision"] = "attempted"
        else:
            entry["adopted_by"] = None
            entry["final_decision"] = "not_optimized"

    # Sort by GPU share descending.
    out = sorted(
        by_kid.values(),
        key=lambda e: (-(e.get("gpu_pct") or 0.0), e.get("kernel_id") or ""),
    )
    if cap is not None and len(out) > cap:
        return out[:cap]
    return out


def _collect_recommended_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Shape the orchestrator's recommended hot-kernels list.

    Reads ``state.last_trace_analyze.hot_kernels_top15`` and projects each
    entry to the report's recommended-kernel shape.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        list[dict[str, Any]]: One row per recommended kernel (id / name /
        gpu_pct / recommended backends + actions / bottleneck). Empty when no
        trace-analyze recommendations exist.
    """
    sk = state.get("last_trace_analyze") or {}
    if not isinstance(sk, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in sk.get("hot_kernels_top15") or []:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "kernel_id": str(entry.get("kernel_id") or ""),
                "name": str(entry.get("name") or ""),
                "gpu_pct": _to_float(entry.get("gpu_pct")),
                "recommended_backends": list(entry.get("recommended_backends") or []),
                "recommended_actions": list(entry.get("recommended_actions") or []),
                "bottleneck": str(entry.get("bottleneck") or ""),
                "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
            }
        )
    return out


def _collect_optimized_kernels(
    geak: list[dict[str, Any]],
    state: dict[str, Any],
    forge: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fold per-attempt invocations into per-kernel optimization summaries.

    Aggregates all lanes' attempts per kernel (counts, best micro-speedup,
    best artifact, decision history) and cross-references
    ``state.kernel_opt_attempts`` to recover decisions whose on-disk
    verification was rotated away.

    Args:
        geak (list[dict[str, Any]]): GEAK-lane invocations.
        state (dict[str, Any]): Parsed ``state.json``.
        forge (list[dict[str, Any]] | None): Forge-lane invocations.

    Returns:
        list[dict[str, Any]]: Per-kernel optimization summaries sorted by
        ``kernel_id``.
    """
    by_kid: dict[str, dict[str, Any]] = {}
    for invs in (geak, forge or []):
        for inv in invs:
            kid = inv.get("kernel_id") or ""
            if not kid:
                continue
            entry = by_kid.setdefault(
                kid,
                {
                    "kernel_id": kid,
                    "backend": inv.get("backend") or "",
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "best_micro_speedup": None,
                    "last_decision": "",
                    "best_artifact_path": None,
                    "attempts_summary": [],
                },
            )
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
            entry["attempts_summary"].append(
                {
                    "attempt_id": inv.get("attempt_id"),
                    "backend": inv.get("backend"),
                    "decision": inv.get("decision"),
                    "micro_speedup": spd,
                    "ts": inv.get("ts"),
                }
            )
    # Cross-reference the stable task ledger (covers ordinal reuse and rotated
    # on-disk verification), falling back to legacy per-ordinal state.
    ko_attempts = state.get("kernel_opt_task_attempts") or state.get("kernel_opt_attempts") or {}
    if isinstance(ko_attempts, dict):
        for ledger_id, ent in ko_attempts.items():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("current_kernel_id") or ent.get("kernel_id") or ledger_id)
            entry = by_kid.setdefault(
                kid,
                {
                    "kernel_id": kid,
                    "backend": "",
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "best_micro_speedup": None,
                    "last_decision": "",
                    "best_artifact_path": None,
                    "attempts_summary": [],
                },
            )
            entry["total_attempts"] = max(entry["total_attempts"], int(ent.get("attempts", 0)))
            entry["last_decision"] = entry["last_decision"] or str(ent.get("last_decision") or "")
    return sorted(by_kid.values(), key=lambda e: e.get("kernel_id") or "")


def _ledger_entry_is_adopted(ent: dict[str, Any]) -> bool:
    """Report whether a ledger row represents an adopted kernel patch.

    ``validated`` tracks single-kernel gain attribution, not adoption. GEAK
    joint rebench rows and later failed revalidations can be ``validated=False``
    while the kernel was still promoted with a proven overlay or an earlier
    ``KEEP`` attempt.
    """
    if ent.get("last_decision") == "KEEP":
        return True
    source = str(ent.get("source") or "")
    if source != "geak_e2e":
        # Forge / integrate writers stamp REVERT on the entry itself; a prior
        # KEEP attempt must not resurrect a kernel that was later rejected.
        return False
    attempts = ent.get("attempts") or []
    has_keep = any(isinstance(a, dict) and a.get("decision") == "KEEP" for a in attempts)
    if ent.get("overlay_loaded") is True:
        return True
    return has_keep


def _collect_adopted_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect KEEP-promoted (adopted) kernel patch entries.

    Reads ``state.kernel_integrate_attempts`` and keeps entries that were
    promoted (``KEEP`` or proven GEAK overlay / historical ``KEEP``).

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        list[dict[str, Any]]: One row per adopted kernel patch (kernel id,
        patch / target paths, extra server args, validated e2e gain, status,
        adoption timestamp, attempt count). Empty when none were adopted.
    """
    out: list[dict[str, Any]] = []
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for key, ent in integ.items():
            if not isinstance(ent, dict):
                continue
            if not _ledger_entry_is_adopted(ent):
                continue
            out.append(
                {
                    "kernel_id": str(ent.get("kernel_id") or ""),
                    "patch_path": str(ent.get("patch_path") or ""),
                    "target_file": str(ent.get("target_file") or ""),
                    "extra_server_args": str(ent.get("extra_server_args") or ""),
                    "e2e_gain_pct": _to_float(ent.get("best_gain_pct")),
                    # Writers that cannot attribute the measured gain to this
                    # one kernel say so; everything else stays validated, as
                    # every pre-existing writer's row was.
                    "validated": bool(ent.get("validated", True)),
                    "last_status": str(ent.get("last_status") or ""),
                    "adopted_at": str(ent.get("updated_at") or ""),
                    "attempt_count": int(ent.get("attempt_count") or 0),
                    "basis": str(ent.get("basis") or ""),
                    "alignment_status": str(ent.get("alignment_status") or ""),
                }
            )
    return out


def _collect_rejected_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect rejected / retired kernel patch entries.

    Reads ``state.rejected_kernel_patches`` and additionally surfaces any
    ``state.rejected_kernel_ids`` that never made it into the patch list
    (marked with ``reason="retired"``).

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        list[dict[str, Any]]: One row per rejected kernel (id, reason, patch /
        target paths, attempt count, best gain, timestamp).
    """
    out: list[dict[str, Any]] = []
    rejected = state.get("rejected_kernel_patches") or []
    if isinstance(rejected, list):
        for r in rejected:
            if not isinstance(r, dict):
                continue
            out.append(
                {
                    "kernel_id": str(r.get("kernel_id") or ""),
                    "reason": str(r.get("reason") or ""),
                    "patch_path": r.get("patch_path"),
                    "target_file": r.get("target_file"),
                    "attempt_count": int(r.get("attempt_count") or 0),
                    "best_gain_pct": _to_float(r.get("best_gain_pct")),
                    "ts": str(r.get("ts") or ""),
                }
            )
    # also surface rejected_kernel_ids that didn't make it into rejected_kernel_patches
    seen_ids = {entry["kernel_id"] for entry in out if entry.get("kernel_id")}
    for kid in state.get("rejected_kernel_ids") or []:
        kid_s = str(kid or "")
        if not kid_s or kid_s in seen_ids:
            continue
        out.append(
            {
                "kernel_id": kid_s,
                "reason": "retired",
                "patch_path": None,
                "target_file": None,
                "attempt_count": 0,
                "best_gain_pct": None,
                "ts": "",
            }
        )
    return out


def collect_kernel_lifecycle(
    session_dir: Path,
    state: dict[str, Any],
    geak: list[dict[str, Any]],
    warnings: list[str],
    forge: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Collect the kernel-lifecycle section.

    Bundles the five per-kernel views — detected, recommended, optimized,
    adopted, and rejected — into one section.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        geak (list[dict[str, Any]]): GEAK-lane invocations.
        warnings (list[str]): Shared warnings list (mutated in place).
        forge (list[dict[str, Any]] | None): Forge-lane invocations (own lane).

    Returns:
        dict[str, Any]: ``{"detected", "recommended", "optimized", "adopted",
        "rejected"}`` lists.
    """
    forge = forge or []
    return {
        "detected": _collect_detected_kernels(session_dir, state, geak, warnings, forge=forge),
        "recommended": _collect_recommended_kernels(state),
        "optimized": _collect_optimized_kernels(geak, state, forge),
        "adopted": _collect_adopted_kernels(state),
        "rejected": _collect_rejected_kernels(state),
    }


# Kernel Optimization Summary.
_KERNEL_OPT_SUMMARY_REL_PATH = "reports/kernel_optimization_summary.json"


def collect_kernel_optimization_summary(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Mirror ``reports/kernel_optimization_summary.json`` into its section.

    Missing → ``{}`` (quiet); malformed → ``{}`` + warning. Otherwise
    mirrored verbatim (producer additions ride through) apart from shape
    guards on the iterated containers + an added ``report_path``.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place on
            malformed input).

    Returns:
        dict[str, Any]: The mirrored summary section (with a ``report_path``),
        or ``{}`` when the file is absent / not a JSON object.
    """
    path = session_dir / _KERNEL_OPT_SUMMARY_REL_PATH
    if not path.exists():
        # Quiet on absence (mirrors collect_kernel_roofline).
        return {}
    blob = _load_json_safe(path, warnings)
    if not isinstance(blob, dict):
        warnings.append(f"kernel_optimization_summary: {_KERNEL_OPT_SUMMARY_REL_PATH} is not a JSON object")
        return {}

    out = dict(blob)

    raw_by_kernel = out.get("by_kernel")
    if raw_by_kernel is None:
        out["by_kernel"] = []
    elif not isinstance(raw_by_kernel, list):
        warnings.append("kernel_optimization_summary.by_kernel is not a list; dropping entries")
        out["by_kernel"] = []
    else:
        # Drop non-dict rows; pass the rest through verbatim.
        out["by_kernel"] = [r for r in raw_by_kernel if isinstance(r, dict)]

    for key in (
        "totals",
        "rejection_breakdown",
        "unattempted_reason_breakdown",
        "failure_reason_breakdown",
        "dispatch_skip_reason",
        "field_glossary",
    ):
        val = out.get(key)
        if val is not None and not isinstance(val, dict):
            warnings.append(f"kernel_optimization_summary.{key} is not an object; dropping")
            out[key] = {}

    takeaways = out.get("top_takeaways")
    if takeaways is not None and not isinstance(takeaways, list):
        warnings.append("kernel_optimization_summary.top_takeaways is not a list; dropping")
        out["top_takeaways"] = []

    out["report_path"] = _rel(path, session_dir) or _KERNEL_OPT_SUMMARY_REL_PATH
    return out


# Conc Sweep Summary.
_CONC_SWEEP_SUMMARY_REL_PATH = "reports/conc_sweep_summary.json"

_CONC_SWEEP_VARIANT_RE = re.compile(r"^(baseline|optimized)_conc(\d+)$")


def _conc_sweep_successful_pairs(summary: dict[str, Any]) -> int:
    try:
        return int((summary.get("summary") or {}).get("successful_pairs") or 0)
    except (TypeError, ValueError):
        return 0


def _load_conc_variant_point(variant_dir: Path, *, arm: str, conc: int) -> dict[str, Any]:
    """Best-effort point extraction from a conc_sweep variant workspace."""
    result_paths = sorted(
        variant_dir.rglob("inferencex_result.json"),
        key=safe_mtime,
        reverse=True,
    )
    for result_path in result_paths:
        data = _load_json_safe(result_path, [])
        if not isinstance(data, dict):
            continue
        tput = data.get("output_throughput")
        if tput is None:
            tput = data.get("total_output_throughput")
        try:
            tput_f = float(tput)
        except (TypeError, ValueError):
            continue
        return {
            "arm": arm,
            "conc": conc,
            "status": "succeeded",
            "output_throughput": tput_f,
            "request_throughput": data.get("request_throughput"),
            "total_token_throughput": data.get("total_token_throughput"),
            "raw_result_path": _rel(
                result_path,
                variant_dir.parents[2] if len(variant_dir.parents) >= 3 else variant_dir,
            ),
        }
    return {
        "arm": arm,
        "conc": conc,
        "status": "failed",
        "output_throughput": None,
    }


def _recover_conc_sweep_summary_from_runs(
    session_dir: Path,
    warnings: list[str],
    *,
    metric_key: str = GRADED_OUTPUT,
) -> dict[str, Any]:
    """Recover a conc_sweep summary from raw run workspaces when the report is stale.

    ``metric_key`` is the axis the recovered speedups are taken on. A recovery
    that supersedes a report has to reuse that report's own axis, or the
    section silently swaps one quantity for another.
    """
    runs_dir = session_dir / "runs" / "conc_sweep"
    if not runs_dir.exists():
        return {}
    tasks = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=safe_mtime,
        reverse=True,
    )
    best_payload: dict[str, Any] = {}
    best_pairs = 0
    for task_dir in tasks:
        baseline_points: list[dict[str, Any]] = []
        optimized_points: list[dict[str, Any]] = []
        try:
            for variant_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                match = _CONC_SWEEP_VARIANT_RE.match(variant_dir.name)
                if not match:
                    continue
                arm, conc_text = match.groups()
                point = _load_conc_variant_point(variant_dir, arm=arm, conc=int(conc_text))
                if arm == "baseline":
                    baseline_points.append(point)
                else:
                    optimized_points.append(point)
        except OSError:
            continue
        if not baseline_points and not optimized_points:
            continue
        baseline_points.sort(key=lambda p: p["conc"])
        optimized_points.sort(key=lambda p: p["conc"])
        comparison, summary = conc_pair_comparison(baseline_points, optimized_points, metric_key=metric_key)
        pairs = int(summary.get("successful_pairs") or 0)
        payload = {
            "schema_version": "recovered-v1",
            "status": "succeeded" if pairs else "failed",
            "source": "recovered_from_runs",
            "workspace": task_dir.as_posix(),
            "baseline": {"points": baseline_points},
            "optimized": {"points": optimized_points},
            "comparison": comparison,
            "summary": summary,
            "report_path": _rel(task_dir, session_dir) or task_dir.as_posix(),
        }
        if pairs > best_pairs:
            best_payload = payload
            best_pairs = pairs
    return best_payload


def collect_conc_sweep_summary(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Mirror ``reports/conc_sweep_summary.json`` into its section.

    Missing → recover from ``runs/conc_sweep``, returning the recovered
    payload + a warning when it yields successful pairs, else ``{}``;
    malformed → ``{}`` + warning. Otherwise mirrored verbatim (only
    ``comparison`` shape-guarded) + ``report_path``, except that a mirrored
    report with zero successful pairs is superseded by the recovered payload
    (``source="recovered_from_runs"``, ``schema_version="recovered-v1"``, the
    mirrored path kept as ``original_report_path``) when recovery finds pairs,
    taken on the axis that report named. A recovery standing in for an absent
    report has no axis to read and takes the default.
    Do not synthesize the optional blocks the producer omits when
    ``status="skipped"``.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place on
            malformed input).

    Returns:
        dict[str, Any]: The mirrored conc-sweep summary (with a
        ``report_path``), the run-workspace recovery payload, or ``{}`` when
        the file is absent / not a JSON object and recovery finds no pairs.
    """
    path = session_dir / _CONC_SWEEP_SUMMARY_REL_PATH
    if not path.exists():
        recovered = _recover_conc_sweep_summary_from_runs(session_dir, warnings)
        if _conc_sweep_successful_pairs(recovered) > 0:
            warnings.append(
                "conc_sweep_summary: reports/conc_sweep_summary.json absent; recovered from runs/conc_sweep"
            )
            return recovered
        return {}
    blob = _load_json_safe(path, warnings)
    if not isinstance(blob, dict):
        warnings.append(f"conc_sweep_summary: {_CONC_SWEEP_SUMMARY_REL_PATH} is not a JSON object")
        return {}

    out = dict(blob)

    comparison = out.get("comparison")
    if comparison is not None and not isinstance(comparison, list):
        warnings.append("conc_sweep_summary.comparison is not a list; dropping entries")
        out["comparison"] = []

    out["report_path"] = _rel(path, session_dir) or _CONC_SWEEP_SUMMARY_REL_PATH

    # Fall back to run-workspace recovery only when the authoritative report
    # has no successful pairs (also skips the full runs/ scan on the healthy path).
    if _conc_sweep_successful_pairs(out) == 0:
        reported_metric = str((out.get("summary") or {}).get("metric") or "").strip()
        recovered = _recover_conc_sweep_summary_from_runs(
            session_dir, warnings, metric_key=reported_metric or GRADED_OUTPUT
        )
        if _conc_sweep_successful_pairs(recovered) > 0:
            recovered["original_report_path"] = out["report_path"]
            warnings.append("conc_sweep_summary: recovered successful conc_sweep data from runs/conc_sweep")
            return recovered
    return out


# Optimization stack — raw KEEP ledger passthrough with the full per-entry
# evidence the summarised sections drop.
def collect_optimization_stack(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mirror ``state.optimization_stack[]`` to the breakdown; never raises.

    Each entry is stamped ``validated`` (within
    ``cumulative_gain_validated_stack_len``). Returns ``[]`` when absent.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        list[dict[str, Any]]: One normalized stack entry per KEEP (each flagged
        ``validated``), or ``[]`` when the stack is absent.
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list):
        return []
    try:
        validated_len = int(state.get("cumulative_gain_validated_stack_len") or 0)
    except (TypeError, ValueError):
        validated_len = 0
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        out.append(_normalize_optimization_stack_entry(entry, validated=idx < validated_len))
    return out


def _normalize_optimization_stack_entry(
    raw: dict[str, Any],
    *,
    validated: bool,
) -> dict[str, Any]:
    """Coerce one stack entry to the schema shape; unknown fields pass through verbatim.

    Args:
        raw (dict[str, Any]): One raw ``optimization_stack`` entry.
        validated (bool): Whether the entry falls within the validated stack
            length.

    Returns:
        dict[str, Any]: The coerced stack entry with optional evidence fields
        (gemm-tuning and collective) included only when present in ``raw``.
    """
    # Known fields — coerced types
    out: dict[str, Any] = {
        "action": str(raw.get("action") or ""),
        "variant_name": str(raw.get("variant_name") or ""),
        "candidate_extra_server_args": str(raw.get("candidate_extra_server_args") or ""),
        "extra_envs": dict(raw.get("extra_envs") or {}),
        "tput": _to_float(raw.get("tput")),
        "ts": str(raw.get("ts") or ""),
        "workspace": raw.get("workspace"),
        "validated": bool(validated),
    }
    # gemm_tuning-specific evidence (optional).
    if "engine" in raw:
        out["engine"] = str(raw.get("engine") or "")
    if "tuned_file" in raw:
        out["tuned_file"] = str(raw.get("tuned_file") or "")
    if "final_report_path" in raw:
        out["final_report_path"] = str(raw.get("final_report_path") or "")
    if "source" in raw:
        out["source"] = str(raw.get("source") or "")
    if "gain_pct" in raw:
        out["gain_pct"] = _to_float(raw.get("gain_pct"))
    if "kernel_id" in raw:
        out["kernel_id"] = str(raw.get("kernel_id") or "")
    if "fingerprint" in raw:
        out["fingerprint"] = str(raw.get("fingerprint") or "")
    if "provenance" in raw:
        out["provenance"] = str(raw.get("provenance") or "")
    if "task_id" in raw:
        out["task_id"] = str(raw.get("task_id") or "")
    if "source_phase" in raw:
        out["source_phase"] = str(raw.get("source_phase") or "")
    if "operation_kind" in raw:
        out["operation_kind"] = str(raw.get("operation_kind") or "")
    if "scope" in raw:
        out["scope"] = str(raw.get("scope") or "")
    # collective-specific evidence (optional); ``integration_id`` /
    # ``collective_attempt_id`` join the entry back to its campaign record.
    if "collective_op" in raw:
        out["collective_op"] = str(raw.get("collective_op") or "")
    if "world_size" in raw:
        out["world_size"] = raw.get("world_size")
    if "collective_attempt_id" in raw:
        out["collective_attempt_id"] = str(raw.get("collective_attempt_id") or "")
    if "integration_id" in raw:
        out["integration_id"] = str(raw.get("integration_id") or "")
    return out


def _resolve_gemm_engine(record: dict[str, Any]) -> str:
    """Resolve the GEMM-tuning engine label for a run/stack record.

    Prefer ``engine``, then ``backend`` (the forge lane records its tuner
    there), then fall back to ``geak`` for records that carry neither.
    """
    return str(record.get("engine") or record.get("backend") or "geak")


def collect_gemm_tuning(state: dict[str, Any]) -> dict[str, Any]:
    """Build the top-level ``gemm_tuning`` section from session state; never raises.

    Assembles one run per ``state.gemm_tuning_attempts[]`` entry (falling back
    to ``last_gemm_tuning`` when the history is absent), tags each with its
    tuning ``engine`` (``geak`` today, ``forge`` later), and cross-references
    ``optimization_stack`` so a run whose ``tuned_file`` was kept is marked
    ``adopted`` with the kept gain. Gain is mirrored here for the optimization
    layer while ``attribution`` remains the authoritative roll-up.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        dict[str, Any]: A ``GemmTuning`` envelope (``runs`` + adopted summary),
        or ``{}`` when the session ran no GEMM tuning.
    """
    attempts = state.get("gemm_tuning_attempts")
    if not isinstance(attempts, list) or not attempts:
        last = state.get("last_gemm_tuning")
        attempts = [last] if isinstance(last, dict) and last else []
    if not attempts:
        return {}

    # tuned_file -> (gain_pct, validated) for KEEPs already in the stack.
    adopted_gain: dict[str, dict[str, Any]] = {}
    stack = state.get("optimization_stack")
    validated_len = 0
    try:
        validated_len = int(state.get("cumulative_gain_validated_stack_len") or 0)
    except (TypeError, ValueError):
        validated_len = 0
    if isinstance(stack, list):
        for idx, item in enumerate(stack):
            if not isinstance(item, dict) or item.get("action") != "gemm_tuning":
                continue
            tf = str(item.get("tuned_file") or "").strip()
            if not tf:
                continue
            adopted_gain[tf] = {
                "gain_pct": _to_float(item.get("gain_pct")),
                "validated": idx < validated_len,
                "engine": _resolve_gemm_engine(item),
            }

    baseline_tput = _to_float(state.get("baseline_tput"))
    knob_tp = state.get("tp")
    knob_conc = state.get("conc")
    knob_isl = state.get("isl")
    knob_osl = state.get("osl")
    gpu_type = str(state.get("gpu_type") or "").strip()
    precision = str(state.get("precision") or "").strip()
    framework = str(state.get("framework") or "").strip()

    runs: list[dict[str, Any]] = []
    for raw in attempts:
        if not isinstance(raw, dict):
            continue
        engine = _resolve_gemm_engine(raw)
        e2e_gain_pct = _to_float(raw.get("e2e_gain_pct"))
        speedup = _to_float(raw.get("best_speedup"))
        gain_pct: float | None = None
        tuned_tput: float | None = None
        if e2e_gain_pct is not None:
            gain_pct = e2e_gain_pct
            if baseline_tput is not None:
                tuned_tput = baseline_tput * (1.0 + e2e_gain_pct / 100.0)
        elif speedup is not None:
            gain_pct = (speedup - 1.0) * 100.0
            if baseline_tput is not None:
                tuned_tput = baseline_tput * speedup
        tuned_file = str(raw.get("tuned_file") or "").strip()
        adopted = tuned_file in adopted_gain
        # Prefer the kept gain (validated e2e) when this run was adopted.
        if adopted and adopted_gain[tuned_file].get("gain_pct") is not None:
            gain_pct = adopted_gain[tuned_file]["gain_pct"]

        run: dict[str, Any] = {
            "engine": engine,
            "status": str(raw.get("status") or ""),
            "decision": str(raw.get("decision") or ""),
            "source": str(raw.get("source") or ""),
            "ts": str(raw.get("ts") or ""),
            "duration_sec": next(
                (
                    _to_float(raw.get(field))
                    for field in ("duration_sec", "elapsed_sec", "elapsed")
                    if _to_float(raw.get(field)) is not None
                ),
                None,
            ),
            "error_class": str(raw.get("error_class") or ""),
            "error": str(raw.get("error") or raw.get("reason") or ""),
            "precision": str(raw.get("precision") or precision),
            "framework": str(raw.get("framework") or framework),
            "gpu_type": str(raw.get("gpu_type") or gpu_type),
            "baseline_tput": baseline_tput,
            "best_speedup": speedup,
            "e2e_gain_pct": e2e_gain_pct,
            "gain_pct": gain_pct,
            "tuned_tput": tuned_tput,
            "tuned_file": tuned_file,
            "final_report_path": str(raw.get("final_report_path") or ""),
            "workspace": str(raw.get("workspace") or ""),
            "adopted": adopted,
        }
        for knob, val in (("tp", knob_tp), ("conc", knob_conc), ("isl", knob_isl), ("osl", knob_osl)):
            raw_knob = raw.get(knob)
            chosen = raw_knob if raw_knob not in (None, "") else val
            if chosen not in (None, ""):
                try:
                    run[knob] = int(chosen)
                except (TypeError, ValueError):
                    continue  # skip non-numeric knob values
        if raw.get("libtype"):
            run["libtype"] = str(raw.get("libtype"))
        # Surface why a run skipped (e.g. dense fp8 missing GEMM shapes).
        if raw.get("skip_reason"):
            run["skip_reason"] = str(raw.get("skip_reason"))
        if isinstance(raw.get("tuners_skipped"), list) and raw.get("tuners_skipped"):
            run["tuners_skipped"] = raw["tuners_skipped"]
        if isinstance(raw.get("summary"), dict):
            run["summary"] = raw["summary"]
        if isinstance(raw.get("parameters"), dict):
            run["parameters"] = raw["parameters"]
        if isinstance(raw.get("candidates"), list):
            run["candidates"] = raw["candidates"]
        if isinstance(raw.get("shapes"), list):
            run["shapes"] = raw["shapes"]
        runs.append(run)

    if not runs:
        return {}

    adopted_engine = ""
    adopted_tuned_file = ""
    total_gain_pct = 0.0
    for run in runs:
        if run.get("adopted"):
            adopted_engine = run.get("engine") or adopted_engine
            adopted_tuned_file = run.get("tuned_file") or adopted_tuned_file
            if isinstance(run.get("gain_pct"), (int, float)):
                total_gain_pct += float(run["gain_pct"])

    return {
        "runs": runs,
        "adopted_engine": adopted_engine,
        "adopted_tuned_file": adopted_tuned_file,
        "total_gain_pct": round(total_gain_pct, 2),
    }


#: Integrate-gate evidence copied verbatim onto every collective record. The
#: E2E verdict, not the microbenchmark, decides whether a campaign is adopted.
_COLLECTIVE_INTEGRATION_FIELDS = (
    "integration_id",
    "integration_decision",
    "patch_cleanup_status",
    "integration_result_status",
    "integration_revert_status",
    "integration_finalize_status",
    "integration_recovery_action",
    "integration_error_class",
    "integration_error",
    "integration_report_path",
    "integration_workspace",
    "integration_ts",
)


def _normalize_collective_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one collective campaign record into the exported shape."""
    world_size_raw = raw.get("world_size")
    try:
        world_size = int(world_size_raw) if world_size_raw not in (None, "") else None
    except (TypeError, ValueError):
        world_size = None

    out: dict[str, Any] = {
        "collective_attempt_id": str(raw.get("collective_attempt_id") or ""),
        "experiment_id": str(raw.get("experiment_id") or ""),
        "kernel_id": str(raw.get("kernel_id") or ""),
        "kernel_name": str(raw.get("kernel_name") or ""),
        "collective_op": str(raw.get("collective_op") or ""),
        "world_size": world_size,
        "engine": str(raw.get("engine") or raw.get("backend") or ""),
        "status": str(raw.get("status") or ""),
        "decision": str(raw.get("decision") or ""),
        "kept": bool(raw.get("kept")),
        "salvaged": bool(raw.get("salvaged")),
        "requires_e2e_validation": bool(raw.get("requires_e2e_validation")),
        "iterations": raw.get("iterations"),
        "kernel_speedup": _to_float(raw.get("kernel_speedup")),
        "gpu_pct": _to_float(raw.get("gpu_pct")),
        "duration_sec": _to_float(raw.get("duration_sec")),
        "ts": str(raw.get("ts") or ""),
        "source_file": str(raw.get("source_file") or ""),
        "kernel_repo": str(raw.get("kernel_repo") or ""),
        "workspace": str(raw.get("workspace") or ""),
        "patch_path": str(raw.get("patch_path") or raw.get("patch") or ""),
        "error_class": str(raw.get("error_class") or ""),
        "error": str(raw.get("error") or ""),
        "integration_gain_pct": _to_float(raw.get("integration_gain_pct")),
        "integration_base_tput": _to_float(raw.get("integration_base_tput")),
        "integration_new_tput": _to_float(raw.get("integration_new_tput")),
    }
    for field in _COLLECTIVE_INTEGRATION_FIELDS:
        value = raw.get(field) or ""
        if not value and field == "patch_cleanup_status":
            # Resume compat: older state.json records use "integration_status".
            value = raw.get("integration_status") or ""
        out[field] = str(value)
    if isinstance(raw.get("bandwidth"), dict):
        out["bandwidth"] = raw["bandwidth"]
    if isinstance(raw.get("artifact_files"), list):
        out["artifact_files"] = [str(item) for item in raw["artifact_files"]]
    return out


def collect_collective(state: dict[str, Any]) -> dict[str, Any]:
    """Build the top-level ``collective`` section from session state; never raises.

    Mirrors the SharedState collective lane fields so a campaign stays auditable
    even when it never reaches ``optimizations`` — a lane that wins its
    microbenchmark but loses the E2E gate leaves no trace there.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        dict[str, Any]: A ``Collective`` envelope (``only_mode`` / ``attempts``
        / ``last``), or ``{}`` when the lane never ran.
    """
    raw_attempts = state.get("collective_attempts")
    if not isinstance(raw_attempts, list):
        raw_attempts = []
    last_raw = state.get("last_collective")
    if not isinstance(last_raw, dict):
        last_raw = {}
    if not raw_attempts and not last_raw:
        return {}

    attempts = [_normalize_collective_record(item) for item in raw_attempts if isinstance(item, dict)]
    envelope: dict[str, Any] = {
        "only_mode": bool(state.get("collective_only_mode")),
        "attempts": attempts,
    }
    if last_raw:
        envelope["last"] = _normalize_collective_record(last_raw)
    return envelope


def collect_source_files(
    session_dir: Path,
    baseline_path: str | None,
    profile_reports: list[str],
    sweep_reports: list[str],
) -> dict[str, Any]:
    """Build the ``source_files`` map of key on-disk artifacts.

    Always surfaces the manifest / state / baseline references and the
    critic / robustness workdirs (when present), and conditionally adds the
    profile / sweep / kernel-attempt list categories when non-empty (empty
    list-valued categories are omitted so the renderer doesn't show
    ``count=0`` rows).

    Args:
        session_dir (Path): Absolute session root.
        baseline_path (str | None): Relative path to the baseline report.
        profile_reports (list[str]): Relative profile report paths.
        sweep_reports (list[str]): Relative sweep report paths.

    Returns:
        dict[str, Any]: The source-files map.
    """
    kernel_attempts = [
        _rel(run_dir / "optimization_attempts.jsonl", session_dir) or str(run_dir / "optimization_attempts.jsonl")
        for run_dir in _kernel_agent_run_dirs(session_dir)
        if (run_dir / "optimization_attempts.jsonl").exists()
    ]
    critic = session_dir / "critic-workdir"
    rob = session_dir / "robustness-workdir"
    out: dict[str, Any] = {
        "manifest": "manifest.json",
        "state": "state.json",
        "baseline_report": baseline_path,
        "critic_workdir": "critic-workdir" if critic.exists() else None,
        "robustness_workdir": "robustness-workdir" if rob.exists() else None,
    }
    # Skip empty list categories so the renderer shows no ``count=0`` rows.
    for key, lst in (
        ("profile_reports", profile_reports),
        ("sweep_reports", sweep_reports),
        ("kernel_attempts", kernel_attempts),
    ):
        if lst:
            out[key] = lst
    return out
