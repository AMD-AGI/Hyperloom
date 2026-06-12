# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-side of the breakdown recorder.

Assembles the per-producer fragments written by :class:`~.recorder.Recorder`
into a ``{section: value}`` mapping ready to drop into the
``session_breakdown.json`` envelope:

* ``singleton`` sections -> the payload of the latest fragment (by ``ts``).
* ``item`` sections      -> payloads concatenated into a list, ordered by
  ``seq`` then ``ts``.

There is no cross-producer conflict resolution because each section has a
single owner. Bad/partial fragments are skipped and noted in ``warnings``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parts_dir(session_dir: Path | str) -> Path:
    from ...session_paths import breakdown_parts_dir  # local: avoid import cycle

    return breakdown_parts_dir(Path(session_dir))


def has_parts(session_dir: Path | str) -> bool:
    """True iff at least one record fragment exists for this session."""
    d = parts_dir(session_dir)
    return d.is_dir() and any(d.glob("*.json"))


def _load(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"recorder: failed to read {path.name}: {exc!r}")
        return None
    if not isinstance(rec, dict):
        warnings.append(f"recorder: {path.name} is not an object")
        return None
    return rec


def assemble_parts(
    session_dir: Path | str,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Return ``{section: list | dict}`` assembled from the spool directory.

    Empty mapping when no fragments exist (caller falls back to collectors).
    """
    warns = warnings if warnings is not None else []
    d = parts_dir(session_dir)
    if not d.is_dir():
        return {}

    items: dict[str, list[dict[str, Any]]] = {}
    singletons: dict[str, dict[str, Any]] = {}

    for path in sorted(d.glob("*.json")):
        rec = _load(path, warns)
        if rec is None:
            continue
        section = rec.get("section")
        if not isinstance(section, str) or not section:
            warns.append(f"recorder: {path.name} missing 'section'")
            continue
        if rec.get("kind") == "singleton":
            prev = singletons.get(section)
            if prev is None or str(rec.get("ts") or "") >= str(prev.get("ts") or ""):
                singletons[section] = rec
        else:
            items.setdefault(section, []).append(rec)

    out: dict[str, Any] = {}
    for section, recs in items.items():
        recs.sort(key=lambda r: (int(r.get("seq") or 0), str(r.get("ts") or "")))
        out[section] = [r.get("payload") for r in recs]
    for section, rec in singletons.items():
        out[section] = rec.get("payload")

    _compose_critic_robustness(out)
    _compose_kernel_journey(out)
    _compose_versions(out)
    return out


def _compose_versions(out: dict[str, Any]) -> None:
    """Fold the ``versions`` item substream into a top-level ``{tool: meta}``
    map (last write per tool wins; the substream is already deduped by tool key
    at record time). No-op when nothing was recorded."""
    rows = out.get("versions")
    if not isinstance(rows, list):
        return
    merged: dict[str, Any] = {}
    for r in rows:
        if isinstance(r, dict):
            tool = str(r.get("tool") or "").lower()
            if tool:
                merged[tool] = r
    out["versions"] = merged


def _compose_critic_robustness(out: dict[str, Any]) -> None:
    """Fold the ``critic_iterations`` / ``robustness_signals`` item substreams
    into the ``critic_robustness`` singleton (shape mirrors
    ``collectors.collect_critic_robustness``). Pops the raw substreams so they
    don't leak into the breakdown envelope."""
    critic_iters = out.pop("critic_iterations", None)
    rob_signals = out.pop("robustness_signals", None)
    if critic_iters is None and rob_signals is None:
        return
    # A directly-recorded singleton (if any) takes precedence over substreams.
    if "critic_robustness" in out:
        return
    critic_iters = critic_iters if isinstance(critic_iters, list) else []
    rob_signals = rob_signals if isinstance(rob_signals, list) else []
    out["critic_robustness"] = {
        "critic_iterations":  critic_iters,
        "robustness_signals": rob_signals,
        "kb_writes_summary":  _kb_writes_summary(critic_iters),
    }


def _compose_kernel_journey(out: dict[str, Any]) -> None:
    """Fold the four kernel-lifecycle item substreams into a single
    kernel-major ``kernel_journey`` view (discovery -> dispatch -> backend
    attempts -> e2e), then pop the raw substreams so they don't leak into the
    envelope. No-op (and ``kernel_journey`` stays absent) when no substream was
    recorded, preserving historical breakdowns byte-for-byte.
    """
    discovery = out.pop("kernel_discovery", None)
    dispatch = out.pop("kernel_dispatch", None)
    backend = out.pop("kernel_backend_result", None)
    e2e = out.pop("kernel_e2e", None)
    if discovery is None and dispatch is None and backend is None and e2e is None:
        return
    # A directly-recorded singleton (if any) takes precedence over substreams.
    if "kernel_journey" in out:
        return

    discovery_runs = [r for r in (discovery or []) if isinstance(r, dict)]
    dispatch_rows = [r for r in (dispatch or []) if isinstance(r, dict)]
    backend_rows = [r for r in (backend or []) if isinstance(r, dict)]
    e2e_rows = [r for r in (e2e or []) if isinstance(r, dict)]

    # Latest discovery snapshot per kernel_id (later runs win).
    discovery_by_kid: dict[str, dict[str, Any]] = {}
    for run in discovery_runs:
        for hk in run.get("hot_kernels") or []:
            if not isinstance(hk, dict):
                continue
            kid = str(hk.get("kernel_id") or "")
            if kid:
                discovery_by_kid[kid] = hk

    dispatch_by_kid = {
        str(r.get("kernel_id") or ""): r for r in dispatch_rows
        if str(r.get("kernel_id") or "")
    }
    e2e_by_kid = {
        str(r.get("kernel_id") or ""): r for r in e2e_rows
        if str(r.get("kernel_id") or "")
    }
    attempts_by_kid: dict[str, list[dict[str, Any]]] = {}
    for r in backend_rows:
        kid = str(r.get("kernel_id") or "")
        if kid:
            attempts_by_kid.setdefault(kid, []).append(r)

    kids: list[str] = []
    for source in (discovery_by_kid, dispatch_by_kid, attempts_by_kid, e2e_by_kid):
        for kid in source:
            if kid and kid not in kids:
                kids.append(kid)

    kernels: list[dict[str, Any]] = []
    for kid in kids:
        disc = discovery_by_kid.get(kid, {})
        disp = dispatch_by_kid.get(kid, {})
        atts = attempts_by_kid.get(kid, [])
        kernel_e2e = e2e_by_kid.get(kid, {})
        kernels.append({
            "kernel_id":        kid,
            "name":             str(disc.get("name") or ""),
            "gpu_pct":          disc.get("gpu_pct"),
            "bound_type":       str(disc.get("bound_type") or ""),
            "source_file":      disc.get("source_file"),
            "micro_speedup":    _best_micro_speedup(atts),
            "discovery":        disc,
            "dispatch":         disp,
            "backend_attempts": atts,
            "e2e":              kernel_e2e,
            "outcome":          _kernel_outcome(disp, atts, kernel_e2e),
        })

    def _gpu(k: dict[str, Any]) -> float:
        v = k.get("gpu_pct")
        try:
            return float(v) if v is not None else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    kernels.sort(key=_gpu, reverse=True)
    out["kernel_journey"] = {
        "discovery_runs": discovery_runs,
        "kernels":        kernels,
    }


def _best_micro_speedup(attempts: list[dict[str, Any]]) -> float | None:
    """Best (max) micro_speedup across a kernel's attempts, or None.

    Surfaces the kernel-level achieved speedup at the journey-entry top level so
    the dashboard can correlate it with ``e2e.e2e_gain_pct`` without digging
    through the attempt ladder.
    """
    best: float | None = None
    for att in attempts:
        if not isinstance(att, dict):
            continue
        v = att.get("micro_speedup")
        try:
            f = float(v) if v is not None else None
        except (TypeError, ValueError):
            f = None
        if f is not None and (best is None or f > best):
            best = f
    return best


def _kernel_outcome(
    dispatch: dict[str, Any],
    attempts: list[dict[str, Any]],
    e2e: dict[str, Any],
) -> str:
    """Coarse per-kernel outcome: adopted / reverted / attempted / dispatched /
    skipped / discovered (in lifecycle-descending precedence)."""
    if e2e:
        decision = str(e2e.get("decision") or "").upper()
        if e2e.get("integrated") or decision in ("KEEP", "ADOPTED"):
            return "adopted"
        if decision in ("REVERT", "REJECTED"):
            return "reverted"
    if attempts:
        return "attempted"
    if dispatch:
        return "dispatched" if dispatch.get("dispatched") else "skipped"
    return "discovered"


def _kb_writes_summary(critic_iters: list[Any]) -> dict[str, Any]:
    """Count each critic iteration's verdict (mirrors the collector)."""
    by_verdict: dict[str, int] = {}
    total = 0
    for entry in critic_iters:
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("verdict") or "").strip().upper()
        if not verdict:
            continue
        total += 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    return {"total": total, "by_verdict": by_verdict}


__all__ = ["assemble_parts", "has_parts", "parts_dir"]
