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
    return out


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
