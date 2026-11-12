"""The KB collector: a campaign's decisions as ``arbor-v1`` Experiences in Hyperloom's Experience KB.

``python3 -m arbor.collect kb --session <dir>`` exports the session (``export.py``, standard library only) to
``<out>/arbor-kb-export.json`` and the files its records link, then collects that document through this package's
own mapping, ``mappings/arbor-v1.yaml``, with ``hyperloom_kb.collect`` when it is importable. The KB is the one
``HYPERLOOM_KB_URL`` and ``HYPERLOOM_KB_TOKEN`` name; without them, or without ``hyperloom_kb``, the export is
written and nothing is published. ``--dry-run`` projects and validates every record without any KB.

Nothing here raises into a campaign: a callback on ``end`` runs it in its own process, and every failure is a line
in the report and a non-zero exit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from arbor.collect.kb.export import export

HERE = Path(__file__).resolve().parent
MAPPING = HERE / "mappings" / "arbor-v1.yaml"
DECLARATION = HERE / "declarations" / "arbor-v1.yaml"
DOCUMENT = "arbor-kb-export.json"
RECEIPT = "collect-receipt.json"


def run(session: Path, out: Path, pack_roots: list[Path] | None = None, campaign_logs: list[Path] | None = None,
        dry_run: bool = False) -> dict[str, Any]:
    """Export ``session`` under ``out`` and collect it; a summary of both, never an exception from collection."""

    document = export(session, out, pack_roots, campaign_logs)
    path = Path(out) / DOCUMENT
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"document": str(path), "units": len(document["units"]),
                               "session": document["session"], "checks": document.get("checks")}
    summary["warnings"] = warnings(document.get("checks") or {})
    if document["session"].get("read_error"):
        summary["collect"] = {"status": "skipped", "why": document["session"]["read_error"]}
        return summary
    summary["collect"] = _collect(path, Path(out) / RECEIPT, dry_run)
    return summary


def warnings(checks: dict) -> list[str]:
    """What a person must look at: a bar replay that disagrees with Arbor, and files left out for credentials."""

    found = []
    best = checks.get("final_best_metric") or {}
    if best.get("agree") is False:
        found.append(f"the replayed bar {best.get('replayed')} disagrees with session.json best_metric "
                     f"{best.get('session')}: every record's best_before may be wrong; fix the replay before "
                     "trusting them")
    for path, hits in (checks.get("credential_omissions") or {}).items():
        found.append(f"{path} left out, never published: credential-shaped values ({'; '.join(hits[:3])})")
    return found


def _collect(document: Path, receipt: Path, dry_run: bool) -> dict[str, Any]:
    try:
        from hyperloom_kb.collect import collect
    except ImportError:
        return {"status": "not_collected", "why": "hyperloom_kb is not installed; the export is written"}
    if not dry_run and not os.environ.get("HYPERLOOM_KB_URL"):
        return {"status": "not_collected", "why": "HYPERLOOM_KB_URL is not set; the export is written"}
    try:
        report = collect(MAPPING, document, dry_run=dry_run, receipt=receipt).to_dict()
    except Exception as exc:  # noqa: BLE001 - a collection failure is reported, never raised into a campaign
        return {"status": "error", "why": f"{type(exc).__name__}: {exc}"[:500]}
    return {"status": "dry_run" if dry_run else "collected", "receipt": str(receipt), "counts": report.get("counts")}
