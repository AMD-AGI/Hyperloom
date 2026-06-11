# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Bundle a session's *consumer-facing* artifacts into a single zip under
``/workspace`` so the Claw sandbox sync picks it up (and ships it to S3 /
the wekafs persist base).

Why this exists
---------------
Hyperloom writes all products under ``session_dir`` (``$USER_DATA_PATH``),
which production launchers frequently point at a wekafs path *outside*
``/workspace``. Claw only syncs ``/workspace``, so those products never
reach Claw storage and the ``/v1/sessions/<sid>/files`` endpoint can't
serve them. Rather than move the whole (multi-hundred-MB) session tree,
this module copies just the small set of result/report/analysis files
(KB–MB total) into one zip placed *inside* ``/workspace``, where Claw's
sync is guaranteed to see it. By default it ALSO drops the same files
*loose* (uncompressed, original relative tree) directly under the dest
root itself (e.g. ``/workspace/session_breakdown.json``,
``/workspace/reports/final.json``), so a consumer can fetch a single
file without unzipping (disable via ``HYPERLOOM_SESSION_PACKAGE_LOOSE=0``).

A processing log (``PACKAGE_MANIFEST.json`` + ``PACKAGE_MANIFEST.txt``)
recording exactly which files were included / missing is written into
the zip itself, so a consumer can audit the bundle without the source
session dir.

Contract
--------
* Best-effort: never raises. On any failure returns ``None`` and logs;
  the caller MUST treat the canonical per-file writes as the source of
  truth and never let a packaging failure mask the real ``stop_reason``.
* Selection is a glob spec (:data:`PACKAGE_GLOBS`) resolved against the
  session dir. ``runs/`` trace blobs and per-turn agent dumps are never
  matched — only the curated result/report set.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Default destination root. Claw mounts the synced workspace at
# ``/workspace`` REGARDLESS of where ``$USER_DATA_PATH`` points, so the
# bundle must be anchored here (not at ``workspace_root()`` which follows
# ``$USER_DATA_PATH`` to wekafs). Overridable via env for non-Claw envs
# and tests.
ENV_PACKAGE_DEST_ROOT = "HYPERLOOM_SESSION_PACKAGE_DEST"
DEFAULT_DEST_ROOT = Path("/workspace")

# In addition to the zip, also lay the same curated files down *loose*
# (uncompressed, keeping their relative tree) directly under the dest
# root, so a consumer can fetch a single file (e.g.
# /v1/sessions/<sid>/files/<rel>) without unzipping. Set to
# "0"/"false"/"no" to write only the zip.
ENV_PACKAGE_LOOSE = "HYPERLOOM_SESSION_PACKAGE_LOOSE"

#: Subdir under the dest root where bundles land.
PACKAGE_SUBDIR = "hyperloom-session-packages"

MANIFEST_JSON_NAME = "PACKAGE_MANIFEST.json"
MANIFEST_TXT_NAME = "PACKAGE_MANIFEST.txt"
PACKAGE_SCHEMA_VERSION = 1

# Curated artifact selection, relative to session_dir. Glob patterns are
# matched against POSIX-style relative paths. ``**`` spans directories.
# Keep this list in sync with the "necessary products" audit; the goal is
# results / reports / analysis only — never the bulky ``runs/`` traces,
# profile blobs, or per-turn agent ``request.json`` dumps.
PACKAGE_GLOBS: tuple[str, ...] = (
    # ── top-level core ────────────────────────────────────────────────
    "session_breakdown.json",
    "state.json",
    "manifest.json",
    # ── reports/ ──────────────────────────────────────────────────────
    "reports/final.json",
    "reports/final.md",
    "reports/optimization_journal.json",
    "reports/kernel_optimization_summary.json",
    "reports/kernel_roofline.json",
    "reports/conc_sweep_summary.json",
    "reports/trace/*.jsonl",
    # ── target analysis ───────────────────────────────────────────────
    "target_analysis/target_baseline.json",
    "target_analysis/target_analysis_report.md",
    # ── coordinator DB ────────────────────────────────────────────────
    "storage/coordinator.db",
    # ── TraceLens analysis/report family (dynamic <ts>/<tl-id> subdirs) ─
    "kernel-agent/runs/**/tracelens/analysis.md",
    "kernel-agent/runs/**/tracelens/tracelens_report.json",
    "kernel-agent/runs/**/tracelens/summary.json",
    "kernel-agent/runs/**/tracelens/priority_data.json",
    "kernel-agent/runs/**/kernel_candidates.json",
    "kernel-agent/runs/**/trace_input_manifest.json",
    "kernel-agent/runs/**/tracelens/category_findings/*.md",
    "kernel-agent/runs/**/tracelens/system_findings/*.md",
    "kernel-agent/runs/**/tracelens/perf_report_csvs/*.csv",
    # ── per-run benchmark reports (small JSON/txt; NOT the trace blobs) ─
    "runs/**/benchmark_report.json",
    "runs/**/summary.txt",
    "runs/**/inferencex_result.json",
    "runs/gemm_tuning/**/final_report.json",
    "runs/gemm_tuning/**/best_results.json",
    "runs/specialist/**/specialist_done.json",
    "runs/recover/**/result.json",
)

# Hard safety caps so a pathological session can't blow up the bundle.
_MAX_FILES = 5000
_MAX_TOTAL_BYTES = 256 * 1024 * 1024  # 256 MB


def _dest_root() -> Path:
    override = (os.environ.get(ENV_PACKAGE_DEST_ROOT) or "").strip()
    return Path(override) if override else DEFAULT_DEST_ROOT


def _loose_enabled() -> bool:
    """Whether to also drop loose (unzipped) copies. Defaults to True."""
    raw = (os.environ.get(ENV_PACKAGE_LOOSE) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _copy_loose_tree(
    included: list[tuple[Path, str, int]],
    manifest: dict,
    loose_dir: Path,
) -> int:
    """Copy each included file into ``loose_dir`` preserving its relative
    tree, plus the two manifest files. Best-effort, per-file isolated:
    one unreadable file never aborts the rest. Returns the count copied.

    Files are overwritten in place (no wholesale wipe of ``loose_dir``):
    the dest is the shared ``/workspace`` root, so deleting it is never
    safe. A stale file from a previous, larger selection is left as-is;
    the per-run ``PACKAGE_MANIFEST`` is the source of truth for what this
    run actually included.
    """
    loose_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, rel, _sz in included:
        dst = loose_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        except OSError:
            log.warning("session package: failed to copy loose file %s", rel)
    try:
        (loose_dir / MANIFEST_JSON_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        (loose_dir / MANIFEST_TXT_NAME).write_text(
            _manifest_text(manifest), encoding="utf-8",
        )
    except OSError:
        log.warning("session package: failed to write loose manifest")
    return copied


def _iter_session_files(session_dir: Path) -> list[Path]:
    """All files under session_dir (one walk), so glob matching is a
    single pass instead of N globs each re-walking the tree."""
    out: list[Path] = []
    for dp, _dn, fn in os.walk(session_dir):
        for f in fn:
            out.append(Path(dp) / f)
    return out


def _select(session_dir: Path) -> tuple[list[Path], list[str]]:
    """Return (matched absolute paths, unmatched globs).

    A glob is reported "unmatched" when it selected zero files — useful
    audit signal in the manifest (e.g. conc_sweep_summary absent because
    the sweep was skipped).
    """
    all_files = _iter_session_files(session_dir)
    rels = {p: p.relative_to(session_dir).as_posix() for p in all_files}

    matched: list[Path] = []
    seen: set[Path] = set()
    unmatched_globs: list[str] = []
    for pattern in PACKAGE_GLOBS:
        hit = False
        for p, rel in rels.items():
            if _glob_match(rel, pattern):
                hit = True
                if p not in seen:
                    seen.add(p)
                    matched.append(p)
        if not hit:
            unmatched_globs.append(pattern)
    matched.sort(key=lambda p: rels[p])
    return matched, unmatched_globs


def _glob_match(rel: str, pattern: str) -> bool:
    """fnmatch with ``**`` spanning ``/``.

    ``fnmatch`` treats ``*`` as spanning ``/`` too, which is too loose
    for single-segment patterns like ``reports/trace/*.jsonl``. Handle
    the two cases explicitly:

    * pattern contains ``**`` → collapse to a permissive regex-ish match
      by replacing ``**`` with a sentinel that fnmatch's ``*`` covers.
    * otherwise → require the path to have the same number of segments,
      matching each segment with fnmatch so ``*`` stays within a segment.
    """
    if "**" in pattern:
        # fnmatch's '*' already spans '/', so '**' == '*' for our purpose.
        collapsed = pattern.replace("**/", "*/").replace("**", "*")
        return fnmatch.fnmatch(rel, collapsed) or fnmatch.fnmatch(rel, pattern.replace("**", "*"))
    pat_parts = pattern.split("/")
    rel_parts = rel.split("/")
    if len(pat_parts) != len(rel_parts):
        return False
    return all(fnmatch.fnmatch(rp, pp) for rp, pp in zip(rel_parts, pat_parts))


def _build_manifest(
    session_dir: Path,
    session_id: str,
    included: list[tuple[str, int]],
    missing_globs: list[str],
) -> dict:
    total = sum(sz for _, sz in included)
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": session_id,
        "session_dir": str(session_dir),
        "included_count": len(included),
        "included_total_bytes": total,
        "included_files": [{"path": rel, "bytes": sz} for rel, sz in included],
        "unmatched_globs": missing_globs,
        "selection_globs": list(PACKAGE_GLOBS),
    }


def _manifest_text(manifest: dict) -> str:
    lines = [
        "Hyperloom session artifact package",
        f"  session_id   : {manifest.get('session_id') or '?'}",
        f"  packaged_at  : {manifest.get('packaged_at_utc')}",
        f"  source dir   : {manifest.get('session_dir')}",
        f"  files        : {manifest.get('included_count')}",
        f"  total bytes  : {manifest.get('included_total_bytes')}",
        "",
        "Included files:",
    ]
    for entry in manifest.get("included_files") or []:
        lines.append(f"  + {entry['path']}  ({entry['bytes']} B)")
    missing = manifest.get("unmatched_globs") or []
    if missing:
        lines.append("")
        lines.append("Selection patterns that matched nothing (informational):")
        for g in missing:
            lines.append(f"  - {g}")
    lines.append("")
    return "\n".join(lines)


def package_session_artifacts(
    session_dir: Path | str,
    *,
    session_id: str = "",
    dest_root: Path | str | None = None,
) -> Path | None:
    """Bundle curated artifacts of ``session_dir`` into one zip under the
    dest root (default ``/workspace/<PACKAGE_SUBDIR>/``).

    Args:
        session_dir: hyperloom session directory (the products live here).
        session_id: used for the zip filename + manifest. Falls back to
            the session dir basename when empty.
        dest_root: override the destination root (defaults to
            ``$HYPERLOOM_SESSION_PACKAGE_DEST`` or ``/workspace``).

    Returns:
        Absolute path to the written zip, or ``None`` on any failure /
        no files matched. Never raises.
    """
    try:
        sd = Path(session_dir).resolve()
        if not sd.is_dir():
            log.warning("session package skipped: session_dir not a dir: %s", sd)
            return None

        sid = (session_id or "").strip() or sd.name
        matched, missing_globs = _select(sd)
        if not matched:
            log.warning("session package skipped: no artifacts matched in %s", sd)
            return None

        # Apply safety caps.
        included: list[tuple[Path, str, int]] = []
        total = 0
        for p in matched:
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if len(included) >= _MAX_FILES or total + sz > _MAX_TOTAL_BYTES:
                log.warning(
                    "session package: hit size/count cap, truncating bundle "
                    "(files=%d, bytes=%d)", len(included), total,
                )
                break
            included.append((p, p.relative_to(sd).as_posix(), sz))
            total += sz

        manifest = _build_manifest(
            sd, sid, [(rel, sz) for _, rel, sz in included], missing_globs,
        )

        root = Path(dest_root).resolve() if dest_root else _dest_root()
        out_dir = root / PACKAGE_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{sid}.zip"

        # Atomic write: build into a temp zip in the same dir, then replace.
        fd, tmp = tempfile.mkstemp(prefix=f".{sid}.", suffix=".zip.tmp", dir=str(out_dir))
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p, rel, _sz in included:
                    try:
                        zf.write(p, arcname=rel)
                    except OSError:
                        log.warning("session package: failed to add %s", rel)
                zf.writestr(MANIFEST_JSON_NAME, json.dumps(manifest, indent=2))
                zf.writestr(MANIFEST_TXT_NAME, _manifest_text(manifest))
            os.replace(tmp_path, target)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

        log.info(
            "session package: wrote %s (%d files, %d bytes pre-zip)",
            target, len(included), total,
        )

        # Also lay the same files down loose (uncompressed, original tree)
        # directly under the dest root (e.g. ``/workspace/``) so a consumer
        # can grab one file without unzip. NOT under the package subdir —
        # straight at the root, preserving each file's relative path.
        if _loose_enabled():
            try:
                copied = _copy_loose_tree(included, manifest, root)
                log.info(
                    "session package: copied %d loose files into %s",
                    copied, root,
                )
            except Exception:  # noqa: BLE001 — loose copy must not mask the zip
                log.exception("session package: loose copy failed (non-fatal)")

        return target
    except Exception:  # noqa: BLE001 — never let packaging mask stop_reason
        log.exception("session package failed (non-fatal)")
        return None
