# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bundle a session's consumer-facing artifacts into a single zip under
``/workspace`` so the Claw sandbox sync picks it up.

Products live under ``session_dir`` (``$USER_DATA_PATH``), often outside
``/workspace`` which is the only path Claw syncs. This copies the small set
of result/report/analysis files into one zip inside ``/workspace``, and by
default also drops them loose (uncompressed, original tree) under the dest
root so a consumer can fetch a single file without unzipping (disable via
``HYPERLOOM_SESSION_PACKAGE_LOOSE=0``). A ``PACKAGE_MANIFEST.{json,txt}``
describing the bundle is written alongside the files it describes.

Contract:
* Best-effort: never raises. On any failure returns ``None`` and logs;
  the caller treats the canonical per-file writes as the source of truth.
* Selection is a glob spec (:data:`PACKAGE_GLOBS`) resolved against the
  session dir; only the curated result/report set is matched.
* A selected entry is only bundled when it is a regular file resolving
  inside the session, so a link planted in this shared-filesystem
  directory cannot pull outside content into the dest root.
* The manifest lists what was written, and ``complete`` is false whenever
  anything selected is missing from it.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import tempfile
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from ..session.paths import is_path_within

log = logging.getLogger(__name__)

# Default destination root. Claw mounts the synced workspace at ``/workspace``
# regardless of where ``$USER_DATA_PATH`` points, so anchor the bundle here.
# Overridable via env for non-Claw envs and tests.
ENV_PACKAGE_DEST_ROOT = "HYPERLOOM_SESSION_PACKAGE_DEST"
DEFAULT_DEST_ROOT = Path("/workspace")

# Also lay the curated files down loose (uncompressed, relative tree) under the
# dest root so a consumer can fetch one file without unzipping.
# Set "0"/"false"/"no" to write only the zip.
ENV_PACKAGE_LOOSE = "HYPERLOOM_SESSION_PACKAGE_LOOSE"

#: Subdir under the dest root where bundles land.
PACKAGE_SUBDIR = "hyperloom-session-packages"

MANIFEST_JSON_NAME = "PACKAGE_MANIFEST.json"
MANIFEST_TXT_NAME = "PACKAGE_MANIFEST.txt"
PACKAGE_SCHEMA_VERSION = 2

# Curated artifact selection, relative to session_dir. Glob patterns match
# POSIX-style relative paths; ``**`` spans directories. Results / reports /
# analysis only — never the bulky ``runs/`` traces or per-turn agent dumps.
PACKAGE_GLOBS: tuple[str, ...] = (
    # ── top-level core ────────────────────────────────────────────────
    "session_breakdown.json",
    "state.json",
    "manifest.json",
    "current_setting.sh",
    # ── reports/ ──────────────────────────────────────────────────────
    "reports/enablement/**",
    "reports/final.json",
    "reports/final.md",
    "reports/optimization_journal.json",
    "reports/kernel_optimization_summary.json",
    "reports/kernel_roofline.json",
    "reports/conc_sweep_summary.json",
    "reports/sbd_v6/timeline/*.json",
    "reports/sbd_v6/write_warnings.jsonl",
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
    """Resolve the destination root for session packages.

    Returns:
        The path from ``ENV_PACKAGE_DEST_ROOT`` when set, otherwise the
        default destination root.
    """
    override = (os.environ.get(ENV_PACKAGE_DEST_ROOT) or "").strip()
    return Path(override) if override else DEFAULT_DEST_ROOT


def _loose_enabled() -> bool:
    """Whether to also drop loose (unzipped) copies. Defaults to True.

    Returns:
        ``True`` unless the env override disables loose copies.
    """
    raw = (os.environ.get(ENV_PACKAGE_LOOSE) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _copy_loose_tree(
    included: list[tuple[Path, str, int]],
    loose_dir: Path,
) -> tuple[list[tuple[str, int]], list[str]]:
    """Copy each included file into ``loose_dir`` preserving its relative
    tree. Best-effort, per-file isolated: one unreadable file never aborts
    the rest. Files are overwritten in place (no wholesale wipe of the
    shared dest root).

    Args:
        included: Tuples of ``(source path, relative path, size)`` to copy.
        loose_dir: Destination root for the loose tree.

    Returns:
        The ``(relative path, size)`` pairs that landed and the relative
        paths that could not be copied.
    """
    loose_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, int]] = []
    failed: list[str] = []
    for src, rel, sz in included:
        dst = loose_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append((rel, sz))
        except OSError:
            failed.append(rel)
            log.warning("session package: failed to copy loose file %s", rel)
    return copied, failed


def _write_loose_manifest(loose_dir: Path, manifest: dict) -> None:
    """Write the manifest pair describing the loose tree.

    Args:
        loose_dir: Destination root holding the loose tree.
        manifest: Manifest dict describing what actually landed there.
    """
    try:
        (loose_dir / MANIFEST_JSON_NAME).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        (loose_dir / MANIFEST_TXT_NAME).write_text(
            _manifest_text(manifest),
            encoding="utf-8",
        )
    except OSError:
        log.warning("session package: failed to write loose manifest")


def _is_packageable(path: Path, session_dir: Path) -> bool:
    """Whether ``path`` is a regular file resolving inside ``session_dir``.

    Refuses a symlink whose target escapes the session, and sockets and
    devices, which are not artifacts and cannot be archived.

    Args:
        path: Candidate file discovered under ``session_dir``.
        session_dir: The resolved session root.

    Returns:
        Whether the entry is safe to bundle.
    """
    return is_path_within(path, session_dir) and path.is_file()


def _iter_session_files(session_dir: Path) -> list[Path]:
    """All files under session_dir (one walk), so glob matching is a
    single pass instead of N globs each re-walking the tree.

    Symlinked directories are not descended into, so only an entry itself
    can be a link.

    Args:
        session_dir: Root directory to walk.

    Returns:
        Absolute paths of every file found under ``session_dir``.
    """
    out: list[Path] = []
    for dp, _dn, fn in os.walk(session_dir):
        for f in fn:
            out.append(Path(dp) / f)
    return out


def _select(session_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    """Return (matched absolute paths, unmatched globs, refused paths).

    A glob is reported "unmatched" when it selected zero files — useful
    audit signal in the manifest (e.g. conc_sweep_summary absent because
    the sweep was skipped). A glob that only selected entries failing the
    session boundary still counts as a hit, and those entries are returned
    separately so the manifest can name them.

    Args:
        session_dir: Session directory whose files are matched against the
            package globs.

    Returns:
        A tuple of the matched absolute paths, the patterns that matched
        nothing, and the relative paths refused by the boundary check.
    """
    all_files = _iter_session_files(session_dir)
    rels = {p: p.relative_to(session_dir).as_posix() for p in all_files}

    matched: list[Path] = []
    seen: set[Path] = set()
    unmatched_globs: list[str] = []
    refused: set[str] = set()
    for pattern in PACKAGE_GLOBS:
        hit = False
        for p, rel in rels.items():
            if not _glob_match(rel, pattern):
                continue
            hit = True
            if not _is_packageable(p, session_dir):
                refused.add(rel)
                log.warning(
                    "session package: refusing %s (not a regular file inside the session)",
                    rel,
                )
                continue
            if p not in seen:
                seen.add(p)
                matched.append(p)
        if not hit:
            unmatched_globs.append(pattern)
    matched.sort(key=lambda p: rels[p])
    return matched, unmatched_globs, sorted(refused)


def _glob_match(rel: str, pattern: str) -> bool:
    """fnmatch with ``**`` spanning ``/``.

    ``fnmatch`` treats ``*`` as spanning ``/`` too, which is too loose
    for single-segment patterns like ``reports/trace/*.jsonl``. Handle
    the two cases explicitly:

    * pattern contains ``**`` → collapse to a permissive regex-ish match
      by replacing ``**`` with a sentinel that fnmatch's ``*`` covers.
    * otherwise → require the path to have the same number of segments,
      matching each segment with fnmatch so ``*`` stays within a segment.

    Args:
        rel: POSIX-style relative path to test.
        pattern: Glob pattern, possibly containing ``**``.

    Returns:
        ``True`` when ``rel`` matches ``pattern``.
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
    *,
    truncated: bool = False,
    dropped_files: list[str] | None = None,
    failed_files: list[str] | None = None,
    refused_files: list[str] | None = None,
) -> dict:
    """Build the manifest dict describing a session package.

    ``included_files`` names what was verified written, so a consumer can
    treat it as the contents rather than the intent.

    Args:
        session_dir: Source session directory.
        session_id: Identifier of the session.
        included: ``(relative_path, size_bytes)`` pairs that were written.
        missing_globs: Selection globs that matched no files.
        truncated: Whether a size/count cap stopped the bundle short.
        dropped_files: Files omitted due to truncation.
        failed_files: Selected files whose write failed.
        refused_files: Selected entries rejected by the session boundary.

    Returns:
        A JSON-serializable manifest mapping.
    """
    total = sum(sz for _, sz in included)
    dropped = list(dropped_files or [])
    failed = list(failed_files or [])
    refused = list(refused_files or [])
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
        # True when a size/count cap stopped the bundle short (consult dropped_files).
        "truncated": truncated,
        "dropped_files": dropped,
        # Selected but absent: writes that failed, and entries refused for
        # resolving outside the session or not being regular files.
        "failed_files": failed,
        "refused_files": refused,
        # One flag a consumer can gate on instead of checking each list.
        "complete": not (truncated or dropped or failed or refused),
    }


def _manifest_text(manifest: dict) -> str:
    """Render a manifest dict as a human-readable text summary.

    Args:
        manifest: Manifest mapping produced by :func:`_build_manifest`.

    Returns:
        A multi-line plain-text description of the package contents.
    """
    lines = [
        "Hyperloom session artifact package",
        f"  session_id   : {manifest.get('session_id') or '?'}",
        f"  packaged_at  : {manifest.get('packaged_at_utc')}",
        f"  source dir   : {manifest.get('session_dir')}",
        f"  files        : {manifest.get('included_count')}",
        f"  total bytes  : {manifest.get('included_total_bytes')}",
        f"  complete     : {manifest.get('complete')}",
        f"  truncated    : {manifest.get('truncated')}",
        "",
        "Included files (verified written):",
    ]
    for entry in manifest.get("included_files") or []:
        lines.append(f"  + {entry['path']}  ({entry['bytes']} B)")
    dropped = manifest.get("dropped_files") or []
    if dropped:
        lines.append("")
        lines.append("DROPPED (bundle hit size/count cap — package is INCOMPLETE):")
        for d in dropped:
            lines.append(f"  ! {d}")
    failed = manifest.get("failed_files") or []
    if failed:
        lines.append("")
        lines.append("FAILED TO WRITE (selected but NOT in this package):")
        for f in failed:
            lines.append(f"  ! {f}")
    refused = manifest.get("refused_files") or []
    if refused:
        lines.append("")
        lines.append("REFUSED (not a regular file inside the session; NOT in this package):")
        for r in refused:
            lines.append(f"  ! {r}")
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
        no files matched. Never raises. A returned zip may still be
        partial; its manifest's ``complete`` field says which.
    """
    try:
        sd = Path(session_dir).resolve()
        if not sd.is_dir():
            log.warning("session package skipped: session_dir not a dir: %s", sd)
            return None

        sid = (session_id or "").strip() or sd.name
        matched, missing_globs, refused = _select(sd)
        if not matched:
            log.warning("session package skipped: no artifacts matched in %s", sd)
            return None

        # Apply safety caps. On hitting a cap, record what got dropped and
        # flag the manifest as truncated.
        included: list[tuple[Path, str, int]] = []
        total = 0
        truncated = False
        dropped: list[str] = []
        for i, p in enumerate(matched):
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if len(included) >= _MAX_FILES or total + sz > _MAX_TOTAL_BYTES:
                truncated = True
                dropped = [q.relative_to(sd).as_posix() for q in matched[i:]]
                log.warning(
                    "session package: hit size/count cap, TRUNCATING bundle "
                    "(included=%d, bytes=%d, dropped=%d). Manifest flagged "
                    "truncated=true.",
                    len(included),
                    total,
                    len(dropped),
                )
                break
            included.append((p, p.relative_to(sd).as_posix(), sz))
            total += sz

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
                written: list[tuple[str, int]] = []
                write_failures: list[str] = []
                for p, rel, sz in included:
                    try:
                        zf.write(p, arcname=rel)
                        written.append((rel, sz))
                    except OSError:
                        write_failures.append(rel)
                        log.warning("session package: failed to add %s", rel)
                # Built here so it describes the members that exist, not the
                # ones that were selected.
                manifest = _build_manifest(
                    sd,
                    sid,
                    written,
                    missing_globs,
                    truncated=truncated,
                    dropped_files=dropped,
                    failed_files=write_failures,
                    refused_files=refused,
                )
                zf.writestr(MANIFEST_JSON_NAME, json.dumps(manifest, indent=2))
                zf.writestr(MANIFEST_TXT_NAME, _manifest_text(manifest))
            os.replace(tmp_path, target)
        except Exception:
            with suppress(OSError):
                tmp_path.unlink()
            raise

        log.info(
            "session package: wrote %s (%d files, %d bytes pre-zip, complete=%s)",
            target,
            len(written),
            total,
            manifest["complete"],
        )

        # Also lay the same files down loose (uncompressed, original tree)
        # straight under the dest root so a consumer can grab one file without
        # unzip. It succeeds or fails independently of the zip, so it carries
        # its own manifest.
        if _loose_enabled():
            try:
                copied, loose_failures = _copy_loose_tree(included, root)
                _write_loose_manifest(
                    root,
                    _build_manifest(
                        sd,
                        sid,
                        copied,
                        missing_globs,
                        truncated=truncated,
                        dropped_files=dropped,
                        failed_files=loose_failures,
                        refused_files=refused,
                    ),
                )
                log.info(
                    "session package: copied %d loose files into %s",
                    len(copied),
                    root,
                )
            except Exception:  # noqa: BLE001 — loose copy must not mask the zip
                log.exception("session package: loose copy failed (non-fatal)")

        return target
    except Exception:  # noqa: BLE001 — never let packaging mask stop_reason
        log.exception("session package failed (non-fatal)")
        return None
