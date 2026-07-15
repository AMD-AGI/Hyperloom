# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared, dependency-light audit helpers for framework-agent.

Pure building blocks (diff parsing, local-file resolution, symbol extraction,
verdict assembly, patch-text acquisition) used by both
:mod:`hyperloom.agents.framework.audit` and
:mod:`hyperloom.agents.framework.cross_framework`. Kept standalone so neither
imports the other (breaks the import cycle).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import now_iso


log = logging.getLogger(__name__)


_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)")
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class FileChange:
    """One file's hunks parsed out of a unified diff."""

    path: str
    is_new: bool = False
    is_deleted: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


def parse_unified_diff(patch_text: str) -> list[FileChange]:
    """Parse a unified diff into per-file added/removed/context line groups.

    Args:
        patch_text: The unified diff (``diff --git`` / ``--- a/`` / ``+++ b/``).

    Returns:
        One :class:`FileChange` per file section, in first-seen order.
    """
    changes: list[FileChange] = []
    current: FileChange | None = None
    for raw in (patch_text or "").splitlines():
        if raw.startswith("diff --git"):
            # New file section; the +++ line below sets the canonical path.
            current = FileChange(path="")
            changes.append(current)
            continue
        if current is None:
            # Tolerate a diff with no leading "diff --git".
            if raw.startswith("--- ") or raw.startswith("+++ "):
                current = FileChange(path="")
                changes.append(current)
            else:
                continue
        if raw.startswith("new file mode"):
            current.is_new = True
            continue
        if raw.startswith("deleted file mode"):
            current.is_deleted = True
            continue
        if raw.startswith("+++ "):
            current.path = _strip_diff_path(raw[4:].strip())
            continue
        if raw.startswith("--- "):
            if not current.path:
                current.path = _strip_diff_path(raw[4:].strip())
            continue
        if raw.startswith("@@"):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            current.added.append(raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            current.removed.append(raw[1:])
        elif raw.startswith(" "):
            current.context.append(raw[1:])
    # Drop empty/placeholder sections.
    return [c for c in changes if c.path and c.path != "/dev/null"]


def _strip_diff_path(token: str) -> str:
    """Normalize a diff path token (strip ``a/``/``b/`` prefix + tab suffix)."""
    token = token.split("\t", 1)[0].strip()
    if token in ("/dev/null", ""):
        return token
    if token.startswith(("a/", "b/")):
        return token[2:]
    return token


def _resolve_local_file(path: str, roots: list[Path]) -> Path | None:
    """Resolve a diff path to an existing file under one of ``roots``.

    Tries strip levels 0..3 (the diff path often carries the package dir as its
    leading component while the root *is* that package dir), across all roots.

    Args:
        path: The diff's file path (e.g. ``vllm/model_executor/models/x.py``).
        roots: Candidate framework source roots.

    Returns:
        The first existing local path, or ``None``.
    """
    parts = Path(path).parts
    for root in roots:
        for strip in range(0, min(4, len(parts))):
            cand = root.joinpath(*parts[strip:])
            if cand.is_file():
                return cand
    return None


def _symbols(lines: list[str]) -> list[str]:
    """Extract def/class names from a set of lines (strongest match signal)."""
    syms: list[str] = []
    for ln in lines:
        m = _DEF_RE.match(ln) or _CLASS_RE.match(ln)
        if m:
            syms.append(m.group(1))
    return syms


def _verdict(
    *,
    candidate_id: str,
    semantic_status: str,
    applicability: str,
    confidence: float,
    evidence: list[dict[str, Any]],
    risks: list[str],
    recommended_next_step: str,
    layer: str = "static",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the semantic_audit payload in the canonical schema."""
    return {
        "candidate_id": str(candidate_id),
        "semantic_status": semantic_status,
        "applicability": applicability,
        "confidence": round(float(confidence), 4),
        "evidence": evidence,
        "risks": risks,
        "recommended_next_step": recommended_next_step,
        "layer": layer,
        "metrics": metrics or {},
        "ts": now_iso(timespec="auto"),
    }


def _fetch_diff_url(diff_url: str, work_dir: Path) -> str:
    """Fetch a unified diff from an ``http(s)://`` or ``file://`` URL (best-effort).

    Args:
        diff_url: The diff URL (GitHub ``.diff`` or ``file://`` path).
        work_dir: Unused except as a hint; kept for symmetry / future caching.

    Returns:
        The diff text, or ``""`` on any failure.
    """
    del work_dir
    try:
        from urllib.request import urlopen

        if diff_url.startswith("file://"):
            p = Path(diff_url[len("file://") :])
            return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        if diff_url.startswith(("http://", "https://")):
            with urlopen(diff_url, timeout=30) as resp:  # noqa: S310 — public PR diff
                return resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — best-effort fetch
        log.warning("phase-audit: diff_url fetch failed (%s): %r", diff_url, exc)
    return ""


def _obtain_patch_text(request: dict[str, Any], work_dir: Path) -> tuple[str, str]:
    """Resolve the PR's unified diff text from the request / network.

    Resolution order: inline ``diff_text`` → ``patches_path`` file →
    gbrain PR KB → ``diff_url`` → ``fetch_pr_audit_material`` via
    primus_cortex (when URL + PR number present).

    Args:
        request: The phase-audit request.
        work_dir: Where fetched ``pr.patches`` / ``pr_files.json`` land.

    Returns:
        ``(patch_text, source)``; ``patch_text`` is ``""`` when unavailable.
    """
    diff_text = str(request.get("diff_text") or "")
    if diff_text.strip():
        return diff_text, "inline"

    patches_path = str(request.get("patches_path") or "").strip()
    if patches_path:
        p = Path(patches_path).expanduser()
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace"), "patches_path"
            except OSError:
                pass

    candidate = request.get("candidate") or {}

    # gbrain PR KB (primary): synthesize the unified diff from the KB files
    # page, degrading to "" on any miss so audit never sees a partial diff.
    import os

    if (os.environ.get("PR_KB_ENABLE", "1") or "1").strip() != "0":
        kb_repo = str(request.get("repo_url") or candidate.get("repo") or "").strip()
        kb_pr = candidate.get("pr_number")
        kb_slug = str(candidate.get("pr_kb_files_slug") or "").strip()
        if kb_repo and (kb_slug or isinstance(kb_pr, int)):
            try:
                from .gbrain_page_client import build_gbrain_page_client_from_env
                from .pr_kb import fetch_pr_kb_diff

                client = build_gbrain_page_client_from_env()
                if client is not None:
                    text, src = fetch_pr_kb_diff(kb_repo, kb_pr, client=client, slug=kb_slug)
                    if text.strip():
                        return text, src
            except Exception as exc:  # noqa: BLE001 — best-effort primary
                log.warning("phase-audit: pr_kb diff fetch failed: %r", exc)

    # diff_url (GitHub ``.diff`` / file://), fetched best-effort.
    diff_url = str(request.get("diff_url") or candidate.get("diff_url") or "").strip()
    if diff_url:
        text = _fetch_diff_url(diff_url, work_dir)
        if text.strip():
            return text, "diff_url"

    repo_url = str(request.get("repo_url") or candidate.get("repo") or "").strip()
    pr_number = candidate.get("pr_number")
    primus_url = str(request.get("primus_cortex_url") or "").strip()
    if not primus_url:
        import os

        primus_url = os.environ.get("PRIMUS_CORTEX_PR_API", "").strip()
    if repo_url and isinstance(pr_number, int) and primus_url:
        try:
            from .runtime.tools_api import fetch_pr_audit_material

            work_dir.mkdir(parents=True, exist_ok=True)
            paths = fetch_pr_audit_material(
                repo_url,
                pr_number,
                out_dir=work_dir,
                primus_cortex_url=primus_url,
            )
            patches_file = Path(paths.get("patches_path") or "")
            if patches_file.is_file():
                return patches_file.read_text(encoding="utf-8", errors="replace"), "primus_cortex"
        except Exception as exc:  # noqa: BLE001 — best-effort fetch
            log.warning("phase-audit: fetch_pr_audit_material failed: %r", exc)
    return "", "unavailable"
