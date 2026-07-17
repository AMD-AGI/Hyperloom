# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR KB page parsing + diff synthesis (consumer side).

Reads the markdown pages the ``Primus-Claw/pr-kb`` worker writes to gbrain
(``pr-kb-meta/ pr-kb-files/ pr-kb-index/``) and adapts them to the shapes
framework-agent already consumes:

* :func:`fetch_pr_kb_diff` — files page -> unified diff text (P1, for audit).
* :func:`parse_index_prs` — candidate discovery inputs (P2).

All helpers are defensive: missing keys / unparseable blocks yield empty
results so callers fall back to Cortex / GitHub.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .gbrain_page_client import GbrainPageClient, GbrainPageError
from .pr_kb_slug import files_slug, index_slug, normalise_repo

log = logging.getLogger(__name__)


def _page_markdown(page: dict[str, Any]) -> str:
    """Extract page body markdown across known gbrain key spellings.

    ``compiled_truth`` is the field ``get_page`` returns on this gbrain
    deployment; the rest are defensive fallbacks.
    """
    for key in ("compiled_truth", "markdown", "content", "body", "text", "page_content"):
        val = page.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_fenced_json(markdown: str, marker: str) -> Any:
    """Return the JSON payload of a ```json fence under a ``## <marker>`` head."""
    if not markdown:
        return None
    idx = markdown.find(marker)
    if idx == -1:
        return None
    fence = markdown.find("```json", idx)
    if fence == -1:
        return None
    end = markdown.find("```", fence + 7)
    if end == -1:
        return None
    try:
        return json.loads(markdown[fence + 7 : end])
    except json.JSONDecodeError:
        return None


def _parse_frontmatter(markdown: str) -> dict[str, Any]:
    """Parse a minimal YAML frontmatter block (flat scalars + flow lists)."""
    if not markdown.lstrip().startswith("---"):
        return {}
    body = markdown.lstrip()
    end = body.find("\n---", 3)
    if end == -1:
        return {}
    fm: dict[str, Any] = {}
    for line in body[3:end].splitlines():
        line = line.rstrip()
        if not line or line.startswith("#") or ":" not in line or line.startswith("  "):
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        val = raw.strip()
        if not key:
            continue
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            fm[key] = [v.strip().strip('"') for v in inner.split(",") if v.strip()] if inner else []
        else:
            fm[key] = val.strip('"')
    return fm


def synthesize_unified_diff(patches: list[dict[str, Any]]) -> str:
    """Build a git-style unified diff from PR KB ``Patches JSON`` entries.

    Skips entries flagged ``patch_omitted`` (binary / too large). Returns an
    empty string when nothing usable remains.

    Args:
        patches: The parsed ``## Patches JSON`` list.

    Returns:
        Concatenated unified-diff text (empty when no usable patch).
    """
    out: list[str] = []
    for entry in patches:
        if not isinstance(entry, dict) or entry.get("patch_omitted"):
            continue
        patch = entry.get("patch")
        filename = entry.get("filename")
        if not patch or not filename:
            continue
        status = str(entry.get("status") or "")
        old = "/dev/null" if status == "added" else f"a/{filename}"
        new = "/dev/null" if status == "removed" else f"b/{filename}"
        out.append(f"diff --git a/{filename} b/{filename}")
        out.append(f"--- {old}")
        out.append(f"+++ {new}")
        out.append(patch if patch.endswith("\n") else patch + "\n")
    return "\n".join(out).strip() + "\n" if out else ""


def parse_files_page(page: dict[str, Any]) -> dict[str, Any]:
    """Return ``{files_truncated, patch_omitted_any, patches}`` from a files page."""
    md = _page_markdown(page)
    fm = _parse_frontmatter(md)
    patches = _extract_fenced_json(md, "## Patches JSON") or []
    if not isinstance(patches, list):
        patches = []
    truncated = str(fm.get("files_truncated") or "").lower() == "true"
    omitted_any = any(isinstance(p, dict) and p.get("patch_omitted") for p in patches)
    return {"files_truncated": truncated, "patch_omitted_any": omitted_any, "patches": patches}


def fetch_pr_kb_diff(
    repo: str,
    pr_number: int | str,
    *,
    client: GbrainPageClient,
    slug: str | None = None,
) -> tuple[str, str]:
    """Fetch + synthesize a PR's unified diff from its PR KB files page.

    Returns ``("", "")`` on any miss / truncation / omission so the caller
    falls back to a full source (``diff_url`` / Cortex), never a partial diff.

    Args:
        repo: Repo full name or URL.
        pr_number: PR number.
        client: A configured :class:`GbrainPageClient`.
        slug: Optional precomputed files-page slug (``candidate.pr_kb_files_slug``).
            When provided it is used verbatim and ``repo``/``pr_number``
            derivation is skipped, so a candidate carrying only the slug still
            resolves its KB diff when ``pr_number`` is absent.

    Returns:
        ``(diff_text, "gbrain_pr_kb")`` on success, else ``("", "")``.
    """
    files_page_slug = (slug or "").strip()
    if not files_page_slug:
        repo_n = normalise_repo(repo)
        if not repo_n or pr_number in (None, ""):
            return "", ""
        files_page_slug = files_slug(repo_n, pr_number)
    try:
        page = client.get_page(files_page_slug)
    except GbrainPageError as exc:
        log.warning("pr_kb: get_page(%s) failed: %r", files_page_slug, exc)
        return "", ""
    if not page:
        return "", ""
    parsed = parse_files_page(page)
    # Never feed a partial diff downstream; degrade to the full diff_url.
    if parsed["files_truncated"] or parsed["patch_omitted_any"]:
        log.info("pr_kb: files page %s truncated/omitted; degrading to diff_url", files_page_slug)
        return "", ""
    diff = synthesize_unified_diff(parsed["patches"])
    return (diff, "gbrain_pr_kb") if diff.strip() else ("", "")


def parse_index_prs(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``## PRs JSON`` list from an index page (empty on miss)."""
    data = _extract_fenced_json(_page_markdown(page), "## PRs JSON")
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


__all__ = [
    "synthesize_unified_diff",
    "parse_files_page",
    "fetch_pr_kb_diff",
    "parse_index_prs",
    "index_slug",
]
