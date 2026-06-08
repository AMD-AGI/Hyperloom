# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Source-files manifest renderer — what files this breakdown was
built from. Helps operators replay or audit a session.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer


@register_renderer("source_files")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the source-files manifest section.

    Lists, per file kind, how many artifacts the breakdown was built from
    plus a short preview, so a session can be replayed or audited. Skipped
    when no source-files manifest is present.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered source-files section.
    """
    sf = breakdown.get("source_files") or {}
    if not sf:
        return RenderedSection(
            section_id="source_files",
            title="Source Files",
            key_facts=[],
            markdown_block="",
            skipped=True,
        )

    rows = []
    for k, v in sorted(sf.items()):
        if isinstance(v, list):
            preview = ", ".join(str(x) for x in v[:3]) if v else "—"
            rows.append([k, len(v), preview])
        elif v in (None, "", []):
            # Skip empty entries entirely rather than rendering ``—`` /
            # ``—`` rows that add visual noise without information.
            continue
        else:
            rows.append([k, "1", str(v)])
    md = md_table(["kind", "count", "first values"], rows)
    facts = [f"source_files manifest captures {len(rows)} file kind(s)."]
    return RenderedSection(
        section_id="source_files",
        title="Source Files",
        key_facts=facts,
        markdown_block=md,
        skipped=False,
    )
