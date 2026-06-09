# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Source-files manifest renderer — what files this breakdown was
built from. Helps operators replay or audit a session.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_table, register_renderer


@register_renderer("source_files")
def render(breakdown: dict[str, Any]) -> RenderedSection:
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
            # Skip empty entries rather than render noisy ``—`` rows.
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
