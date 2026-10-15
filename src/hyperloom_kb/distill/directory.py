# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write a KB's visible Distills into a directory the KB owns, one markdown page per Distill, for a Curated Wiki to
search and cite.

Each sync makes a schema's subdirectory a mirror of the Distills the KB shows: it writes the page of each, and removes
the page of one no longer visible, so a page edited by hand is written back. A page's YAML frontmatter carries its ids
and its evidence summary; its body is the claim, so the ids never become text a search matches. Only files named like
a Distill's page are written or removed, each replaced whole.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hyperloom_kb.distill.model import FormDeclaration
from hyperloom_kb.remote import RemoteClient
from hyperloom_kb.schema import JsonValue

PAGE_NAME = re.compile(r"dst-[0-9a-f]{32}\.md")
PAGE_COUNTS = ("created", "updated", "unchanged", "removed")


def schema_directory(directory: Path, schema_ref: str) -> Path:
    """The subdirectory of ``directory`` holding ``schema_ref``'s pages."""

    return directory / f"schema-{schema_ref.rsplit(':', 1)[-1][:16]}"


def _value(value: Any) -> str:
    """A claim value as a page shows it: text as written, a file by its name, several values joined, and a number or
    a boolean as JSON writes it."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value["name"])
    if isinstance(value, list):
        return ", ".join(_value(item) for item in value)
    return json.dumps(value)


_COUNT_LABELS = {
    "supports": "supporting",
    "contradicts": "contradicting",
    "irrelevant": "irrelevant",
    "undetermined": "undetermined",
}


def _counts(counts: Mapping[str, Any], *names: str) -> str:
    return ", ".join(f"{counts[name]} {_COUNT_LABELS[name]}" for name in names) + f", from {counts['runs']} runs"


def render(listed: Mapping[str, Any], read: Mapping[str, Any], form: FormDeclaration) -> str:
    """The page of one Distill: ``listed`` is its listing item, ``read`` its read, and ``form`` its claim's form."""

    entry, drift = read["entry"], read["drift"]
    snapshot = entry["evidence"]["stats"]
    frontmatter: dict[str, JsonValue] = {
        "id": entry["id"],
        "version": entry["version"],
        "schema_ref": entry["schema_ref"],
        "form": listed["form_name"],
        "form_ref": entry["form_ref"],
        "tags": entry["tags"],
        "scope": entry["scope"],
        "evidence_status": drift["evidence_status"],
        "needs_attention": drift["needs_attention"],
        "evidence": {"snapshot": snapshot, "current": drift["current"]},
        "representatives": read["representatives"],
        "relations": entry["relations"],
    }
    # JSON is YAML, so each value reads the same to a YAML parser and needs no YAML library to write.
    lines = ["---", *(f"{key}: {json.dumps(value, sort_keys=True)}" for key, value in frontmatter.items()), "---", ""]
    summary = form.summary_field
    lines += [f"# {listed['summary'] or form.name}", ""]
    for field in form.claim:
        if field != summary and field.name in entry["claim"]:
            lines += [f"**{field.name}:** {_value(entry['claim'][field.name])}", ""]
    lines += [
        f"Evidence when written: {_counts(snapshot, 'supports', 'contradicts', 'irrelevant')}.",
        f"Evidence at this sync ({drift['evidence_status']}): "
        f"{_counts(drift['current'], 'supports', 'contradicts', 'undetermined')}.",
        "",
        "The complete evidence: `hyperloom-kb distill get` with this page's id, `--follow-evidence` for its records.",
        "",
    ]
    return "\n".join(lines)


def _write_page(path: Path, page: str) -> str:
    """Write ``page`` at ``path`` unless it holds it already; what changed, of ``PAGE_COUNTS``."""

    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == page:
        return "unchanged"
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(page, encoding="utf-8")
    os.replace(staged, path)
    return "created" if existing is None else "updated"


def sync_directory(
    client: RemoteClient, directory: Path, schema_refs: Iterable[str], *, representatives: int | None = None
) -> dict[str, JsonValue]:
    """Mirror each of ``schema_refs``' visible Distills into its subdirectory of ``directory``; what each page did,
    and the Distills that need their author's attention."""

    schemas: dict[str, JsonValue] = {}
    attention: list[JsonValue] = []
    for schema_ref in schema_refs:
        target = schema_directory(directory, schema_ref)
        target.mkdir(parents=True, exist_ok=True)
        forms = {
            str(item["form_ref"]): FormDeclaration.from_dict(item["form"])
            for item in client.distill_forms(schema_ref)["forms"]
        }
        counts: Counter[str] = Counter()
        written: set[str] = set()
        for listed in client.list_distills(schema_ref)["items"]:
            entry_id = str(listed["entry_id"])
            read = client.distill(entry_id, representatives=representatives)
            name = f"{entry_id}.md"
            counts[_write_page(target / name, render(listed, read, forms[str(listed["form_ref"])]))] += 1
            written.add(name)
            if read["drift"]["needs_attention"]:
                attention.append(
                    {
                        "schema_ref": schema_ref,
                        "entry_id": entry_id,
                        "summary": listed["summary"],
                        "evidence_status": read["drift"]["evidence_status"],
                    }
                )
        for page in target.iterdir():
            if PAGE_NAME.fullmatch(page.name) and page.name not in written:
                page.unlink()
                counts["removed"] += 1
        schemas[schema_ref] = {"directory": str(target), **{key: counts[key] for key in PAGE_COUNTS}}
    return {"directory": str(directory), "schemas": schemas, "needs_attention": attention}


__all__ = ["PAGE_COUNTS", "PAGE_NAME", "render", "schema_directory", "sync_directory"]
