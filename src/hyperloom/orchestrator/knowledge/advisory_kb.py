# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Human-editable, framework-partitioned advisory knowledge base.

Knowledge lives as markdown under ``advisory_kb/`` (shipped in the repo, so it is
version-controlled and reviewable):

    advisory_kb/
      generic/*.md        -> reaches ALL frameworks (framework-agnostic reasoning)
      vllm/*.md           -> only vLLM runs
      sglang/*.md         -> only SGLang runs
      atom/*.md           -> only ATOM runs

At PRELUDE the loader reads ``generic/`` + ``<run framework>/`` and nothing else,
so **the folder is the routing gate** — a fact in ``generic/`` reaches every run;
a fact in ``vllm/`` reaches only vLLM runs. This mirrors the framework-agent KB's
``path_for_framework`` directory-partition convention.

Everything here is ADVISORY: parsed entries feed the specialist "KB CONTEXT"
prompt section and advisory ``gaps[]``; the Critic still gates every kept change.
Nothing here can reject a config.

Markdown parse contract (human-forgiving)
-----------------------------------------
Each ``##`` heading starts one entry. The heading text is the claim (``what`` for
hints; also the checklist ``detect`` lead). An optional labeled block underneath
supplies structured fields, one per line::

    ## VLLM_ROCM_USE_AITER=1 is the master AITER switch (off by default)
    - kind: hint
    - source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
    - impact: throughput
    - accuracy_risk: none
    - domain_tags: framework

    ## rocm.fp4.aiter_master_switch_gap
    - kind: checklist
    - source: cph-perf-tuning:KNOWLEDGE.md#0.2.1, session:gpt-oss-120b/20260729T193315Z
    - applies_when: gpu=rocm, precision=fp4
    - domain_hint: kernel_switch_specialist
    - source_dirs: vllm/model_executor/layers/quantization/, vllm/model_executor/layers/fused_moe/
    - consequence: <text>
    - bridge: <text>
    - detect: <text>            # optional; falls back to the heading + body prose

Unstructured prose in the body is retained as the entry's ``body`` and appended to
``detect``/``what`` so nothing a human writes is lost. A missing ``source`` drops
the entry (advisory KB must stay attributable), matching ``research_hints``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("hyperloom.advisory_kb")

_KNOWN_FRAMEWORKS = ("vllm", "sglang", "atom")
_GENERIC = "generic"


def kb_root() -> Path:
    """Resolve the advisory-KB root each call (env override for tests/deploys).

    Order: (1) ``HYPERLOOM_ADVISORY_KB_DIR`` env; (2) the shipped
    ``advisory_kb/`` dir next to this module (development / packaged default).

    Returns:
        The resolved advisory-KB root path.
    """
    explicit = os.environ.get("HYPERLOOM_ADVISORY_KB_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path(__file__).resolve().parent / "advisory_kb"


@dataclass
class _Entry:
    """One parsed markdown entry (heading + labeled fields + body prose)."""

    heading: str
    fields: dict[str, str] = field(default_factory=dict)
    body: str = ""
    partition: str = ""  # "generic" | "vllm" | "sglang" | "atom"
    rel_path: str = ""


def _normalize_framework(framework: str) -> str:
    """Lower/strip a framework label to a known partition name (or "")."""
    fw = (framework or "").strip().lower()
    return fw if fw in _KNOWN_FRAMEWORKS else fw  # unknown passes through (folder just won't exist)


def _parse_markdown(text: str, *, partition: str, rel_path: str) -> list[_Entry]:
    """Split one markdown doc into entries on ``##`` headings.

    Args:
        text: Raw markdown.
        partition: The owning partition ("generic"/"vllm"/...).
        rel_path: Path relative to the KB root (for provenance/debug).

    Returns:
        Parsed entries (possibly empty).
    """
    entries: list[_Entry] = []
    cur: _Entry | None = None
    body_lines: list[str] = []

    def _flush() -> None:
        if cur is not None:
            cur.body = "\n".join(body_lines).strip()
            entries.append(cur)

    for line in text.splitlines():
        m = re.match(r"^##\s+(.*\S)\s*$", line)
        if m:
            _flush()
            cur = _Entry(heading=m.group(1).strip(), partition=partition, rel_path=rel_path)
            body_lines = []
            continue
        if cur is None:
            continue  # preamble before the first heading (title/intro) — ignored
        fm = re.match(r"^\s*-\s*([A-Za-z_]+)\s*:\s*(.*)$", line)
        if fm:
            key = fm.group(1).strip().lower()
            val = fm.group(2).strip()
            cur.fields[key] = val
        else:
            body_lines.append(line)
    _flush()
    return entries


def _load_partition(root: Path, partition: str) -> list[_Entry]:
    """Read all ``*.md`` under ``<root>/<partition>/`` (sorted, fail-soft)."""
    pdir = root / partition
    if not pdir.is_dir():
        return []
    out: list[_Entry] = []
    for md in sorted(pdir.glob("*.md")):
        if md.name.upper() == "README.MD":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            log.warning("advisory_kb: failed to read %s", md)
            continue
        out.extend(_parse_markdown(text, partition=partition, rel_path=f"{partition}/{md.name}"))
    return out


def load_entries(framework: str) -> list[_Entry]:
    """Return parsed entries for ``generic/`` + ``<framework>/`` (routing gate).

    Args:
        framework: The run's framework (``vllm``/``sglang``/``atom``); empty
            resolves to generic-only.

    Returns:
        The applicable parsed entries.
    """
    root = kb_root()
    fw = _normalize_framework(framework)
    entries = _load_partition(root, _GENERIC)
    if fw and fw != _GENERIC:
        entries.extend(_load_partition(root, fw))
    return entries


def _split_csv(val: str) -> list[str]:
    return [p.strip() for p in val.split(",") if p.strip()]


def hints_from_markdown(framework: str) -> list[dict[str, object]]:
    """Return advisory research-hint dicts from ``generic/`` + ``<framework>/``.

    Only entries whose ``kind`` is ``hint`` (or unset) become hints. A missing
    ``source`` drops the entry (attributability invariant). The returned shape
    matches what ``research_hints._coerce_hint`` accepts.

    Args:
        framework: The run's framework.

    Returns:
        Hint dicts: ``{what, expected_impact, accuracy_risk, source, domain_tags,
        framework, status}``.
    """
    hints: list[dict[str, object]] = []
    for e in load_entries(framework):
        kind = (e.fields.get("kind") or "hint").lower()
        if kind != "hint":
            continue
        source = e.fields.get("source", "").strip()
        if not source:
            continue  # advisory KB must stay attributable
        what = e.heading
        if e.body:
            what = f"{what} {e.body}".strip()
        hints.append(
            {
                "what": what,
                "expected_impact": e.fields.get("impact", "").strip(),
                "accuracy_risk": e.fields.get("accuracy_risk", "").strip(),
                "source": source,
                "domain_tags": _split_csv(e.fields.get("domain_tags", "")),
                "framework": "" if e.partition == _GENERIC else e.partition,
                "status": e.fields.get("status", "proposed").strip() or "proposed",
            }
        )
    return hints


def _parse_applies_when(val: str) -> dict[str, str]:
    """Parse ``gpu=rocm, precision=fp4`` into a dict (empty-tolerant)."""
    out: dict[str, str] = {}
    for part in val.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower()
            v = v.strip().lower()
            if k and v:
                out[k] = v
    return out


def checklist_from_markdown(framework: str) -> list[dict[str, object]]:
    """Return static-recon checklist entries from ``generic/`` + ``<framework>/``.

    Only entries whose ``kind`` is ``checklist`` become checklist items. Each is
    returned as a plain dict mirroring ``static_recon_checklist.ChecklistEntry``
    fields; ``applies_when`` is a dict the checklist's ``entries_for`` filters on
    (gpu/precision). The framework itself is already gated by the folder, so no
    per-record framework key is needed.

    Args:
        framework: The run's framework.

    Returns:
        Checklist entry dicts.
    """
    out: list[dict[str, object]] = []
    for e in load_entries(framework):
        if (e.fields.get("kind") or "").lower() != "checklist":
            continue
        source = e.fields.get("source", "").strip()
        if not source:
            continue
        detect = e.fields.get("detect", "").strip() or e.body.strip()
        out.append(
            {
                "id": e.heading.strip(),
                "applies_when": _parse_applies_when(e.fields.get("applies_when", "")),
                "detect": detect,
                "consequence": e.fields.get("consequence", "").strip(),
                "bridge": e.fields.get("bridge", "").strip(),
                "domain_hint": e.fields.get("domain_hint", "freeform").strip() or "freeform",
                "source_dirs": tuple(_split_csv(e.fields.get("source_dirs", ""))),
                "evidence": tuple(_split_csv(source)),
            }
        )
    return out


__all__ = [
    "kb_root",
    "load_entries",
    "hints_from_markdown",
    "checklist_from_markdown",
]
