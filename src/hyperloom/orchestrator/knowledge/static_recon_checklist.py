# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seed checklist for the static-recon specialist.

The static-recon specialist is a read-only PRELUDE sub-agent that greps the
framework source tree (vLLM / SGLang) for *un-bridged capability switches* —
fast paths that *should* be enabled for the current ``(model_class, gpu_type,
precision)`` but are silently disabled by a predicate (e.g. a CUDA-only
``*_supported()`` helper returning ``False`` on ROCm). This module seeds the
specialist with a curated list of known patterns to look for.

Each :class:`ChecklistEntry` describes one known "bridge opportunity":
- ``id`` — stable slug, used to build the gap canonical id.
- ``applies_when`` — coarse predicate over ``{gpu, precision}`` (``"*"`` = any)
  so an entry is only handed to a run where it could plausibly fire.
- ``detect`` — what to grep for and how to confirm the path is disabled.
- ``consequence`` — what regression the disabled path causes.
- ``bridge`` — advisory sketch of the fix.
- ``domain_hint`` — which EXPLORE specialist domain should pick up the gap.
- ``evidence`` — provenance (PR / session) the pattern was distilled from.

A small, hand-curated starter set keyed to validated findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistEntry:
    """One known un-bridged-capability pattern the static-recon specialist hunts.

    Attributes:
        id: Stable slug; used to build ``gap.static_recon.<id>``.
        applies_when: Coarse match dict over ``{"gpu": ..., "precision": ...}``.
            Values are lower-cased substrings; ``"*"`` (or absent) matches any.
        detect: Grep target + confirmation instruction handed to the specialist.
        consequence: The regression caused while the path stays disabled.
        bridge: Advisory sketch of the fix.
        domain_hint: EXPLORE specialist domain to route the seeded gap to.
        source_dirs: Source subdirectories to point the specialist at (relative
            to a framework source root); rendered as ``source_hint_directories``.
        evidence: Provenance refs (PR / session) the pattern was distilled from.
    """

    id: str
    applies_when: dict[str, str]
    detect: str
    consequence: str
    bridge: str
    domain_hint: str = "freeform"
    source_dirs: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


# Checklist entries now live as HUMAN-EDITABLE MARKDOWN under
# ``advisory_kb/<framework>/checklist.md`` (routed by folder). They are loaded
# and converted to :class:`ChecklistEntry` objects on demand. Keep entries
# grounded in a validated finding (each carries a ``source``/``evidence`` ref).


def _load_checklist(framework: str = "") -> tuple["ChecklistEntry", ...]:
    """Build ChecklistEntry objects from the advisory markdown KB.

    Reads ``advisory_kb/generic/`` + ``advisory_kb/<framework>/`` checklist
    entries (the folder is the framework routing gate) and adapts each parsed
    dict into a :class:`ChecklistEntry`.

    Args:
        framework: The run's framework (``vllm``/``sglang``/``atom``); empty
            resolves to the generic partition only.

    Returns:
        The checklist entries (possibly empty).
    """
    from . import advisory_kb as _advisory_kb

    out: list[ChecklistEntry] = []
    for d in _advisory_kb.checklist_from_markdown(framework):
        out.append(
            ChecklistEntry(
                id=str(d.get("id") or "").strip(),
                applies_when={str(k): str(v) for k, v in (d.get("applies_when") or {}).items()},
                detect=str(d.get("detect") or ""),
                consequence=str(d.get("consequence") or ""),
                bridge=str(d.get("bridge") or ""),
                domain_hint=str(d.get("domain_hint") or "freeform") or "freeform",
                source_dirs=tuple(d.get("source_dirs") or ()),
                evidence=tuple(d.get("evidence") or ()),
            )
        )
    return tuple(e for e in out if e.id)


def _matches(entry_val: str, run_val: str) -> bool:
    """Return True when a checklist ``applies_when`` value matches the run value.

    ``"*"`` / empty matches anything. Otherwise the run value is tokenized on
    non-alphanumeric boundaries (so ``"fp8_e4m3"`` -> ``{"fp8", "e4m3"}``) and
    the entry value matches iff it equals the whole run value or is one of its
    tokens. This is deliberately NOT a loose substring match, so ``"fp8"`` does
    NOT match ``"mxfp8"`` and vice versa (they are distinct precisions).
    """
    entry_val = (entry_val or "").strip().lower()
    if not entry_val or entry_val == "*":
        return True
    run_val = (run_val or "").strip().lower()
    if not run_val:
        return False
    if entry_val == run_val:
        return True
    tokens = {t for t in re.split(r"[^a-z0-9]+", run_val) if t}
    return entry_val in tokens


def _gpu_family(gpu_type: str) -> str:
    """Map a GPU type label to a coarse family token used by ``applies_when``.

    AMD Instinct parts (``MI300X`` / ``MI325X`` / ``MI355X`` / ``gfx94*`` /
    ``gfx95*``) map to ``"rocm"``; everything else passes through lower-cased so
    a CUDA/NVIDIA entry could match in the future.
    """
    g = (gpu_type or "").strip().lower()
    if g.startswith("mi") or g.startswith("gfx") or "rocm" in g or "amd" in g:
        return "rocm"
    return g


def entries_for(
    *, model_class: str = "", gpu_type: str = "", precision: str = "", framework: str = ""
) -> list[ChecklistEntry]:
    """Return the checklist entries applicable to a ``(framework, gpu, precision)``.

    Framework routing is by folder (``generic/`` + ``<framework>/``); gpu and
    precision are then filtered per-entry via ``applies_when``. ``model_class``
    is currently advisory (entries gate on gpu/precision only); it is accepted
    now so the signature is stable when model-class-specific entries are added.

    Args:
        model_class: Categorical model class (advisory; reserved for future use).
        gpu_type: GPU type label (e.g. ``"MI300X"``), normalized to a family.
        precision: Workload precision (e.g. ``"fp8"`` / ``"mxfp8"`` / ``"fp4"``).
        framework: The run's framework, selecting which markdown KB folders load.

    Returns:
        The matching :class:`ChecklistEntry` list (possibly empty).
    """
    gpu_fam = _gpu_family(gpu_type)
    out: list[ChecklistEntry] = []
    for e in _load_checklist(framework):
        if not _matches(e.applies_when.get("gpu", "*"), gpu_fam):
            continue
        if not _matches(e.applies_when.get("precision", "*"), precision):
            continue
        out.append(e)
    return out


def source_hint_directories_for(
    *, model_class: str = "", gpu_type: str = "", precision: str = "", framework: str = ""
) -> tuple[str, ...]:
    """Return the de-duplicated source subdirectories to point the specialist at.

    Built from the ``source_dirs`` of every applicable checklist entry, in first
    -seen order so the prompt's navigation hint is stable.

    Args:
        model_class: Categorical model class (advisory; reserved for future use).
        gpu_type: GPU type label.
        precision: Workload precision.

    Returns:
        Ordered tuple of relative source subdirectories (possibly empty).
    """
    seen: set[str] = set()
    out: list[str] = []
    for e in entries_for(
        model_class=model_class, gpu_type=gpu_type, precision=precision, framework=framework
    ):
        for d in e.source_dirs:
            d = (d or "").strip()
            if d and d not in seen:
                seen.add(d)
                out.append(d)
    return tuple(out)


def render_checklist_for_prompt(entries: list[ChecklistEntry]) -> str:
    """Render checklist entries as a Markdown block for the specialist prompt.

    Args:
        entries: The applicable checklist entries (from :func:`entries_for`).

    Returns:
        A Markdown string, or ``""`` when there are no entries.
    """
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries:
        lines.append(f"- **{e.id}** (domain_hint=`{e.domain_hint}`)")
        lines.append(f"  - detect: {e.detect}")
        lines.append(f"  - consequence: {e.consequence}")
        lines.append(f"  - bridge: {e.bridge}")
        if e.source_dirs:
            lines.append(f"  - look under: {', '.join(e.source_dirs)}")
        if e.evidence:
            lines.append(f"  - evidence: {', '.join(e.evidence)}")
    return "\n".join(lines)


def checklist_as_dicts(entries: list[ChecklistEntry]) -> list[dict[str, object]]:
    """Serialize checklist entries to plain dicts (for task params / persistence).

    Args:
        entries: The applicable checklist entries.

    Returns:
        A JSON-serializable list of dicts mirroring the dataclass fields.
    """
    out: list[dict[str, object]] = []
    for e in entries:
        out.append(
            {
                "id": e.id,
                "applies_when": dict(e.applies_when),
                "detect": e.detect,
                "consequence": e.consequence,
                "bridge": e.bridge,
                "domain_hint": e.domain_hint,
                "source_dirs": list(e.source_dirs),
                "evidence": list(e.evidence),
            }
        )
    return out


def filter_entries_for_model(entries: list[ChecklistEntry], model_info: dict) -> list[ChecklistEntry]:
    """Filter checklist entries based on model metadata.

    Currently gates ``rocm.moe.shared_expert_fusion`` on
    ``model_info["has_shared_expert"]`` so the entry does not appear for
    ROCm + MXFP8 runs whose model has no always-on shared expert.

    Args:
        entries: Candidate entries (typically from :func:`entries_for`).
        model_info: The session ``model_info`` dict (may be empty).

    Returns:
        Filtered list; entries not requiring model-level gating pass through
        unchanged.
    """
    if model_info.get("has_shared_expert"):
        return list(entries)
    return [e for e in entries if e.id != "rocm.moe.shared_expert_fusion"]


__all__ = [
    "ChecklistEntry",
    "entries_for",
    "filter_entries_for_model",
    "source_hint_directories_for",
    "render_checklist_for_prompt",
    "checklist_as_dicts",
]
