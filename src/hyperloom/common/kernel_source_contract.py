# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""On-disk audit view of hot-kernel source resolution.

Source resolution used to exist only as a scatter of fields mutated onto
candidate rows by a dozen functions, with no single place to read "which file
did we decide this kernel lives in, and how sure are we". This module defines a
standalone, versioned artifact for exactly that question.

**Scope: this is an audit view, not the stage contract.** The contract between
analysis and optimization remains ``kernel_candidates.json`` -- it alone carries
what dispatch needs (shapes, ``reusable_native_kernel``, ``skip_reason``,
``source_type``, recommended backends, task groups, dispatchability). This
artifact answers one narrow question and is deliberately not sufficient to
decide what to optimize.

The review tier revises entries here, but a revision only reaches the pipeline
because ``apply_resolution_entries_to_candidates()`` folds it back onto the
candidates and re-runs classification. Editing this file after the fact changes
nothing downstream.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

#: Bump the major on any field removal or meaning change; consumers gate on it.
#: 1.1.0 adds ``reason_class`` to every entry (additive; readers of 1.0.0 are
#: unaffected).
SOURCE_RESOLUTION_SCHEMA_VERSION = "1.1.0"

#: Canonical artifact name, relative to the analysis run directory. Mirrored by
#: ``tracelens_analysis._SOURCE_RESOLUTION_NAME`` for the standalone path.
SOURCE_RESOLUTION_FILENAME = "kernel_source_resolution.json"

#: How a location was decided, best evidence first. ``llm_review`` outranks the
#: deterministic tiers only because it runs last and sees their output; it is
#: not more trusted, which is why the tier it overrode is kept in
#: ``previous_method``.
#:
#: ``active_finder`` runs *before* the curated map: it demangles the device
#: kernel symbol and looks it up in the currently-installed framework source
#: tree, so it self-heals across file moves/renames and version drift. The
#: curated map remains the fallback when the symbol is absent from the live
#: index. ``symbol_index`` is the same resolution surfaced by the bypass route.
METHOD_ACTIVE_FINDER = "active_finder"
METHOD_SYMBOL_INDEX = "symbol_index"
METHOD_CURATED = "op_to_source"
METHOD_TRACE = "trace_python_stack"
METHOD_GREP = "name_grep"
METHOD_LLM_FALLBACK = "llm_fallback"
METHOD_LLM = "llm_review"
METHOD_REJECTED = "rejected_non_path_sentinel"
METHOD_UNRESOLVED = "unresolved"

KNOWN_METHODS = frozenset(
    {
        METHOD_ACTIVE_FINDER,
        METHOD_SYMBOL_INDEX,
        METHOD_CURATED,
        METHOD_TRACE,
        METHOD_GREP,
        METHOD_LLM_FALLBACK,
        METHOD_LLM,
        METHOD_REJECTED,
        METHOD_UNRESOLVED,
    }
)

#: Why a kernel is not dispatchable, as a judgeable class rather than prose.
#: ``method`` says how a location was decided; this says why there is nothing to
#: dispatch. The split that matters to a consumer is between the classes below
#: that can never be optimized from source -- a consumer can skip those for free
#: -- and ``source_not_resolved``, which is the only one worth spending a search
#: budget on.
CLASS_RESOLVED = "resolved"
CLASS_LAUNCH_API_ONLY = "launch_api_only"
CLASS_VENDOR_BINARY = "vendor_binary"
CLASS_DISPATCH_SHIM = "dispatch_shim"
CLASS_NON_PATCHABLE_NAME = "non_patchable_name"
CLASS_NOT_REWRITABLE_VERDICT = "not_rewritable_verdict"
CLASS_SOURCE_NOT_RESOLVED = "source_not_resolved"
CLASS_UNKNOWN = "unknown"

KNOWN_REASON_CLASSES = frozenset(
    {
        CLASS_RESOLVED,
        CLASS_LAUNCH_API_ONLY,
        CLASS_VENDOR_BINARY,
        CLASS_DISPATCH_SHIM,
        CLASS_NON_PATCHABLE_NAME,
        CLASS_NOT_REWRITABLE_VERDICT,
        CLASS_SOURCE_NOT_RESOLVED,
        CLASS_UNKNOWN,
    }
)

#: Classes where no amount of searching produces an editable kernel body. A
#: consumer deciding where to spend a budget can drop these without measuring.
UNSALVAGEABLE_REASON_CLASSES = frozenset(
    {
        CLASS_LAUNCH_API_ONLY,
        CLASS_VENDOR_BINARY,
        CLASS_DISPATCH_SHIM,
        CLASS_NON_PATCHABLE_NAME,
        CLASS_NOT_REWRITABLE_VERDICT,
    }
)

#: Ordered longest-context-first: the first substring that matches a skip reason
#: decides the class, so a more specific phrase must precede a broader one.
_REASON_CLASS_MARKERS: tuple[tuple[str, str], ...] = (
    ("launch api, not a kernel", CLASS_LAUNCH_API_ONLY),
    ("source file not resolved", CLASS_SOURCE_NOT_RESOLVED),
    ("prebuilt assembly compute-core", CLASS_VENDOR_BINARY),
    ("vendor binary", CLASS_VENDOR_BINARY),
    ("vendor backend library", CLASS_VENDOR_BINARY),
    ("precompiled binary", CLASS_VENDOR_BINARY),
    ("vendor dispatch wrapper", CLASS_DISPATCH_SHIM),
    ("dispatch shim", CLASS_DISPATCH_SHIM),
    ("non-patchable kernel name marker", CLASS_NON_PATCHABLE_NAME),
)

#: Every entry carries these, so a consumer can rely on presence without
#: defaulting. Values may be empty; the keys may not be absent.
REQUIRED_ENTRY_KEYS = (
    "kernel_id",
    "name",
    "gpu_pct",
    "source_file",
    "method",
    "reason",
    "reason_class",
)


def classify_skip_reason(*, reusable: Any, skip_reason: Any, source_file: Any = "") -> str:
    """Reduce a candidate's routing verdict to one judgeable class.

    Reads only the two fields the gate itself produces, so this stays a pure
    function of the verdict rather than a second opinion on it.

    Args:
        reusable: The gate's ``reusable_native_kernel`` verdict.
        skip_reason: The gate's prose explanation, empty when reusable.
        source_file: Resolved path, used only to disambiguate an empty reason.

    Returns:
        One of :data:`KNOWN_REASON_CLASSES`.
    """
    if reusable is True:
        return CLASS_RESOLVED
    text = str(skip_reason or "").strip().lower()
    if not text:
        # No verdict text: an unresolved path is the one case we can still name.
        return CLASS_SOURCE_NOT_RESOLVED if not str(source_file or "").strip() else CLASS_UNKNOWN
    for marker, reason_class in _REASON_CLASS_MARKERS:
        if marker in text:
            return reason_class
    # The active finder's own verdict is prefixed by the gate and carries an
    # arbitrary tail, so it is matched last and by prefix.
    if text.startswith("source:"):
        return CLASS_NOT_REWRITABLE_VERDICT
    return CLASS_UNKNOWN


REQUIRED_DOCUMENT_KEYS = ("schema_version", "generated_by", "entries")


def make_entry(
    *,
    kernel_id: str,
    name: str,
    gpu_pct: float,
    source_file: str = "",
    source_line: int | None = None,
    source_function: str = "",
    method: str = METHOD_UNRESOLVED,
    confidence: float | None = None,
    reason: str = "",
    rejected_value: str = "",
    previous_source_file: str = "",
    previous_method: str = "",
    reason_class: str = "",
) -> dict[str, Any]:
    """Build one resolution entry with every required key present.

    Args:
        kernel_id: Stable-within-run candidate id (``k001``).
        name: Kernel symbol as the profiler reported it.
        gpu_pct: Share of GPU time, used to rank what is worth resolving.
        source_file: Resolved path, or ``""`` when unresolved.
        source_line: 1-based line when the tier produced one.
        source_function: Enclosing function when the tier produced one.
        method: One of :data:`KNOWN_METHODS`.
        confidence: 0..1 when the tier reports one; ``None`` for deterministic
            tiers, which are either right or silent.
        reason: Human-readable note -- why this path, or why none.
        rejected_value: Placeholder that was zeroed, kept for audit.
        previous_source_file: Location replaced by a model review.
        previous_method: Resolution tier replaced by a model review.

    Returns:
        The entry dict.
    """
    if isinstance(gpu_pct, bool) or not isinstance(gpu_pct, (int, float)) or not math.isfinite(float(gpu_pct)):
        safe_gpu_pct = 0.0
    else:
        safe_gpu_pct = float(gpu_pct)
    entry = {
        "kernel_id": str(kernel_id or ""),
        "name": str(name or ""),
        "gpu_pct": safe_gpu_pct,
        "source_file": str(source_file or ""),
        "source_line": source_line if isinstance(source_line, int) else None,
        "source_function": str(source_function or ""),
        "method": str(method or METHOD_UNRESOLVED),
        "confidence": confidence,
        "reason": str(reason or ""),
        "rejected_value": str(rejected_value or ""),
        # Derived from the caller's verdict when not supplied, so an entry is
        # never written without a class a consumer can act on.
        "reason_class": str(reason_class or "")
        or classify_skip_reason(
            reusable=bool(source_file),
            skip_reason=reason,
            source_file=source_file,
        ),
    }
    if previous_source_file:
        entry["previous_source_file"] = str(previous_source_file)
        entry["previous_method"] = str(previous_method or "")
    return entry


def summarize_resolution(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Count located kernels and group the rest by why they are undispatchable.

    A bare located/total count cannot say whether the unlocated kernels were
    worth chasing, so this also reports the GPU share behind them and splits it
    into what a search could still rescue and what nothing can.

    Args:
        entries: Source-resolution entries, as written to the artifact.

    Returns:
        ``total`` / ``located`` counts, per-class counts, the GPU share behind
        undispatchable kernels, and its ``recoverable`` / ``unsalvageable``
        split.
    """
    by_class: dict[str, int] = {}
    located = 0
    undispatchable_gpu_pct = 0.0
    recoverable = 0
    unsalvageable = 0
    rows = [entry for entry in entries if isinstance(entry, dict)]
    for entry in rows:
        reason_class = str(entry.get("reason_class") or CLASS_UNKNOWN)
        by_class[reason_class] = by_class.get(reason_class, 0) + 1
        if reason_class == CLASS_RESOLVED:
            located += 1
            continue
        gpu_pct = entry.get("gpu_pct")
        if isinstance(gpu_pct, (int, float)) and not isinstance(gpu_pct, bool) and math.isfinite(float(gpu_pct)):
            undispatchable_gpu_pct += float(gpu_pct)
        if reason_class in UNSALVAGEABLE_REASON_CLASSES:
            unsalvageable += 1
        else:
            recoverable += 1
    return {
        "total": len(rows),
        "located": located,
        "by_class": by_class,
        "undispatchable_gpu_pct": undispatchable_gpu_pct,
        "recoverable": recoverable,
        "unsalvageable": unsalvageable,
    }


def make_document(
    entries: list[dict[str, Any]],
    *,
    generated_by: str,
    model_name: str = "",
    framework: str = "",
) -> dict[str, Any]:
    """Wrap ``entries`` in the versioned envelope."""
    return {
        "schema_version": SOURCE_RESOLUTION_SCHEMA_VERSION,
        "generated_by": str(generated_by or ""),
        "model_name": str(model_name or ""),
        "framework": str(framework or ""),
        "entries": list(entries),
    }


def validate_document(doc: Any) -> list[str]:
    """Return a list of contract violations; empty means the document is valid.

    Reports every problem rather than raising on the first, so a producer test
    failure names all of them at once.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"document is {type(doc).__name__}, expected dict"]
    for key in REQUIRED_DOCUMENT_KEYS:
        if key not in doc:
            problems.append(f"document missing required key {key!r}")
    version = str(doc.get("schema_version") or "")
    if version and version.split(".")[0] != SOURCE_RESOLUTION_SCHEMA_VERSION.split(".")[0]:
        problems.append(f"schema_version {version!r} has a different major than {SOURCE_RESOLUTION_SCHEMA_VERSION!r}")
    entries = doc.get("entries")
    if not isinstance(entries, list):
        problems.append(f"entries is {type(entries).__name__}, expected list")
        return problems
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"entries[{i}] is {type(entry).__name__}, expected dict")
            continue
        for key in REQUIRED_ENTRY_KEYS:
            if key not in entry:
                problems.append(f"entries[{i}] missing required key {key!r}")
        kernel_id = str(entry.get("kernel_id") or "")
        if not kernel_id:
            problems.append(f"entries[{i}] has empty kernel_id")
        elif kernel_id in seen_ids:
            problems.append(f"entries[{i}] has duplicate kernel_id {kernel_id!r}")
        else:
            seen_ids.add(kernel_id)
        gpu_pct = entry.get("gpu_pct")
        if gpu_pct is not None:
            if isinstance(gpu_pct, bool) or not isinstance(gpu_pct, (int, float)) or not math.isfinite(float(gpu_pct)):
                problems.append(f"entries[{i}] has invalid gpu_pct {gpu_pct!r}; expected a finite number")
        source_line = entry.get("source_line")
        if source_line is not None and not isinstance(source_line, int):
            problems.append(f"entries[{i}] has invalid source_line {source_line!r}; expected int or null")
        method = str(entry.get("method") or "")
        if method and method not in KNOWN_METHODS:
            problems.append(f"entries[{i}] has unknown method {method!r}")
        reason_class = str(entry.get("reason_class") or "")
        if reason_class and reason_class not in KNOWN_REASON_CLASSES:
            problems.append(f"entries[{i}] has unknown reason_class {reason_class!r}")
        src = str(entry.get("source_file") or "")
        if src and method in {METHOD_UNRESOLVED, METHOD_REJECTED}:
            problems.append(f"entries[{i}] has a source_file but method is {method}")
        if not src and method not in {METHOD_UNRESOLVED, METHOD_REJECTED}:
            problems.append(f"entries[{i}] has method {method!r} but no source_file")
        confidence = entry.get("confidence")
        if confidence is not None:
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                problems.append(
                    f"entries[{i}] has invalid confidence {confidence!r}; expected a finite number in [0, 1]"
                )
    return problems


#: TraceLens reports a call site as "path.py(247): fn_name"; the line and
#: function ride along in the same string.
_LINE_SUFFIX_RE = re.compile(r"^(?P<path>.+?)\((?P<line>\d+)\)\s*(?::\s*(?P<function>.*))?$")


def split_line_suffix(path: str) -> tuple[str, int | None, str]:
    """Split a TraceLens call-site string into path, line and function."""
    text = (path or "").strip()
    if not text:
        return "", None, ""
    match = _LINE_SUFFIX_RE.match(text)
    if match is None:
        return text, None, ""
    return (
        match.group("path").strip(),
        int(match.group("line")),
        str(match.group("function") or "").strip(),
    )


def strip_line_suffix(path: str) -> str:
    """Return the bare file path from a possibly line-annotated one.

    ``/repo/moe.py(247): _grouped_gemm`` -> ``/repo/moe.py``. Measured on a real
    session, 29 of 36 resolved entries carried this suffix, so an existence
    check that skips this step rejects paths that are perfectly real.
    """
    return split_line_suffix(path)[0]


def canonical_source_path(path: str, roots: tuple[str, ...]) -> str:
    """Return the validated canonical target for ``path``, or ``""``.

    The file must exist on this host **and** sit under a known framework root.

    Requiring existence is the point. "Under a root" alone is trivially
    satisfiable by a plausible-looking invention -- a model can emit
    ``<root>/python/sglang/srt/layers/attention/does_not_exist.py`` and pass a
    root-prefix check, which would hand the backend a fabricated file and
    reintroduce exactly the wrong-source failure this pipeline exists to
    prevent. A deterministic tier may legitimately resolve to a path that is
    absent here (the analysis host often lacks the serving container's
    filesystem), and such a path is kept with a ``source_file_missing_on_disk``
    breadcrumb -- but that latitude is not extended to a generated rewrite.
    """
    text = strip_line_suffix(path)
    if not text or not os.path.isfile(text):
        return ""
    real = Path(os.path.realpath(text))
    for root in roots:
        if not root:
            continue
        resolved_root = Path(os.path.realpath(str(root)))
        try:
            real.relative_to(resolved_root)
            return str(real)
        except ValueError:
            continue
    return ""


def path_is_acceptable(path: str, roots: tuple[str, ...]) -> bool:
    """Whether a rewriting tier may write ``path`` as a resolved location."""
    return bool(canonical_source_path(path, roots))
