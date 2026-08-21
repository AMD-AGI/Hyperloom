###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Model review of an already-produced source-resolution artifact.

The deterministic tiers fail in a way that a fallback cannot catch: they do not
come up empty, they come up *confidently wrong*. Measured over historical
sessions, only 59% of verifiable resolutions actually mention the kernel they
claim to define, and ``aten::fill_`` alone has been resolved to four unrelated
business files -- every one of them a real, existing, root-resident source file
that passes every mechanical check. A tier gated on "source_file is empty"
never sees any of those.

So this tier reviews *every* entry rather than only the blanks, and may rewrite
a location the deterministic tiers already filled in.

Two properties keep the added freedom from becoming a new failure mode:

* **Nothing is taken on faith.** A rewritten path must exist on disk or sit
  under a known framework root (:func:`path_is_acceptable`). This does not
  check correctness -- it stops an invented path from being written.
* **Nothing is destroyed.** Every revision keeps ``previous_source_file`` and
  ``previous_method``, so a bad review is auditable and reversible.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable

from _llm_source_context import launcher_stack
from _llm_source_fallback import (
    _complete,
    _preview,
    _resolve_model,
    _resolve_provider,
    _safe_exception_label,
    llm_source_audit,
    source_preview_authorised,
)

try:
    from hyperloom.common import kernel_source_contract as _KSC
except ImportError:  # pragma: no cover - standalone invocation
    _KSC = None  # type: ignore[assignment]

#: Reviewing a long tail of sub-percent kernels costs tokens and changes
#: nothing anyone will act on.
_DEFAULT_MIN_GPU_PCT = 1.0

#: Cap on entries per request, so one call stays within a sane context.
_DEFAULT_MAX_ENTRIES = 40

_DEFAULT_TIMEOUT_SEC = 180.0

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

_ACTION_KEEP = "keep"
_ACTION_REWRITE = "rewrite"
_ACTION_UNRESOLVE = "unresolve"
_ACTIONS = frozenset({_ACTION_KEEP, _ACTION_REWRITE, _ACTION_UNRESOLVE})

_PROMPT_HEADER = """\
You are auditing an automated mapping from GPU kernel symbols to the source
file that defines each kernel. The mapping was produced by heuristics that are
known to fail in a specific way: when a kernel has no source of its own (a
PyTorch built-in such as aten::fill_, or a bare launch API), the heuristic
often attributes it to whichever business file happened to call it. Such an
entry looks perfectly plausible -- the path exists and holds real code -- but
the file does not define that kernel.

For each entry decide one of:
  keep       the file plausibly defines this kernel
  rewrite    the file is wrong and you know a better path
  unresolve  this kernel has no single defining source file, or the current
             path is wrong and you do not know the right one

Prefer "unresolve" over a guess. A wrong path costs an entire optimization
attempt; an empty one just falls through.

Reply with JSON only:
{"revisions": [{"kernel_id": "...", "action": "keep|rewrite|unresolve",
                "source_file": "...", "reason": "..."}]}
"source_file" is required for "rewrite" and ignored otherwise. Include every
entry you were given.
"""


def _gpu_pct(entry: dict[str, Any]) -> float:
    """GPU share of one entry, treating an unreadable value as zero.

    Ranking and the floor both read this. Letting a malformed value raise would
    escape into the caller's blanket handler and switch the whole tier off for
    the run, so one bad row would cost every other row its review.
    """
    try:
        return float(entry.get("gpu_pct") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _entry_block(
    entry: dict[str, Any],
    with_preview: bool,
    framework_roots: tuple[str, ...],
) -> str:
    """Render one entry, with a head of its current file when readable."""
    src = str(entry.get("source_file") or "")
    lines = [
        f"kernel_id: {entry.get('kernel_id')}",
        f"symbol: {entry.get('name')}",
        f"gpu_pct: {entry.get('gpu_pct')}",
        f"current_source_file: {src or '(unresolved)'}",
        f"resolved_by: {entry.get('method')}",
    ]
    stack = launcher_stack(entry)
    if stack:
        # The call site is often the only thing separating a kernel from the
        # business file that merely calls it.
        lines.append("launcher_stack:\n" + "\n".join(f"  {f}" for f in stack))
    bare = _KSC.strip_line_suffix(src) if _KSC else src
    canonical = _KSC.canonical_source_path(bare, framework_roots) if with_preview and bare and _KSC else ""
    if canonical:
        lines.append(f"file_head:\n```\n{_preview(canonical)}\n```")
    return "\n".join(lines)


def build_review_prompt(
    entries: list[dict[str, Any]],
    *,
    with_preview: bool | None = None,
    context_block: str = "",
    framework_roots: tuple[str, ...] = (),
) -> str:
    """Render the review request for ``entries``.

    ``with_preview`` defaults to the operator's egress decision (see
    :func:`source_preview_authorised`) rather than to True, so repository source
    is never shipped to the provider by omission.
    """
    if with_preview is None:
        with_preview = source_preview_authorised()
    head = _PROMPT_HEADER
    if context_block:
        head = f"{head}\n{context_block}\n"
    blocks = [_entry_block(e, with_preview, framework_roots) for e in entries]
    return head + "\nEntries:\n\n" + "\n\n---\n\n".join(blocks)


def parse_revisions(text: str) -> tuple[bool, list[dict[str, Any]], str]:
    """Extract ``(parsed, revisions, error)`` from a model reply.

    ``parsed`` distinguishes an unreadable reply from a reply that revised
    nothing; conflating them would report a broken call as a clean audit.
    """
    if not isinstance(text, str):
        return False, [], "reply is not text"
    match = _JSON_BLOCK_RE.search(text or "")
    if not match:
        return False, [], "no JSON object in reply"
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return False, [], "unparseable JSON"
    if not isinstance(payload, dict):
        return False, [], "JSON payload is not an object"
    revisions = payload.get("revisions")
    if not isinstance(revisions, list):
        return False, [], "payload has no 'revisions' list"
    out = [r for r in revisions if isinstance(r, dict)]
    return True, out, ""


def _apply_revision(
    entry: dict[str, Any],
    revision: dict[str, Any],
    roots: tuple[str, ...],
) -> str:
    """Apply one revision in place; return a note describing what happened."""
    action = str(revision.get("action") or "").strip().lower()
    reason = str(revision.get("reason") or "").strip()
    if action not in _ACTIONS:
        return f"{entry.get('kernel_id')}: ignored unknown action {action!r}"
    if action == _ACTION_KEEP:
        return ""
    previous_file = str(entry.get("source_file") or "")
    previous_method = str(entry.get("method") or "")

    if action == _ACTION_UNRESOLVE:
        if not previous_file:
            return ""
        entry["previous_source_file"] = previous_file
        entry["previous_method"] = previous_method
        entry["source_file"] = ""
        entry["source_line"] = None
        entry["source_function"] = ""
        entry["method"] = _KSC.METHOD_UNRESOLVED if _KSC else "unresolved"
        entry["reason"] = f"llm_review unresolved: {reason or 'no defining source'}"
        return f"{entry.get('kernel_id')}: unresolved (was {previous_file})"

    picked_raw = str(revision.get("source_file") or "").strip()
    if not picked_raw:
        return f"{entry.get('kernel_id')}: rewrite without a path, ignored"
    if _KSC is None:
        # The mechanical floor lives in the contract module. Without it a
        # rewrite cannot be verified at all, and writing an unverifiable path is
        # the failure this tier exists to prevent -- so an unusable guard denies
        # the rewrite instead of waving it through.
        return f"{entry.get('kernel_id')}: rejected rewrite, path contract unavailable"
    picked, source_line, source_function = _KSC.split_line_suffix(picked_raw)
    previous_bare = _KSC.strip_line_suffix(previous_file)
    if picked == previous_bare:
        return ""
    # The mechanical floor: a rewrite may not conjure a location. This is not a
    # correctness check, only a guard against invention.
    canonical = _KSC.canonical_source_path(picked, roots)
    if not canonical:
        return f"{entry.get('kernel_id')}: rejected unverifiable path {picked_raw!r}"
    if canonical == previous_bare:
        return ""
    entry["previous_source_file"] = previous_file
    entry["previous_method"] = previous_method
    entry["source_file"] = canonical
    entry["source_line"] = source_line
    entry["source_function"] = source_function
    entry["method"] = _KSC.METHOD_LLM
    entry["reason"] = f"llm_review rewrote: {reason or 'no reason given'}"
    return f"{entry.get('kernel_id')}: {previous_file or '(none)'} -> {canonical}"


def review_resolution_document(
    doc: dict[str, Any],
    *,
    framework_roots: tuple[str, ...] = (),
    min_gpu_pct: float = _DEFAULT_MIN_GPU_PCT,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    model: str | None = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    context_block: str = "",
    log: Callable[[str], None] | None = None,
    complete: Callable[[str, str, float], str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Review and possibly revise the resolution table in ``doc``.

    Args:
        doc: A source-resolution document; revised in place and returned.
        framework_roots: Roots a rewritten path may live under when it does not
            exist on this host.
        min_gpu_pct: Entries below this GPU share are not worth a review.
        max_entries: Hardest-hitting N entries reviewed in one call.
        model: Chat model; defaults to ``$HYPERLOOM_LLM_SOURCE_MODEL``, then
            the selected provider's model setting.
        timeout_sec: Per-call ceiling; there is no retry.
        log: Optional ``callable(str)`` for diagnostics.
        complete: Injection point for the completion call (tests).

    Returns:
        ``(doc, notes)``; ``notes`` records every applied and rejected revision.
    """

    def _say(message: str) -> None:
        if callable(log):
            log(f"llm_source_review: {message}")

    entries = doc.get("entries") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not entries:
        return doc, ["no entries to review"]

    reviewable_items = [
        (index, entry)
        for index, entry in enumerate(entries)
        if isinstance(entry, dict) and _gpu_pct(entry) >= min_gpu_pct
    ]
    reviewable_items.sort(key=lambda item: -_gpu_pct(item[1]))
    reviewable_items = reviewable_items[: max(1, int(max_entries))]
    reviewable = [entry for _, entry in reviewable_items]
    if not reviewable:
        return doc, [f"no entry at or above {min_gpu_pct}% GPU share"]

    sent_ids = [str(entry.get("kernel_id") or "") for entry in reviewable]
    duplicate_sent = sorted(kernel_id for kernel_id in set(sent_ids) if sent_ids.count(kernel_id) > 1)
    if not all(sent_ids) or duplicate_sent:
        note = "entries sent for review have missing or duplicate kernel_id values" + (
            f": {duplicate_sent}" if duplicate_sent else ""
        )
        return doc, [note]
    try:
        provider = _resolve_provider() if complete is None else ""
        chosen_model = _resolve_model(model or "", provider)
    except RuntimeError as exc:
        audit = llm_source_audit(model=model or "")
        audit["outcome"] = "configuration_error"
        doc.setdefault("llm_audit", {})["review"] = audit
        detail = _safe_exception_label(exc)
        _say(f"configuration failed: {detail}")
        return doc, [f"llm configuration failed: {detail}"]
    if not chosen_model:
        audit = llm_source_audit(model=model or "")
        audit["outcome"] = "configuration_error"
        doc.setdefault("llm_audit", {})["review"] = audit
        _say("no source-resolution model configured")
        return doc, ["no model configured"]

    audit = llm_source_audit(model=chosen_model)
    audit["outcome"] = "requested"
    doc.setdefault("llm_audit", {})["review"] = audit
    caller = complete or _complete
    try:
        reply = caller(
            build_review_prompt(
                reviewable,
                context_block=context_block,
                framework_roots=framework_roots,
            ),
            chosen_model,
            timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001 - advisory tier, never fatal
        audit["outcome"] = "call_error"
        detail = _safe_exception_label(exc)
        _say(f"call failed: {detail}")
        return doc, [f"llm call failed: {detail}"]

    try:
        parsed, revisions, error = parse_revisions(reply)
    except Exception as exc:  # noqa: BLE001 - malformed replies remain advisory
        detail = _safe_exception_label(exc)
        audit["outcome"] = "unusable_reply"
        _say(f"unusable reply: {detail}")
        return doc, [f"unusable reply: {detail}"]
    if not parsed:
        audit["outcome"] = "unusable_reply"
        _say(f"unusable reply: {error}")
        return doc, [f"unusable reply: {error}"]

    received_ids = [str(revision.get("kernel_id") or "") for revision in revisions]
    duplicate_ids = sorted(kernel_id for kernel_id in set(received_ids) if received_ids.count(kernel_id) > 1)
    missing_ids = sorted(set(sent_ids) - set(received_ids))
    extra_ids = sorted(set(received_ids) - set(sent_ids))
    if duplicate_ids or missing_ids or extra_ids:
        details = []
        if missing_ids:
            details.append(f"missing={missing_ids[:8]}")
        if duplicate_ids:
            details.append(f"duplicate={duplicate_ids[:8]}")
        if extra_ids:
            details.append(f"extra/unknown kernel_id={extra_ids[:8]}")
        note = "revision set does not match entries sent: " + "; ".join(details)
        audit["outcome"] = "protocol_error"
        doc["reviewed_by"] = "llm_source_review"
        doc["review_notes"] = [note]
        _say(note)
        return doc, [note]

    notes: list[str] = []
    try:
        staged_entries = copy.deepcopy(entries)
        staged_by_id = {kernel_id: staged_entries[index] for kernel_id, (index, _) in zip(sent_ids, reviewable_items)}
        for revision in revisions:
            # The batch check above already rejected any id that was not sent,
            # so every revision still here names a staged entry.
            entry = staged_by_id[str(revision.get("kernel_id") or "")]
            note = _apply_revision(entry, revision, framework_roots)
            if note:
                notes.append(note)
    except Exception as exc:  # noqa: BLE001 - discard the entire staged batch
        detail = _safe_exception_label(exc)
        note = f"revision validation failed: {detail}"
        audit["outcome"] = "validation_error"
        _say(note)
        return doc, [note]

    doc["entries"] = staged_entries
    doc["reviewed_by"] = "llm_source_review"
    doc["review_notes"] = notes
    audit["outcome"] = "completed"
    _say(f"reviewed {len(reviewable)} entr(ies), {len(notes)} change(s)")
    return doc, notes
