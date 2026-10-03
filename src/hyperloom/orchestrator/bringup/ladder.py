# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn raw bring-up streams into one typed :class:`BootObservation`."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from hyperloom.common import failure_signature as rules
from hyperloom.common.bringup import (
    BootObservation,
    Excerpt,
    LadderStage,
    TerminalFrame,
    failure_digest,
    normalise_file_rel,
    redact,
    render_excerpt,
)
from hyperloom.orchestrator.bringup.trees import TreeIdentity, tree_roots

#: Identifies observations produced here in downstream artifacts.
PRODUCER = "bringup.ladder"

#: Characters of context materialised around a match.
EXCERPT_WIDTH = 480

#: Stream names, in the order they are consulted.
SERVER_LOG = "server_log"
WRAPPER_STDERR = "wrapper_stderr"
WRAPPER_STDOUT = "wrapper_stdout"

#: Milestones witnessed by a line in the server's own log; case-insensitive
#: substrings. A milestone with no line a framework prints exactly once per boot
#: is absent rather than witnessed by one that also appears during serving.
_PROGRESS_MARKERS: tuple[tuple[LadderStage, str], ...] = (
    (LadderStage.CONFIG_VALIDATE, "server_args="),
    (LadderStage.CONFIG_VALIDATE, "initializing an llm engine"),
    (LadderStage.CONFIG_VALIDATE, "initializing a v1 llm engine"),
    (LadderStage.WEIGHTS_LOADING, "loading weights"),
    (LadderStage.WEIGHTS_LOADING, "loading safetensors checkpoint"),
    (LadderStage.WEIGHTS_LOADED, "loading weights took"),
    (LadderStage.WEIGHTS_LOADED, "load weight end"),
    (LadderStage.WEIGHTS_LOADED, "model loading took"),
    (LadderStage.ENGINE_INIT, "kv cache"),
    (LadderStage.ENGINE_INIT, "max_total_num_tokens"),
    (LadderStage.ENGINE_INIT, "gpu blocks:"),
    (LadderStage.GRAPH_CAPTURE, "capture cuda graph"),
    (LadderStage.GRAPH_CAPTURE, "graph capturing finished"),
    (LadderStage.HTTP_READY, "application startup complete"),
    (LadderStage.HTTP_READY, "uvicorn running on"),
    (LadderStage.HTTP_READY, "the server is fired up and ready to roll"),
)

#: The marker substrings alone, for a reader that must guarantee its text still
#: witnesses every milestone after it drops the bulk of a log.
PROGRESS_MARKER_SUBSTRINGS: tuple[str, ...] = tuple(marker for _stage, marker in _PROGRESS_MARKERS)

#: Where each enablement failure kind sits on the ladder. Kinds absent here
#: carry no fixed position and are placed relative to observed progress.
_KIND_STAGE: Mapping[str, LadderStage] = {
    rules.SERVE_FLAG: LadderStage.ARGV_PARSE,
    rules.IMPORT_ERROR: LadderStage.IMPORT,
    rules.MISSING_MODEL_ARCH: LadderStage.CONFIG_VALIDATE,
    rules.TOKENIZER_ERROR: LadderStage.CONFIG_VALIDATE,
    rules.UNSUPPORTED_DTYPE: LadderStage.CONFIG_VALIDATE,
    rules.MISSING_WEIGHT: LadderStage.WEIGHTS_LOADING,
    rules.SHAPE_MISMATCH: LadderStage.WEIGHTS_LOADING,
    rules.NOT_IMPLEMENTED: LadderStage.ENGINE_INIT,
    rules.CAPABILITY_DISABLED: LadderStage.ENGINE_INIT,
    rules.HIP_KERNEL_MISSING: LadderStage.ENGINE_INIT,
    rules.RESOURCE_CONSTRAINT: LadderStage.ENGINE_INIT,
    rules.KERNEL_RESOURCE_LIMIT: LadderStage.ENGINE_INIT,
    rules.ACCURACY_BELOW_FLOOR: LadderStage.ACCURACY_OK,
    rules.EVAL_GENERATION_PATHOLOGY: LadderStage.ACCURACY_OK,
    rules.EVAL_RUNTIME_FAILURE: LadderStage.ACCURACY_OK,
}

#: Kinds no source change can repair: the host lacks the resources asked for.
_ENV_FAULT_KINDS: frozenset[str] = frozenset({rules.RESOURCE_CONSTRAINT})

_LADDER: tuple[LadderStage, ...] = tuple(LadderStage)

# Traceback structure, not failure classification: keying a failure needs the
# frame's line number, which the enablement table's frame regex does not capture.
_TB_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_TB_HEADER = "Traceback (most recent call last):"
_EXC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt))\b")
# Multi-process servers tag every child line with its role and pid and route
# tracebacks through their logger, interleaving several processes' tracebacks.
_LOG_LINE = re.compile(
    r"^(?:\((?P<proc>[^()\s]+ pid=\d+)\)[ \t]?)?"
    r"(?:(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) [\d-]+ [\d:.,]+ \[[^\]\n]*\] ?)?"
    r"(?P<body>.*)$",
    re.MULTILINE,
)
# A traceback logged at these levels was caught, and the process carried on.
_CAUGHT_LEVELS = frozenset({"DEBUG", "INFO", "WARNING"})


@dataclass(frozen=True)
class _Traceback:
    """One printed traceback: its innermost frame, where that frame starts, its exception."""

    anchor: int
    frame: tuple[str, str] | None = None
    exc_type: str = ""
    exc_line: str = ""


def _next_stage(stage: LadderStage) -> LadderStage:
    """Return the milestone above ``stage``, or ``stage`` when it is the last."""
    index = _LADDER.index(stage)
    return _LADDER[min(index + 1, len(_LADDER) - 1)]


def _roots_of(trees: Sequence[TreeIdentity] | Sequence[str] | None) -> tuple[str, ...]:
    """Return the directories frames are normalised against, longest first."""
    if not trees:
        return ()
    if isinstance(trees[0], TreeIdentity):
        return tree_roots([t for t in trees if isinstance(t, TreeIdentity)])
    return tuple(sorted((str(t).rstrip("/") for t in trees if str(t).strip()), key=len, reverse=True))


def _witness_progress(text: str) -> tuple[LadderStage | None, dict[str, str]]:
    """Scan ``text`` for milestone markers."""
    lowered = text.lower()
    witness: dict[str, str] = {}
    deepest: LadderStage | None = None
    for stage, marker in _PROGRESS_MARKERS:
        if marker not in lowered:
            continue
        witness.setdefault(stage.name, marker)
        if deepest is None or stage > deepest:
            deepest = stage
    return deepest, witness


def _root_traceback(text: str) -> _Traceback | None:
    """Return the traceback the failure started from.

    Python prints a chained cause before the exception it caused, and a
    multi-process server prints a child's traceback before the parent's re-raise,
    so the root is the first traceback that names a frame, even one whose header
    was cut off. A bare exception line counts only when no traceback names a frame.
    Lines logged below ERROR are skipped.
    """
    open_by_proc: dict[str, _Traceback] = {}
    bare: _Traceback | None = None
    for line in _LOG_LINE.finditer(text):
        if line.group("level") in _CAUGHT_LEVELS:
            continue
        start, proc, body = line.start(), line.group("proc") or "", line.group("body").rstrip()
        if body.startswith(_TB_HEADER):
            open_by_proc[proc] = _Traceback(anchor=start)
            continue
        frame = _TB_FRAME.search(body)
        if frame is not None:
            block = open_by_proc.get(proc) or _Traceback(anchor=start)
            open_by_proc[proc] = replace(block, anchor=start, frame=(frame.group(1), frame.group(2)))
            continue
        exc = _EXC_LINE.match(body)
        if exc is None:
            continue
        block = open_by_proc.pop(proc, None)
        if block is not None and block.frame is not None:
            return replace(block, exc_type=exc.group(1), exc_line=body)
        bare = _Traceback(anchor=start, exc_type=exc.group(1), exc_line=body)
    unterminated = [block for block in open_by_proc.values() if block.frame is not None]
    if unterminated:
        return min(unterminated, key=lambda block: block.anchor)
    return bare


def _terminal_frame(root: _Traceback | None, roots: Sequence[str]) -> TerminalFrame | None:
    """Return the root traceback's innermost frame and its exception type."""
    if root is None:
        return None
    if root.frame is None:
        return TerminalFrame(exc_type=root.exc_type)
    path, line = root.frame
    file_rel = normalise_file_rel(path, roots)
    module = file_rel
    if module.endswith(".py"):
        module = module[: -len(".py")]
    if module.endswith("/__init__"):
        module = module[: -len("/__init__")]
    module = module.replace("/", ".").lstrip(".")
    return TerminalFrame(exc_type=root.exc_type, module=module, file_rel=file_rel, line=int(line))


def _failure_site(text: str, signature: rules.FailureSignature, root: _Traceback | None) -> tuple[int, str]:
    """Return where the excerpt is anchored and the line that states the failure.

    The anchor keeps the frames above a rule match in the excerpt, where the
    offending file is read from; the line alone keys which failure it is.
    """
    head = signature.raw_excerpt.strip()[:40] if signature.is_actionable else ""
    if head:
        # ``raw_excerpt`` is whitespace-collapsed; match it back with a
        # whitespace-flexible pattern so the anchor lands on the real line.
        pattern = r"\s+".join(re.escape(tok) for tok in head.split())
        found = re.search(pattern, text)
        if found is not None:
            return found.start(), text[found.start() :].partition("\n")[0].strip()
    if root is None:
        return len(text), ""
    return root.anchor, root.exc_line


def _classified_streams(
    server_log: str,
    wrapper_stderr: str,
    wrapper_stdout: str,
) -> tuple[tuple[str, str, rules.FailureSignature], ...]:
    """Return ``(stream_name, text, signature)`` per non-empty stream."""
    out: list[tuple[str, str, rules.FailureSignature]] = []
    for name, text in (
        (SERVER_LOG, server_log),
        (WRAPPER_STDERR, wrapper_stderr),
        (WRAPPER_STDOUT, wrapper_stdout),
    ):
        if text.strip():
            out.append((name, text, rules.classify_failure(text)))
    return tuple(out)


def classify(
    *,
    server_log: str,
    server_elapsed_sec: float,
    wrapper_stderr: str = "",
    wrapper_stdout: str = "",
    trees: Sequence[TreeIdentity] | Sequence[str] | None = None,
    session_root: str = "",
) -> BootObservation:
    """Classify one bring-up attempt into a :class:`BootObservation`.

    Args:
        server_log: Full text of the server child's log.
        server_elapsed_sec: Seconds from server start to this observation, on
            the server child's clock.
        wrapper_stderr: Launcher stderr, used only as fallback.
        wrapper_stdout: Launcher stdout, used only as fallback.
        trees: Pinned trees (or roots) frames are normalised against.
        session_root: Absolute session root redacted out of the excerpt.

    Returns:
        BootObservation: The typed observation. ``stage_failed`` is ``None``
        only when no stream carries a failure.
    """
    roots = _roots_of(trees)
    redact_roots = (session_root,) if session_root.strip() else ()
    streams = _classified_streams(server_log, wrapper_stderr, wrapper_stdout)

    witnessed: LadderStage | None = None
    progress_witness: dict[str, str] = {}
    for _name, text, _sig in streams:
        stage, witness = _witness_progress(text)
        for key, marker in witness.items():
            progress_witness.setdefault(key, marker)
        if stage is not None and (witnessed is None or stage > witnessed):
            witnessed = stage

    chosen: tuple[str, str, rules.FailureSignature] | None = None
    for entry in streams:
        if entry[2].is_actionable:
            chosen = entry
            break
    if chosen is None:
        # No rule fired anywhere: keep the highest-precedence stream with
        # content so the observation still carries a frame and an excerpt.
        chosen = streams[0] if streams else None

    floor = LadderStage.PROCESS_START if streams else LadderStage.ARGV_PARSE
    stage_reached = witnessed if witnessed is not None else floor

    if chosen is None:
        return BootObservation(
            producer=PRODUCER,
            stage_reached=stage_reached,
            stage_failed=None,
            progress_witness=progress_witness or None,
            server_elapsed_sec=server_elapsed_sec,
        )

    stream_name, text, signature = chosen
    root = _root_traceback(text)
    frame = _terminal_frame(root, roots)
    has_failure = signature.is_actionable or frame is not None

    if not has_failure:
        return BootObservation(
            producer=PRODUCER,
            stage_reached=stage_reached,
            stage_failed=None,
            progress_witness=progress_witness or None,
            evidence_ref=stream_name,
            server_elapsed_sec=server_elapsed_sec,
        )

    mapped = _KIND_STAGE.get(signature.kind)
    if mapped is None:
        # Unplaced by the vocabulary: the first milestone not witnessed.
        mapped = _next_stage(witnessed) if witnessed is not None else _next_stage(floor)
    elif witnessed is not None and mapped <= witnessed:
        # A witnessed milestone completed, so the failure is above it however
        # the rule is normally placed.
        mapped = _next_stage(witnessed)

    anchor, failure_line = _failure_site(text, signature, root)
    excerpt: Excerpt = render_excerpt(
        text,
        anchor=anchor,
        width=EXCERPT_WIDTH,
        stream=stream_name,
        redact_roots=redact_roots,
    )

    return BootObservation(
        producer=PRODUCER,
        stage_reached=min(stage_reached, mapped),
        stage_failed=mapped,
        progress_witness=progress_witness or None,
        terminal_frame=frame,
        matched_marker=signature.kind if signature.is_actionable else "",
        failure_line=redact(failure_line, roots=redact_roots),
        excerpt=excerpt,
        evidence_ref=stream_name,
        server_elapsed_sec=server_elapsed_sec,
        env_fault=(signature.kind if signature.kind in _ENV_FAULT_KINDS else None),
    )


def observation_summary(observation: BootObservation) -> dict[str, Any]:
    """Return the observation as a flat, JSON-safe record.

    Args:
        observation: The observation to flatten.

    Returns:
        dict[str, Any]: :meth:`BootObservation.to_dict` output plus the failure
        digest under ``failure_digest``, empty when nothing failed.
    """
    payload = observation.to_dict()
    payload["failure_digest"] = failure_digest(observation) if observation.stage_failed is not None else ""
    return payload


__all__ = [
    "PRODUCER",
    "SERVER_LOG",
    "WRAPPER_STDERR",
    "WRAPPER_STDOUT",
    "classify",
    "observation_summary",
]
