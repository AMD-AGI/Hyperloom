# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn raw bring-up streams into one typed :class:`BootObservation`.

The classifier is pure: everything it reads is handed to it. It borrows the
failure vocabulary from the enablement rule table and maps each rule onto the
boot ladder, so which wall was hit and how far the boot got are answered by one
pass over the same text. The server child's own log is read before the wrapper's
streams, which report a child's death rather than the cause of it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from hyperloom.agents.framework import enablement as rules
from hyperloom.common.bringup import (
    BootObservation,
    Excerpt,
    LadderStage,
    TerminalFrame,
    failure_digest,
    normalise_file_rel,
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
_EXC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt))\b", re.MULTILINE)


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
    """Scan ``text`` for milestone markers.

    Returns the deepest milestone witnessed (``None`` when none was) and, per
    milestone, the marker that witnessed it.
    """
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


def _terminal_frame(text: str, roots: Sequence[str]) -> TerminalFrame | None:
    """Extract the innermost traceback frame and its exception type.

    Returns ``None`` when ``text`` carries neither a traceback frame nor an
    exception line. ``roots`` are the pinned roots paths are relativised to.
    """
    frames = _TB_FRAME.findall(text)
    exc_matches = _EXC_LINE.findall(text)
    exc_type = exc_matches[-1] if exc_matches else ""
    if not frames:
        return TerminalFrame(exc_type=exc_type) if exc_type else None
    path, line, _func = frames[-1]
    file_rel = normalise_file_rel(path, roots)
    module = file_rel
    if module.endswith(".py"):
        module = module[: -len(".py")]
    if module.endswith("/__init__"):
        module = module[: -len("/__init__")]
    module = module.replace("/", ".").lstrip(".")
    return TerminalFrame(exc_type=exc_type, module=module, file_rel=file_rel, line=int(line))


def _anchor_for(text: str, signature: rules.FailureSignature) -> int:
    """Return the character offset the excerpt window should be anchored at.

    Prefers the matched rule text, then the innermost traceback frame, then the
    end of the stream. An unmatched signature's own excerpt is not used: it is
    cut end-relative, so its window shifts whenever the log grows.
    """
    head = signature.raw_excerpt.strip()[:40] if signature.is_actionable else ""
    if head:
        # ``raw_excerpt`` is whitespace-collapsed; match it back with a
        # whitespace-flexible pattern so the anchor lands on the real line.
        pattern = r"\s+".join(re.escape(tok) for tok in head.split())
        found = re.search(pattern, text)
        if found is not None:
            return found.start()
    frames = list(_TB_FRAME.finditer(text))
    if frames:
        return frames[-1].start()
    return len(text)


def _classified_streams(
    server_log: str,
    wrapper_stderr: str,
    wrapper_stdout: str,
) -> tuple[tuple[str, str, rules.FailureSignature], ...]:
    """Return ``(stream_name, text, signature)`` per non-empty stream.

    Server log first, then wrapper stderr, then wrapper stdout.
    """
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

    Pure: no filesystem, process or network access. An attempt that matches no
    rule is still fully described -- its terminal frame and excerpt give it a
    stable :func:`~hyperloom.common.bringup.failure_digest`.

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
    frame = _terminal_frame(text, roots)
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

    excerpt: Excerpt = render_excerpt(
        text,
        anchor=_anchor_for(text, signature),
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
