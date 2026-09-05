# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed boot observations for a served-model bring-up attempt.

A bring-up advances along a fixed ladder of observable milestones or stops at
one; this module carries that observation as data. :func:`failure_digest` hashes
only the shape of a failure -- never a path, timestamp, pid, or other per-run
identifier -- so the same failure on two hosts collapses to one dedup key.

Standard library only apart from :mod:`hyperloom.common.env_safety`, which is
itself stdlib-only: imported by both the launcher and the orchestrator.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

#: Replaces an absolute session path in an excerpt.
SESSION_PLACEHOLDER = "<session>"

#: Replaces an absolute path inside a digest message template.
PATH_PLACEHOLDER = "<path>"

#: Prefix for a frame whose file lies outside every pinned tree.
EXTERNAL_PREFIX = "<external>"


class LadderStage(IntEnum):
    """Ordered milestones a server bring-up passes through.

    Values are spaced so a milestone can be inserted without renumbering its
    neighbours; ordering comparisons are how progress is judged.
    """

    ARGV_PARSE = 10
    PROCESS_START = 20
    IMPORT = 30
    CONFIG_VALIDATE = 40
    WEIGHTS_LOADING = 50
    WEIGHTS_LOADED = 60
    ENGINE_INIT = 70
    GRAPH_CAPTURE = 80
    HTTP_READY = 90
    GENERATES = 100
    ACCURACY_OK = 110

    @classmethod
    def from_name(cls, name: str) -> "LadderStage":
        """Return the stage named ``name``.

        Args:
            name: A member name, case-insensitive.

        Returns:
            LadderStage: The matching stage.

        Raises:
            ValueError: When ``name`` names no stage.
        """
        key = name.strip().upper()
        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"unknown ladder stage: {name!r}") from None


def _stage_name(stage: LadderStage | None) -> str:
    """Return a stage's member name, or ``""`` for ``None``."""
    return stage.name if stage is not None else ""


def _stage_or_none(name: Any) -> LadderStage | None:
    """Parse an optional serialized stage name back into a stage."""
    if name in (None, ""):
        return None
    return LadderStage.from_name(str(name))


def redact(text: str, *, roots: Sequence[str] = ()) -> str:
    """Replace absolute session paths and secret values in ``text``.

    Everything this module materialises is cut from a server log, a wrapper's
    stderr or a probe's own message, and each of those routinely carries the
    launch environment -- so an excerpt is a place secrets reach disk, the
    session package and an LLM prompt. It goes through the one rule set every
    other on-disk text in the tree uses
    (:func:`hyperloom.common.env_safety.redact_secret_values`) rather than a
    second, weaker one here.

    Args:
        text: Raw text to redact.
        roots: Absolute session roots to replace. Longer roots are replaced
            first so a nested root does not shadow its parent.

    Returns:
        str: ``text`` with every root replaced by :data:`SESSION_PLACEHOLDER`
        and every recognised secret value masked.
    """
    out = text
    for root in sorted({r.rstrip("/") for r in roots if r.strip()}, key=len, reverse=True):
        out = out.replace(root, SESSION_PLACEHOLDER)
    return redact_secret_values(out)


@dataclass(frozen=True)
class Excerpt:
    """A materialised, redacted snippet of a stream, with its origin.

    Attributes:
        text: The redacted snippet.
        stream: Name of the stream it came from (``"server_log"``,
            ``"wrapper_stderr"``, ...).
        byte_start: Offset of the snippet's first byte within that stream.
        byte_end: Offset one past the snippet's last byte.
    """

    text: str
    stream: str
    byte_start: int
    byte_end: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict of the four fields."""
        return {
            "text": self.text,
            "stream": self.stream,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Excerpt":
        """Rebuild an excerpt from :meth:`to_dict` output.

        Raises:
            KeyError: When a field :meth:`to_dict` writes is absent.
            ValueError: When an offset field does not parse as an integer.
        """
        return cls(
            text=str(raw["text"]),
            stream=str(raw["stream"]),
            byte_start=int(raw["byte_start"]),
            byte_end=int(raw["byte_end"]),
        )


#: Fraction of the window placed before the anchor line.
_ANCHOR_LEAD = 0.25


def render_excerpt(
    text: str,
    *,
    anchor: int,
    width: int,
    stream: str = "",
    redact_roots: Sequence[str] = (),
) -> Excerpt:
    """Materialise a window of ``text`` anchored at a match offset.

    The window is placed around ``anchor``, never at the end of ``text``, so it
    does not move as the log grows.

    Args:
        text: The full stream text.
        anchor: Character offset of the match to centre the window on.
        width: Window width in characters.
        stream: Name recorded on the excerpt.
        redact_roots: Session roots passed to :func:`redact`.

    Returns:
        Excerpt: The redacted window and the byte range it covers.
    """
    span = max(1, int(width))
    lead = int(span * _ANCHOR_LEAD)
    start = max(0, min(int(anchor), len(text)) - lead)
    end = min(len(text), start + span)
    return Excerpt(
        text=redact(text[start:end], roots=redact_roots).strip(),
        stream=stream,
        byte_start=len(text[:start].encode("utf-8", "replace")),
        byte_end=len(text[:end].encode("utf-8", "replace")),
    )


def normalise_file_rel(path: str, roots: Sequence[str]) -> str:
    """Return ``path`` relative to the longest pinned root that contains it.

    Purely textual: no filesystem access, no symlink resolution, no ``..``
    collapsing, so a root absent on this host still normalises identically.

    Args:
        path: An absolute or relative source path from a terminal frame.
        roots: Pinned tree roots to strip.

    Returns:
        str: The path relative to its containing root,
            ``"<external>/<basename>"`` when no root contains it, ``""`` for
            empty input.
    """
    p = path.strip()
    if not p:
        return ""
    best = ""
    for root in roots:
        prefix = root.strip().rstrip("/")
        if not prefix:
            continue
        prefix += "/"
        if p.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    if best:
        return p[len(best) :]
    return f"{EXTERNAL_PREFIX}/{p.rsplit('/', 1)[-1]}"


@dataclass(frozen=True)
class TerminalFrame:
    """The innermost source frame of a failure.

    Attributes:
        exc_type: Exception class name, e.g. ``"ValueError"``.
        module: Dotted module name the frame belongs to, or ``""``.
        file_rel: Frame file, already passed through
            :func:`normalise_file_rel`.
        line: 1-based line number, or ``0`` when unknown.
    """

    exc_type: str = ""
    module: str = ""
    file_rel: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict of the four fields."""
        return {
            "exc_type": self.exc_type,
            "module": self.module,
            "file_rel": self.file_rel,
            "line": self.line,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TerminalFrame":
        """Rebuild a frame from :meth:`to_dict` output.

        Raises:
            KeyError: When a field :meth:`to_dict` writes is absent.
            ValueError: When ``line`` does not parse as an integer.
        """
        return cls(
            exc_type=str(raw["exc_type"]),
            module=str(raw["module"]),
            file_rel=str(raw["file_rel"]),
            line=int(raw["line"]),
        )


@dataclass(frozen=True)
class BootObservation:
    """What one bring-up attempt was observed to do.

    Attributes:
        producer: What produced the observation (the classifier or probe name).
        stage_reached: Deepest milestone witnessed.
        stage_failed: Milestone the attempt stopped at, or ``None``.
        progress_witness: Milestone name to the evidence that witnessed it.
        terminal_frame: Innermost frame of the failure, when one was found.
        matched_marker: Identifier of the rule or marker that fired, or ``""``.
        excerpt: Materialised evidence for ``stage_failed``.
        evidence_ref: Pointer to the full artifact the excerpt was cut from.
        server_elapsed_sec: Seconds from server-process start to the
            observation, on the server child's own clock.
        env_fault: Set when the attempt failed for an environment reason no
            source change can repair.
    """

    producer: str
    stage_reached: LadderStage
    stage_failed: LadderStage | None = None
    progress_witness: Mapping[str, str] | None = None
    terminal_frame: TerminalFrame | None = None
    matched_marker: str = ""
    excerpt: Excerpt | None = None
    evidence_ref: str = ""
    server_elapsed_sec: float = 0.0
    env_fault: str | None = None

    @property
    def booted(self) -> bool:
        """True when nothing failed and the server answered requests."""
        return self.stage_failed is None and self.stage_reached >= LadderStage.HTTP_READY

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, stages written by name."""
        return {
            "producer": self.producer,
            "stage_reached": self.stage_reached.name,
            "stage_failed": _stage_name(self.stage_failed),
            "progress_witness": dict(self.progress_witness) if self.progress_witness is not None else None,
            "terminal_frame": self.terminal_frame.to_dict() if self.terminal_frame is not None else None,
            "matched_marker": self.matched_marker,
            "excerpt": self.excerpt.to_dict() if self.excerpt is not None else None,
            "evidence_ref": self.evidence_ref,
            "server_elapsed_sec": self.server_elapsed_sec,
            "env_fault": self.env_fault,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BootObservation":
        """Rebuild an observation from :meth:`to_dict` output.

        Raises:
            KeyError: When a field :meth:`to_dict` writes is absent.
            ValueError: When a stage field names no stage, or a numeric field
                does not parse.
        """
        witness = raw["progress_witness"]
        frame = raw["terminal_frame"]
        excerpt = raw["excerpt"]
        env_fault = raw["env_fault"]
        return cls(
            producer=str(raw["producer"]),
            stage_reached=LadderStage.from_name(str(raw["stage_reached"])),
            stage_failed=_stage_or_none(raw["stage_failed"]),
            progress_witness={str(k): str(v) for k, v in witness.items()} if witness is not None else None,
            terminal_frame=TerminalFrame.from_dict(frame) if frame is not None else None,
            matched_marker=str(raw["matched_marker"]),
            excerpt=Excerpt.from_dict(excerpt) if excerpt is not None else None,
            evidence_ref=str(raw["evidence_ref"]),
            server_elapsed_sec=float(raw["server_elapsed_sec"]),
            env_fault=str(env_fault) if env_fault is not None else None,
        )


# Applied before the digit mask: a path's own digits must not survive as `#`.
_ABS_PATH = re.compile(r"(?:/[A-Za-z0-9._+-]+){2,}")
_HEX = re.compile(r"0[xX][0-9a-fA-F]+")
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")

#: Wide enough for an error's distinguishing clause, narrow enough to drop a
#: trailing dump of shapes.
_TEMPLATE_WIDTH = 200


def _message_template(text: str) -> str:
    """Collapse whitespace and mask paths, hex addresses and digit runs, then
    lower-case and truncate, leaving only the part identical across runs."""
    body = _WS.sub(" ", text).strip()
    body = _ABS_PATH.sub(PATH_PLACEHOLDER, body)
    body = _HEX.sub("#", body)
    body = _DIGITS.sub("#", body)
    return body.lower()[:_TEMPLATE_WIDTH]


def failure_digest(observation: BootObservation) -> str:
    """Return a stable dedup key for the failure in ``observation``.

    The key covers the failed stage, the terminal frame's exception type, module
    and normalised file, and the masked message template, and nothing else.

    Args:
        observation: The observation to key.

    Returns:
        str: A 64-character lowercase sha256 hex digest.
    """
    frame = observation.terminal_frame
    excerpt = observation.excerpt
    parts = (
        _stage_name(observation.stage_failed),
        frame.exc_type if frame is not None else "",
        frame.module if frame is not None else "",
        frame.file_rel if frame is not None else "",
        _message_template(excerpt.text if excerpt is not None else ""),
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


__all__ = [
    "BootObservation",
    "Excerpt",
    "LadderStage",
    "TerminalFrame",
    "failure_digest",
    "normalise_file_rel",
    "redact",
    "render_excerpt",
]
