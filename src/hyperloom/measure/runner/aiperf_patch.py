# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The replay client of an AgentX test keeps the output token count of a reply it found nothing in.

aiperf turns a reply that arrived with a 2xx status and held nothing it could use (usage chunks, empty deltas and the
``[DONE]`` marker only) into an error record, and keeps the input token count of that record alone: the server's output
token count, which the parser has just read from the usage chunk, is dropped, and only a metric flagged error-only is
computed for an error record. A round's export therefore cannot tell a server that produced nothing from one that
produced tokens the client could not use (``measure.agentx_requests``).

The InferenceX commit Hyperloom pins carries the client as the git submodule ``utils/aiperf`` and installs it editable
into an environment made for each job, so the source of that submodule is what a round runs. :func:`patch_replay_client`
edits two files of it: the parser keeps the token counts of the invalid record whole, and a new error-only metric
(``error_osl``, the output length as ``output_sequence_length`` is) puts the count on the record's export line. A record
the client counted successful, and every other field of a failed one, are what they were.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from hyperloom.common.outside_facts import (
    AIPERF_ERROR_OUTPUT_LENGTH_METRIC,
    AIPERF_OUTPUT_LENGTH_FILE,
    AIPERF_PARSER_FILE,
    AIPERF_SUBMODULE,
)

__all__ = ["ReplayClientPatchError", "patch_replay_client"]

_UTF8 = "utf-8"
_SUBMODULE_TIMEOUT_SEC = 300
_PROJECT_FILE = "pyproject.toml"
_ERROR_DETAIL_CHARS = 300

#: The branch of the parser that returns a parsed record found invalid as an error record, as the pinned client has it,
#: and what replaces it: the record's token counts, which hold the server's output counts, are kept whole.
_PARSER_OLD = """\
                    # Parsed record was invalid, return as error record
                    return ParsedResponseRecord(
                        request=record.request,
                        responses=[],
                        token_counts=TokenCounts(
                            input=record.token_counts.input
                            if record.token_counts
                            else None
                        ),
"""
_PARSER_NEW = """\
                    # Parsed record was invalid, return as error record. Its token
                    # counts are kept whole: the server's output count tells an
                    # empty reply from a short one (see error_osl).
                    return ParsedResponseRecord(
                        request=record.request,
                        responses=[],
                        token_counts=record.token_counts or TokenCounts(),
"""
_OUTPUT_LENGTH_CLASS = "class OutputSequenceLengthMetric("
#: The metric that puts that count on the export line of a failed request, as ``ErrorInputSequenceLengthMetric`` does
#: the input count; appended to the module of the output length metric it extends.
_METRIC_NEW = f'''

class ErrorOutputSequenceLengthMetric(OutputSequenceLengthMetric):
    """
    Output Sequence Length of a request that resulted in an error, when the record
    carries one: a reply that arrived with nothing the client could use is an error
    record that keeps the token counts parsed from it. Absent for any other error.
    """

    tag = "{AIPERF_ERROR_OUTPUT_LENGTH_METRIC}"
    header = "Error Output Sequence Length"
    short_header = "Error OSL"
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.ERROR_ONLY
    )
    console_group = MetricConsoleGroup.NONE
'''
_METRIC_MARK = f'tag = "{AIPERF_ERROR_OUTPUT_LENGTH_METRIC}"'
_SUBMODULE_DECLARED = re.compile(rf"^\s*path\s*=\s*{re.escape(AIPERF_SUBMODULE)}\s*$", re.MULTILINE)


class ReplayClientPatchError(Exception):
    """The replay client's source is not there, or is not the text the patch anchors on."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding=_UTF8)
    except (OSError, UnicodeError) as exc:
        raise ReplayClientPatchError(f"{path.name} cannot be read ({type(exc).__name__})") from exc


def _write(path: Path, text: str) -> None:
    """Replace ``path`` whole or not at all, keeping its mode."""
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        scratch.write_text(text, encoding=_UTF8)
        shutil.copymode(path, scratch)
        os.replace(scratch, path)
    except OSError as exc:
        scratch.unlink(missing_ok=True)
        raise ReplayClientPatchError(f"{path.name} cannot be written ({type(exc).__name__})") from exc


def _parser_text(path: Path, text: str) -> str | None:
    """The parser with its invalid-record branch patched; ``None`` when it already is."""
    if _PARSER_NEW in text:
        return None
    if text.count(_PARSER_OLD) != 1:
        raise ReplayClientPatchError(
            f"{path.name} does not hold the branch the patch replaces exactly once, so the client is not the one pinned"
        )
    return text.replace(_PARSER_OLD, _PARSER_NEW)


def _metric_text(path: Path, text: str) -> str | None:
    """The output length metric module with the error-only metric added; ``None`` when it already is."""
    if _METRIC_MARK in text:
        return None
    if _OUTPUT_LENGTH_CLASS not in text:
        raise ReplayClientPatchError(f"{path.name} does not define the output length metric the patch extends")
    return text.rstrip("\n") + "\n" + _METRIC_NEW


def _init_submodule(root: Path) -> None:
    """Fetch the client's submodule the way Magpie does before a native AgentX run, so it can be patched first."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "submodule", "update", "--init", "--recursive", "--", AIPERF_SUBMODULE],
            capture_output=True,
            text=True,
            timeout=_SUBMODULE_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReplayClientPatchError(f"the {AIPERF_SUBMODULE} submodule could not be fetched ({exc})") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[-_ERROR_DETAIL_CHARS:]
        raise ReplayClientPatchError(f"the {AIPERF_SUBMODULE} submodule could not be fetched ({detail})")


def _client_dir(root: Path) -> Path:
    """The replay client's source below an InferenceX checkout, fetched when the checkout declares it and has none."""
    client = root / AIPERF_SUBMODULE
    if not (client / _PROJECT_FILE).is_file():
        declaration = root / ".gitmodules"
        if not declaration.is_file() or not _SUBMODULE_DECLARED.search(_read(declaration)):
            raise ReplayClientPatchError(f"the InferenceX checkout has no {AIPERF_SUBMODULE} submodule")
        _init_submodule(root)
    if not (client / _PROJECT_FILE).is_file():
        raise ReplayClientPatchError(f"the {AIPERF_SUBMODULE} submodule holds no {_PROJECT_FILE}")
    return client


def patch_replay_client(inferencex: Path | str) -> None:
    """Make the replay client of the InferenceX checkout at ``inferencex`` keep the output token count of a reply it found
    nothing in, and export it as ``error_osl``.

    Idempotent: a client that already has the patch is left as it is. Both files are checked before either is written, so
    a client whose text is not the pinned one is changed in neither. Raises :class:`ReplayClientPatchError` when the
    checkout has no client source (the submodule is fetched when the checkout declares one) or the source is not the
    text the patch anchors on.
    """
    client = _client_dir(Path(inferencex))
    parser, metric = client / AIPERF_PARSER_FILE, client / AIPERF_OUTPUT_LENGTH_FILE
    edits = (
        (metric, _metric_text(metric, _read(metric))),
        (parser, _parser_text(parser, _read(parser))),
    )
    for path, text in edits:
        if text is not None:
            _write(path, text)
