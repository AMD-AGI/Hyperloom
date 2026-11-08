# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The staged AgentX client keeps the output token count of a reply it found nothing in, exported as ``error_osl``.

aiperf, as agentx-harness ships it, turns a reply that arrived with a 2xx status and held nothing it could use (usage
chunks, empty deltas and the ``[DONE]`` marker only) into an error record and keeps only the input token count of that
record; and only a metric flagged error-only is computed for an error record. Its export therefore cannot tell a server
that produced nothing from one that produced tokens the client could not use.

:func:`patch_aiperf` edits two files of the installed client: the parser keeps the token counts of the invalid record
whole, and a new error-only metric (:data:`ERROR_OSL_METRIC`, the output length as ``output_sequence_length`` is) puts
the count on the record's export line. The owning distribution's ``RECORD`` is rewritten for both files, so the staging
manifest, which hashes every ``RECORD``, names the patched bytes. Staging installs the client, then patches it; a client
whose text is not the one the patch anchors on fails the staging.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path

#: The metric the patched client adds to the export line of a no-content failure.
ERROR_OSL_METRIC = "error_osl"
PARSER_FILE = "aiperf/records/inference_result_parser.py"
METRIC_FILE = "aiperf/metrics/types/output_sequence_length_metric.py"
#: The files the patch rewrites, relative to the client environment's site-packages.
PATCHED_FILES = (PARSER_FILE, METRIC_FILE)

_UTF8 = "utf-8"
#: The branch of the parser that returns a parsed record found invalid as an error record, and what replaces it.
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
#: Appended to the module of the output length metric it extends, as ``ErrorInputSequenceLengthMetric`` is to the input
#: one; the class registers itself with aiperf's metric registry when the module is imported.
_METRIC_NEW = f'''

class ErrorOutputSequenceLengthMetric(OutputSequenceLengthMetric):
    """
    Output Sequence Length of a request that resulted in an error, when the record
    carries one: a reply that arrived with nothing the client could use is an error
    record that keeps the token counts parsed from it. Absent for any other error.
    """

    tag = "{ERROR_OSL_METRIC}"
    header = "Error Output Sequence Length"
    short_header = "Error OSL"
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.ERROR_ONLY
    )
    console_group = MetricConsoleGroup.NONE
'''
#: Changes whenever the edit does; the staging key carries it, so a changed patch is staged anew.
PATCH_SHA256 = hashlib.sha256("\0".join((_PARSER_OLD, _PARSER_NEW, _METRIC_NEW)).encode(_UTF8)).hexdigest()


class AiperfPatchError(ValueError):
    """The installed client is not the text the patch anchors on, or its ``RECORD`` does not list the files."""


def _read(path: Path) -> str:
    """``path``'s text with its line ends as they are, so a rewritten ``RECORD`` differs only in the patched rows."""
    try:
        return path.read_bytes().decode(_UTF8)
    except (OSError, UnicodeError) as exc:
        raise AiperfPatchError(f"{path.name} cannot be read ({type(exc).__name__})") from exc


def _rows(record: Path) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(_read(record), newline="")) if row]


def _patched_parser(path: Path) -> str:
    text = _read(path)
    if text.count(_PARSER_OLD) != 1:
        raise AiperfPatchError(f"{path.name} does not hold the branch the patch replaces exactly once")
    return text.replace(_PARSER_OLD, _PARSER_NEW)


def _patched_metric(path: Path) -> str:
    text = _read(path)
    if _OUTPUT_LENGTH_CLASS not in text:
        raise AiperfPatchError(f"{path.name} does not define the output length metric the patch extends")
    return text.rstrip("\n") + "\n" + _METRIC_NEW


def _owning_record(site: Path) -> Path:
    """The ``RECORD`` of the distribution that installed both files the patch rewrites."""
    for record in sorted(site.glob("*.dist-info/RECORD")):
        listed = {row[0] for row in _rows(record)}
        if listed.issuperset(PATCHED_FILES):
            return record
    raise AiperfPatchError(f"no distribution in {site} records both {' and '.join(PATCHED_FILES)}")


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _rewrite_record(record: Path, written: dict[str, bytes]) -> None:
    """Rewrite the hash and size of each file in ``written``; pip writes ``RECORD`` with the csv module's defaults."""
    rows = _rows(record)
    for row in rows:
        if row[0] in written:
            row[1:] = [_record_hash(written[row[0]]), str(len(written[row[0]]))]
    sink = io.StringIO(newline="")
    csv.writer(sink).writerows(rows)
    record.write_bytes(sink.getvalue().encode(_UTF8))


def patch_aiperf(site: Path) -> None:
    """Patch the aiperf installed in the site-packages directory ``site`` to export ``error_osl``.

    Both files are checked before either is written, so a client whose text is not the anchored one is changed in
    neither.

    Raises:
        AiperfPatchError: A file is absent or is not the text the patch anchors on, or no ``RECORD`` lists it.
    """
    record = _owning_record(site)
    texts = {PARSER_FILE: _patched_parser(site / PARSER_FILE), METRIC_FILE: _patched_metric(site / METRIC_FILE)}
    written = {relative: text.encode(_UTF8) for relative, text in texts.items()}
    for relative, data in written.items():
        (site / relative).write_bytes(data)
    _rewrite_record(record, written)


__all__ = ["ERROR_OSL_METRIC", "PATCHED_FILES", "PATCH_SHA256", "AiperfPatchError", "patch_aiperf"]
