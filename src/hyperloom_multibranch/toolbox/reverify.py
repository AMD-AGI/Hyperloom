# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a recorded result must still say once it is derived again from its verified bundle (design 4.4, L26).

Verifying a bundle proves the recorded hashes name an authentic bundle; it does not prove that the recorded numbers are
the bundle's. A notebook entry whose throughput was edited and whose hashes were left alone verifies perfectly. So the
rounds and the accuracy are derived from the bundle's own files again and every field is compared with the record:
:func:`first_value_difference` and :func:`first_hash_difference` name the first field that differs, or return ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence

from pathlib import Path

from hyperloom.contracts.measurement import AccuracyValue, RoundValue, TestResult
from hyperloom.measure.decode import OUTPUTS_PATH, DecodePassError, parse_outputs
from hyperloom.measure.manifest import ResultManifest

__all__ = ["first_decode_difference", "first_hash_difference", "first_value_difference", "first_warning_difference"]

_ROUND_FIELDS = ("phase", "concurrency", "completed_requests", "failed_requests")
_HASH_FIELDS = ("spec_fingerprint", "surface_sha256")


def first_hash_difference(recorded: TestResult, manifest: ResultManifest) -> str | None:
    """The first of the manifest's own fields that the record names differently."""
    for name in _HASH_FIELDS:
        if getattr(recorded, name) != getattr(manifest, name):
            return f"{name}: recorded {getattr(recorded, name)!r}, the bundle has {getattr(manifest, name)!r}"
    return None


def first_value_difference(
    recorded: TestResult, rounds: Sequence[RoundValue], accuracy: Sequence[AccuracyValue]
) -> str | None:
    """The first round or accuracy field that differs between the record and the values derived from the bundle."""
    return _round_difference(recorded.rounds, tuple(rounds)) or _accuracy_difference(recorded.accuracy, tuple(accuracy))


def first_decode_difference(recorded: TestResult, manifest: ResultManifest, landed: Path) -> str | None:
    """The decode fields of the record when they are not what the signed bundle holds.

    The bundle holds a pass when its manifest lists ``decode/outputs.jsonl``; the record must then name that file and
    the identity of the header the signature covers, and a record that names a pass the bundle does not hold is refused.
    """
    listed = any(f.path == OUTPUTS_PATH for f in manifest.files)
    if not listed:
        if recorded.decode_outputs or recorded.decode_identity:
            return f"decode_outputs: recorded {recorded.decode_outputs!r}, the bundle lists no {OUTPUTS_PATH}"
        return None
    if recorded.decode_outputs != OUTPUTS_PATH:
        return f"decode_outputs: recorded {recorded.decode_outputs!r}, the bundle holds {OUTPUTS_PATH!r}"
    try:
        signed = parse_outputs((Path(landed) / OUTPUTS_PATH).read_text(encoding="utf-8"))
    except (DecodePassError, OSError, ValueError) as exc:
        return f"decode_outputs: {exc}"
    if recorded.decode_identity != signed.identity:
        return f"decode_identity: recorded {recorded.decode_identity!r}, the bundle says {signed.identity!r}"
    return None


def first_warning_difference(recorded: TestResult, derived: Sequence[str]) -> str | None:
    """The warning list of the record when it is not the list derived from the bundle, as the field's name and both."""
    if recorded.nonfatal_warnings != tuple(derived):
        return f"nonfatal_warnings: recorded {list(recorded.nonfatal_warnings)!r}, the bundle has {list(derived)!r}"
    return None


def _round_difference(recorded: tuple[RoundValue, ...], derived: tuple[RoundValue, ...]) -> str | None:
    if len(recorded) != len(derived):
        return f"rounds: recorded {len(recorded)}, the bundle has {len(derived)}"
    for index, (have, want) in enumerate(zip(recorded, derived, strict=True)):
        for name in _ROUND_FIELDS:
            if getattr(have, name) != getattr(want, name):
                return (
                    f"rounds[{index}].{name}: recorded {getattr(have, name)!r}, the bundle has {getattr(want, name)!r}"
                )
        for metric in sorted(set(have.metrics) | set(want.metrics)):
            if have.metrics.get(metric) != want.metrics.get(metric):
                return (
                    f"rounds[{index}].metrics.{metric}: recorded {have.metrics.get(metric)!r}, "
                    f"the bundle has {want.metrics.get(metric)!r}"
                )
    return None


def _accuracy_difference(recorded: tuple[AccuracyValue, ...], derived: tuple[AccuracyValue, ...]) -> str | None:
    if len(recorded) != len(derived):
        return f"accuracy: recorded {len(recorded)} scores, the bundle has {len(derived)}"
    for index, (have, want) in enumerate(zip(recorded, derived, strict=True)):
        for name in ("task", "metric", "score", "version"):
            if getattr(have, name) != getattr(want, name):
                return f"accuracy[{index}].{name}: recorded {getattr(have, name)!r}, the bundle has {getattr(want, name)!r}"
    return None
