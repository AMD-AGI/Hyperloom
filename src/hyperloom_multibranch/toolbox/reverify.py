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
from dataclasses import fields

from hyperloom.contracts.measurement import AccuracyValue, RequestClasses, RoundValue, TestResult
from hyperloom.measure.manifest import ResultManifest

__all__ = [
    "first_hash_difference",
    "first_request_classes_difference",
    "first_value_difference",
    "first_warning_difference",
]

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


def first_warning_difference(recorded: TestResult, derived: Sequence[str]) -> str | None:
    """The warning list of the record when it is not the list derived from the bundle, as the field's name and both."""
    if recorded.nonfatal_warnings != tuple(derived):
        return f"nonfatal_warnings: recorded {list(recorded.nonfatal_warnings)!r}, the bundle has {list(derived)!r}"
    return None


def first_request_classes_difference(recorded: TestResult, derived: Sequence[RequestClasses]) -> str | None:
    """The first field of the request classes the record holds that differs from the ones derived from the bundle.

    A record that holds none is not compared: it claims nothing, and a consumer that needs the classes reads it as
    having none.
    """
    held = recorded.request_classes
    if not held:
        return None
    if len(held) != len(derived):
        return f"request_classes: recorded {len(held)}, the bundle has {len(derived)}"
    for index, (have, want) in enumerate(zip(held, derived, strict=True)):
        for field in fields(RequestClasses):
            if getattr(have, field.name) != getattr(want, field.name):
                return (
                    f"request_classes[{index}].{field.name}: recorded {getattr(have, field.name)!r}, "
                    f"the bundle has {getattr(want, field.name)!r}"
                )
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
