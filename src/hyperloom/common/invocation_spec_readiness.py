# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether an invocation spec is complete enough to drive driver preparation.

The spec builder does not fail when evidence is missing: it marks the document
``status: "partial"`` and lists what is absent in ``missing_fields``. Nothing
read either field, so the rewrite route admitted a partial spec, the producer
kept its placeholder driver, and the run burned its whole budget before exiting
non-zero. This module is the missing reader.

Kept out of the backends package on purpose -- that tree is excluded from
coverage, and this is the decision worth covering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"

REASON_MISSING = "invocation_spec_missing"
REASON_UNREADABLE = "invocation_spec_unreadable"
REASON_PARTIAL = "invocation_spec_partial"


@dataclass(frozen=True)
class SpecReadiness:
    """Verdict on one spec file, with the reason a rejection can be reported by."""

    ready: bool
    reason: str = ""
    detail: str = ""
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


def evaluate_spec_readiness(spec_path: str | Path | None) -> SpecReadiness:
    """Decide whether a spec file can drive driver preparation.

    Args:
        spec_path: Path to the invocation spec JSON, or an empty value.

    Returns:
        A ready verdict, or a rejection carrying one of the module's reason
        codes. Unreadable and partial are reported separately: the first is an
        environment problem, the second is missing upstream evidence.
    """
    raw = str(spec_path or "").strip()
    if not raw:
        return SpecReadiness(False, REASON_MISSING, "no invocation spec path was supplied")
    path = Path(raw)
    if not path.is_file():
        return SpecReadiness(False, REASON_MISSING, f"no invocation spec at {raw}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return SpecReadiness(False, REASON_UNREADABLE, f"could not read invocation spec {raw}: {error}")
    if not isinstance(document, dict):
        return SpecReadiness(False, REASON_UNREADABLE, f"invocation spec is not a JSON object: {raw}")
    status = str(document.get("status") or "").strip().lower()
    missing = _missing_fields(document)
    # An absent status predates the field; treat it as complete so an older spec
    # keeps working, and let the explicit `partial` marker be the only rejection.
    if status == STATUS_PARTIAL:
        listed = ", ".join(missing) if missing else "unspecified"
        return SpecReadiness(
            False,
            REASON_PARTIAL,
            f"invocation spec at {raw} is partial; missing: {listed}",
            missing_fields=missing,
        )
    return SpecReadiness(True, missing_fields=missing)


def _missing_fields(document: dict) -> tuple[str, ...]:
    """Read the builder's own list of what it could not fill in."""
    listed = document.get("missing_fields")
    if not isinstance(listed, list):
        return ()
    return tuple(str(item).strip() for item in listed if str(item or "").strip())
