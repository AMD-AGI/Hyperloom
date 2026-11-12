# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checks the measurement contract types validate their fields with, and the error they raise.

``measurement``, ``consistency`` and ``long_context`` share them; every type validates on construction and reads its
document strictly.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from hyperloom_common.contract_fields import ContractFields


class MeasurementError(ValueError):
    """A test request, a test result or one of their documents breaks the contract."""


_FIELDS = ContractFields(MeasurementError)
_enum = _FIELDS.enum
_text = _FIELDS.text
_optional_text = _FIELDS.optional_text
_sha256 = _FIELDS.sha256
_optional_sha256 = _FIELDS.optional_sha256
_int = _FIELDS.integer
_optional_int = _FIELDS.optional_integer
_number = _FIELDS.number
_optional_number = _FIELDS.optional_number
_mapping = _FIELDS.mapping
_items = _FIELDS.items
_strings = _FIELDS.string_tuple
_bool = _FIELDS.boolean
_optional_bool = _FIELDS.optional_boolean


def _relative_dir(value: object, what: str) -> str:
    text = _text(value, what) if value != "" else ""
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise MeasurementError(f"{what} must be a relative path inside the session directory, got {value!r}")
    return text
