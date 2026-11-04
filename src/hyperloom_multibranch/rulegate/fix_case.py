# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The reproduction case a correctness fix is declared with, as the test that measures the fix carries it.

A case is the directory attachment ``FIX_CASE_ID``: a tar with ``repro.sh`` at its top, the script the repro runner
(``hyperloom.measure.repro``) runs against a live server and reads by its exit code, and ``DIAGNOSIS.md``, the defect
the specialist found. Equal inputs pack to equal bytes, so a case stored twice is one blob.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass

from hyperloom.contracts.measurement import AttachmentRef, TestRequest

#: The attachment id of a test's case; the toolbox unpacks it under ``attach/repro-case/``.
FIX_CASE_ID = "repro-case"
#: ``hyperloom.measure.repro.CASE_SCRIPT``, which cannot be imported where ``resource`` is missing.
CASE_SCRIPT = "repro.sh"
DIAGNOSIS_FILE = "DIAGNOSIS.md"
_DIR = "dir"
_EXECUTABLE = 0o755
_READABLE = 0o644


class CaseError(ValueError):
    """The input or the bytes are not a case; the message says what is missing."""


@dataclass(frozen=True)
class FixCase:
    repro_sh: str
    diagnosis: str


def _member(name: str, text: str, mode: int) -> tuple[tarfile.TarInfo, io.BytesIO]:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info, io.BytesIO(data)


def pack_case(repro_sh: str, diagnosis: str) -> bytes:
    """The tar of a case; ``repro_sh`` exits 0 when the defect does not occur and 1 when it does."""
    if not repro_sh.strip() or not diagnosis.strip():
        raise CaseError(f"a case needs both a {CASE_SCRIPT} and a diagnosis")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        archive.addfile(*_member(CASE_SCRIPT, repro_sh, _EXECUTABLE))
        archive.addfile(*_member(DIAGNOSIS_FILE, diagnosis, _READABLE))
    return buffer.getvalue()


def read_case(data: bytes) -> FixCase:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            texts = {}
            for name in (CASE_SCRIPT, DIAGNOSIS_FILE):
                member = archive.extractfile(name) if name in archive.getnames() else None
                if member is None:
                    raise CaseError(f"the case holds no {name} at its top")
                texts[name] = member.read().decode("utf-8")
    except tarfile.TarError as exc:
        raise CaseError(f"the case is not a tar archive: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CaseError(f"a case is UTF-8 text: {exc}") from exc
    return FixCase(texts[CASE_SCRIPT], texts[DIAGNOSIS_FILE])


def case_of(request: TestRequest) -> AttachmentRef | None:
    """The case ``request`` carries, or ``None`` when it declares no fix."""
    case = next((attachment for attachment in request.attachments if attachment.id == FIX_CASE_ID), None)
    if case is not None and case.kind != _DIR:
        raise CaseError(f"the attachment {FIX_CASE_ID} is a {case.kind}; a case is a directory")
    return case


__all__ = ["CASE_SCRIPT", "DIAGNOSIS_FILE", "FIX_CASE_ID", "CaseError", "FixCase", "case_of", "pack_case", "read_case"]
