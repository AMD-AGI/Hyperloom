# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A repro case as a request carries it: a directory attachment whose blob is in the session's blob store."""

from __future__ import annotations

import io
import tarfile
from typing import Any

from hyperloom.contracts.measurement import AttachmentRef


def case_attachment(blobs: Any, script: str, *, attachment_id: str = "case-1", name: str = "repro.sh") -> AttachmentRef:
    """Pack ``script`` as ``name`` into a tar directory attachment, store it in ``blobs`` and return its reference."""
    data = script.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size, info.mode = len(data), 0o644  # an attachment carries no executable bit
        tar.addfile(info, io.BytesIO(data))
    return AttachmentRef(attachment_id, "dir", blobs.put(buffer.getvalue()))
