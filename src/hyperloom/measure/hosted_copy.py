# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copying a hosted result's listed files out of the sandbox, again when a copy comes back short or long.

A transfer of the platform's file API can end with fewer (or more) bytes than the file holds and no error. The byte
count :meth:`~hyperloom.sandbox.socket.Sandbox.copy_out` returns is compared with the size the signed manifest lists;
a file whose count differs is copied again, at most :data:`RECOPIES` more times. A count that still differs raises
:class:`DownloadIncomplete`: the download, not the result, is what failed, and the attempt ends as an infrastructure
failure (``result_download_incomplete``) that is retried in a new sandbox. Only a download whose every count matches
goes on to verification, whose refusals (size, hash, signature, a missing file) stay final.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom.measure.manifest import ManifestFile
from hyperloom.sandbox.socket import Sandbox

__all__ = ["RECOPIES", "DownloadIncomplete", "copy_listed"]

log = logging.getLogger(__name__)

#: How many more times a result file is copied when the bytes received do not add up to its listed size.
RECOPIES = 2


class DownloadIncomplete(Exception):
    """A listed file whose every copy received another byte count than the manifest lists."""

    def __init__(self, path: str, received: int, listed: int) -> None:
        super().__init__(f"{path}: {received} bytes received, the manifest lists {listed}")
        self.path, self.received, self.listed = path, received, listed


def copy_listed(handle: Sandbox, files_root: str, entry: ManifestFile, target: Path) -> None:
    """Copy one listed file to ``target``; a copy whose byte count is not the listed size is made again, at most
    :data:`RECOPIES` more times, and one that still differs raises :class:`DownloadIncomplete`."""
    source = f"{files_root}/{entry.path}"
    for copy in range(1, RECOPIES + 2):
        received = handle.copy_out(source, target)
        if received == entry.size:
            return
        then = "copying it again" if copy <= RECOPIES else "the download is incomplete"
        log.warning(
            "%s: copy %d received %s bytes, the manifest lists %d; %s", entry.path, copy, received, entry.size, then
        )
    raise DownloadIncomplete(entry.path, received, entry.size)
