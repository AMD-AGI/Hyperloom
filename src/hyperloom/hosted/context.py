# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the hosted entry hands to the part of the process that starts the plug host.

The entry reads the host's inputs once and then removes the host-only ones from ``os.environ`` (see
:func:`hyperloom.hosted.host_info.take_host_info`).  Whatever builds a sandbox provider, a measurement client or an
archive writer later must therefore take its addresses, keys and template names from :attr:`HostedRun.host_info`,
never from ``os.environ``: an E2B provider built with ``from_environ`` on the cleaned environment finds no key, and
every role is then refused its sandbox.

``plug_lock`` is the hosted lock layer, ``{socket: {"impl": ...}}``, in the shape of
:attr:`hyperloom.plugs.manifest.ManifestSources.locked`; the code that resolves the manifest applies it last.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.hosted.campaign import CampaignSpec
from hyperloom.hosted.host_info import HostInfo


@dataclass(frozen=True)
class HostedRun:
    """The facts of one hosted run, fixed before ``optimize`` starts."""

    spec: CampaignSpec
    host_info: HostInfo
    plug_lock: Mapping[str, Mapping[str, str]]
    session_dir: Path
    archive_root: Path


class _Slot:
    run: HostedRun | None = None


_slot = _Slot()


def install(run: HostedRun) -> None:
    """Make *run* what :func:`current` returns; the entry calls this once, before ``optimize`` starts."""
    _slot.run = run


def current() -> HostedRun | None:
    """The hosted run of this process, or ``None`` when the process was not started by the hosted entry."""
    return _slot.run


def clear() -> None:
    """Forget the installed run (tests, and an entry that returns into a longer-lived process)."""
    _slot.run = None
