# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The events socket and its local implementation (4.11)."""

from __future__ import annotations

import logging

from hyperloom.plugs import ManifestSources, PlugHost, builtin_registry, resolve_manifest
from hyperloom.plugs.events import ProgressEvents
from hyperloom.plugs.events_local import LocalProgressEvents


def test_the_local_implementation_satisfies_the_socket_protocol() -> None:
    assert isinstance(LocalProgressEvents(), ProgressEvents)


def test_the_local_implementation_only_writes_a_debug_log_line(caplog, tmp_path) -> None:
    with caplog.at_level(logging.DEBUG, logger="hyperloom.events"):
        events = LocalProgressEvents()
        events.emit("hyperloom.warning", {"code": "x"})
        events.flush()
    assert [r.getMessage() for r in caplog.records] == ["event hyperloom.warning {'code': 'x'}"]
    assert list(tmp_path.iterdir()) == []


def test_the_default_host_fills_events_with_the_local_implementation() -> None:
    registry = builtin_registry()
    host = PlugHost(resolve_manifest(registry, ManifestSources()), registry, session_dir=None)
    host.start()
    assert isinstance(host.get("events"), LocalProgressEvents)
    host.close()
