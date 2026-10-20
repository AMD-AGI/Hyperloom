# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local E2B-compatible sandbox service, for testing the E2B provider with no Claw behind it.

It is a stand-in for what the official ``e2b`` SDK talks to: a stdlib control plane (create, get, list, kill, timeout,
connect, with per-role keys, template checks and idempotency keys), a router that sends every data-plane request to the
right sandbox by its ``E2b-Sandbox-Id`` header, and one real ``envd`` per sandbox, each in private pid and mount
namespaces. Commands and files therefore run through the real envd protocol; only the control plane is simulated, and
only the parts of the Claw sandbox service that a stand-in can honestly reproduce (see :mod:`.service`).

Running it needs root (envd switches accounts for every command) and an ``envd`` binary named by
``HYPERLOOM_E2B_ENVD``: build ``packages/envd`` of e2b-dev/infra with ``GOWORK=off CGO_ENABLED=0 go build``
(about 20 s). :func:`unavailable_reason` says why a machine cannot run it, for a skip message.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Mapping
from pathlib import Path

from hyperloom.sandbox.socket import Role
from hyperloom.sandbox.tests.e2b_standin.envd_host import ENVD_ENV, EnvdHost, interpreter_binds, unavailable_reason
from hyperloom.sandbox.tests.e2b_standin.http_api import serve_api
from hyperloom.sandbox.tests.e2b_standin.proxy import serve_proxy
from hyperloom.sandbox.tests.e2b_standin.service import ControlPlane, StandInTemplate

__all__ = ["ENVD_ENV", "StandIn", "StandInTemplate", "interpreter_binds", "unavailable_reason"]

_ROLES: tuple[Role, ...] = ("measurement", "subrun", "dev")


class StandIn:
    """The running stand-in: its two addresses, one API key per role, and the control plane behind them.

    Args:
        envd: The envd binary.
        workdir: An empty directory for sandbox roots and logs.
        templates: The templates that exist.
        max_sandboxes: Per role, how many sandboxes may be live at once (a create over it gets 429
            ``grant_quota_exceeded``).
        page_size: The most sandboxes one page of a listing holds.

    Use it as a context manager; leaving it kills every sandbox it started.
    """

    def __init__(
        self,
        envd: Path,
        workdir: Path,
        templates: list[StandInTemplate],
        *,
        max_sandboxes: Mapping[Role, int] | None = None,
        page_size: int = 100,
    ) -> None:
        self.keys: dict[Role, str] = {role: f"e2b_{secrets.token_hex(20)}" for role in _ROLES}
        self.host = EnvdHost(envd, workdir)
        self.plane = ControlPlane(self.host, templates, self.keys, max_sandboxes=max_sandboxes)
        self._api = serve_api(self.plane, page_size=page_size)
        self._proxy = serve_proxy(self.plane)
        self.api_url = f"http://127.0.0.1:{self._api.server_address[1]}"
        self.sandbox_url = f"http://127.0.0.1:{self._proxy.server_address[1]}"
        self._threads = [
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
            for server in (self._api, self._proxy)
        ]
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        """Stop serving and kill every sandbox; raises if one of the processes this stand-in started is still alive."""
        for server in (self._api, self._proxy):
            server.shutdown()
            server.server_close()
        self.plane.close()
        if self.host.alive():
            raise RuntimeError(f"processes started by the stand-in are still alive: {self.host.alive()}")

    def __enter__(self) -> StandIn:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
