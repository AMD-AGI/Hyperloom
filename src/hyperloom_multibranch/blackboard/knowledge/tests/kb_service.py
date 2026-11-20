# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A live Experience KB service for the tests that publish to one, and a client of it."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

from hyperloom_kb import (
    ExperienceDeclaration,
    ExperienceHTTPService,
    HTTPServiceConfig,
    RemoteClient,
    RemoteConfig,
    create_http_server,
)
from hyperloom_kb.database import open_database

TOKEN = "multibranch-test-token"


@contextlib.contextmanager
def experience_service(tmp_path: Path, declaration: ExperienceDeclaration) -> Iterator[RemoteClient]:
    """A service started on ``declaration`` under ``tmp_path``, stopped on exit."""

    home = tmp_path / "service"
    service = ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), declaration, database=open_database(home))
    server = create_http_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield RemoteClient(RemoteConfig(f"http://{host!s}:{port}", TOKEN, spool_root=tmp_path / "spool"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
