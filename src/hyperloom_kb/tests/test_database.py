# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A service's database answers its next request once the server is back, as after a failover to a new primary."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hyperloom_kb.database import Database
from hyperloom_kb.tests.postgres_fixtures import requires_embedded_postgres
from hyperloom_kb.embedded_postgres import start_embedded_postgres


@requires_embedded_postgres
def test_the_first_request_after_the_server_restarts_is_answered() -> None:
    home = Path(tempfile.mkdtemp(prefix="hyperloom-kb-restart-"))
    home.chmod(0o755)
    server = start_embedded_postgres(home)
    database = Database(server.conninfo, max_size=2)
    try:
        with database.transaction() as connection:
            connection.execute("CREATE TABLE kept (value TEXT)")
            connection.execute("INSERT INTO kept VALUES ('before the restart')")
        server.stop()
        server = start_embedded_postgres(home)
        with database.transaction() as connection:
            kept = connection.execute("SELECT value FROM kept").fetchall()
    finally:
        database.close()
        server.stop()
        shutil.rmtree(home, ignore_errors=True)

    assert kept == [{"value": "before the restart"}]
