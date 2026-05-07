"""Sanity checks on the embedded migration set.

We don't apply migrations against a real Postgres in CI here (that
lives in the integration suite), but we do verify that:

* the migration package is loadable as resources;
* every shipped file is named ``NNNN_description.sql``;
* the SQL contains the canonical table names declared in the design.

This catches breakage from renames, missing `__init__.py`, or stray
files much earlier than an integration run.
"""

from __future__ import annotations

import re
from importlib import resources


_MIGRATION_NAME_RE = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
_REQUIRED_TABLES = ("sessions", "session_pod_assignment", "session_events")


def _migration_files() -> list[str]:
    package = resources.files("robustness_server.store.migrations")
    return sorted(entry.name for entry in package.iterdir() if entry.name.endswith(".sql"))


def test_migration_filenames_follow_convention() -> None:
    files = _migration_files()
    assert files, "expected at least one migration file"
    for name in files:
        assert _MIGRATION_NAME_RE.match(name), f"bad migration name: {name}"


def test_initial_migration_declares_required_tables() -> None:
    package = resources.files("robustness_server.store.migrations")
    initial = package.joinpath("0001_init.sql")
    sql = initial.read_text(encoding="utf-8").lower()
    for table in _REQUIRED_TABLES:
        assert f"create table if not exists {table}" in sql, f"missing table {table}"
