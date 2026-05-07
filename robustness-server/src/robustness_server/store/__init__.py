"""Persistence layer.

Wraps an asyncpg connection pool, applies embedded SQL migrations on
startup, and exposes thin repository helpers that the API + services
share. We avoid an ORM intentionally — the schema is small, queries are
mostly plain CRUD, and the migrations live in a sibling SQL folder for
zero-magic deployment.
"""

from .assignments_repo import AssignmentsRepository
from .events_repo import EventsRepository
from .pool import Database, get_database, install_database, set_database_for_test
from .sessions_repo import SessionsRepository

__all__ = [
    "AssignmentsRepository",
    "Database",
    "EventsRepository",
    "SessionsRepository",
    "get_database",
    "install_database",
    "set_database_for_test",
]
