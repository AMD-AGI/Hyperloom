"""Load Experiences into an in-memory store and build query views over them."""

from __future__ import annotations

from hyperloom_kb.query_view import QueryView, QueryViewBuilder
from hyperloom_kb.schema import Experience, ExperienceDeclaration
from hyperloom_kb.storage import ExperienceStore


def submit_complete(declaration: ExperienceDeclaration, experiences: ExperienceStore, *items: Experience) -> None:
    """Validate each Experience against ``declaration`` and insert it."""

    for item in items:
        declaration.validate(item)
        experiences.insert_complete(item)


def build_view(declaration: ExperienceDeclaration, experiences: ExperienceStore) -> QueryView:
    """A view over every stored Experience of the declaration's schema."""

    return QueryViewBuilder().build(declaration, experiences.list_experiences(declaration.schema_ref))
