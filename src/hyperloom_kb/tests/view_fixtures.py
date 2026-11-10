"""Load Experiences into an in-memory store and publish query views over them."""

from __future__ import annotations

from hyperloom_kb.query_view import QueryViewBuilder, QueryViewRef, QueryViewStore
from hyperloom_kb.schema import Experience, ExperienceDeclaration
from hyperloom_kb.storage import ExperienceStore


def submit_complete(declaration: ExperienceDeclaration, experiences: ExperienceStore, *items: Experience) -> None:
    """Validate each Experience against ``declaration`` and insert it."""

    for item in items:
        declaration.validate(item)
        experiences.insert_complete(item)


def publish_view(
    declaration: ExperienceDeclaration,
    experiences: ExperienceStore,
    views: QueryViewStore,
    *,
    fuzzy_ready: bool = False,
    semantic_ready: bool = False,
) -> QueryViewRef:
    """Build a view over every stored Experience of the declaration's schema and make it current."""

    records = experiences.list_experiences(declaration.schema_ref)
    view = QueryViewBuilder().build(
        declaration,
        records,
        fuzzy_ready=fuzzy_ready,
        semantic_ready=semantic_ready,
    )
    return views.publish_view(view)
