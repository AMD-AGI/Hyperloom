"""Backend-neutral canonical Experience service."""

from __future__ import annotations

from hyperloom_kb.schema import Experience, ExperienceDeclaration, ExperienceStatus
from hyperloom_kb.storage import (
    ExperienceStore,
    InsertResult,
    SchemaRegistry,
    StorageContractError,
    StoredExperience,
)


class UnknownSchemaRef(StorageContractError):
    """Raised when an Experience references an unregistered schema."""


class CompleteExperienceRequired(StorageContractError):
    """Raised when immutable submission receives a non-terminal Experience."""


class ExperienceService:
    """Register exact schemas and atomically submit immutable Experiences."""

    def __init__(self, schemas: SchemaRegistry, experiences: ExperienceStore) -> None:
        self._schemas = schemas
        self._experiences = experiences

    def register_schema(self, declaration: ExperienceDeclaration) -> str:
        return self._schemas.register_schema(declaration)

    def get_schema(self, schema_ref: str) -> ExperienceDeclaration | None:
        return self._schemas.get_schema(schema_ref)

    def submit_complete(self, experience: Experience) -> InsertResult:
        if experience.status is not ExperienceStatus.COMPLETE:
            raise CompleteExperienceRequired("submit_complete requires a complete Experience")
        declaration = self._schemas.get_schema(experience.schema_ref)
        if declaration is None:
            raise UnknownSchemaRef(experience.schema_ref)
        declaration.validate(experience)
        return self._experiences.insert_complete(experience)

    def get_experience(self, experience_id: str) -> StoredExperience | None:
        return self._experiences.get_experience(experience_id)

    def list_experiences(self, schema_ref: str) -> tuple[StoredExperience, ...]:
        if self._schemas.get_schema(schema_ref) is None:
            raise UnknownSchemaRef(schema_ref)
        return self._experiences.list_experiences(schema_ref)


__all__ = [
    "CompleteExperienceRequired",
    "ExperienceService",
    "UnknownSchemaRef",
]
