# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a read shapes its result, and lexical fuzzy recall over a Query View."""

from __future__ import annotations

import math
import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import cast

from hyperloom_kb.query_view import QueryView, RetrievalCapability
from hyperloom_kb.retrieval import (
    CandidateHit,
)
from hyperloom_kb.schema import (
    Experience,
    ExperienceDeclaration,
    FieldRole,
    FieldValue,
    FileRef,
    JsonValue,
    default_search_weight,
)
from hyperloom_kb.storage import ExperienceStore

LEXICAL_FUZZY_PROVIDER_REF = "hyperloom-kb.lexical-fuzzy@v3"


class RepresentativePolicy(str, Enum):
    MATCHED_FIRST = "matched_first"
    FIRST_MEMBER = "first_member"


@dataclass(frozen=True)
class ResultShape:
    """How a read ranks the groups it recalled, which member stands for each, and how many and how much it renders."""

    ranking_policy_ref: str = "weighted-signal-sum@v1"
    representative_policy: RepresentativePolicy = RepresentativePolicy.MATCHED_FIRST
    max_groups: int = 20
    render_budget_chars: int | None = 4_000


class LexicalFuzzyProvider:
    """Deterministic dependency-free fuzzy recall over pinned Experiences."""

    capability = RetrievalCapability.FUZZY
    provider_ref = LEXICAL_FUZZY_PROVIDER_REF

    def __init__(self, experiences: ExperienceStore, declaration: ExperienceDeclaration) -> None:
        self._experiences = experiences
        self._declaration = declaration

    def recall(
        self,
        request: dict[str, JsonValue],
        view: QueryView,
    ) -> tuple[CandidateHit, ...]:
        query = _normalized_text(str(request.get("text") or ""))
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()
        documents: list[tuple[str, _SearchDocument]] = []
        document_frequency: dict[str, int] = {}
        for experience_id in view.visible_experience_ids:
            stored = self._experiences.get_experience(experience_id)
            if stored is None or stored.content_hash != view.experience_hashes[experience_id]:
                continue
            document = _search_document(self._declaration, stored.experience)
            documents.append((experience_id, document))
            for token in set().union(*(tokens for _, tokens in document.fields)):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        visible_count = len(documents)

        query_weight = sum(_idf(token, visible_count, document_frequency) for token in query_tokens)
        hits: list[CandidateHit] = []
        max_field_weight = max((weight for _, document in documents for weight, _ in document.fields), default=1.0)
        for experience_id, document in documents:
            matched: list[str] = []
            weighted_match = 0.0
            for token in query_tokens:
                token_field_weight = max(
                    (weight for weight, tokens in document.fields if token in tokens),
                    default=0.0,
                )
                if token_field_weight:
                    matched.append(token)
                    weighted_match += _idf(token, visible_count, document_frequency) * token_field_weight
            coverage = weighted_match / (query_weight * max_field_weight) if query_weight else 0.0
            summary_similarity = SequenceMatcher(None, query, document.summary).ratio()
            identity_compatibility = _identity_compatibility(query_tokens, document.identity)
            score = (0.9 * coverage + 0.1 * summary_similarity) * identity_compatibility
            if score <= 0:
                continue
            hits.append(
                CandidateHit(
                    experience_id,
                    self.capability,
                    score,
                    {
                        "provider_ref": self.provider_ref,
                        "query_view_id": view.ref.view_id,
                        "identity_compatibility": identity_compatibility,
                        "matched_tokens": cast(list[JsonValue], sorted(matched)),
                    },
                )
            )
        return tuple(sorted(hits, key=lambda hit: (-(hit.score or 0.0), hit.experience_id)))


def _normalized_text(value: str) -> str:
    normalized = value.casefold()
    replacements = {
        "8 bit": " fp8 ",
        "8-bit": " fp8 ",
        "key value": " kv ",
        "query key normalization": " qk norm ",
        "rotary embedding": " rope ",
        "memory traffic": " bandwidth ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


_TOKEN_RE = re.compile(r"[a-z]+[0-9]+[a-z]*|[0-9]+[a-z]+|[a-z]+|[0-9]+")
_TOKEN_ALIASES = {
    "traffic": "bandwidth",
    "streaming": "stream",
    "detokenization": "detokeniz",
    "normalization": "norm",
}


def _tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(value):
        token = _TOKEN_ALIASES.get(raw, raw)
        if token.startswith("qwen") and token != "qwen":
            tokens.add("qwen")
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        if len(token) >= 2:
            tokens.add(token)
    return tokens


#: Notes and the objective belong to the KB rather than to a declaration, so their weights are the KB's.
_NOTES_WEIGHT = 1.5
_OBJECTIVE_WEIGHT = 0.5


@dataclass(frozen=True)
class _SearchDocument:
    #: Each searched field's weight and tokens.
    fields: tuple[tuple[float, frozenset[str]], ...]
    #: The identity's tokens, which a query's model size or hardware must not contradict.
    identity: frozenset[str]
    #: The change summary a query is compared with.
    summary: str


def _text_of(value: FieldValue) -> str:
    if isinstance(value, tuple):
        return " ".join(_text_of(item) for item in value)
    return "" if isinstance(value, FileRef) else str(value)


def _field_tokens(value: FieldValue) -> frozenset[str]:
    return frozenset(_tokens(_normalized_text(_text_of(value))))


def _search_document(declaration: ExperienceDeclaration, experience: Experience) -> _SearchDocument:
    fields: list[tuple[float, frozenset[str]]] = []
    for category, item, weight in declaration.search_fields():
        if category == "identity":
            continue
        value = getattr(experience, category).get(item.name)
        if value is not None:
            fields.append((weight, _field_tokens(value)))
    # Identity keys no field declares are searched at the identity's weight too.
    for name, value in experience.identity.items():
        declared = declaration.field("identity", name)
        weight = declared.search_weight("identity") if declared is not None else default_search_weight("identity")
        if weight:
            fields.append((weight, _field_tokens(value)))
    if experience.notes:
        fields.append(
            (_NOTES_WEIGHT, _field_tokens(tuple(f"{label} {text}" for label, text in experience.notes.items())))
        )
    objective = declaration.objective(experience.objective)
    described = f"{experience.objective} {objective.description if objective is not None else ''}"
    fields.append((_OBJECTIVE_WEIGHT, _field_tokens(described)))
    summary_field = declaration.role_field("change", FieldRole.SUMMARY)
    summary = experience.change.get(summary_field.name, "") if summary_field is not None else ""
    return _SearchDocument(
        fields=tuple(fields),
        identity=frozenset().union(*(_field_tokens(value) for value in experience.identity.values())),
        summary=_normalized_text(_text_of(summary)),
    )


def _idf(token: str, count: int, frequencies: dict[str, int]) -> float:
    return math.log((count + 1) / (frequencies.get(token, 0) + 1)) + 1.0


_MODEL_SIZE_RE = re.compile(r"^[0-9]+b$")
_GPU_RE = re.compile(r"^mi[0-9]+x$")


def _identity_compatibility(
    query_tokens: AbstractSet[str],
    identity_tokens: AbstractSet[str],
) -> float:
    """Penalize explicit model-size or hardware contradictions."""
    for pattern in (_MODEL_SIZE_RE, _GPU_RE):
        query_values = {token for token in query_tokens if pattern.fullmatch(token)}
        identity_values = {token for token in identity_tokens if pattern.fullmatch(token)}
        if query_values and identity_values and query_values.isdisjoint(identity_values):
            return 0.1
    return 1.0


__all__ = [
    "LEXICAL_FUZZY_PROVIDER_REF",
    "LexicalFuzzyProvider",
    "RepresentativePolicy",
    "ResultShape",
]
