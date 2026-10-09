# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Versioned retrieval policy execution over policy-neutral primitives."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

from hyperloom_kb.query_view import QueryView, QueryViewRef, RetrievalCapability
from hyperloom_kb.retrieval import (
    CandidateHit,
    CandidateProvider,
    CapabilityUnavailable,
    GroupCandidate,
    LocalRetrievalService,
    RetrievalResult,
)
from hyperloom_kb.schema import (
    Experience,
    ExperienceDeclaration,
    FieldRole,
    FieldValue,
    FileRef,
    JsonScalar,
    JsonValue,
    default_search_weight,
)
from hyperloom_kb.storage import ExperienceStore

RETRIEVAL_POLICY_VERSION = 1
LEXICAL_FUZZY_PROVIDER_REF = "hyperloom-kb.lexical-fuzzy@v3"
_POLICY_ID_PREFIX = "retrieval-config:sha256:"
_SCHEMA_REF_PREFIX = "schema:sha256:"
_RANKING_POLICIES = frozenset(
    {
        "capability-priority@v1",
        "group-key@v1",
        "max-hit-score@v1",
        "weighted-signal-sum@v1",
    }
)


class RetrievalPolicyError(ValueError):
    """Raised when a versioned retrieval policy cannot be executed faithfully."""


class RepresentativePolicy(str, Enum):
    MATCHED_FIRST = "matched_first"
    FIRST_MEMBER = "first_member"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _config_id(manifest: Mapping[str, JsonValue]) -> str:
    digest = hashlib.sha256(f"retrieval-config\0{_canonical(manifest)}".encode()).hexdigest()
    return f"{_POLICY_ID_PREFIX}{digest}"


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RetrievalPolicyError(f"{name} must be non-empty")
    return text


def _schema_ref(value: Any) -> str:
    text = _required_text(value, "schema_ref")
    if not text.startswith(_SCHEMA_REF_PREFIX) or len(text) != len(_SCHEMA_REF_PREFIX) + 64:
        raise RetrievalPolicyError("schema_ref is invalid")
    return text


def _json_object(value: Any, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise RetrievalPolicyError(f"{name} must be an object")
    try:
        normalized = json.loads(_canonical(value))
    except (TypeError, ValueError) as exc:
        raise RetrievalPolicyError(f"{name} must contain JSON values") from exc
    if not isinstance(normalized, dict):
        raise RetrievalPolicyError(f"{name} must be an object")
    return normalized


def _scalar_object(value: Any, name: str) -> dict[str, JsonScalar]:
    normalized = _json_object(value, name)
    if any(isinstance(item, (dict, list)) for item in normalized.values()):
        raise RetrievalPolicyError(f"{name} values must be scalar")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class QueryRequest:
    schema_ref: str
    exact_where: dict[str, JsonScalar] = field(default_factory=dict)
    filter_where: dict[str, JsonScalar] = field(default_factory=dict)
    intent: str = ""
    context: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_ref", _schema_ref(self.schema_ref))
        object.__setattr__(
            self,
            "exact_where",
            _scalar_object(self.exact_where, "exact_where"),
        )
        object.__setattr__(
            self,
            "filter_where",
            _scalar_object(self.filter_where, "filter_where"),
        )
        object.__setattr__(self, "intent", str(self.intent or "").strip())
        object.__setattr__(self, "context", _json_object(self.context, "context"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_ref": self.schema_ref,
            "exact_where": dict(self.exact_where),
            "filter_where": dict(self.filter_where),
            "intent": self.intent,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, value: Any) -> QueryRequest:
        if not isinstance(value, dict):
            raise RetrievalPolicyError("QueryRequest must be an object")
        return cls(
            schema_ref=str(value.get("schema_ref") or ""),
            exact_where=_scalar_object(value.get("exact_where", {}), "exact_where"),
            filter_where=_scalar_object(value.get("filter_where", {}), "filter_where"),
            intent=str(value.get("intent") or ""),
            context=_json_object(value.get("context", {}), "context"),
        )


@dataclass(frozen=True)
class RetrievalConfiguration:
    configuration_id: str
    schema_ref: str
    policy_version: str
    capabilities: tuple[RetrievalCapability, ...]
    limits: dict[RetrievalCapability, int]
    provider_refs: dict[RetrievalCapability, str]
    ranking_policy_ref: str
    representative_policy: RepresentativePolicy
    max_groups: int
    render_budget_chars: int | None
    version: int = RETRIEVAL_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_ref", _schema_ref(self.schema_ref))
        object.__setattr__(
            self,
            "policy_version",
            _required_text(self.policy_version, "policy_version"),
        )
        capabilities = tuple(self.capabilities)
        if set(capabilities) != set(RetrievalCapability) or len(capabilities) != len(RetrievalCapability):
            raise RetrievalPolicyError("capabilities must enumerate exact, filter, fuzzy, and semantic once")
        object.__setattr__(self, "capabilities", capabilities)
        limits = dict(self.limits)
        if set(limits) != set(RetrievalCapability):
            raise RetrievalPolicyError("limits must cover every retrieval capability")
        if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 for limit in limits.values()):
            raise RetrievalPolicyError("retrieval limits must be positive integers")
        object.__setattr__(self, "limits", limits)
        provider_refs = dict(self.provider_refs)
        if any(
            capability not in {RetrievalCapability.FUZZY, RetrievalCapability.SEMANTIC} for capability in provider_refs
        ):
            raise RetrievalPolicyError("provider_refs may pin only fuzzy or semantic providers")
        if any(not str(ref or "").strip() for ref in provider_refs.values()):
            raise RetrievalPolicyError("provider_refs must be non-empty")
        object.__setattr__(self, "provider_refs", provider_refs)
        if self.ranking_policy_ref not in _RANKING_POLICIES:
            raise RetrievalPolicyError("ranking_policy_ref is not implemented")
        if not isinstance(self.representative_policy, RepresentativePolicy):
            raise RetrievalPolicyError("representative_policy is invalid")
        if isinstance(self.max_groups, bool) or self.max_groups < 1:
            raise RetrievalPolicyError("max_groups must be positive")
        if self.render_budget_chars is not None and (
            isinstance(self.render_budget_chars, bool) or self.render_budget_chars < 1
        ):
            raise RetrievalPolicyError("render_budget_chars must be positive or None")
        if self.version != RETRIEVAL_POLICY_VERSION:
            raise RetrievalPolicyError("unsupported RetrievalConfiguration version")
        if self.configuration_id != _config_id(self.to_manifest()):
            raise RetrievalPolicyError("configuration_id does not match manifest content")

    @classmethod
    def create(
        cls,
        schema_ref: str,
        policy_version: str,
        *,
        capabilities: tuple[RetrievalCapability, ...] = tuple(RetrievalCapability),
        limits: Mapping[RetrievalCapability, int] | None = None,
        provider_refs: Mapping[RetrievalCapability, str] | None = None,
        ranking_policy_ref: str = "group-key@v1",
        representative_policy: RepresentativePolicy = RepresentativePolicy.MATCHED_FIRST,
        max_groups: int = 20,
        render_budget_chars: int | None = 4_000,
    ) -> RetrievalConfiguration:
        normalized_limits = dict(limits or {capability: 20 for capability in RetrievalCapability})
        normalized_providers = dict(provider_refs or {})
        manifest: dict[str, JsonValue] = {
            "version": RETRIEVAL_POLICY_VERSION,
            "schema_ref": schema_ref,
            "policy_version": policy_version,
            "capabilities": [capability.value for capability in capabilities],
            "limits": {
                capability.value: limit
                for capability, limit in sorted(
                    normalized_limits.items(),
                    key=lambda item: item[0].value,
                )
            },
            "provider_refs": {
                capability.value: ref
                for capability, ref in sorted(
                    normalized_providers.items(),
                    key=lambda item: item[0].value,
                )
            },
            "ranking_policy_ref": ranking_policy_ref,
            "representative_policy": representative_policy.value,
            "max_groups": max_groups,
            "render_budget_chars": render_budget_chars,
        }
        return cls(
            configuration_id=_config_id(manifest),
            schema_ref=schema_ref,
            policy_version=policy_version,
            capabilities=capabilities,
            limits=normalized_limits,
            provider_refs=normalized_providers,
            ranking_policy_ref=ranking_policy_ref,
            representative_policy=representative_policy,
            max_groups=max_groups,
            render_budget_chars=render_budget_chars,
        )

    def to_manifest(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "schema_ref": self.schema_ref,
            "policy_version": self.policy_version,
            "capabilities": [capability.value for capability in self.capabilities],
            "limits": {
                capability.value: limit
                for capability, limit in sorted(
                    self.limits.items(),
                    key=lambda item: item[0].value,
                )
            },
            "provider_refs": {
                capability.value: ref
                for capability, ref in sorted(
                    self.provider_refs.items(),
                    key=lambda item: item[0].value,
                )
            },
            "ranking_policy_ref": self.ranking_policy_ref,
            "representative_policy": self.representative_policy.value,
            "max_groups": self.max_groups,
            "render_budget_chars": self.render_budget_chars,
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {"configuration_id": self.configuration_id, **self.to_manifest()}

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalConfiguration:
        if not isinstance(value, dict):
            raise RetrievalPolicyError("RetrievalConfiguration must be an object")
        raw_capabilities = value.get("capabilities")
        raw_limits = value.get("limits")
        raw_providers = value.get("provider_refs")
        if (
            not isinstance(raw_capabilities, list)
            or not isinstance(raw_limits, dict)
            or not isinstance(raw_providers, dict)
        ):
            raise RetrievalPolicyError("RetrievalConfiguration structures are invalid")
        raw_budget = value.get("render_budget_chars", 0)
        return cls(
            configuration_id=str(value.get("configuration_id") or ""),
            schema_ref=str(value.get("schema_ref") or ""),
            policy_version=str(value.get("policy_version") or ""),
            capabilities=tuple(RetrievalCapability(str(item)) for item in raw_capabilities),
            limits={RetrievalCapability(str(capability)): int(limit) for capability, limit in raw_limits.items()},
            provider_refs={RetrievalCapability(str(capability)): str(ref) for capability, ref in raw_providers.items()},
            ranking_policy_ref=str(value.get("ranking_policy_ref") or ""),
            representative_policy=RepresentativePolicy(str(value.get("representative_policy") or "")),
            max_groups=int(value.get("max_groups", 0)),
            render_budget_chars=None if raw_budget is None else int(raw_budget),
            version=int(value.get("version", 0)),
        )


@dataclass(frozen=True)
class QueryExecutionResult:
    configuration_id: str
    request: QueryRequest
    retrieval: RetrievalResult
    ranked_group_keys: tuple[str, ...]
    representative_ids: tuple[str, ...]
    candidate_count: int
    latency_ms: float

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "configuration_id": self.configuration_id,
            "request": self.request.to_dict(),
            "view": self.retrieval.view.to_dict(),
            "groups": [
                {
                    "group_key": group.group_key,
                    "member_ids": list(group.member_ids),
                    "annotations": group.annotations.to_dict(),
                    "hits": [
                        {
                            "experience_id": hit.experience_id,
                            "capability": hit.capability.value,
                            "score": hit.score,
                            "details": dict(hit.details),
                        }
                        for hit in group.hits
                    ],
                }
                for group in self.retrieval.groups
            ],
            "ranked_group_keys": list(self.ranked_group_keys),
            "representative_ids": list(self.representative_ids),
            "rendered": {
                "text": self.retrieval.rendered.text,
                "rendered_refs": [reference.to_dict() for reference in self.retrieval.rendered.rendered_refs],
                "truncated": self.retrieval.rendered.truncated,
            },
            "executed_capabilities": [capability.value for capability in self.retrieval.executed_capabilities],
            "unavailable_capabilities": [capability.value for capability in self.retrieval.unavailable_capabilities],
            "candidate_count": self.candidate_count,
            "latency_ms": self.latency_ms,
        }


class RetrievalRunner:
    """Execute one explicit configuration without hiding policy in primitives."""

    def __init__(
        self,
        service: LocalRetrievalService,
        *,
        provider_refs: Mapping[RetrievalCapability, str] | None = None,
    ) -> None:
        self._service = service
        self._provider_refs = dict(provider_refs or {})

    def execute(
        self,
        request: QueryRequest,
        configuration: RetrievalConfiguration,
        *,
        view: QueryViewRef | None = None,
    ) -> QueryExecutionResult:
        if request.schema_ref != configuration.schema_ref:
            raise RetrievalPolicyError("request and configuration schema_ref differ")
        for capability, expected in configuration.provider_refs.items():
            if self._provider_refs.get(capability) != expected:
                raise RetrievalPolicyError(f"{capability.value} provider does not match pinned provider_ref")
        started = time.perf_counter()
        lease = self._service.acquire_view(request.schema_ref, view=view)
        candidate_sets: list[tuple[CandidateHit, ...]] = []
        executed: list[RetrievalCapability] = []
        unavailable: list[RetrievalCapability] = []
        provider_request: dict[str, JsonValue] = {
            "text": request.intent,
            "context": dict(request.context),
            "exact_where": dict(request.exact_where),
            "filter_where": dict(request.filter_where),
        }
        try:
            for capability in configuration.capabilities:
                limit = configuration.limits[capability]
                try:
                    if capability is RetrievalCapability.EXACT:
                        hits = (
                            self._service.exact(
                                request.exact_where,
                                view=lease.view,
                                lease_id=lease.lease_id,
                                limit=limit,
                            )
                            if request.exact_where
                            else ()
                        )
                    elif capability is RetrievalCapability.FILTER:
                        hits = (
                            self._service.query(
                                request.filter_where,
                                view=lease.view,
                                lease_id=lease.lease_id,
                                limit=limit,
                            )
                            if request.filter_where
                            else ()
                        )
                    else:
                        hits = self._service.recall(
                            capability,
                            provider_request,
                            view=lease.view,
                            lease_id=lease.lease_id,
                            limit=limit,
                        )
                except CapabilityUnavailable:
                    unavailable.append(capability)
                    continue
                executed.append(capability)
                candidate_sets.append(tuple(hits))
            groups = self._service.organize(
                tuple(candidate_sets),
                view=lease.view,
                lease_id=lease.lease_id,
            )
            ranked = _rank_groups(groups, configuration.ranking_policy_ref)[: configuration.max_groups]
            representatives = _representatives(ranked, configuration.representative_policy)
            rendered = self._service.render(
                representatives,
                view=lease.view,
                lease_id=lease.lease_id,
                budget_chars=configuration.render_budget_chars,
            )
            retrieval = self._service.result(
                ranked,
                rendered,
                view=lease.view,
                lease_id=lease.lease_id,
                executed_capabilities=tuple(executed),
                unavailable_capabilities=tuple(unavailable),
            )
        finally:
            self._service.leases.release(lease.lease_id)
        candidate_count = len({hit.experience_id for candidates in candidate_sets for hit in candidates})
        return QueryExecutionResult(
            configuration.configuration_id,
            request,
            retrieval,
            tuple(group.group_key for group in retrieval.groups),
            representatives,
            candidate_count,
            (time.perf_counter() - started) * 1000,
        )


def _rank_groups(
    groups: tuple[GroupCandidate, ...],
    policy_ref: str,
) -> tuple[GroupCandidate, ...]:
    if policy_ref == "group-key@v1":
        return tuple(sorted(groups, key=lambda group: group.group_key))
    if policy_ref == "max-hit-score@v1":
        return tuple(
            sorted(
                groups,
                key=lambda group: (
                    -max((hit.score or 0.0 for hit in group.hits), default=0.0),
                    group.group_key,
                ),
            )
        )
    if policy_ref == "capability-priority@v1":
        priority = {
            RetrievalCapability.EXACT: 0,
            RetrievalCapability.FILTER: 1,
            RetrievalCapability.FUZZY: 2,
            RetrievalCapability.SEMANTIC: 3,
        }
        return tuple(
            sorted(
                groups,
                key=lambda group: (
                    min((priority[hit.capability] for hit in group.hits), default=4),
                    -max((hit.score or 0.0 for hit in group.hits), default=0.0),
                    group.group_key,
                ),
            )
        )
    raise RetrievalPolicyError(f"unsupported ranking policy {policy_ref!r}")


def _representatives(
    groups: tuple[GroupCandidate, ...],
    policy: RepresentativePolicy,
) -> tuple[str, ...]:
    selected: list[str] = []
    for group in groups:
        if policy is RepresentativePolicy.FIRST_MEMBER:
            selected.append(group.member_ids[0])
            continue
        hits = sorted(
            group.hits,
            key=lambda hit: (
                -(hit.score or 0.0),
                hit.experience_id,
                hit.capability.value,
            ),
        )
        selected.append(hits[0].experience_id if hits else group.member_ids[0])
    return tuple(selected)


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
        *,
        limit: int,
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
        return tuple(
            sorted(
                hits,
                key=lambda hit: (-(hit.score or 0.0), hit.experience_id),
            )[:limit]
        )


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


@dataclass(frozen=True)
class SemanticHit:
    experience_id: str
    score: float
    details: dict[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class SemanticSearchBackend(Protocol):
    provider_ref: str

    def search(self, text: str, *, limit: int) -> tuple[SemanticHit, ...]:
        """Return provider-ranked semantic hits by canonical Experience id."""


class SemanticCandidateProvider:
    """Adapt a versioned semantic backend to the CandidateProvider contract."""

    capability = RetrievalCapability.SEMANTIC

    def __init__(self, backend: SemanticSearchBackend) -> None:
        self._backend = backend
        self.provider_ref = _required_text(backend.provider_ref, "semantic provider_ref")

    def recall(
        self,
        request: dict[str, JsonValue],
        view: QueryView,
        *,
        limit: int,
    ) -> tuple[CandidateHit, ...]:
        text = str(request.get("text") or "").strip()
        if not text:
            return ()
        return tuple(
            CandidateHit(
                hit.experience_id,
                self.capability,
                float(hit.score),
                {"provider_ref": self.provider_ref, **hit.details},
            )
            for hit in self._backend.search(text, limit=limit)
        )


def provider_refs(
    providers: tuple[CandidateProvider, ...],
) -> dict[RetrievalCapability, str]:
    """Return the version pins exposed by configured providers."""
    refs: dict[RetrievalCapability, str] = {}
    for provider in providers:
        value = str(getattr(provider, "provider_ref", "") or "").strip()
        if value:
            refs[provider.capability] = value
    return refs


__all__ = [
    "LEXICAL_FUZZY_PROVIDER_REF",
    "RETRIEVAL_POLICY_VERSION",
    "LexicalFuzzyProvider",
    "QueryExecutionResult",
    "QueryRequest",
    "RepresentativePolicy",
    "RetrievalConfiguration",
    "RetrievalPolicyError",
    "RetrievalRunner",
    "SemanticCandidateProvider",
    "SemanticHit",
    "SemanticSearchBackend",
    "provider_refs",
]
