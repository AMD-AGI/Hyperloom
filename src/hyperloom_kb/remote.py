# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HTTP client and write-SDK facade for a shared Experience service."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import logging
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any

from hyperloom_kb.identity import derive_experience_id
from hyperloom_kb.schema import (
    Alternative,
    Change,
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    JsonScalar,
    JsonValue,
    Outcome,
    Provenance,
    RenderedRef,
)

log = logging.getLogger(__name__)

# Statuses about the request itself, which no retry changes: a proxy in front of a service answers an oversized
# record with 413, and that record must not hold back every write after it.
_PERMANENT_HTTP_STATUSES = frozenset(
    {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.CONFLICT,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        HTTPStatus.REQUEST_URI_TOO_LONG,
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    }
)
_SYNC_COUNTS = ("created", "unchanged", "skipped")
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class RemoteClientError(RuntimeError):
    """Raised for an unavailable, invalid, or rejecting Experience service."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class RemoteConfig:
    base_url: str
    token: str
    timeout_seconds: float = 120.0
    spool_root: Path = field(default_factory=lambda: Path("~/.cache/hyperloom/kb-spool").expanduser())

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.base_url:
            raise RemoteClientError("HYPERLOOM_KB_URL must be configured", retryable=False)
        if not self.token:
            raise RemoteClientError("HYPERLOOM_KB_TOKEN must be configured", retryable=False)
        if self.timeout_seconds <= 0:
            raise RemoteClientError("timeout_seconds must be positive", retryable=False)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        spool_root: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> RemoteConfig | None:
        values = os.environ if env is None else env
        base_url = str(values.get("HYPERLOOM_KB_URL") or "").strip()
        if not base_url:
            return None
        config = cls(base_url=base_url, token=str(values.get("HYPERLOOM_KB_TOKEN") or ""))
        if spool_root is not None:
            config = replace(config, spool_root=Path(spool_root))
        return config if timeout_seconds is None else replace(config, timeout_seconds=timeout_seconds)


@dataclass(frozen=True)
class RemoteReadResult:
    read_id: str
    status: str
    prompt_block: str
    rendered_refs: tuple[RenderedRef, ...]
    warnings: tuple[str, ...]
    experiences: tuple[dict[str, JsonValue], ...] = ()
    # ``{"ref", "bytes", "content"}`` for each ``change.content`` the prompt block references instead of inlining.
    contents: tuple[dict[str, JsonValue], ...] = ()


@dataclass(frozen=True)
class RemoteWriteResult:
    status: str
    experience_id: str
    content_hash: str = ""


@dataclass(frozen=True)
class ListPage:
    items: tuple[dict[str, JsonValue], ...]
    next_cursor: int
    has_more: bool


def _int(value: JsonValue, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemoteClientError(f"Experience service {name} is invalid")
    return value


class RemoteClient:
    """Authenticated client with fail-open reads and durably spooled writes."""

    def __init__(
        self,
        config: RemoteConfig,
        *,
        opener: Any = None,
    ) -> None:
        self.config = config
        # An environment proxy cannot reach this host's loopback, and urllib proxies it unless NO_PROXY lists it.
        self._opener = opener or (_DIRECT.open if is_loopback(config.base_url) else urllib.request.urlopen)
        # Set by a write the service could not take, cleared by one it took: until then every publish spools without
        # a request, so an unreachable service costs one timeout rather than one per write.
        self._spooling = False

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        data = None if body is None else json.dumps(body).encode()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.token}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            with exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RemoteClientError(
                f"Experience service returned HTTP {exc.code}: {detail}",
                retryable=exc.code not in _PERMANENT_HTTP_STATUSES,
            ) from exc
        except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
            raise RemoteClientError(f"Experience service request failed with {type(exc).__name__}") from exc
        if not isinstance(payload, dict):
            raise RemoteClientError("Experience service response is not an object")
        return payload

    def health(self) -> dict[str, JsonValue]:
        return self._request("GET", "/health")

    def read(
        self,
        decision: str,
        context: Mapping[str, JsonValue],
        *,
        outcome: str | None = None,
        limit: int | None = None,
        schema_ref: str | None = None,
        content_inline_limit: int | None = None,
        render_budget_chars: int | None = None,
    ) -> RemoteReadResult:
        body: dict[str, JsonValue] = {"decision": decision, "context": dict(context)}
        if outcome is not None:
            body["outcome"] = outcome
        if limit is not None:
            body["limit"] = limit
        if schema_ref is not None:
            body["schema_ref"] = schema_ref
        if content_inline_limit is not None:
            body["content_inline_limit"] = content_inline_limit
        if render_budget_chars is not None:
            body["render_budget_chars"] = render_budget_chars
        try:
            payload = self._request("POST", "/v1/read", body)
            refs = payload.get("rendered_refs")
            experiences = payload.get("experiences")
            warnings = payload.get("warnings")
            contents = payload.get("contents", [])
            if not (
                isinstance(refs, list)
                and isinstance(experiences, list)
                and isinstance(warnings, list)
                and isinstance(contents, list)
            ):
                raise RemoteClientError("Experience read response is invalid")
            return RemoteReadResult(
                read_id=str(payload.get("read_id") or ""),
                status=str(payload.get("status") or ""),
                prompt_block=str(payload.get("prompt_block") or ""),
                rendered_refs=tuple(RenderedRef.from_dict(item) for item in refs),
                warnings=tuple(str(item) for item in warnings),
                experiences=tuple(item for item in experiences if isinstance(item, dict)),
                contents=tuple(item for item in contents if isinstance(item, dict)),
            )
        except (RemoteClientError, ValueError) as exc:
            return RemoteReadResult(
                read_id="",
                status="unavailable",
                prompt_block="",
                rendered_refs=(),
                warnings=(str(exc),),
            )

    @staticmethod
    def _write_body(experience: Experience, declaration: ExperienceDeclaration | None) -> dict[str, JsonValue]:
        body: dict[str, JsonValue] = {"experience": experience.to_dict()}
        if declaration is not None:
            body["declaration"] = declaration.to_dict()
        return body

    def write(self, experience: Experience, *, declaration: ExperienceDeclaration | None = None) -> RemoteWriteResult:
        """Write one complete Experience, raising on any failure; ``declaration`` registers a schema the service lacks."""

        payload = self._request("PUT", f"/v1/experiences/{experience.id}", self._write_body(experience, declaration))
        return self._write_result(payload, experience.id)

    def publish(self, experience: Experience, *, declaration: ExperienceDeclaration | None = None) -> RemoteWriteResult:
        """Write one complete Experience; spool only retryable failures, with the declaration they need later.

        Once a write on this client fails retryably, later publishes spool without a request until ``flush_spool``
        delivers one.
        """

        if not self._spooling:
            try:
                return self.write(experience, declaration=declaration)
            except RemoteClientError as exc:
                if not exc.retryable:
                    raise
                self._spooling = True
        self._spool(experience.id, self._write_body(experience, declaration))
        return RemoteWriteResult("spooled", experience.id)

    def _spool(self, experience_id: str, body: dict[str, JsonValue]) -> None:
        root = self.config.spool_root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        digest = hashlib.sha256(experience_id.encode()).hexdigest()
        target = root / f"spool-{digest}.json"
        encoded = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            dir=root,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)

    def _reject_spooled(self, path: Path, reason: str) -> None:
        rejected = self.config.spool_root / "rejected"
        rejected.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(path, rejected / path.name)
        log.warning("Experience service rejected spooled write %s: %s", path.name, reason)

    @staticmethod
    def _write_result(
        payload: Mapping[str, JsonValue],
        experience_id: str,
    ) -> RemoteWriteResult:
        return RemoteWriteResult(
            status=str(payload.get("status") or ""),
            experience_id=str(payload.get("experience_id") or experience_id),
            content_hash=str(payload.get("content_hash") or ""),
        )

    def flush_spool(self) -> tuple[RemoteWriteResult, ...]:
        """Replay spooled writes until the service is unavailable again.

        Permanently rejected or unreadable spool files move to ``rejected/`` so they
        never block later writes.
        """

        root = self.config.spool_root
        if not root.exists():
            return ()
        results: list[RemoteWriteResult] = []
        for path in sorted(root.glob("spool-*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                experience = Experience.from_dict(value["experience"])
                declaration = (
                    None if "declaration" not in value else ExperienceDeclaration.from_dict(value["declaration"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._reject_spooled(path, f"unreadable spool file: {exc}")
                continue
            try:
                result = self.write(experience, declaration=declaration)
            except RemoteClientError as exc:
                if exc.retryable:
                    self._spooling = True
                    break
                self._reject_spooled(path, str(exc))
                continue
            self._spooling = False
            path.unlink()
            results.append(result)
        return tuple(results)

    @staticmethod
    def _page_query(after: int, limit: int, schema_ref: str | None) -> str:
        values: dict[str, str | int] = {"after": after, "limit": limit}
        if schema_ref is not None:
            values["schema_ref"] = schema_ref
        return urllib.parse.urlencode(values)

    def list_experiences(self, *, after: int = 0, limit: int = 100, schema_ref: str | None = None) -> ListPage:
        payload = self._request("GET", f"/v1/list?{self._page_query(after, limit, schema_ref)}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise RemoteClientError("Experience list response is invalid")
        return ListPage(
            items=tuple(item for item in items if isinstance(item, dict)),
            next_cursor=_int(payload.get("next_cursor"), "next_cursor"),
            has_more=payload.get("has_more") is True,
        )

    def export_page(self, *, after: int = 0, limit: int = 100, schema_ref: str | None = None) -> ListPage:
        """One page of complete records, ``{"sequence", "experience"}``, in the service's write order."""

        payload = self._request("GET", f"/v1/export?{self._page_query(after, limit, schema_ref)}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise RemoteClientError("Experience export response is invalid")
        return ListPage(
            items=tuple(item for item in items if isinstance(item, dict)),
            next_cursor=_int(payload.get("next_cursor"), "next_cursor"),
            has_more=payload.get("has_more") is True,
        )

    def push(self) -> dict[str, JsonValue]:
        """Ask this service to send every Experience written here and not yet pushed to its global KB."""

        return self._sync("/v1/push")

    def pull(self) -> dict[str, JsonValue]:
        """Ask this service to store every global-KB Experience of its declaration it does not hold yet."""

        return self._sync("/v1/pull")

    def _sync(self, path: str) -> dict[str, JsonValue]:
        # Each request handles one bounded batch, so no single request outlives the client timeout.
        totals = dict.fromkeys(_SYNC_COUNTS, 0)
        rejected: list[JsonValue] = []
        while True:
            report = self._request("POST", path, {})
            for key in _SYNC_COUNTS:
                totals[key] += _int(report.get(key, 0), key)
            batch_rejected = report.get("rejected")
            rejected.extend(batch_rejected if isinstance(batch_rejected, list) else ())
            if report.get("status") != "completed" or report.get("has_more") is not True:
                return {**report, **totals, "rejected": rejected}


class RemoteExperienceSession:
    """Lifecycle builder that publishes one complete remote Experience."""

    def __init__(
        self,
        client: RemoteClient,
        declaration: ExperienceDeclaration,
        experience: Experience,
    ) -> None:
        self._client = client
        self._declaration = declaration
        self._experience = experience

    @property
    def record(self) -> Experience:
        return self._experience

    def decide(
        self,
        *,
        reasoning: str,
        change: Change,
        alternatives: tuple[Alternative, ...] = (),
        rendered_refs: tuple[RenderedRef, ...] = (),
    ) -> Experience:
        self._experience = replace(
            self._experience,
            reasoning=reasoning,
            change=change,
            alternatives=tuple(alternatives),
            rendered_refs=tuple(rendered_refs),
        )
        return self._experience

    def complete(
        self,
        *,
        outcome: Outcome,
        reflection: str,
        completed_at: datetime | None = None,
    ) -> Experience:
        self._experience = replace(
            self._experience,
            status=ExperienceStatus.COMPLETE,
            outcome=outcome,
            reflection=reflection,
            completed_at=completed_at or datetime.now(timezone.utc),
        )
        self._declaration.validate(self._experience)
        return self._experience

    def publish(self) -> RemoteWriteResult:
        return self._client.publish(self._experience, declaration=self._declaration)


class RemoteExperienceKB:
    """Write-SDK-compatible facade backed by the Experience HTTP service."""

    enabled = True
    degraded = False
    reason = ""

    def __init__(
        self,
        client: RemoteClient,
        declaration: ExperienceDeclaration,
    ) -> None:
        self.client = client
        self.declaration = declaration

    @property
    def schema_ref(self) -> str:
        return self.declaration.schema_ref

    @classmethod
    def from_env(
        cls,
        declaration: ExperienceDeclaration,
        env: Mapping[str, str] | None = None,
    ) -> RemoteExperienceKB | None:
        config = RemoteConfig.from_env(env)
        if config is None:
            return None
        return cls(RemoteClient(config), declaration)

    def begin(
        self,
        *,
        run_id: str,
        seq: int,
        identity: dict[str, JsonScalar],
        objective: str,
        baseline_identity: dict[str, JsonScalar],
        baseline_value: float,
        provenance: Provenance,
        preconditions: tuple[str, ...] = (),
        parent_id: str = "",
        supersedes: str = "",
        created_at: datetime | None = None,
    ) -> RemoteExperienceSession:
        experience = Experience(
            id=derive_experience_id(provenance.producer, run_id, seq),
            run_id=run_id,
            seq=seq,
            created_at=created_at or datetime.now(timezone.utc),
            identity=identity,
            objective=objective,
            baseline_identity=baseline_identity,
            baseline_value=baseline_value,
            provenance=provenance,
            schema_ref=self.declaration.schema_ref,
            preconditions=tuple(preconditions),
            parent_id=parent_id,
            supersedes=supersedes,
        )
        self.declaration.validate(experience)
        return RemoteExperienceSession(self.client, self.declaration, experience)

    def read(
        self,
        decision: str,
        context: Mapping[str, JsonValue],
        *,
        outcome: str | None = None,
        limit: int | None = None,
    ) -> RemoteReadResult:
        return self.client.read(decision, context, outcome=outcome, limit=limit, schema_ref=self.schema_ref)


__all__ = [
    "ListPage",
    "RemoteClient",
    "RemoteClientError",
    "RemoteConfig",
    "RemoteExperienceKB",
    "RemoteExperienceSession",
    "RemoteReadResult",
    "RemoteWriteResult",
    "is_loopback",
]
