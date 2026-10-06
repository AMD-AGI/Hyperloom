"""Collection runtime: the configured Experience service, or a no-op."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

from hyperloom_kb.config import PACKAGED_DECLARATION, load_declaration
from hyperloom_kb.knowledge_read import ReadResult, ReadStatus
from hyperloom_kb.remote import (
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    RemoteExperienceKB,
)
from hyperloom_kb.schema import (
    Alternative,
    Change,
    ExperienceDeclaration,
    JsonValue,
    Outcome,
    RenderedRef,
)

log = logging.getLogger(__name__)


class NoOpExperienceSession:
    """Session-shaped object that deliberately records nothing."""

    @property
    def record(self) -> None:
        return None

    def decide(
        self,
        *,
        reasoning: str,
        change: Change,
        alternatives: tuple[Alternative, ...] = (),
        rendered_refs: tuple[RenderedRef, ...] = (),
    ) -> None:
        return None

    def complete(
        self,
        *,
        outcome: Outcome,
        reflection: str,
        completed_at: datetime | None = None,
        notes: Mapping[str, str] | None = None,
    ) -> None:
        return None

    def publish(self) -> None:
        return None


class NoOpExperienceKB:
    """Disabled collector returned when no Experience service is configured."""

    enabled = False
    schema_ref = ""

    def __init__(self, reason: str = "not_configured", *, degraded: bool = False) -> None:
        self.reason = reason
        self.degraded = degraded

    def begin(self, **_kwargs: Any) -> NoOpExperienceSession:
        return NoOpExperienceSession()

    def read(
        self,
        decision: str,
        context: Mapping[str, JsonValue],
    ) -> ReadResult:
        return ReadResult(ReadStatus.DISABLED, "", ())


ConfiguredExperienceKB: TypeAlias = NoOpExperienceKB | RemoteExperienceKB


def experience_kb_from_env(
    env: Mapping[str, str] | None = None,
    *,
    spool_root: Path | None = None,
    declaration: ExperienceDeclaration | None = None,
    timeout_seconds: float | None = None,
) -> ConfiguredExperienceKB:
    """Bootstrap the Experience service named by ``HYPERLOOM_KB_URL``, or a no-op facade.

    The collector writes ``declaration``'s schema, the packaged one by default; the service registers it on first write.
    """

    remote_config = RemoteConfig.from_env(
        os.environ if env is None else env, spool_root=spool_root, timeout_seconds=timeout_seconds
    )
    if remote_config is None:
        return NoOpExperienceKB()
    client = RemoteClient(remote_config)
    try:
        client.flush_spool()
    except (RemoteClientError, OSError) as exc:
        log.warning("Experience service spool flush deferred: %s", exc)
    return RemoteExperienceKB(client, declaration or load_declaration(PACKAGED_DECLARATION))


__all__ = [
    "ConfiguredExperienceKB",
    "NoOpExperienceKB",
    "NoOpExperienceSession",
    "experience_kb_from_env",
]
