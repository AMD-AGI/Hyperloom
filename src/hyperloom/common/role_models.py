# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which model, endpoint and wire protocol a role calls: ``--role-models`` / ``$HYPERLOOM_ROLE_MODELS``.

The spec is JSON, inline or in a file, keyed by role or by ``role@PHASE``::

    {"orchestration@KERNEL_AGENT": {"model": "glm-5.3-flash", "base_url": "http://127.0.0.1:4000"},
     "critic": {"model": "glm-5.3-flash", "base_url": "http://127.0.0.1:4000/v1", "protocol": "openai"}}

A role with no route keeps the model and endpoint it was launched with. A route never carries a
secret: ``api_key_env`` names the variable that holds the key. A route to another endpoint that
names none gets a placeholder key and no gateway headers, so the launch credential is never sent
to an endpoint it was not issued for.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROLE_MODELS_ENV = "HYPERLOOM_ROLE_MODELS"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOL_OPENAI = "openai"

#: The protocols each routable role speaks, its default first, and whether it may be routed per phase.
ROLE_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "orchestration": (PROTOCOL_ANTHROPIC,),
    "critic": (PROTOCOL_ANTHROPIC, PROTOCOL_OPENAI),
    "specialist": (PROTOCOL_ANTHROPIC,),
    "scorer": (PROTOCOL_OPENAI,),
}
PHASE_ROUTED_ROLES: frozenset[str] = frozenset({"orchestration"})

PLACEHOLDER_KEY = "hyperloom-route-no-key"
_CREDENTIAL_ENVS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
)
# Claude Code picks these for background and sub-agent requests; an endpoint serving one model
# answers them with that model.
_CLAUDE_MODEL_ENVS: tuple[str, ...] = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


class RoleModelsError(ValueError):
    """A ``--role-models`` spec that cannot be honoured."""


@dataclass(frozen=True)
class RoleModel:
    """The model, endpoint, protocol and key variable one role (or role in one phase) uses."""

    model: str
    base_url: str = ""
    protocol: str = PROTOCOL_ANTHROPIC
    api_key_env: str = ""

    def env(self, source: Mapping[str, str]) -> dict[str, str]:
        """``source`` with this route's endpoint and credential in place of the launch ones."""
        env = dict(source)
        if not self.base_url and not self.api_key_env:
            return env
        key = ""
        if self.api_key_env:
            key = (source.get(self.api_key_env) or "").strip()
            if not key:
                raise RoleModelsError(f"route for {self.model!r}: {self.api_key_env} is not set")
        if self.base_url:
            for name in _CREDENTIAL_ENVS:
                env.pop(name, None)
            key = key or PLACEHOLDER_KEY
        if self.protocol == PROTOCOL_OPENAI:
            if self.base_url:
                env["OPENAI_BASE_URL"] = self.base_url
            env["OPENAI_API_KEY"] = key
            return env
        if self.base_url:
            # Not CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: with it Claude Code 2.1.197 leaves the SDK MCP
            # tools (emit_intent, the context tools) out of every request.
            env["ANTHROPIC_BASE_URL"] = self.base_url
            for name in _CLAUDE_MODEL_ENVS:
                env[name] = self.model
        env["ANTHROPIC_API_KEY"] = key
        env["ANTHROPIC_AUTH_TOKEN"] = key
        return env

    def describe(self) -> dict[str, str]:
        """What the manifest records: the endpoint host only, never a key."""
        out = {"model": self.model, "protocol": self.protocol}
        if self.base_url:
            out["endpoint"] = urlsplit(self.base_url).hostname or "configured"
        if self.api_key_env:
            out["api_key_env"] = self.api_key_env
        return out


@dataclass(frozen=True)
class RoleModels:
    """Routes keyed by ``role`` or ``role@PHASE``; the phase key wins."""

    routes: Mapping[str, RoleModel] = field(default_factory=dict)

    def resolve(self, role: str, phase: str = "") -> RoleModel | None:
        phase = str(phase or "").strip().upper()
        if phase:
            routed = self.routes.get(f"{role}@{phase}")
            if routed is not None:
                return routed
        return self.routes.get(role)

    def has_role(self, role: str) -> bool:
        return any(key.split("@", 1)[0] == role for key in self.routes)

    def describe(self) -> dict[str, dict[str, str]]:
        return {key: route.describe() for key, route in sorted(self.routes.items())}

    def __bool__(self) -> bool:
        return bool(self.routes)


def _load_spec(spec: str) -> dict:
    text = spec.strip()
    if not text.startswith("{"):
        path = Path(text).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RoleModelsError(f"--role-models: {spec!r} is neither JSON nor a readable file ({exc})") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RoleModelsError(f"--role-models: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise RoleModelsError("--role-models: expected a JSON object keyed by role or role@PHASE")
    return data


def parse_role_models(spec: str | None, *, phases: Iterable[str] = ()) -> RoleModels:
    """Parse and validate a spec; empty or ``None`` routes nothing."""
    if not spec or not str(spec).strip():
        return RoleModels()
    known_phases = {str(p).upper() for p in phases}
    routes: dict[str, RoleModel] = {}
    for raw_key, raw in _load_spec(str(spec)).items():
        role, _, phase = str(raw_key).strip().partition("@")
        if role not in ROLE_PROTOCOLS:
            raise RoleModelsError(f"--role-models: unknown role {role!r} (routable: {sorted(ROLE_PROTOCOLS)})")
        if phase:
            phase = phase.upper()
            if role not in PHASE_ROUTED_ROLES:
                raise RoleModelsError(f"--role-models: {role!r} is routed per role only, not per phase")
            if known_phases and phase not in known_phases:
                raise RoleModelsError(f"--role-models: unknown phase {phase!r} (phases: {sorted(known_phases)})")
        if not isinstance(raw, dict) or not str(raw.get("model") or "").strip():
            raise RoleModelsError(f"--role-models: {raw_key!r} needs an object with a 'model'")
        unknown = set(raw) - {"model", "base_url", "protocol", "api_key_env"}
        if unknown:
            raise RoleModelsError(f"--role-models: {raw_key!r} has unknown field(s) {sorted(unknown)}")
        protocol = str(raw.get("protocol") or ROLE_PROTOCOLS[role][0]).strip().lower()
        if protocol not in ROLE_PROTOCOLS[role]:
            raise RoleModelsError(f"--role-models: {role!r} speaks {list(ROLE_PROTOCOLS[role])}, not {protocol!r}")
        routes[f"{role}@{phase}" if phase else role] = RoleModel(
            model=str(raw["model"]).strip(),
            base_url=str(raw.get("base_url") or "").strip().rstrip("/"),
            protocol=protocol,
            api_key_env=str(raw.get("api_key_env") or "").strip(),
        )
    return RoleModels(routes)


def role_models_from_env(*, phases: Iterable[str] = ()) -> RoleModels:
    return parse_role_models(os.environ.get(ROLE_MODELS_ENV), phases=phases)


__all__ = [
    "PHASE_ROUTED_ROLES",
    "PLACEHOLDER_KEY",
    "PROTOCOL_ANTHROPIC",
    "PROTOCOL_OPENAI",
    "ROLE_MODELS_ENV",
    "ROLE_PROTOCOLS",
    "RoleModel",
    "RoleModels",
    "RoleModelsError",
    "parse_role_models",
    "role_models_from_env",
]
