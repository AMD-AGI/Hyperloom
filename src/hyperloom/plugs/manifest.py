# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manifests: reading them, layering them, expanding variables, fingerprinting the result."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from hyperloom.plugs.errors import ManifestError
from hyperloom.plugs.registry import PlugRegistry

SCHEMA = "hyperloom.plugs/v1"
BASE_PROFILE = "local"
CUSTOM_PROFILE = "custom"

#: The sockets a hosted run locks. Nothing a session directory, the environment or the command line says reaches them.
LOCKABLE_SOCKETS = ("measurement", "sandbox", "credentials", "events", "gpu_token")

#: Layer names, in the order they are applied (4.2.2). The environment layer is added with the migration of the
#: environment variables; it sits between ``builtin`` and ``profile``.
LAYER_BUILTIN = "builtin"
LAYER_RESOLVED = "resolved"
LAYER_PROFILE = "profile"
LAYER_MANIFEST = "manifest"
LAYER_OVERRIDES = "overrides"
LAYER_CLI = "cli"
LAYER_LOCKED = "locked"

_PROFILES_DIR = Path(__file__).parent / "profiles"
_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ENTRY_KEYS = frozenset({"impl", "config"})


@dataclass(frozen=True)
class SocketSetting:
    """What fills one socket: the implementation, its configuration as written, and the layer that chose it."""

    impl: str
    config: Mapping[str, Any]
    layer: str


@dataclass(frozen=True)
class LockMismatch:
    """A session file named another implementation than the one the hosted run locked."""

    socket: str
    file_impl: str
    locked_impl: str


@dataclass(frozen=True)
class ResolvedManifest:
    """The outcome of layering: one setting per socket in use."""

    profile: str
    sockets: Mapping[str, SocketSetting]
    lock_mismatches: tuple[LockMismatch, ...] = field(default=())

    @property
    def is_builtin_default(self) -> bool:
        """Whether nothing but the built-in profile contributed."""
        return self.profile == BASE_PROFILE and all(s.layer == LAYER_BUILTIN for s in self.sockets.values())


def _load_text_document(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {str(path)!r}: {exc}") from exc
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ManifestError(f"manifest {str(path)!r} is not valid JSON: {exc}") from exc


def validate_document(doc: Any, source: str) -> dict[str, Any]:
    """Check one manifest document's shape; returns it as a dict."""
    if not isinstance(doc, Mapping):
        raise ManifestError(f"{source}: a manifest must be a mapping")
    if doc.get("schema") != SCHEMA:
        raise ManifestError(f"{source}: schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    sockets = doc.get("sockets", {})
    if not isinstance(sockets, Mapping):
        raise ManifestError(f"{source}: sockets must be a mapping")
    for name, entry in sockets.items():
        _validate_entry(name, entry, source)
    return dict(doc)


def _validate_entry(name: str, entry: Any, source: str) -> None:
    if not isinstance(entry, Mapping):
        raise ManifestError(f"{source}: socket {name!r} must be a mapping")
    extra = sorted(set(entry) - _ENTRY_KEYS)
    if extra:
        raise ManifestError(f"{source}: socket {name!r} has unknown keys {extra}; allowed: impl, config")
    impl = entry.get("impl")
    if impl is not None and not (isinstance(impl, str) and impl):
        raise ManifestError(f"{source}: socket {name!r} impl must be a non-empty string")
    if not isinstance(entry.get("config", {}), Mapping):
        raise ManifestError(f"{source}: socket {name!r} config must be a mapping")


def load_manifest_file(path: Path | str) -> dict[str, Any]:
    """Read and validate a manifest or override file (YAML or JSON)."""
    path = Path(path)
    return validate_document(_load_text_document(path), str(path))


def builtin_profile_names() -> tuple[str, ...]:
    """Names of the profiles that ship with Hyperloom."""
    return tuple(sorted(p.stem for p in _PROFILES_DIR.glob("*.json")))


def load_builtin_profile(name: str) -> dict[str, Any]:
    """Read and validate the shipped profile ``name``."""
    path = _PROFILES_DIR / f"{name}.json"
    if not _PROFILE_NAME.fullmatch(name) or not path.is_file():
        raise ManifestError(f"unknown profile {name!r}; available: {', '.join(builtin_profile_names())}")
    return validate_document(_load_text_document(path), f"profile {name}")


def _merge_config(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Dicts merge key by key, lists and scalars replace, ``None`` deletes the key."""
    merged = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping):
            existing = merged.get(key)
            merged[key] = _merge_config(existing if isinstance(existing, Mapping) else {}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_layer(
    current: Mapping[str, SocketSetting],
    doc: Mapping[str, Any],
    layer: str,
    registry: PlugRegistry,
) -> dict[str, SocketSetting]:
    """Lay ``doc`` over ``current``: later wins, ``impl`` replaces as a whole and takes its old ``config`` with it."""
    result = dict(current)
    known = registry.socket_names()
    for name, entry in doc.get("sockets", {}).items():
        if name not in known:
            raise ManifestError(f"{layer}: unknown socket {name!r}; defined: {', '.join(sorted(known))}")
        previous = result.get(name)
        impl = entry.get("impl") or (previous.impl if previous else None)
        if impl is None:
            raise ManifestError(f"{layer}: socket {name!r} sets no impl and no earlier layer chose one")
        keeps_config = previous is not None and previous.impl == impl
        base_config = previous.config if keeps_config else {}
        result[name] = SocketSetting(impl, _merge_config(base_config, entry.get("config", {})), layer)
    return result


def cli_document(plugs: Iterable[str], configs: Iterable[str]) -> dict[str, Any]:
    """Build the command-line layer from ``--plug SOCKET=IMPL`` and ``--plug-config SOCKET.KEY=VALUE`` values."""
    sockets: dict[str, dict[str, Any]] = {}
    for item in plugs:
        socket, sep, impl = item.partition("=")
        if not (sep and socket.strip() and impl.strip()):
            raise ManifestError(f"--plug expects SOCKET=IMPL, got {item!r}")
        sockets.setdefault(socket.strip(), {})["impl"] = impl.strip()
    for item in configs:
        target, sep, raw = item.partition("=")
        socket, dot, key = target.partition(".")
        if not (sep and dot and socket.strip() and key.strip()):
            raise ManifestError(f"--plug-config expects SOCKET.KEY=VALUE, got {item!r}")
        sockets.setdefault(socket.strip(), {}).setdefault("config", {})[key.strip()] = _json_or_text(raw)
    return {"schema": SCHEMA, "sockets": sockets}


def _json_or_text(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _validated_lock(locked: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    for name, entry in locked.items():
        if name not in LOCKABLE_SOCKETS:
            raise ManifestError(f"socket {name!r} cannot be locked; lockable: {', '.join(LOCKABLE_SOCKETS)}")
        _validate_entry(name, entry, "lock")
        if not entry.get("impl"):
            raise ManifestError(f"lock: socket {name!r} must name an impl")
    return dict(locked)


def apply_lock(
    current: Mapping[str, SocketSetting],
    locked: Mapping[str, Mapping[str, Any]],
    resumed: Mapping[str, Any] | None,
) -> tuple[dict[str, SocketSetting], tuple[LockMismatch, ...]]:
    """Replace the locked sockets outright, and report each that ``resumed`` (a session file) chose differently."""
    result = dict(current)
    mismatches: list[LockMismatch] = []
    file_sockets = (resumed or {}).get("sockets", {})
    for name, entry in _validated_lock(locked).items():
        file_impl = (file_sockets.get(name) or {}).get("impl")
        if file_impl and file_impl != entry["impl"]:
            mismatches.append(LockMismatch(name, file_impl, entry["impl"]))
        result[name] = SocketSetting(entry["impl"], copy.deepcopy(dict(entry.get("config", {}))), LAYER_LOCKED)
    return result, tuple(mismatches)


@dataclass(frozen=True)
class ManifestSources:
    """Every input to the layering, each optional."""

    profile: str | None = None
    manifest_path: Path | str | None = None
    overrides_path: Path | str | None = None
    cli_plugs: tuple[str, ...] = ()
    cli_configs: tuple[str, ...] = ()
    #: A previous run's resolved file, already validated; the base of the profile layer on a resume.
    resumed: Mapping[str, Any] | None = None
    #: The hosted run's lock: ``{socket: {"impl": ..., "config": {...}}}``, applied last.
    locked: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def resolve_manifest(registry: PlugRegistry, sources: ManifestSources = ManifestSources()) -> ResolvedManifest:
    """Layer every source, in order, into the manifest a host starts from."""
    settings: dict[str, SocketSetting] = {}
    profile = BASE_PROFILE
    # The explicit layers 3 and up, each as (layer name, document, profile name it sets).
    stages: list[tuple[str, dict[str, Any], str | None]] = [
        (LAYER_BUILTIN, load_builtin_profile(BASE_PROFILE), None),
    ]
    if sources.resumed is not None:
        stages.append((LAYER_RESOLVED, dict(sources.resumed), str(sources.resumed.get("profile") or CUSTOM_PROFILE)))
    if sources.profile:
        stages.append((LAYER_PROFILE, load_builtin_profile(sources.profile), sources.profile))
    if sources.manifest_path:
        doc = load_manifest_file(sources.manifest_path)
        stages.append((LAYER_MANIFEST, doc, str(doc.get("profile") or CUSTOM_PROFILE)))
    if sources.overrides_path:
        stages.append((LAYER_OVERRIDES, load_manifest_file(sources.overrides_path), None))
    if sources.cli_plugs or sources.cli_configs:
        stages.append((LAYER_CLI, cli_document(sources.cli_plugs, sources.cli_configs), None))
    for layer, doc, named in stages:
        settings = apply_layer(settings, doc, layer, registry)
        profile = named or profile
    mismatches: tuple[LockMismatch, ...] = ()
    if sources.locked:
        settings, mismatches = apply_lock(settings, sources.locked, sources.resumed)
    return ResolvedManifest(profile=profile, sockets=settings, lock_mismatches=mismatches)


def expand_variables(value: Any, env: Mapping[str, str], where: str) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in every string of ``value``; a missing variable is an error."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is None:
                raise ManifestError(f"{where}: environment variable {name} is not set and has no default")
            return default

        return _VARIABLE.sub(_sub, value)
    if isinstance(value, Mapping):
        return {k: expand_variables(v, env, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [expand_variables(v, env, where) for v in value]
    return value


def expand_manifest(manifest: ResolvedManifest, env: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Each socket's configuration with its variables expanded against ``env`` (default: the process environment)."""
    env = os.environ if env is None else env
    return {name: expand_variables(dict(s.config), env, f"socket {name}") for name, s in manifest.sockets.items()}


def fingerprint(profile: str, rows: Mapping[str, Mapping[str, Any]]) -> str:
    """First 16 hex digits of the sha256 of the canonical JSON of ``{profile, sockets: {name: {impl, origin, config}}}``."""
    payload = {"profile": profile, "sockets": {name: dict(row) for name, row in rows.items()}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
