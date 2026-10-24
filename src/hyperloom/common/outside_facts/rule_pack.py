# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule pack: facts about projects outside this repository, read from one YAML document and checked against
their own examples before anything uses them.

The pack is data. ``HYPERLOOM_RULE_PACK`` names the file a session reads, and without it the reference copy beside this
module is read, so a fact that changed upstream is changed by publishing a pack and not by changing code. The file is
read once per process: its bytes are hashed and parsed together, so the digest names exactly what was parsed.

A pack that cannot be used is refused, with an error that names the file and the entry at fault, and nothing falls back
to an empty or a default pack, because an unreadable fact must never read as "no constraint". Refused are a file that is
missing or is not one YAML document, a repeated key, a YAML alias or merge key, an entry of an unknown kind or without
its ``project``, ``reason`` and ``examples``, a pattern that does not compile or fails one of its own examples, and an
entry the code reads that the pack lacks. The kinds are a closed list: a rule that needs one more is a change to this
module with its own evaluator and tests, never a new pack.

An entry may also declare ``image_check``: the checks the environment a session runs in can answer about the fact
(``hyperloom.common.outside_facts.image_check_spec``). They are read with the entry, a reference to another entry's name
is resolved here, and each check is pinned to its own examples, so a check that refuses what the real tool prints, or
names a unit the pack does not hold, is refused with the entry.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import yaml

from hyperloom.common.outside_facts.image_check_eval import check_examples
from hyperloom.common.outside_facts.image_check_spec import CheckSpec, ImageCheckSpecError, parse_checks

__all__ = [
    "DEFAULT_PACK_PATH",
    "PACK_ENV",
    "EnvDefault",
    "Kind",
    "RulePack",
    "RulePackError",
    "RulePackIdentity",
    "default_pack",
    "load_pack",
    "rule_pack_identity",
]

#: The session input that names the pack file.
PACK_ENV = "HYPERLOOM_RULE_PACK"
#: The pack a session reads when ``PACK_ENV`` is not set; it ships with the package.
DEFAULT_PACK_PATH = Path(__file__).parent / "packs" / "default.yaml"


class RulePackError(RuntimeError):
    """The pack cannot be used; the message names the file and, when one entry is at fault, that entry."""


class RulePackIdentity(TypedDict):
    """What a session records of the pack it ran with: the pack's own version and the digest of its bytes."""

    version: str
    digest: str


class Kind(Enum):
    """The closed list of entry kinds. The last two are reserved: no rule reads one yet, so a pack that holds one is
    refused until the rule that does arrives with its evaluator."""

    NAMES = "names"
    LOG_REGEX = "log_regex"
    ENV_DEFAULT = "env_default"
    ARG_VALUE = "arg_value"
    KERNEL_TABLE = "kernel_table"


class EnvDefault(NamedTuple):
    """An environment variable and the value a launch sets."""

    name: str
    value: str


@dataclass(frozen=True)
class Entry:
    """One fact of the pack, after its fields and examples have been checked."""

    kind: Kind
    body: Mapping[str, Any]
    project: str = ""
    image_checks: tuple[CheckSpec, ...] = ()


_COMMON_FIELDS = frozenset({"kind", "project", "reason", "examples"})
_IMAGE_CHECK_FIELD = "image_check"
_NAMES_FIELD = "names"
_PATTERN_FIELD = "pattern"
_ENV_FIELD = "env"
_VALUE_FIELD = "value"
_MATCH = "match"
_NO_MATCH = "no_match"
_MERGE_TAG = "tag:yaml.org,2002:merge"


@dataclass(frozen=True)
class RulePack:
    """A loaded pack. The code reads an entry through the accessor of its kind; an entry it asks for that the pack does
    not hold, or holds as another kind, is refused."""

    path: Path
    version: str
    digest: str
    entries: Mapping[str, Entry]

    def names(self, entry_id: str, *roles: str) -> dict[str, str]:
        """The names the entry holds under ``roles``; a role it lacks is refused."""
        held = self._entry(entry_id, Kind.NAMES).body[_NAMES_FIELD]
        lacking = [role for role in roles if role not in held]
        if lacking:
            raise RulePackError(f"{self._where(entry_id)}: has no name for the role(s) {lacking}, which the code reads")
        return {role: held[role] for role in roles}

    def pattern(self, entry_id: str) -> str:
        """The regular-expression source of a ``log_regex`` entry."""
        return self._entry(entry_id, Kind.LOG_REGEX).body[_PATTERN_FIELD]

    def env_default(self, entry_id: str) -> EnvDefault:
        """The variable and value of an ``env_default`` entry."""
        body = self._entry(entry_id, Kind.ENV_DEFAULT).body
        return EnvDefault(body[_ENV_FIELD], body[_VALUE_FIELD])

    def _where(self, entry_id: str) -> str:
        return f"rule pack {self.path}: entry {entry_id!r}"

    def _entry(self, entry_id: str, kind: Kind) -> Entry:
        entry = self.entries.get(entry_id)
        if entry is None:
            raise RulePackError(f"rule pack {self.path}: has no entry {entry_id!r}, which the code reads")
        if entry.kind is not kind:
            raise RulePackError(
                f"{self._where(entry_id)}: is a {entry.kind.value} entry, the code reads a {kind.value}"
            )
        return entry


def _text(where: str, what: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RulePackError(f"{where}: {what} must be a non-empty string (quote a value YAML would read as a number)")
    return value


def _lines(where: str, what: str, value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RulePackError(f"{where}: {what} must be a list with at least one example")
    return [_text(where, what, line) for line in value]


def _check_names(where: str, body: Mapping[str, Any], examples: object) -> None:
    held = body[_NAMES_FIELD]
    if not isinstance(held, dict) or not held:
        raise RulePackError(f"{where}: {_NAMES_FIELD} must be a non-empty mapping of role to name")
    for role, name in held.items():
        _text(where, f"the role name {role!r}", role)
        _text(where, f"the name for role {role!r}", name)
    _lines(where, "examples (where the names are used)", examples)


def _check_env_default(where: str, body: Mapping[str, Any], examples: object) -> None:
    _text(where, _ENV_FIELD, body[_ENV_FIELD])
    _text(where, _VALUE_FIELD, body[_VALUE_FIELD])
    _lines(where, "examples (where the default is set)", examples)


def _check_log_regex(where: str, body: Mapping[str, Any], examples: object) -> None:
    source = _text(where, _PATTERN_FIELD, body[_PATTERN_FIELD])
    try:
        compiled = re.compile(source)
    except re.error as exc:
        raise RulePackError(f"{where}: pattern {source!r} does not compile: {exc}") from exc
    if not isinstance(examples, dict) or set(examples) != {_MATCH, _NO_MATCH}:
        raise RulePackError(f"{where}: examples must hold the lists {_MATCH!r} and {_NO_MATCH!r}")
    for line in _lines(where, f"examples.{_MATCH}", examples[_MATCH]):
        if not compiled.search(line):
            raise RulePackError(f"{where}: the must-match example {line!r} does not match pattern {source!r}")
    for line in _lines(where, f"examples.{_NO_MATCH}", examples[_NO_MATCH]):
        if compiled.search(line):
            raise RulePackError(f"{where}: the must-not-match example {line!r} matches pattern {source!r}")


#: The fields each readable kind holds beside the common ones, and the check of its fields and examples.
_READABLE: dict[Kind, tuple[frozenset[str], Callable[[str, Mapping[str, Any], object], None]]] = {
    Kind.NAMES: (frozenset({_NAMES_FIELD}), _check_names),
    Kind.LOG_REGEX: (frozenset({_PATTERN_FIELD}), _check_log_regex),
    Kind.ENV_DEFAULT: (frozenset({_ENV_FIELD, _VALUE_FIELD}), _check_env_default),
}


def _parse_entry(where: str, raw: object) -> Entry:
    if not isinstance(raw, dict):
        raise RulePackError(f"{where}: must be a mapping")
    try:
        kind = Kind(raw.get("kind"))
    except ValueError:
        raise RulePackError(
            f"{where}: kind {raw.get('kind')!r} is not one of {[kind.value for kind in Kind]}"
        ) from None
    if kind not in _READABLE:
        raise RulePackError(f"{where}: kind {kind.value} is reserved; no rule reads one yet")
    own_fields, check = _READABLE[kind]
    expected = _COMMON_FIELDS | own_fields
    if set(raw) - {_IMAGE_CHECK_FIELD} != expected:
        raise RulePackError(
            f"{where}: a {kind.value} entry holds exactly {sorted(expected)} "
            f"(and may add {_IMAGE_CHECK_FIELD!r}); found {sorted(map(str, raw))}"
        )
    _text(where, "project", raw["project"])
    _text(where, "reason", raw["reason"])
    body = {name: raw[name] for name in own_fields}
    check(where, body, raw["examples"])
    return Entry(kind, body, raw["project"])


def _refuse_ambiguous_mapping(path: Path, node: yaml.MappingNode, where: str) -> None:
    if any(key.tag == _MERGE_TAG for key, _ in node.value):
        raise RulePackError(f"rule pack {path}: {where} uses a YAML merge key, which a pack does not allow")
    keys = [key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise RulePackError(f"rule pack {path}: {where}: the key(s) {repeated} are written twice")


def _children(node: yaml.Node, trail: str) -> list[tuple[yaml.Node, str]]:
    if isinstance(node, yaml.SequenceNode):
        return [(item, trail) for item in node.value]
    children: list[tuple[yaml.Node, str]] = []
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            label = key.value if isinstance(key, yaml.ScalarNode) else ""
            children += [(key, trail), (value, ".".join(filter(None, (trail, label))))]
    return children


def _refuse_ambiguous_yaml(path: Path, root: yaml.Node | None) -> None:
    """A pack is flat data. YAML lets a repeated key win without a word, and lets an alias or a merge key stand in for a
    field, so a pack that uses any of them holds facts that no reader of the file can be sure of."""
    seen: set[int] = set()
    pending = [] if root is None else [(root, "")]
    while pending:
        node, trail = pending.pop()
        if id(node) in seen:
            raise RulePackError(
                f"rule pack {path}: uses a YAML alias, which a pack does not allow; write the value out"
            )
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            _refuse_ambiguous_mapping(path, node, trail or "the top level")
        pending.extend(_children(node, trail))


def _name_lookup(entries: Mapping[str, Entry]) -> Callable[[str], str]:
    """Reads ``<entry>.<role>`` as the name an entry of the pack holds: a role of a names entry, or the variable or the
    value of an env_default entry."""

    def lookup(reference: str) -> str:
        entry_id, _, role = reference.partition(".")
        entry = entries.get(entry_id)
        if entry is None or entry.kind not in (Kind.NAMES, Kind.ENV_DEFAULT):
            raise ImageCheckSpecError(f"{reference!r} names no names or env_default entry")
        held = entry.body[_NAMES_FIELD] if entry.kind is Kind.NAMES else entry.body
        if role not in held or not isinstance(held[role], str):
            raise ImageCheckSpecError(f"{reference!r}: entry {entry_id!r} holds no name for {role!r}")
        return held[role]

    return lookup


def _with_image_checks(path: Path, facts: Mapping[str, Any], entries: Mapping[str, Entry]) -> dict[str, Entry]:
    lookup = _name_lookup(entries)
    checked = {}
    for entry_id, entry in entries.items():
        if _IMAGE_CHECK_FIELD not in facts[entry_id]:
            checked[entry_id] = entry
            continue
        try:
            specs = parse_checks(facts[entry_id][_IMAGE_CHECK_FIELD], lookup)
            for index, spec in enumerate(specs):
                _pin_to_examples(index, spec)
        except ImageCheckSpecError as exc:
            raise RulePackError(f"rule pack {path}: entry {entry_id!r}: {exc}") from exc
        checked[entry_id] = replace(entry, image_checks=specs)
    return checked


def _pin_to_examples(index: int, spec: CheckSpec) -> None:
    try:
        check_examples(spec)
    except ImageCheckSpecError as exc:
        raise ImageCheckSpecError(f"{_IMAGE_CHECK_FIELD}[{index}]: {exc}") from exc


def load_pack(path: Path) -> RulePack:
    """Read, parse and check the pack at ``path``; raise ``RulePackError`` rather than return a pack that is partial."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RulePackError(f"rule pack {path}: cannot be read: {exc}") from exc
    try:
        _refuse_ambiguous_yaml(path, yaml.compose(raw, Loader=yaml.SafeLoader))
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RulePackError(f"rule pack {path}: is not one valid YAML document: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"version", "facts"}:
        raise RulePackError(f"rule pack {path}: must be a mapping with exactly the keys 'version' and 'facts'")
    version = _text(f"rule pack {path}", "version", document["version"])
    facts = document["facts"]
    if not isinstance(facts, dict) or not facts:
        raise RulePackError(f"rule pack {path}: 'facts' must be a non-empty mapping of entry id to entry")
    entries = {}
    for entry_id, raw_entry in facts.items():
        entries[_text(f"rule pack {path}", "an entry id", entry_id)] = _parse_entry(
            f"rule pack {path}: entry {entry_id!r}", raw_entry
        )
    entries = _with_image_checks(path, facts, entries)
    return RulePack(path, version, "sha256:" + hashlib.sha256(raw).hexdigest(), entries)


@functools.cache
def default_pack() -> RulePack:
    """The pack this process reads: the file ``PACK_ENV`` names, or the reference copy. Loaded once, on first use."""
    configured = os.environ.get(PACK_ENV)
    if configured is not None and not configured.strip():
        raise RulePackError(f"{PACK_ENV} is set but empty; unset it to read the reference pack, or name a pack file")
    return load_pack(Path(configured) if configured else DEFAULT_PACK_PATH)


def rule_pack_identity() -> RulePackIdentity:
    """The version and digest of the pack this process reads, for the session manifest and the knowledge base."""
    pack = default_pack()
    return {"version": pack.version, "digest": pack.digest}
