# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The normalised identity of a staging (``inference-suite-normalised/1``): what two hosts compare to tell whether their
AgentX client and eval are the same software.

A staging's file manifest names bytes, and some of those bytes hold the path the staging was built at (a ``RECORD``, a
script's shebang), so two hosts never stage byte-identical trees. The normalised identity leaves every path out: the
InferenceX pin, the client interpreter's version and the sha256 of the build it was unpacked from, and each installed
distribution as its canonical name, version and the sha256 of the wheel it was installed from.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from hyperloom_common.contract_fields import ContractFields, field_names

NORMALISED_SCHEMA = "inference-suite-normalised/1"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_NAME_SEPARATORS = re.compile(r"[-_.]+")
_SCHEMA_KEY = "schema"


class NormalisedError(ValueError):
    """A normalised identity breaks its contract."""


_FIELDS = ContractFields(NormalisedError)


def canonical_name(name: str) -> str:
    """The distribution name as PEP 503 compares it."""
    return _NAME_SEPARATORS.sub("-", name).lower()


@dataclass(frozen=True)
class PackageIdentity:
    """One installed distribution: its canonical name, version and the sha256 of the wheel installed."""

    name: str
    version: str
    wheel_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonical_name(_FIELDS.text(self.name, "a package name")))
        _FIELDS.text(self.version, f"the version of {self.name}")
        _FIELDS.sha256(self.wheel_sha256, f"the wheel sha256 of {self.name}")

    def describe(self) -> str:
        return f"{self.version} (wheel sha256 {self.wheel_sha256})"

    def to_document(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "wheel_sha256": self.wheel_sha256}

    @classmethod
    def from_document(cls, document: object) -> PackageIdentity:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "a package identity")
        return cls(*(fields.get(key) for key in keys))


@dataclass(frozen=True)
class NormalisedManifest:
    """The InferenceX pin, the client interpreter and every installed distribution, sorted by name."""

    inferencex: str
    interpreter_version: str
    interpreter_sha256: str
    packages: tuple[PackageIdentity, ...]

    def __post_init__(self) -> None:
        _FIELDS.matching(_COMMIT, self.inferencex, "the InferenceX pin")
        _FIELDS.text(self.interpreter_version, "the interpreter version")
        _FIELDS.sha256(self.interpreter_sha256, "the interpreter build sha256")
        packages = tuple(sorted(self.packages, key=lambda package: package.name))
        for package in packages:
            if not isinstance(package, PackageIdentity):
                raise NormalisedError(f"a package must be a PackageIdentity, got {package!r}")
        _FIELDS.unique([package.name for package in packages], "package names")
        object.__setattr__(self, "packages", packages)

    def with_packages(self, extra: Sequence[PackageIdentity]) -> NormalisedManifest:
        return NormalisedManifest(
            self.inferencex, self.interpreter_version, self.interpreter_sha256, (*self.packages, *extra)
        )

    def to_document(self) -> dict[str, object]:
        return {
            _SCHEMA_KEY: NORMALISED_SCHEMA,
            "inferencex": self.inferencex,
            "interpreter_version": self.interpreter_version,
            "interpreter_sha256": self.interpreter_sha256,
            "packages": [package.to_document() for package in self.packages],
        }

    @classmethod
    def from_document(cls, document: object) -> NormalisedManifest:
        keys = field_names(cls)
        fields = _FIELDS.known(document, (_SCHEMA_KEY, *keys), "the normalised manifest")
        if fields.get(_SCHEMA_KEY) != NORMALISED_SCHEMA:
            raise NormalisedError(
                f"the normalised manifest schema is {fields.get(_SCHEMA_KEY)!r}, not {NORMALISED_SCHEMA}"
            )
        packages = tuple(
            PackageIdentity.from_document(item) for item in _FIELDS.items(fields.get("packages"), "packages")
        )
        return cls(
            fields.get("inferencex"), fields.get("interpreter_version"), fields.get("interpreter_sha256"), packages
        )

    def sha256(self) -> str:
        """The sha256 of the canonical document: equal for two stagings exactly when they are the same software."""
        return hashlib.sha256(
            json.dumps(self.to_document(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def differences(self, other: NormalisedManifest) -> tuple[str, ...]:
        """One line per thing that differs from ``other``, each starting with what it names: ``inferencex``,
        ``interpreter`` or the package's name. Empty when the two are the same."""
        return tuple(self._differences(other))

    def _differences(self, other: NormalisedManifest) -> Iterator[str]:
        if self.inferencex != other.inferencex:
            yield f"inferencex {self.inferencex} here, {other.inferencex} there"
        mine = (self.interpreter_version, self.interpreter_sha256)
        theirs = (other.interpreter_version, other.interpreter_sha256)
        if mine != theirs:
            yield f"interpreter {mine[0]} (build sha256 {mine[1]}) here, {theirs[0]} (build sha256 {theirs[1]}) there"
        here = {package.name: package for package in self.packages}
        there = {package.name: package for package in other.packages}
        for name in sorted(here.keys() | there.keys()):
            if here.get(name) != there.get(name):
                yield f"{name} {_described(here.get(name))} here, {_described(there.get(name))} there"


def _described(package: PackageIdentity | None) -> str:
    return "not installed" if package is None else package.describe()


__all__ = ["NORMALISED_SCHEMA", "NormalisedError", "NormalisedManifest", "PackageIdentity", "canonical_name"]
