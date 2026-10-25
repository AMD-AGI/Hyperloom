# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The signed record of one native build: ``manifest.json`` for a build that made its outputs, ``failed.json`` for a
deterministic compile error.

A ready record lists each output that differs from the image's own file (path below the package, sha256, size, mode),
the toolchain facts the recipe reported, when it was built, ``build_seconds`` (CPU seconds of the compile, the unit build
accounting is kept in) and ``wall_seconds``. A failed record carries the failure instead of outputs, so every test of the
same key gets the same ``compile_error`` without a rebuild.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.measure.docs import Reader, canonical_json, parse_json_object, relative_path, string_map
from hyperloom.measure.jit.native.key import BuildInputs

__all__ = ["FAILED", "LOG_NAME", "READY", "RECORD_FILES", "BuildRecord", "OutputFile"]

RECORD_SCHEMA = "hyperloom.native-build-record/v1"
READY = "ready"
FAILED = "failed"
#: The record and signature file of each status; the record is the last file written into an entry.
RECORD_FILES: Mapping[str, tuple[str, str]] = {
    READY: ("manifest.json", "manifest.sig"),
    FAILED: ("failed.json", "failed.sig"),
}
LOG_NAME = "log.txt"
EXECUTABLE, PLAIN = "0755", "0644"
_HEX = re.compile(r"[0-9a-f]{64}")
_OUTPUTS = "outputs"


@dataclass(frozen=True)
class OutputFile:
    path: str
    sha256: str
    size: int
    mode: str

    def document(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size, "mode": self.mode}

    @classmethod
    def from_document(cls, value: Any, path: str) -> OutputFile:
        reader = Reader(value, path)
        output = cls(
            relative_path(reader.raw("path"), reader.at("path")),
            str(reader.text("sha256", pattern=_HEX)),
            int(reader.integer("size", minimum=0) or 0),
            str(reader.choice("mode", (EXECUTABLE, PLAIN))),
        )
        reader.done()
        return output


@dataclass(frozen=True)
class BuildRecord:
    """One build under :attr:`key`; :attr:`key` is stated in the record and must equal the hash of :attr:`inputs`."""

    key: str
    inputs: BuildInputs
    status: str
    outputs: tuple[OutputFile, ...]
    toolchain: Mapping[str, str]
    built_at: str
    build_seconds: float
    wall_seconds: float
    failure: str = ""

    def document(self) -> dict[str, object]:
        return {
            "schema": RECORD_SCHEMA,
            "key": self.key,
            "inputs": self.inputs.document(),
            "status": self.status,
            _OUTPUTS: [output.document() for output in self.outputs],
            "toolchain": dict(self.toolchain),
            "built_at": self.built_at,
            "build_seconds": self.build_seconds,
            "wall_seconds": self.wall_seconds,
            "failure": self.failure,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.document()).encode("utf-8")

    @classmethod
    def parse(cls, raw: bytes) -> BuildRecord:
        """Read a record; :class:`~hyperloom.measure.docs.DocumentError` for anything outside the schema."""
        reader = Reader(parse_json_object(raw, what="build record"), "record")
        reader.choice("schema", (RECORD_SCHEMA,))
        record = cls(
            key=str(reader.text("key", pattern=_HEX)),
            inputs=reader.section("inputs", BuildInputs.from_document),
            status=str(reader.choice("status", (READY, FAILED))),
            outputs=reader.sequence(_OUTPUTS, OutputFile.from_document),
            toolchain=reader.section("toolchain", string_map),
            built_at=str(reader.text("built_at")),
            build_seconds=float(reader.number("build_seconds", minimum=0) or 0.0),
            wall_seconds=float(reader.number("wall_seconds", minimum=0) or 0.0),
            failure=reader.string("failure"),
        )
        reader.done()
        return record
