# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's plugs.resolved.json (4.2.2, the resume paragraph)."""

from __future__ import annotations

import json

import pytest

from hyperloom.plugs import (
    Implementation,
    ManifestError,
    ManifestSources,
    PlugHost,
    builtin_registry,
    resolve_manifest,
)
from hyperloom.plugs.resolved import RESOLVED_FILENAME, read_resolved, write_resolved


class _Spy:
    def __init__(self):
        self.events = []

    def emit(self, type, data):
        self.events.append((type, dict(data)))

    def flush(self):
        pass


def _registry(spy):
    registry = builtin_registry()
    registry.register(Implementation("events", "spy", lambda ctx: spy))
    registry.register(Implementation("measurement", "hosted", lambda ctx: "hosted-measurement"))
    registry.register(Implementation("measurement", "local", lambda ctx: "local-measurement"))
    registry.register(Implementation("sandbox", "e2b", lambda ctx: "e2b-sandbox"))
    registry.register(Implementation("sandbox", "local", lambda ctx: "local-sandbox"))
    return registry


def _start(registry, session_dir, **sources):
    manifest = resolve_manifest(registry, ManifestSources(resumed=read_resolved(session_dir), **sources))
    host = PlugHost(manifest, registry, session_dir=session_dir)
    host.start()
    write_resolved(session_dir, manifest, host.fingerprint())
    return manifest, host


def test_a_socket_resumes_from_the_file(tmp_path) -> None:
    registry = _registry(_Spy())
    _start(registry, tmp_path, cli_plugs=("sandbox=local",))[1].close()
    manifest, host = _start(registry, tmp_path)
    assert host.get("sandbox") == "local-sandbox"
    assert manifest.sockets["sandbox"].layer == "resolved"


def test_a_command_line_layer_still_overrides_the_file_on_resume(tmp_path) -> None:
    registry = _registry(_Spy())
    _start(registry, tmp_path, cli_plugs=("sandbox=local",))[1].close()
    _, host = _start(registry, tmp_path, cli_plugs=("sandbox=e2b",))
    assert host.get("sandbox") == "e2b-sandbox"


def test_a_damaged_resolved_file_is_refused_rather_than_ignored(tmp_path) -> None:
    (tmp_path / RESOLVED_FILENAME).write_text("{oops", encoding="utf-8")
    with pytest.raises(ManifestError, match="fix or remove the file to resume"):
        read_resolved(tmp_path)


def test_a_resolved_file_naming_an_undefined_socket_is_refused(tmp_path) -> None:
    (tmp_path / RESOLVED_FILENAME).write_text(
        json.dumps({"schema": "hyperloom.plugs/v1", "sockets": {"warp": {"impl": "x"}}}), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="resolved: unknown socket 'warp'"):
        resolve_manifest(builtin_registry(), ManifestSources(resumed=read_resolved(tmp_path)))


def test_no_file_means_no_resume_layer(tmp_path) -> None:
    assert read_resolved(tmp_path) is None


def test_the_file_keeps_variables_unexpanded_and_records_layers_and_fingerprint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_TEST_TOKEN", "s3cret")
    registry = _registry(_Spy())
    manifest, host = _start(
        registry, tmp_path, cli_configs=("measurement.token=${HYPERLOOM_TEST_TOKEN}",), cli_plugs=("measurement=local",)
    )
    text = (tmp_path / RESOLVED_FILENAME).read_text(encoding="utf-8")
    assert "s3cret" not in text
    doc = json.loads(text)
    assert doc["sockets"]["measurement"] == {"impl": "local", "config": {"token": "${HYPERLOOM_TEST_TOKEN}"}}
    assert doc["layers"]["measurement"] == "cli"
    assert doc["fingerprint"] == host.fingerprint() and doc["profile"] == manifest.profile
