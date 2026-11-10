# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted lock layer and the session's plugs.resolved.json (4.2.2 rule 6 and the resume paragraph)."""

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
from hyperloom.plugs.host import EVENT_LOCK_MISMATCH
from hyperloom.plugs.manifest import LockMismatch
from hyperloom.plugs.resolved import RESOLVED_FILENAME, read_resolved, write_resolved

LOCK = {"measurement": {"impl": "hosted", "config": {"max_parallel_tests": 8}}, "events": {"impl": "spy"}}


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


def _edit(session_dir, mutate):
    path = session_dir / RESOLVED_FILENAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_a_locked_socket_is_replaced_outright_and_marked_locked() -> None:
    registry = _registry(_Spy())
    sources = ManifestSources(cli_plugs=("measurement=local",), cli_configs=("measurement.other=1",), locked=LOCK)
    measurement = resolve_manifest(registry, sources).sockets["measurement"]
    assert (measurement.impl, dict(measurement.config), measurement.layer) == (
        "hosted",
        {"max_parallel_tests": 8},
        "locked",
    )


def test_no_other_layer_reaches_a_locked_socket(tmp_path) -> None:
    registry = _registry(_Spy())
    overrides = tmp_path / "o.json"
    overrides.write_text(json.dumps({"schema": "hyperloom.plugs/v1", "sockets": {"events": {"impl": "local"}}}))
    sources = ManifestSources(profile="hosted-gpu", overrides_path=overrides, cli_plugs=("events=local",), locked=LOCK)
    assert resolve_manifest(registry, sources).sockets["events"].impl == "spy"


def test_only_the_five_lockable_sockets_can_be_locked() -> None:
    with pytest.raises(
        ManifestError,
        match=r"socket 'storage' cannot be locked; lockable: measurement, sandbox, credentials, events, gpu_token",
    ):
        resolve_manifest(builtin_registry(), ManifestSources(locked={"storage": {"impl": "x"}}))


def test_a_lock_entry_must_name_an_impl() -> None:
    with pytest.raises(ManifestError, match="lock: socket 'sandbox' must name an impl"):
        resolve_manifest(builtin_registry(), ManifestSources(locked={"sandbox": {"config": {}}}))


def test_resume_after_editing_the_resolved_file_still_uses_the_locked_implementation(tmp_path) -> None:
    spy = _Spy()
    registry = _registry(spy)
    first, host = _start(registry, tmp_path, locked=LOCK)
    assert host.get("measurement") == "hosted-measurement"
    host.close()

    _edit(tmp_path, lambda doc: doc["sockets"]["measurement"].update(impl="local"))
    second, resumed_host = _start(registry, tmp_path, locked=LOCK)

    assert resumed_host.get("measurement") == "hosted-measurement"
    assert second.sockets["measurement"].impl == "hosted"
    assert second.sockets["measurement"].layer == "locked"
    assert second.lock_mismatches == (LockMismatch("measurement", "local", "hosted"),)
    assert (EVENT_LOCK_MISMATCH, {"socket": "measurement", "file_impl": "local", "locked_impl": "hosted"}) in spy.events
    assert json.loads((tmp_path / RESOLVED_FILENAME).read_text())["sockets"]["measurement"]["impl"] == "hosted"


def test_a_sandbox_swapped_for_a_local_one_in_the_file_does_not_survive_a_resume(tmp_path) -> None:
    registry = _registry(_Spy())
    lock = {**LOCK, "sandbox": {"impl": "e2b"}}
    _start(registry, tmp_path, locked=lock)[1].close()
    _edit(tmp_path, lambda doc: doc["sockets"]["sandbox"].update(impl="local"))
    manifest, host = _start(registry, tmp_path, locked=lock)
    assert host.get("sandbox") == "e2b-sandbox"
    assert [m.socket for m in manifest.lock_mismatches] == ["sandbox"]


def test_a_locked_socket_deleted_from_the_file_comes_back_on_resume(tmp_path) -> None:
    registry = _registry(_Spy())
    _start(registry, tmp_path, locked=LOCK)[1].close()
    _edit(tmp_path, lambda doc: doc["sockets"].pop("measurement"))
    manifest, host = _start(registry, tmp_path, locked=LOCK)
    assert host.get("measurement") == "hosted-measurement"
    assert manifest.lock_mismatches == ()


def test_an_edited_config_of_a_locked_socket_loses_without_a_mismatch_event(tmp_path) -> None:
    registry = _registry(_Spy())
    _start(registry, tmp_path, locked=LOCK)[1].close()
    _edit(tmp_path, lambda doc: doc["sockets"]["measurement"].update(config={"max_parallel_tests": 999}))
    manifest, _ = _start(registry, tmp_path, locked=LOCK)
    assert dict(manifest.sockets["measurement"].config) == {"max_parallel_tests": 8}
    assert manifest.lock_mismatches == ()


def test_an_unlocked_socket_resumes_from_the_file(tmp_path) -> None:
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
