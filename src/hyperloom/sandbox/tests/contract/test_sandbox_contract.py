# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sandbox contract: what any ``SandboxProvider`` must do, through the socket only.

Nothing here looks at processes, directories or any other implementation
detail, so the suite can be pointed at a different implementation by adding
a parameter to the ``harness`` fixture. A test marked ``claw_policy`` checks
what only the Claw sandbox service (and the local provider, which reproduces
it) enforces; an implementation that cannot, such as the E2B stand-in, skips it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hyperloom.sandbox.socket import (
    AVOID_NODES_KEY,
    IDEMPOTENCY_KEY,
    NODE_NAME_KEY,
    SAME_NODE_KEY,
    SIGNING_PUBLIC_KEY,
    CreateOptions,
    OutputSinks,
    SandboxDenied,
    SandboxGone,
)


def _wait_for_file(sandbox, path: str, seconds: float = 20.0) -> bytes:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return sandbox.read_file(path)
        except FileNotFoundError:
            time.sleep(0.05)
    raise AssertionError(f"{path} did not appear within {seconds}s")


def _create(h, template: str, role: str, **kw):
    return h.provider.create(
        template, role=role, timeout_sec=120, metadata=kw.pop("metadata", {}), options=CreateOptions(**kw)
    )


def test_info_carries_caller_metadata(harness) -> None:
    sb = _create(harness, harness.sealed_template, "measurement", metadata={"hyperloom.request-id": "r1"})
    info = sb.info()
    assert info.sandbox_id == sb.sandbox_id
    assert info.template == harness.sealed_template
    assert info.metadata["hyperloom.request-id"] == "r1"


@pytest.mark.claw_policy
def test_info_carries_platform_metadata(harness) -> None:
    info = _create(harness, harness.sealed_template, "measurement").info()
    assert info.metadata[NODE_NAME_KEY]
    assert SAME_NODE_KEY not in info.metadata


def test_sealed_runner_answers_through_in_and_out(harness) -> None:
    sb = _create(harness, harness.sealed_template, "measurement")
    sb.write_file("/in/spec.json", b'{"id": 1}')
    assert _wait_for_file(sb, "/out/status.json") == b'{"id": 1}'
    assert "status.json" in sb.list_dir("/out")


@pytest.mark.claw_policy
def test_sealed_policy_refuses_commands_and_other_paths(harness) -> None:
    sb = _create(harness, harness.sealed_template, "measurement")
    with pytest.raises(SandboxDenied):
        sb.run("echo hi")
    with pytest.raises(SandboxDenied):
        sb.run("echo hi", background=True)
    for bad in ("/out/x", "/work/x", "/etc/x", "/in/../out/x", "/in"):
        with pytest.raises(SandboxDenied):
            sb.write_file(bad, b"x")
    for bad in ("/in/spec.json", "/work", "/etc/passwd", "/out/../in/a"):
        with pytest.raises(SandboxDenied):
            sb.read_file(bad)
    with pytest.raises(SandboxDenied):
        sb.list_dir("/in")


def test_sealed_template_refuses_caller_environment(harness) -> None:
    with pytest.raises(SandboxDenied):
        _create(harness, harness.sealed_template, "measurement", envs={"X": "1"})


@pytest.mark.claw_policy
def test_signing_identity_is_per_sandbox_and_platform_only(harness) -> None:
    a = _create(harness, harness.sealed_template, "measurement")
    b = _create(harness, harness.sealed_template, "measurement")
    key_a, key_b = a.info().metadata[SIGNING_PUBLIC_KEY], b.info().metadata[SIGNING_PUBLIC_KEY]
    assert key_a.startswith("-----BEGIN PUBLIC KEY-----")
    assert key_a != key_b
    with pytest.raises(SandboxDenied):
        _create(harness, harness.sealed_template, "measurement", metadata={SIGNING_PUBLIC_KEY: key_a})
    dev = _create(harness, harness.dev_template, "dev")
    assert SIGNING_PUBLIC_KEY not in dev.info().metadata


@pytest.mark.claw_policy
def test_platform_metadata_keys_cannot_be_forged(harness) -> None:
    for key in (NODE_NAME_KEY, SAME_NODE_KEY, SIGNING_PUBLIC_KEY):
        with pytest.raises(SandboxDenied):
            _create(harness, harness.dev_template, "dev", metadata={key: "x"})


def test_roles_only_use_their_own_templates(harness) -> None:
    with pytest.raises(SandboxDenied):
        _create(harness, harness.sealed_template, "dev")
    with pytest.raises(SandboxDenied):
        _create(harness, harness.dev_template, "measurement")
    with pytest.raises(SandboxDenied):
        _create(harness, "no-such-template", "dev")


def test_a_key_only_sees_its_own_sandboxes(harness) -> None:
    sb = _create(harness, harness.dev_template, "dev", metadata={"hyperloom.purpose": "p"})
    assert harness.provider.connect(sb.sandbox_id, role="dev").sandbox_id == sb.sandbox_id
    with pytest.raises(SandboxGone):
        harness.provider.connect(sb.sandbox_id, role="subrun")
    assert [i.sandbox_id for i in harness.provider.find({"hyperloom.purpose": "p"}, role="dev")] == [sb.sandbox_id]
    assert harness.provider.find({"hyperloom.purpose": "p"}, role="subrun") == []
    assert harness.provider.find({"hyperloom.purpose": "other"}, role="dev") == []


def test_dev_sandbox_runs_commands_and_files(harness) -> None:
    sb = _create(harness, harness.dev_template, "dev", envs={"GREETING": "hello"})
    lines: list[str] = []
    result = sb.run("echo $GREETING; echo oops >&2; exit 3", sinks=OutputSinks(on_stdout=lines.append))
    assert (result.exit_code, result.stdout, result.stderr) == (3, "hello\n", "oops\n")
    assert lines == ["hello\n"]
    sb.write_file("/work/a/b.txt", b"data")
    assert sb.read_file("/work/a/b.txt") == b"data"
    assert sb.list_dir("/work/a") == ["b.txt"]
    with pytest.raises(FileNotFoundError):
        sb.read_file("/work/missing")


def test_copy_out_streams_to_dest(harness, tmp_path: Path) -> None:
    sb = _create(harness, harness.dev_template, "dev")
    payload = b"x" * (3 * 1024 * 1024 + 17)
    sb.write_file("/work/big.bin", payload)
    dest = tmp_path / "big.bin"
    assert sb.copy_out("/work/big.bin", dest) == len(payload)
    assert dest.read_bytes() == payload


def test_background_command_returns_at_once(harness) -> None:
    sb = _create(harness, harness.dev_template, "dev")
    started = time.monotonic()
    result = sb.run("sleep 30", background=True)
    assert result.exit_code == 0
    assert time.monotonic() - started < 5


def test_a_new_sandbox_starts_empty_and_isolated(harness) -> None:
    first = _create(harness, harness.dev_template, "dev")
    first.write_file("/work/leftover.txt", b"x")
    second = _create(harness, harness.dev_template, "dev")
    assert second.sandbox_id != first.sandbox_id
    assert second.list_dir("/work") == []


def test_kill_ends_the_sandbox_and_is_repeatable(harness) -> None:
    sb = _create(harness, harness.dev_template, "dev")
    sb.kill()
    sb.kill()
    with pytest.raises(SandboxGone):
        sb.info()
    with pytest.raises(SandboxGone):
        sb.run("true")
    with pytest.raises(SandboxGone):
        sb.read_file("/work/x")
    with pytest.raises(SandboxGone):
        harness.provider.connect(sb.sandbox_id, role="dev")
    assert harness.provider.find({}, role="dev") == []


def test_a_killed_sealed_sandbox_cannot_be_reused(harness) -> None:
    sb = _create(harness, harness.sealed_template, "measurement")
    sb.kill()
    with pytest.raises(SandboxGone):
        sb.write_file("/in/spec.json", b"{}")


#: How long an ended sandbox may stay visible: the service reaps expired sandboxes on a 30 s sweep, and deleting the
#: sandbox's workload takes a second or two more, so up to about 32 s; 35 s leaves a margin over that bound.
_TIMEOUT_REAP_SEC = 35


def test_timeout_ends_the_sandbox(harness) -> None:
    sb = harness.provider.create(harness.dev_template, role="dev", timeout_sec=1, metadata={})
    deadline = time.monotonic() + _TIMEOUT_REAP_SEC
    while time.monotonic() < deadline:
        try:
            sb.info()
        except SandboxGone:
            return
        time.sleep(0.1)
    raise AssertionError("sandbox outlived its timeout")


def test_set_timeout_extends_the_lifetime(harness) -> None:
    sb = harness.provider.create(harness.dev_template, role="dev", timeout_sec=1, metadata={})
    sb.set_timeout(120)
    time.sleep(2.5)
    assert sb.info().sandbox_id == sb.sandbox_id


def test_idempotency_key_returns_the_same_sandbox(harness) -> None:
    meta = {IDEMPOTENCY_KEY: "k-1"}
    one = _create(harness, harness.dev_template, "dev", metadata=meta)
    two = _create(harness, harness.dev_template, "dev", metadata=meta)
    assert two.sandbox_id == one.sandbox_id
    other = _create(harness, harness.dev_template, "dev", metadata={IDEMPOTENCY_KEY: "k-2"})
    assert other.sandbox_id != one.sandbox_id


def _assert_same_node_marks_where_it_landed(metadata: dict[str, str], avoided: set[str]) -> None:
    """``same-node`` is ``"true"`` exactly when the sandbox landed on a node it was to avoid, and absent otherwise."""
    if metadata[NODE_NAME_KEY] in avoided:
        assert metadata[SAME_NODE_KEY] == "true", metadata
    else:
        assert SAME_NODE_KEY not in metadata, metadata


@pytest.mark.claw_policy
def test_avoiding_this_node_still_creates_and_marks_same_node(harness) -> None:
    """Avoiding a node never refuses the create: the platform places the sandbox elsewhere when it can, and on the
    avoided node (marked ``same-node``) when it cannot, as a single-node platform must."""
    node = (
        harness.provider.create(harness.dev_template, role="dev", timeout_sec=60, metadata={})
        .info()
        .metadata[NODE_NAME_KEY]
    )
    sb = _create(harness, harness.dev_template, "dev", metadata={AVOID_NODES_KEY: f"elsewhere,{node}"})
    _assert_same_node_marks_where_it_landed(dict(sb.info().metadata), {"elsewhere", node})
    away = _create(harness, harness.dev_template, "dev", metadata={AVOID_NODES_KEY: "elsewhere"})
    _assert_same_node_marks_where_it_landed(dict(away.info().metadata), {"elsewhere"})
