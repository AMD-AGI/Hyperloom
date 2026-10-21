# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local sandbox sees its framework roots as they were at start and writes only to a private layer of its own."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from hyperloom.sandbox.launcher import OverlaySetup, SetupFailed, StatusPipe, launcher_argv, read_setup_answer
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate, ProcessUsers
from hyperloom.sandbox.socket import CreateOptions, SandboxUnavailable
from hyperloom.sandbox.tests.framework_overlay import make_framework, overlay_unavailable_reason

_TOUCH_THE_FRAMEWORK = """
cd "$FW"
cat aiter/configs/tuned.csv > "$HYPERLOOM_SANDBOX_ROOT/out/seen.txt"
printf 'damaged\\n' > aiter/configs/tuned.csv
rm sglang/framework_victim
echo new > aiter/new.so
chmod 600 sglang/framework_other
( cat aiter/configs/tuned.csv > "$HYPERLOOM_SANDBOX_ROOT/out/after.txt" )
echo ok > "$HYPERLOOM_SANDBOX_ROOT/out/done"
sleep 600
"""
_PRISTINE = "M,N,K\n64,5120,5120\n"


@pytest.fixture
def overlay_ready(tmp_path_factory: pytest.TempPathFactory) -> None:
    reason = overlay_unavailable_reason(tmp_path_factory.mktemp("probe"))
    if reason is not None:
        pytest.skip(f"this machine cannot give a process a private mount namespace: {reason}")


@pytest.fixture
def provider(tmp_path: Path):
    made: list[LocalSandboxProvider] = []

    def build(templates, **kwargs) -> LocalSandboxProvider:
        kwargs.setdefault("base_dir", tmp_path / "sandboxes")
        built = LocalSandboxProvider(templates, hostname="node-a", **kwargs)
        made.append(built)
        return built

    yield build
    for built in made:
        built.close()


def _template(root: Path, script: str = _TOUCH_THE_FRAMEWORK, **kwargs) -> LocalTemplate:
    kwargs.setdefault("framework_roots", (str(root),))
    return LocalTemplate(
        "runner", "measurement", command=("/bin/sh", "-c", script), sealed=True, envs={"FW": str(root)}, **kwargs
    )


def _create(built: LocalSandboxProvider):
    return built.create("runner", role="measurement", timeout_sec=120, metadata={}, options=CreateOptions(wait_sec=30))


def _out(sandbox, name: str, seconds: float = 20.0) -> bytes:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return sandbox.read_file(f"/out/{name}")
        except FileNotFoundError:
            time.sleep(0.05)
    raise AssertionError(f"{name} did not appear")


def _framework_is_untouched(root: Path) -> None:
    assert (root / "aiter/configs/tuned.csv").read_text() == _PRISTINE
    assert (root / "sglang/framework_victim").read_text() == "present\n"
    assert stat.S_IMODE((root / "sglang/framework_other").stat().st_mode) != 0o600
    assert not (root / "aiter/new.so").exists()


# -- the template --------------------------------------------------------------------------------------------------


def test_a_template_takes_framework_roots_only_as_plain_absolute_directories_of_a_sealed_sandbox() -> None:
    for bad in ("relative/dir", "/", "/a/../b", "/with,comma", "/with:colon", "/with space", "/trailing/"):
        with pytest.raises(ValueError, match="plain absolute directory"):
            LocalTemplate("t", "measurement", sealed=True, framework_roots=(bad,))
    with pytest.raises(ValueError, match="is inside"):
        LocalTemplate("t", "measurement", sealed=True, framework_roots=("/a", "/a/b"))
    with pytest.raises(ValueError, match="need a sealed template"):
        LocalTemplate("t", "dev", framework_roots=("/a",))
    with pytest.raises(ValueError, match="framework_upper_mib"):
        LocalTemplate("t", "measurement", sealed=True, framework_roots=("/a",), framework_upper_mib=0)
    with pytest.raises(ValueError, match="platform variables"):
        LocalTemplate("t", "dev", envs={"HYPERLOOM_FRAMEWORK_ROOTS": "/x"})
    built = LocalTemplate.from_mapping(
        "r", {"role": "measurement", "sealed": True, "framework_roots": ["/a", "/b"], "framework_upper_mib": 64}
    )
    assert (built.framework_roots, built.framework_upper_mib) == (("/a", "/b"), 64)


# -- failing closed ------------------------------------------------------------------------------------------------


def test_a_root_that_is_not_a_directory_fails_the_create_and_starts_nothing(provider, tmp_path: Path) -> None:
    built = provider([_template(tmp_path / "no-such-root", script="echo started > $HYPERLOOM_SANDBOX_ROOT/out/ran")])
    with pytest.raises(SandboxUnavailable, match="is not a directory"):
        _create(built)
    assert built.find({}, role="measurement") == []
    assert not list((tmp_path / "sandboxes").glob("*/fs/out/ran"))


@pytest.mark.skipif(os.geteuid() == 0, reason="the probe below is for an account that cannot make a mount namespace")
def test_an_account_that_cannot_make_a_mount_namespace_gets_no_sandbox(provider, tmp_path: Path) -> None:
    built = provider([_template(make_framework(tmp_path))])
    with pytest.raises(SandboxUnavailable, match="framework isolation failed: unshare"):
        _create(built)


@pytest.mark.skipif(os.geteuid() != 0, reason="mounts need root")
def test_a_mount_below_a_root_stays_where_it_was_and_the_rest_is_private(tmp_path: Path) -> None:
    root = make_framework(tmp_path)
    (root / "nested").mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    answer, answer_writer = os.pipe()
    status, status_writer = os.pipe()
    setup = OverlaySetup((str(root),), str(scratch), 4, answer_writer)
    command = ["/bin/sh", "-c", 'cat "$1/nested/shared.txt" > "$2"; echo late > "$1/aiter/late"', "sh", str(root)]
    argv = launcher_argv(sys.executable, status_writer, [*command, str(tmp_path / "seen")], setup=setup)
    # The nested mount is made in a private mount namespace of its own, which ends with the process.
    wrapper = 'mount -t tmpfs nested "$1/nested" && echo shared > "$1/nested/shared.txt" && shift && exec "$@"'
    unshare = ["unshare", "-m", "--propagation", "private", "/bin/sh", "-c", wrapper, "sh", str(root), *argv]
    with subprocess.Popen(unshare, pass_fds=(status_writer, answer_writer)) as proc:
        os.close(answer_writer)
        os.close(status_writer)
        try:
            read_setup_answer(answer)
            assert StatusPipe(status).wait_exit(proc, 30) == 0
        finally:
            proc.kill()
    assert (tmp_path / "seen").read_text() == "shared\n"
    _framework_is_untouched(root)
    assert not (root / "aiter/late").exists()


def test_a_launcher_that_never_answers_is_a_failed_setup(monkeypatch) -> None:
    monkeypatch.setattr("hyperloom.sandbox.launcher._SETUP_TIMEOUT_SEC", 0.1)
    reader, writer = os.pipe()
    try:
        with pytest.raises(SetupFailed, match="did not answer"):
            read_setup_answer(reader)
    finally:
        os.close(writer)


# -- the isolation -------------------------------------------------------------------------------------------------


def test_what_a_sandbox_writes_to_its_framework_root_is_seen_by_it_alone(overlay_ready, provider, tmp_path) -> None:
    root = make_framework(tmp_path)
    built = provider([_template(root)])
    first = _create(built)
    assert _out(first, "seen.txt") == _PRISTINE.encode()
    _out(first, "done")
    assert _out(first, "after.txt") == b"damaged\n", "the sandbox sees its own writes"
    _framework_is_untouched(root)

    second = _create(built)  # while the first is still running: no sandbox sees another's writes
    assert _out(second, "seen.txt") == _PRISTINE.encode()
    _out(second, "done")
    first.kill()
    second.kill()
    _framework_is_untouched(root)
    mounts = Path("/proc/self/mountinfo").read_text()
    assert "hl-framework" not in mounts, "the private mounts are not in this namespace"


def test_a_root_that_is_a_link_is_private_where_it_points_and_seen_through_the_link(
    overlay_ready, provider, tmp_path
) -> None:
    root = make_framework(tmp_path)
    link = tmp_path / "framework-link"
    link.symlink_to(root)
    sandbox = _create(provider([_template(link)]))
    assert _out(sandbox, "seen.txt") == _PRISTINE.encode()
    assert _out(sandbox, "after.txt") == b"damaged\n"
    sandbox.kill()
    _framework_is_untouched(root)


def test_the_layer_for_the_writes_has_the_size_the_template_gives_it(overlay_ready, provider, tmp_path) -> None:
    root = make_framework(tmp_path)
    script = (
        'dd if=/dev/zero of="$FW/big" bs=1M count=4 2>/dev/null; echo $? > "$HYPERLOOM_SANDBOX_ROOT/out/rc"; sleep 600'
    )
    sandbox = _create(provider([_template(root, script=script, framework_upper_mib=1)]))
    assert _out(sandbox, "rc").strip() != b"0"
    sandbox.kill()
    assert not (root / "big").exists()


def test_a_sandbox_of_another_account_gets_the_same_view(overlay_ready, provider) -> None:
    if os.geteuid() != 0:
        pytest.skip("separate accounts need root")
    parent = Path(tempfile.mkdtemp(prefix="hl-fw-"))
    parent.chmod(0o755)
    root = make_framework(parent)
    for path in (root, *root.rglob("*")):
        path.chmod(0o777 if path.is_dir() else 0o666)
    script = f'id -u > "$HYPERLOOM_SANDBOX_ROOT/out/uid"; {_TOUCH_THE_FRAMEWORK}'
    users = ProcessUsers(sealed_user=(60001, 60001))
    built = provider([_template(root, script=script)], users=users, base_dir=parent / "sandboxes")
    sandbox = _create(built)
    assert _out(sandbox, "uid").strip() == b"60001"
    assert _out(sandbox, "seen.txt") == _PRISTINE.encode()
    _out(sandbox, "done")
    sandbox.kill()
    assert (root / "aiter/configs/tuned.csv").read_text() == _PRISTINE
    assert (root / "sglang/framework_victim").exists() and not (root / "aiter/new.so").exists()
