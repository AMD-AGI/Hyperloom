# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local test sandboxes confined by their provider's plan, with a stand-in bubblewrap: which tier launches a sandbox's
processes, the policy each gets, how unconfined is said, and that a hosted run refuses.

The stand-in records its command line and environment and runs the command after ``--``, setting the ``--setenv``
marks and the ``--chdir`` directory; no namespace is made (the real launches are ``test_sandbox_confined_bwrap.py``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import (
    BWRAP,
    NONE,
    REFUSED,
    TEST_ACCOUNT,
    BwrapIsolation,
    IsolationRefused,
    NoIsolation,
)
from hyperloom.sandbox import confined, local_sandbox
from hyperloom.sandbox.card_occupancy import SANDBOX_TAG_ENV
from hyperloom.sandbox.confined import UNCONFINED_TESTS, local_test_isolation
from hyperloom.sandbox.launcher import LAUNCHER_SOURCE
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate, ProcessUsers, RunnerPlan
from hyperloom.sandbox.socket import SandboxDenied, SandboxUnavailable
from hyperloom_common.bwrap import BwrapProbe, BwrapSetupError

FAKE_BWRAP = r"""#!{python} -I
import json, os, sys
args = sys.argv[1:]
with open({log!r}, "a") as handle:
    handle.write(json.dumps({{"argv": args, "env": dict(os.environ)}}) + "\n")
if {fail!r}:
    print("bwrap: Can't bind mount /oldroot/x on /newroot/x: No such file or directory", file=sys.stderr)
    sys.exit(1)
end = args.index("--")
env, index = dict(os.environ), 0
while index < end:
    if args[index] == "--setenv":
        env[args[index + 1]] = args[index + 2]
        index += 3
    elif args[index] == "--chdir":
        os.chdir(args[index + 1])
        index += 2
    else:
        index += 1
os.execvpe(args[end + 1], args[end + 1 :], env)
"""
#: A runner: says it started, then waits to be stopped.
RUNNER = "import time; print('runner started', flush=True); time.sleep(600)"


class FakeBwrap:
    """A stand-in ``bwrap`` at ``path`` whose launches are recorded in ``log``."""

    def __init__(self, directory: Path, *, fail: bool = False) -> None:
        self.path = directory / "bwrap"
        self.log = directory / "bwrap.log"
        self.path.write_text(FAKE_BWRAP.format(python=sys.executable, log=str(self.log), fail=fail))
        self.path.chmod(0o755)

    def launches(self) -> list[dict[str, object]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def isolation(self, version: tuple[int, int, int] = (0, 13, 1)) -> BwrapIsolation:
        return BwrapIsolation(BwrapProbe(str(self.path), version, ()))


def _given(isolation: BwrapIsolation | NoIsolation) -> Callable[..., BwrapIsolation | NoIsolation]:
    """``choose_test_isolation`` that gives ``isolation`` for a provider and decides a sandbox's refusal as the real
    one does."""

    def choose(*, known: BwrapProbe | None = None) -> BwrapIsolation | NoIsolation:
        return isolation if known is None else agent_isolation.choose_test_isolation(known=known)

    return choose


def _flag_pairs(argv: list[str]) -> list[tuple[str, ...]]:
    """The mounts and settings of a recorded command line before ``--``, each with its arguments."""
    arity = {
        "--bind": 2,
        "--ro-bind": 2,
        "--ro-bind-try": 2,
        "--dev-bind-try": 2,
        "--symlink": 2,
        "--setenv": 2,
        "--tmpfs": 1,
        "--proc": 1,
        "--chdir": 1,
        "--perms": 1,
        "--remount-ro": 1,
        "--userns": 1,
        "--pidns": 1,
        "--cap-drop": 1,
        "--json-status-fd": 1,
    }
    found, index = [], 0
    end = argv.index("--")
    while index < end:
        flag = argv[index]
        count = arity.get(flag, 0)
        found.append((flag, *argv[index + 1 : index + 1 + count]))
        index += 1 + count
    return found


_MOUNTS = ("--bind", "--ro-bind", "--ro-bind-try", "--dev-bind-try", "--symlink", "--tmpfs", "--proc")


def _assert_read_only_root(pairs: list[tuple[str, ...]]) -> None:
    """The system trees are shown read-only, however a host's credential locations make ``/usr`` bound (whole, or entry
    by entry around a file a link in a home resolves to); the cards' devices are bound on the sandbox's own ``/dev``;
    and the root is made read-only once everything is mounted."""
    usr = [pair for pair in pairs if pair[0] in _MOUNTS and (pair[-1] == "/usr" or pair[-1].startswith("/usr/"))]
    assert usr, "the system tree is shown"
    assert all(pair[0] in ("--ro-bind-try", "--symlink") for pair in usr), "nothing of it is writable"
    dev = pairs.index(("--tmpfs", "/dev"))
    devices = [pairs.index(("--dev-bind-try", node, node)) for node in ("/dev/kfd", "/dev/dri")]
    assert all(index > dev for index in devices), "the cards' devices are bound on the sandbox's /dev, after its tmpfs"
    mounts = [index for index, pair in enumerate(pairs) if pair[0] in _MOUNTS]
    assert pairs.index(("--remount-ro", "/")) > max(mounts), "the root is read-only once everything is mounted"


def _system_file() -> Path:
    """A regular file below ``/usr`` that no sandbox here needs: what a link in a home may resolve to."""
    for directory in (Path("/usr/share/common-licenses"), Path("/usr/share/doc")):
        found = (
            sorted(p for p in directory.glob("*") if p.is_file() and not p.is_symlink()) if directory.is_dir() else []
        )
        if found:
            return found[0]
    pytest.skip("no regular file below /usr/share/common-licenses or /usr/share/doc on this host")


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that has not been marked hosted, has said nothing yet, and has no app state directory."""
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(confined, "_said", set())
    monkeypatch.delenv("HYPERLOOM_STATE_DIR", raising=False)


@pytest.fixture
def area() -> Iterator[Path]:
    """A directory outside ``/tmp`` (where bubblewrap before 0.12 binds nothing below its own ``/tmp``)."""
    made = Path(tempfile.mkdtemp(prefix="hl-test-sandbox-", dir="/var/tmp")).resolve()
    yield made
    shutil.rmtree(made, ignore_errors=True)


@pytest.fixture
def providers() -> Iterator[list[LocalSandboxProvider]]:
    made: list[LocalSandboxProvider] = []
    yield made
    for provider in made:
        provider.close()


def _runner(*, gpus: int = 1, code: str = RUNNER) -> LocalTemplate:
    return LocalTemplate(
        "runner", "measurement", command=(sys.executable, "-c", code), gpus=gpus, sealed=True, signing=True
    )


def _wait(predicate: Callable[[], object], seconds: float = 30.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        time.sleep(0.05)


def _log_says(sandbox: object, text: str) -> bool:
    log = Path(sandbox.private) / "runner.log"  # type: ignore[attr-defined]
    return log.exists() and text in log.read_text()


def test_a_runner_with_no_account_runs_in_bubblewrap_with_its_tree_and_key_writable_and_the_plan_read_only(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    model, state = area / "model", area / "state"
    for path in (model, state):
        path.mkdir()
    (model / "weights").write_text("weights")
    (model / "credentials").write_text("SECRET")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(model / "credentials"))
    monkeypatch.setenv("HYPERLOOM_STATE_DIR", str(state))
    base = area / "sandboxes"
    provider = LocalSandboxProvider(
        [_runner()], gpus=(0,), base_dir=base, users=ProcessUsers(plan=RunnerPlan(readable=(model,)))
    )
    providers.append(provider)
    sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    _wait(lambda: _log_says(sandbox, "runner started"))

    [launch] = fake.launches()
    argv, env = launch["argv"], launch["env"]
    assert isinstance(argv, list) and isinstance(env, dict)
    pairs = _flag_pairs(argv)
    directory = base / sandbox.sandbox_id
    fs, private = str(directory / "fs"), str(directory / "private")
    command = argv[argv.index("--") + 1 :]
    assert sandbox.confined
    assert command[:4] == [sys.executable, "-I", "-c", LAUNCHER_SOURCE], "the launcher runs inside"
    assert ("--unshare-user",) in pairs and ("--unshare-pid",) in pairs and ("--cap-drop", "ALL") in pairs
    assert ("--die-with-parent",) not in pairs, "a reap or the timeout ends the sandbox, with its grace"
    assert ("--bind", fs, fs) in pairs and ("--bind", private, private) in pairs, "the runner removes its key"
    assert ("--ro-bind-try", str(model / "weights"), str(model / "weights")) in pairs
    assert not [pair for pair in pairs if str(model / "credentials") in pair], "the app's credential is left out"
    assert ("--ro-bind-try", str(model), str(model)) not in pairs, "the tree holding it is bound entry by entry"
    assert ("--chdir", f"{fs}/work") in pairs
    assert ("--setenv", SANDBOX_TAG_ENV, sandbox.sandbox_id) in pairs and SANDBOX_TAG_ENV not in env
    tmp = [pair for pair in pairs if pair[0] == "--bind" and pair[-1] == "/tmp"]
    shm = [pair for pair in pairs if pair[0] == "--bind" and pair[-1] == "/dev/shm"]
    assert len(tmp) == 1 and len(shm) == 1 and tmp[0][1] != shm[0][1]
    _assert_read_only_root(pairs)
    bound = [
        pair[1] for pair in pairs if pair[0] in ("--bind", "--ro-bind-try", "--ro-bind") and pair[1] != "/dev/null"
    ]
    assert not [path for path in bound if path in (str(base), str(state)) or path.startswith(f"{state}/")]
    assert not [path for path in bound if path.startswith(f"{base}/") and not path.startswith(f"{directory}/")]


def test_a_link_in_the_home_to_a_system_file_leaves_that_file_out_and_the_system_tree_read_only(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A home that holds a program directory on PATH is masked entry by entry around it, each entry where it resolves:
    # a link there to a file below /usr hides that file, so /usr is bound entry by entry without it (as on a CI
    # runner whose home links into /usr).
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    target = _system_file()
    home = area / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / "linked").symlink_to(target)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{home / '.local' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    base = area / "sandboxes"
    provider = LocalSandboxProvider([_runner()], gpus=(0,), base_dir=base, users=ProcessUsers(plan=RunnerPlan()))
    providers.append(provider)
    sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    _wait(lambda: _log_says(sandbox, "runner started"))

    [launch] = fake.launches()
    argv = launch["argv"]
    assert isinstance(argv, list)
    pairs = _flag_pairs(argv)
    assert sandbox.confined
    assert ("--ro-bind-try", "/usr", "/usr") not in pairs, "the system tree is bound entry by entry"
    assert not [pair for pair in pairs if str(target) in pair[1:]], "the file the home links to is left out"
    _assert_read_only_root(pairs)
    assert ("--ro-bind-try", "/usr/bin", "/usr/bin") in pairs, "what is beside it is shown"


def test_a_template_with_no_card_gets_no_card_device(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    provider = LocalSandboxProvider(
        [_runner(gpus=0)], base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan())
    )
    providers.append(provider)
    sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    _wait(lambda: _log_says(sandbox, "runner started"))

    [launch] = fake.launches()
    assert not [arg for arg in launch["argv"] if arg in ("/dev/kfd", "/dev/dri")]  # type: ignore[union-attr]


def test_a_kind_with_an_account_runs_as_that_account_and_never_in_bubblewrap(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    started: list[tuple[list[str], dict[str, object]]] = []
    real = subprocess.Popen

    def popen(argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        started.append((list(argv), dict(kwargs)))
        kept = {name: value for name, value in kwargs.items() if name not in ("user", "group", "extra_groups")}
        return real(argv, **kept)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(local_sandbox.subprocess, "Popen", popen)
    users = ProcessUsers(sealed_user=(os.getuid(), os.getgid()), plan=RunnerPlan())
    provider = LocalSandboxProvider(
        [_runner(), LocalTemplate("dev", "dev")], gpus=(0,), base_dir=area / "sandboxes", users=users
    )
    providers.append(provider)
    runner = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    dev = provider.create("dev", role="dev", timeout_sec=60, metadata={})

    # Only the launchers: deriving a confinement runs the interpreter once to ask where it imports from.
    [(argv, kwargs)] = [(argv, kwargs) for argv, kwargs in started if LAUNCHER_SOURCE in argv]
    assert argv[:4] == [sys.executable, "-I", "-c", LAUNCHER_SOURCE] and kwargs["user"] == os.getuid()
    assert not runner.confined and fake.launches() == [], "the account tier runs it, no bubblewrap"
    assert dev.confined, "the kind no account is named for is confined"


def test_without_a_plan_nothing_is_confined_or_chosen(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(**_kwargs: object) -> BwrapIsolation:
        raise AssertionError("a provider without a plan chose an isolation")

    monkeypatch.setattr(confined, "choose_test_isolation", never)
    provider = LocalSandboxProvider([_runner()], gpus=(0,), base_dir=area / "sandboxes")
    providers.append(provider)
    sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert not sandbox.confined


def test_where_bubblewrap_cannot_sandbox_a_test_tests_run_as_before_and_that_is_said_once(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    missing = BwrapProbe(None, None, ("bubblewrap: not found on PATH",))
    monkeypatch.setattr(agent_isolation, "bwrap_probe", lambda _path: missing)
    code = (
        "import os, signal; os.kill(int(os.environ['APP_PID']), 0); print('app pid seen', flush=True); signal.pause()"
    )
    template = LocalTemplate(
        "runner",
        "measurement",
        command=(sys.executable, "-c", code),
        sealed=True,
        signing=True,
        envs={"APP_PID": str(os.getpid())},
    )
    with caplog.at_level(logging.WARNING):
        for index in range(2):
            provider = LocalSandboxProvider(
                [template], base_dir=area / f"sandboxes-{index}", users=ProcessUsers(plan=RunnerPlan())
            )
            providers.append(provider)
        sandbox = providers[0].create("runner", role="measurement", timeout_sec=60, metadata={})
        _wait(lambda: _log_says(sandbox, "app pid seen"))

    said = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert said == [f"{UNCONFINED_TESTS} (running them as an account: {TEST_ACCOUNT}; bubblewrap: not found on PATH)"]
    assert not sandbox.confined, "the sandbox runs as before: its runner sees the app's processes"


def test_a_probe_that_gave_no_answer_in_time_leaves_only_that_launch_unconfined(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeBwrap(area)
    late = BwrapProbe(str(fake.path), None, (f"bubblewrap at {fake.path}: no answer in 10 s",), transient=True)
    answers = [late, late, fake.isolation().probe]
    asked: list[BwrapProbe] = []

    def answer(_path: str | None) -> BwrapProbe:
        asked.append(answers.pop(0))
        return asked[-1]

    monkeypatch.setattr(agent_isolation, "bwrap_probe", answer)
    with caplog.at_level(logging.WARNING):
        provider = LocalSandboxProvider(
            [_runner(gpus=0)], base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan())
        )
        providers.append(provider)
        made = [provider.create("runner", role="measurement", timeout_sec=60, metadata={}) for _ in range(3)]

    assert [sandbox.confined for sandbox in made] == [False, True, True], "a later launch is chosen for again"
    assert len(asked) == 3, "an answer that came is kept"
    said = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert said == [f"{UNCONFINED_TESTS} (running them as an account: {TEST_ACCOUNT}; {late.problems[0]})"]


def test_a_probe_that_answered_no_is_not_asked_again(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = BwrapProbe(None, None, ("bubblewrap: not found on PATH",))
    asked: list[str | None] = []

    def answer(path: str | None) -> BwrapProbe:
        asked.append(path)
        return missing

    monkeypatch.setattr(agent_isolation, "bwrap_probe", answer)
    provider = LocalSandboxProvider(
        [_runner(gpus=0)], base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan())
    )
    providers.append(provider)
    made = [provider.create("runner", role="measurement", timeout_sec=60, metadata={}) for _ in range(2)]

    assert [sandbox.confined for sandbox in made] == [False, False] and asked == [None]


@pytest.mark.parametrize("layer", [None, "/nonexistent/hyperloom-host/bin/bwrap"])
def test_a_hosted_run_where_bubblewrap_cannot_sandbox_a_test_refuses_the_provider(
    monkeypatch: pytest.MonkeyPatch, layer: str | None
) -> None:
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap=layer))
    made: list[str] = []
    real = tempfile.mkdtemp

    def mkdtemp(*args: object, **kwargs: object) -> str:
        made.append(real(*args, **kwargs))  # type: ignore[arg-type]
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    with pytest.raises(IsolationRefused) as refused:
        LocalSandboxProvider([_runner()], gpus=(0,), users=ProcessUsers(plan=RunnerPlan()))

    named = agent_isolation.NO_LAYER_BWRAP if layer is None else f"bubblewrap at {layer}: No such file or directory"
    assert TEST_ACCOUNT in str(refused.value) and named in str(refused.value)
    assert made and not os.path.exists(made[0]), "the directory the refused provider made is removed"
    assert local_test_isolation() == (REFUSED, refused.value.reasons)


def test_a_sandbox_whose_paths_cannot_be_bound_safely_runs_as_before_locally_and_is_refused_hosted(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    exposing = RunnerPlan(readable=(area,))  # holds the provider's directory, every sandbox's key
    provider = LocalSandboxProvider(
        [_runner()], gpus=(0,), base_dir=area / "sandboxes", users=ProcessUsers(plan=exposing)
    )
    providers.append(provider)
    with caplog.at_level(logging.WARNING):
        sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert not sandbox.confined and fake.launches() == []
    assert f"{area} would expose {area / 'sandboxes'}, which only the app may reach" in caplog.text
    sandbox.kill()

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap=str(fake.path)))
    with pytest.raises(SandboxDenied, match="would expose"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert provider.pool.free_cards() == (0,), "the refused create gave its card back"


@pytest.mark.parametrize("location", ["home", "login-file"])
def test_a_plan_path_that_is_a_credential_location_runs_the_sandbox_as_before_locally_and_is_refused_hosted(
    area: Path,
    providers: list[LocalSandboxProvider],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    location: str,
) -> None:
    """A mask hides what lies below a bound path, never the path itself: bound, the home (or a login file a variable
    names) would show every credential in it unmasked, so its sandbox is not launched in bubblewrap."""
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    home = area / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text("SECRET")
    (home / ".netrc").write_text("SECRET")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NETRC", str(home / ".netrc"))
    named = home if location == "home" else home / ".netrc"
    linked = area / "linked"
    linked.symlink_to(named)  # resolved before it is compared: a link to the location is the location
    provider = LocalSandboxProvider(
        [_runner()], gpus=(0,), base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan(readable=(linked,)))
    )
    providers.append(provider)
    with caplog.at_level(logging.WARNING):
        sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert not sandbox.confined and fake.launches() == []
    assert f"{linked} is a credential location, which its bind would show unmasked" in caplog.text
    sandbox.kill()

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap=str(fake.path)))
    with pytest.raises(SandboxDenied, match="is a credential location"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert fake.launches() == []


def test_a_plan_that_refused_a_tree_runs_the_sandbox_as_before_locally_and_is_refused_hosted(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A plan whose maker found a tree it cannot bind safely (a model link into the home's ``.ssh``) says why; its
    sandboxes are never launched in bubblewrap without it, nor with it."""
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    (area / "model").mkdir()
    why = f"{area}/model/keys links to {area}/home/.ssh, in {area}/home/.ssh, a hidden entry of a home"
    plan = RunnerPlan(readable=(area / "model",), refused=(why,))
    provider = LocalSandboxProvider([_runner()], gpus=(0,), base_dir=area / "sandboxes", users=ProcessUsers(plan=plan))
    providers.append(provider)
    with caplog.at_level(logging.WARNING):
        sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert not sandbox.confined and fake.launches() == []
    assert "runs unconfined, as before: " in caplog.text and why in caplog.text
    sandbox.kill()

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap=str(fake.path)))
    with pytest.raises(SandboxDenied, match="a hidden entry of a home"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert fake.launches() == []
    assert provider.pool.free_cards() == (0,), "the refused create gave its card back"


def test_a_plan_that_would_show_the_app_s_state_runs_the_sandbox_as_before(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    app, state = area / "app", area / "app" / "state"
    state.mkdir(parents=True)
    monkeypatch.setenv("HYPERLOOM_STATE_DIR", str(state))
    plan = RunnerPlan(readable=(app,))
    provider = LocalSandboxProvider([_runner()], gpus=(0,), base_dir=area / "sandboxes", users=ProcessUsers(plan=plan))
    providers.append(provider)
    with caplog.at_level(logging.WARNING):
        sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})

    assert not sandbox.confined and fake.launches() == []
    assert f"{app} would expose {state}, which only the app may reach" in caplog.text


def test_an_older_bubblewrap_gives_a_sealed_runner_below_tmp_a_fresh_tmp_and_runs_any_other_sandbox_as_before(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation((0, 11, 1))))
    base = Path(tempfile.mkdtemp(prefix="hl-test-sandboxes-", dir="/tmp"))
    try:
        provider = LocalSandboxProvider(
            [_runner(), LocalTemplate("dev", "dev")], gpus=(0,), base_dir=base, users=ProcessUsers(plan=RunnerPlan())
        )
        providers.append(provider)
        runner = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
        dev = provider.create("dev", role="dev", timeout_sec=60, metadata={})
        _wait(lambda: _log_says(runner, "runner started"))
        [launch] = fake.launches()
        pairs = _flag_pairs(launch["argv"])  # type: ignore[arg-type]
    finally:
        for made in providers:
            made.close()
        shutil.rmtree(base, ignore_errors=True)

    assert runner.confined and ("--tmpfs", "/tmp") in pairs
    assert not [pair for pair in pairs if pair[-1] == "/tmp" and pair[0] == "--bind"]
    assert [pair for pair in pairs if pair[-1] == "/dev/shm" and pair[0] == "--bind"], "/dev/shm is still its own"
    assert not dev.confined, "commands share the bound /tmp, below which 0.11.1 binds nothing"


def test_a_bubblewrap_that_cannot_set_the_sandbox_up_is_the_runner_s_exit(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBwrap(area, fail=True)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    provider = LocalSandboxProvider(
        [_runner()], gpus=(0,), base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan())
    )
    providers.append(provider)
    sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})

    _wait(lambda: sandbox.startup_exit_code() is not None)
    assert sandbox.startup_exit_code() == 1
    assert _log_says(sandbox, "bwrap: Can't bind mount"), "bubblewrap's own line is in the runner's log"


@pytest.mark.parametrize(
    "failure",
    [
        BwrapSetupError("bwrap: Can't mount proc on /newroot/proc: Operation not permitted"),
        TimeoutError("the holder neither started nor failed in time"),
        ProcessLookupError(3, "the sandbox holder has ended"),
    ],
    ids=["setup", "timeout", "holder-gone"],
)
def test_a_command_whose_holder_cannot_be_had_is_refused_and_never_runs_unconfined(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    provider = LocalSandboxProvider(
        [LocalTemplate("dev", "dev")], base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan())
    )
    providers.append(provider)
    sandbox = provider.create("dev", role="dev", timeout_sec=60, metadata={})

    def fails(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(confined.Holder, "start", fails)
    with pytest.raises(SandboxUnavailable, match="bubblewrap cannot start the process"):
        sandbox.run("touch ran")
    with pytest.raises(SandboxUnavailable, match="bubblewrap cannot start the process"):
        sandbox.run("touch ran-in-background", background=True)
    assert sandbox.confined and fake.launches() == []
    assert not (sandbox.fs / "work" / "ran").exists() and not (sandbox.fs / "work" / "ran-in-background").exists()


def test_a_runner_whose_policy_bubblewrap_must_not_launch_is_refused_and_never_runs_unconfined(
    area: Path, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeBwrap(area)
    monkeypatch.setattr(confined, "choose_test_isolation", _given(fake.isolation()))
    marker = area / "ran"
    provider = LocalSandboxProvider(
        [_runner(code=f"open({str(marker)!r}, 'w').close()")],
        gpus=(0,),
        base_dir=area / "sandboxes",
        users=ProcessUsers(plan=RunnerPlan()),
    )
    providers.append(provider)

    def refuses(*_args: object, **_kwargs: object) -> object:
        raise IsolationRefused(["bubblewrap 0.9.0 cannot bind below a bound /tmp"])

    monkeypatch.setattr(BwrapIsolation, "launch", refuses)
    with pytest.raises(SandboxUnavailable, match="cannot bind below a bound /tmp"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    assert not marker.exists() and fake.launches() == []
    assert provider.pool.free_cards() == (0,), "the refused create gave its card back"


def test_how_a_local_test_is_launched_is_named_with_why(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())
    denied = BwrapProbe("/usr/bin/bwrap", (0, 11, 1), ("bwrap: setting up uid map: Permission denied",))
    monkeypatch.setattr(agent_isolation, "bwrap_probe", lambda _path: ok)
    assert local_test_isolation() == (BWRAP, ())
    monkeypatch.setattr(agent_isolation, "bwrap_probe", lambda _path: denied)
    how, why = local_test_isolation()
    assert how == NONE and why == (
        f"running them as an account: {TEST_ACCOUNT}",
        "bubblewrap (/usr/bin/bwrap, 0.11.1): bwrap: setting up uid map: Permission denied",
    )
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap="/usr/bin/bwrap"))
    assert local_test_isolation() == (REFUSED, why)
