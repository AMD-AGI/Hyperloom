# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of offline staging: the manifest digest a round names, and ``staging_missing`` for anything not staged."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import inference_suite
from inference_suite import cli, staging
from inference_suite.contracts import FailedStep
from inference_suite.pins import (
    INFERENCEX_CLIENT_COMMIT,
    INFERENCEX_TOOL,
    LM_EVAL_DISTRIBUTION,
    LM_EVAL_REQUIREMENT,
    LM_EVAL_VERSION,
    PINNED_TOOLS,
)
from inference_suite.staging import (
    AGENTX_AGGREGATOR_FILE,
    AGENTX_PYTHON,
    AGENTX_REQUIREMENTS,
    CLIENT_FILE,
    CLIENT_MODULE,
    GSM8K_TASK,
    INFERENCEX_PROJECT_DIR,
    MANIFEST_FILE,
    StagedTool,
    StagingError,
    StagingManifest,
    inferencex_root,
    stage,
    tool_root,
    verify,
)

_CLIENT = "infx/bench_serving/benchmark_serving.py"
_CLIENT_TEXT = "print('client')\n"
_INSTALL_SH = Path(__file__).resolve().parents[2] / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
_SRC = Path(__file__).resolve().parents[2]
_SUITE_INSTALL_SH = _SRC / "inference_suite" / "install.sh"
_SHA = "a" * 64
_HARNESS_RECORD = "agentx-venv/lib/python3.12/site-packages/agentx_harness-1.0.13.dist-info/RECORD"
_REQUIREMENTS_TEXT = "agentx-harness==1.0.13\n"


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def _checkout(tmp_path: Path, *, without: str | None = None) -> tuple[Path, str]:
    repo = tmp_path / "InferenceX"
    project = repo / INFERENCEX_PROJECT_DIR
    files = {
        "infx/__init__.py": "",
        "infx/bench_serving/__init__.py": "",
        _CLIENT: _CLIENT_TEXT,
        "infx/bench_serving/KNOWN_LIMITATION.md": "limits\n",
        GSM8K_TASK: "task: gsm8k\n",
        "infx/evals/gpqa_diamond.yaml": "task: gpqa\n",
        "infx/results/__init__.py": "",
        "infx/results/agentic/__init__.py": "",
        AGENTX_AGGREGATOR_FILE: "print('aggregate')\n",
        AGENTX_REQUIREMENTS: _REQUIREMENTS_TEXT,
        "infx/bench/agentic/replay.py": "print('replay')\n",
    }
    for relative, text in files.items():
        if without is not None and relative.startswith(without):
            continue
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    if (project / "infx/bench_serving").is_dir():
        (project / "infx/bench_serving/__pycache__").mkdir()
        (project / "infx/bench_serving/__pycache__/benchmark_serving.cpython-312.pyc").write_bytes(b"\0")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


_REAL_STAGING_STEP = staging._run_staging_step


class _FakeInstall:
    """Stands in for ``python -m venv`` and ``pip install``: lays down the files a real venv would, and records argv.

    git runs for real, so a fetch reads the fixture checkout.
    """

    def __init__(self, *, fail: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, argv: list[str], what: str) -> None:
        if argv[0] == "git":
            _REAL_STAGING_STEP(argv, what)
            return
        self.calls.append(list(argv))
        if self.fail is not None and self.fail in what:
            raise StagingError(f"{what} exited with code 1: no network")
        if argv[0] == sys.executable and argv[1:3] == ["-m", "pip"]:
            return
        if argv[1:3] == ["-m", "venv"]:
            venv = Path(argv[3])
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("interpreter\n")
        else:
            record = Path(argv[0]).parents[1] / Path(_HARNESS_RECORD).relative_to("agentx-venv")
            record.parent.mkdir(parents=True)
            record.write_text("agentx_cli.py,sha256=x,58\n")


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> _FakeInstall:
    fake = _FakeInstall()
    monkeypatch.setattr(staging, "_run_staging_step", fake)
    return fake


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: _FakeInstall) -> tuple[Path, str, str]:
    """A tool cache staged from a fixture checkout, pinned at that checkout's commit."""
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})
    cache = tmp_path / "tools"
    cache.mkdir()
    return cache, commit, stage(repo, cache)


def test_the_stage_digest_is_the_manifest_file_sha256_and_verifies_with_the_pin_recorded(staged) -> None:
    cache, commit, digest = staged

    assert digest == hashlib.sha256((cache / MANIFEST_FILE).read_bytes()).hexdigest()
    report = verify(cache, digest)
    assert (report.failed_step, report.missing) == (None, ())
    assert report.pins == {INFERENCEX_TOOL: commit}
    assert report.manifest_sha256 == digest


def test_staging_copies_the_client_the_task_the_agentx_aggregator_and_the_harness_venv_and_nothing_else(
    staged,
) -> None:
    cache, commit, digest = staged

    root = inferencex_root(cache)
    assert root == tool_root(cache, INFERENCEX_TOOL, commit)
    manifest = StagingManifest.from_document(json.loads((cache / MANIFEST_FILE).read_text()))
    assert set(manifest.tools[INFERENCEX_TOOL].files) == {
        "infx/__init__.py",
        "infx/bench_serving/KNOWN_LIMITATION.md",
        "infx/bench_serving/__init__.py",
        _CLIENT,
        GSM8K_TASK,
        "infx/results/__init__.py",
        "infx/results/agentic/__init__.py",
        AGENTX_AGGREGATOR_FILE,
        AGENTX_REQUIREMENTS,
        AGENTX_PYTHON,
        _HARNESS_RECORD,
    }
    assert (root / _CLIENT).read_text() == _CLIENT_TEXT
    assert not (root / "infx/bench_serving/__pycache__").exists()


def test_the_harness_venv_is_built_in_place_by_the_staging_interpreter_from_the_staged_requirements(
    staged, install: _FakeInstall
) -> None:
    cache, _, _ = staged
    root = inferencex_root(cache)

    assert install.calls == [
        [sys.executable, "-m", "venv", str(root / "agentx-venv")],
        [
            str(root / AGENTX_PYTHON),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--disable-pip-version-check",
            "-r",
            str(root / AGENTX_REQUIREMENTS),
        ],
    ]
    assert (root / AGENTX_REQUIREMENTS).read_text() == _REQUIREMENTS_TEXT


def test_staging_refuses_an_interpreter_the_harness_does_not_support(tmp_path, monkeypatch, install) -> None:
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})
    monkeypatch.setattr(staging, "AGENTX_PYTHON_VERSIONS", ((3, 11),))

    with pytest.raises(StagingError, match="needs Python 3.11"):
        stage(repo, tmp_path / "tools")
    assert install.calls == []
    assert not (tmp_path / "tools" / MANIFEST_FILE).exists()


def test_a_harness_that_cannot_be_installed_leaves_no_manifest(tmp_path, monkeypatch) -> None:
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})
    monkeypatch.setattr(staging, "_run_staging_step", _FakeInstall(fail="installing"))

    with pytest.raises(StagingError, match="installing the AgentX harness exited with code 1: no network"):
        stage(repo, tmp_path / "tools")
    assert not (tmp_path / "tools" / MANIFEST_FILE).exists()


def test_a_failing_staging_step_names_the_step_and_its_stderr() -> None:
    with pytest.raises(StagingError, match="the step exited with code 3: broken"):
        staging._run_staging_step(
            [sys.executable, "-c", "import sys; sys.stderr.write('broken'); sys.exit(3)"], "the step"
        )


def test_staging_the_same_checkout_again_gives_the_same_digest(staged, tmp_path: Path) -> None:
    cache, _, digest = staged

    assert stage(tmp_path / "InferenceX", cache) == digest
    assert not [path.name for path in (cache / INFERENCEX_TOOL).iterdir() if path.name.startswith(".")]


def test_a_cache_without_a_manifest_is_staging_missing(tmp_path: Path) -> None:
    report = verify(tmp_path, _SHA)

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert MANIFEST_FILE in report.detail and "absent" in report.detail


def test_a_manifest_other_than_the_one_the_round_names_is_staging_missing(staged) -> None:
    cache, _, digest = staged

    report = verify(cache, _SHA)

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert digest in report.detail and _SHA in report.detail


@pytest.mark.parametrize(
    ("damage", "state"),
    [(lambda path: path.unlink(), "absent"), (lambda path: path.write_text("edited\n"), "changed since staging")],
)
def test_a_staged_file_absent_or_edited_is_staging_missing_and_named(staged, damage, state) -> None:
    cache, _, digest = staged
    damage(inferencex_root(cache) / GSM8K_TASK)

    report = verify(cache, digest)

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert report.missing == (f"{inferencex_root(cache) / GSM8K_TASK} ({state})",)


def test_a_cache_staged_at_another_commit_than_the_pin_is_staging_missing(staged, monkeypatch) -> None:
    cache, commit, digest = staged
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: "b" * 40})

    report = verify(cache, digest)

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert report.missing == (f"{INFERENCEX_TOOL} at its pin {'b' * 40} (staged at {commit})",)


def test_staging_refuses_a_checkout_off_the_pin(tmp_path: Path, install) -> None:
    repo, commit = _checkout(tmp_path)

    with pytest.raises(StagingError, match=f"checked out at '{commit}'.*pinned at {INFERENCEX_CLIENT_COMMIT}"):
        stage(repo, tmp_path / "tools")
    assert not (tmp_path / "tools" / MANIFEST_FILE).exists()


def test_staging_refuses_a_checkout_without_the_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, commit = _checkout(tmp_path, without="infx/bench_serving")
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})

    with pytest.raises(StagingError, match="has no infx/bench_serving"):
        stage(repo, tmp_path / "tools")
    assert not (tmp_path / "tools").exists()


def test_staging_refuses_a_checkout_whose_client_package_lacks_the_client_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, commit = _checkout(tmp_path, without=_CLIENT)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})

    with pytest.raises(StagingError, match=f"has no {re.escape(_CLIENT)}$"):
        stage(repo, tmp_path / "tools")
    assert not (tmp_path / "tools").exists()


def test_the_client_file_is_the_one_the_client_module_loads() -> None:
    assert (CLIENT_MODULE, CLIENT_FILE) == ("infx.bench_serving.benchmark_serving", _CLIENT)


@pytest.mark.parametrize("required", [_CLIENT, AGENTX_AGGREGATOR_FILE, AGENTX_PYTHON])
def test_a_manifest_at_the_pin_that_does_not_stage_a_file_a_load_runs_is_staging_missing(staged, required) -> None:
    cache, commit, _ = staged
    path = cache / MANIFEST_FILE
    manifest = StagingManifest.from_document(json.loads(path.read_text()))
    files = {name: digest for name, digest in manifest.tools[INFERENCEX_TOOL].files.items() if name != required}
    data = StagingManifest({INFERENCEX_TOOL: StagedTool(commit, files)}).to_bytes()
    path.write_bytes(data)

    report = verify(cache, hashlib.sha256(data).hexdigest())

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert report.missing == (f"{INFERENCEX_TOOL}'s {required} (not staged)",)


def test_the_manifest_round_trips_and_refuses_what_it_does_not_know() -> None:
    manifest = StagingManifest({INFERENCEX_TOOL: StagedTool("c" * 40, {GSM8K_TASK: _SHA})})

    assert StagingManifest.from_document(json.loads(manifest.to_bytes())) == manifest
    document = manifest.to_document()
    with pytest.raises(StagingError, match="unknown key 'extra'"):
        StagingManifest.from_document({**document, "extra": 1})
    with pytest.raises(StagingError, match="schema"):
        StagingManifest.from_document({**document, "schema": "inference-suite-staging/0"})
    with pytest.raises(StagingError, match="commit"):
        StagedTool("main", {GSM8K_TASK: _SHA})
    with pytest.raises(StagingError, match="inside the tool's root"):
        StagedTool("c" * 40, {"../escape": _SHA})
    with pytest.raises(StagingError, match="lists no files"):
        StagedTool("c" * 40, {})


def test_an_unreadable_manifest_is_staging_missing(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_FILE).write_text("not json")
    digest = hashlib.sha256(b"not json").hexdigest()

    report = verify(tmp_path, digest)

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert "readable staging manifest" in report.detail


def test_check_exits_one_naming_staging_missing_then_zero_with_the_digest_once_staged(staged, tmp_path, capsys) -> None:
    cache, _, digest = staged

    assert staging.main(["check", "--tool-cache", str(tmp_path / "empty")]) == 1
    assert capsys.readouterr().err.startswith(f"{FailedStep.STAGING_MISSING.value}: ")
    assert staging.main(["check", "--tool-cache", str(cache)]) == 0
    assert capsys.readouterr().out.strip() == digest


def test_the_manifest_digest_a_host_names_is_the_one_stage_printed_and_none_without_a_manifest(
    staged, tmp_path
) -> None:
    cache, _, digest = staged

    assert inference_suite.manifest_sha256(cache) == digest
    assert inference_suite.manifest_sha256(tmp_path / "empty") is None


def test_pinned_tools_name_the_client_the_base_launch_records() -> None:
    assert dict(PINNED_TOOLS) == {INFERENCEX_TOOL: INFERENCEX_CLIENT_COMMIT}


def _install_function(name: str) -> str:
    text = _INSTALL_SH.read_text(encoding="utf-8")
    header = re.search(rf"(?m)^{re.escape(name)}\(\) \{{", text)
    assert header is not None, f"install.sh has no {name}()"
    block = text[header.start() :]
    return block[: re.search(r"(?m)^\}", block).end()]


def _run_install_staging(cache: Path, *, check_only: int, fetch: str) -> subprocess.CompletedProcess[str]:
    script = f"""
set -euo pipefail
log() {{ echo "LOG $*"; }}
warn() {{ echo "WARN $*"; }}
die() {{ echo "DIE $*"; exit 1; }}
git_fetch_pinned() {{ {fetch}; }}
PYTHON={sys.executable}
INFERENCEX_REPO=https://example.invalid/InferenceX.git
INFERENCE_SUITE_TOOL_CACHE={cache}
CHECK_ONLY={check_only}
DRY_RUN=0
_STALE_KEYS=()
{_install_function("ensure_inference_suite_staging")}
ensure_inference_suite_staging
echo "STALE ${{_STALE_KEYS[*]:-}}"
"""
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120, check=False)


def _cache_at_the_real_pin(cache: Path) -> str:
    root = tool_root(cache, INFERENCEX_TOOL, INFERENCEX_CLIENT_COMMIT)
    files = {}
    staged = ((GSM8K_TASK, "task: gsm8k\n"), (_CLIENT, _CLIENT_TEXT), (AGENTX_AGGREGATOR_FILE, ""), (AGENTX_PYTHON, ""))
    for relative, text in staged:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
        files[relative] = hashlib.sha256(text.encode()).hexdigest()
    data = StagingManifest({INFERENCEX_TOOL: StagedTool(INFERENCEX_CLIENT_COMMIT, files)}).to_bytes()
    (cache / MANIFEST_FILE).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_install_leaves_a_cache_staged_at_the_pin_alone(tmp_path: Path) -> None:
    digest = _cache_at_the_real_pin(tmp_path)

    done = _run_install_staging(tmp_path, check_only=0, fetch="echo FETCHED; return 1")

    assert done.returncode == 0, done.stderr
    assert "FETCHED" not in done.stdout
    assert f"manifest sha256 {digest}" in done.stdout


def test_install_check_only_reports_an_unstaged_cache_as_stale(tmp_path: Path) -> None:
    done = _run_install_staging(tmp_path / "tools", check_only=1, fetch="echo FETCHED; return 1")

    assert done.returncode == 0, done.stderr
    assert "FETCHED" not in done.stdout
    assert "STALE INFERENCE_SUITE_TOOL_CACHE" in done.stdout


def test_install_fetches_the_pinned_commit_and_warns_when_it_cannot(tmp_path: Path) -> None:
    done = _run_install_staging(tmp_path / "tools", check_only=0, fetch='echo "FETCHED $3"; return 1')

    assert done.returncode == 0, done.stderr
    assert f"FETCHED {INFERENCEX_CLIENT_COMMIT}" in done.stdout
    assert "WARN could not fetch InferenceX" in done.stdout and "staging_missing" in done.stdout


def test_install_stages_under_the_install_lock_and_installs_no_magpie_or_inferencex_checkout() -> None:
    text = _INSTALL_SH.read_text(encoding="utf-8")

    assert text.index("\nensure_inference_suite_staging\n") > text.index("\nacquire_install_lock\n")
    assert "magpie" not in text.lower()
    assert "ensure_inferencex" not in text
    assert "INFERENCEX_REF" not in text


def _lm_eval_install() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", LM_EVAL_REQUIREMENT]


@pytest.fixture
def pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: _FakeInstall) -> tuple[Path, Path]:
    """A fixture checkout pinned, lm-eval installed at its pin, and an empty tool cache: what ``stage`` starts from."""
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: LM_EVAL_VERSION}.get)
    return repo, tmp_path / "tools"


def _stage_cli(cache: Path, *source: str) -> int:
    return cli.main(["stage", *source, "--tool-cache", str(cache)])


def test_stage_prints_the_manifest_sha256_and_a_second_run_stages_nothing(pinned, install, capsys) -> None:
    repo, cache = pinned

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_OK
    digest = capsys.readouterr().out.strip()
    assert digest == staging.manifest_sha256(cache)
    assert verify(cache, digest).failed_step is None
    calls = len(install.calls)
    manifest_mtime = (cache / MANIFEST_FILE).stat().st_mtime_ns

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == digest
    assert len(install.calls) == calls
    assert (cache / MANIFEST_FILE).stat().st_mtime_ns == manifest_mtime


def test_stage_without_a_source_fetches_inferencex_at_its_pin(pinned, monkeypatch, capsys) -> None:
    repo, cache = pinned
    monkeypatch.setattr(staging, "INFERENCEX_REPO_URL", str(repo))

    assert _stage_cli(cache) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == staging.manifest_sha256(cache)
    assert (inferencex_root(cache) / _CLIENT).read_text() == _CLIENT_TEXT
    assert sorted(path.name for path in cache.iterdir()) == [INFERENCEX_TOOL, MANIFEST_FILE]


def test_stage_refuses_a_fetch_that_fails_with_staging_missing(pinned, monkeypatch, capsys) -> None:
    _, cache = pinned
    monkeypatch.setattr(staging, "INFERENCEX_REPO_URL", str(cache.parent / "no-such-repo"))

    assert _stage_cli(cache) == cli.EXIT_REFUSED
    assert capsys.readouterr().err.startswith(f"{FailedStep.STAGING_MISSING.value}: fetching InferenceX ")
    assert not (cache / MANIFEST_FILE).exists()
    assert [path.name for path in cache.iterdir()] == []


@pytest.mark.parametrize(
    ("damage", "state"),
    [(lambda path: path.unlink(), "absent"), (lambda path: path.write_text("edited\n"), "changed since staging")],
)
def test_stage_refuses_a_damaged_cache_naming_staging_missing_and_leaves_it_as_found(
    pinned, install, capsys, damage, state
) -> None:
    repo, cache = pinned
    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_OK
    capsys.readouterr()
    task = inferencex_root(cache) / GSM8K_TASK
    damage(task)
    calls = len(install.calls)

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_REFUSED
    assert capsys.readouterr().err.strip() == f"{FailedStep.STAGING_MISSING.value}: {task} ({state})"
    assert len(install.calls) == calls
    assert task.exists() == (state != "absent")


def test_stage_refuses_an_unreadable_manifest(pinned, capsys) -> None:
    repo, cache = pinned
    cache.mkdir()
    (cache / MANIFEST_FILE).write_text("not json")

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_REFUSED
    assert "readable staging manifest" in capsys.readouterr().err
    assert (cache / MANIFEST_FILE).read_text() == "not json"


def test_stage_restages_a_cache_staged_at_a_pin_no_longer_in_force(pinned, monkeypatch, capsys) -> None:
    repo, cache = pinned
    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_OK
    old = capsys.readouterr().out.strip()
    _git(repo, "commit", "-q", "--allow-empty", "-m", "bump")
    commit = _git(repo, "rev-parse", "HEAD").strip()
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_OK
    digest = capsys.readouterr().out.strip()
    assert digest != old
    assert verify(cache, digest).pins == {INFERENCEX_TOOL: commit}


def test_stage_refuses_a_source_off_the_pin_with_staging_missing(pinned, monkeypatch, capsys) -> None:
    repo, cache = pinned
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: "b" * 40})

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.startswith(f"{FailedStep.STAGING_MISSING.value}: ") and f"pinned at {'b' * 40}" in err


@pytest.mark.parametrize("source", [False, True])
def test_stage_without_git_on_path_is_refused_with_staging_missing(pinned, tmp_path, monkeypatch, capsys, source):
    repo, cache = pinned
    monkeypatch.setattr(staging, "_run_staging_step", _REAL_STAGING_STEP)
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))

    assert _stage_cli(cache, *(("--source", str(repo)) if source else ())) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.startswith(f"{FailedStep.STAGING_MISSING.value}: ") and "'git'" in err
    assert not (cache / MANIFEST_FILE).exists()


def test_stage_into_a_cache_that_cannot_be_created_is_refused_with_staging_missing(pinned, tmp_path, capsys) -> None:
    repo, _ = pinned
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    cache = blocker / "tools"

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.startswith(f"{FailedStep.STAGING_MISSING.value}: ") and str(cache) in err


@pytest.mark.parametrize(("installed", "installs"), [(None, True), ("0.4.11", True), (LM_EVAL_VERSION, False)])
def test_stage_installs_lm_eval_at_its_pin_into_the_staging_interpreter_unless_it_is_there(
    pinned, install, monkeypatch, capsys, installed, installs
) -> None:
    repo, cache = pinned
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: installed}.get)

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_OK
    assert (_lm_eval_install() in install.calls) is installs


def test_a_failed_lm_eval_install_is_refused_and_stages_nothing(pinned, monkeypatch, capsys) -> None:
    repo, cache = pinned
    monkeypatch.setattr(staging, "_distribution_version", {}.get)
    monkeypatch.setattr(staging, "_run_staging_step", _FakeInstall(fail=LM_EVAL_REQUIREMENT))

    assert _stage_cli(cache, "--source", str(repo)) == cli.EXIT_REFUSED
    assert LM_EVAL_REQUIREMENT in capsys.readouterr().err
    assert not (cache / MANIFEST_FILE).exists()


def test_the_lm_eval_pin_is_the_release_the_runner_extra_locks() -> None:
    pyproject = (_SRC.parent / "pyproject.toml").read_text(encoding="utf-8")

    assert f'"{LM_EVAL_REQUIREMENT}",' in pyproject


def _run_suite_install(*args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHON"] = sys.executable
    return subprocess.run(
        ["bash", str(_SUITE_INSTALL_SH), *args], env=env, cwd="/", capture_output=True, text=True, check=False
    )


def test_install_sh_is_an_executable_that_takes_exactly_the_tool_cache(tmp_path: Path) -> None:
    assert os.access(_SUITE_INSTALL_SH, os.X_OK)
    for args in ((), (str(tmp_path), "extra")):
        done = _run_suite_install(*args)
        assert done.returncode == cli.EXIT_REFUSED
        assert done.stderr.startswith("usage: install.sh <tool-cache-dir>")


def test_install_sh_stages_with_the_inference_suite_beside_it_and_refuses_a_damaged_cache(tmp_path: Path) -> None:
    _cache_at_the_real_pin(tmp_path)
    task = inferencex_root(tmp_path) / GSM8K_TASK
    task.write_text("edited\n")

    done = _run_suite_install(str(tmp_path))

    assert done.returncode == cli.EXIT_REFUSED, done.stderr
    assert done.stderr.strip() == f"{FailedStep.STAGING_MISSING.value}: {task} (changed since staging)"
