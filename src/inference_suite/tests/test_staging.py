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
from inference_suite import staging
from inference_suite.contracts import FailedStep
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT, INFERENCEX_TOOL, PINNED_TOOLS
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


class _FakeInstall:
    """Stands in for ``python -m venv`` and ``pip install``: lays down the files a real venv would, and records argv."""

    def __init__(self, *, fail: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, argv: list[str], what: str) -> None:
        self.calls.append(list(argv))
        if self.fail is not None and self.fail in what:
            raise StagingError(f"{what} exited with code 1: no network")
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


def test_install_stages_after_ensure_inferencex() -> None:
    text = _INSTALL_SH.read_text(encoding="utf-8")

    assert text.index("\nensure_inference_suite_staging\n") > text.index("\nensure_inferencex\n")
