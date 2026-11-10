# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of offline staging: the manifest digest a round names, and ``staging_missing`` for anything not staged."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import inference_suite
from hyperloom_common.shared_storage import current_umask
from inference_suite import cli, staging
from inference_suite.contracts import FailedStep
from inference_suite.pins import (
    INFERENCEX_CLIENT_COMMIT,
    INFERENCEX_TOOL,
    LM_EVAL_DISTRIBUTION,
    LM_EVAL_REQUIREMENT,
    LM_EVAL_VERSION,
    LM_EVAL_WHEEL_SHA256,
    PINNED_TOOLS,
)
from inference_suite.client_env import BUILD_FILE, STAGED_LOCK
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
from inference_suite.tests import aiperf_source
from inference_suite.tests.client_env_fakes import WHEELS, FakeClient

_CLIENT = "infx/bench_serving/benchmark_serving.py"
_CLIENT_TEXT = "print('client')\n"
_SRC = Path(__file__).resolve().parents[2]
_SUITE_INSTALL_SH = _SRC / "inference_suite" / "install.sh"
_SHA = "a" * 64
_SITE = "agentx-python/lib/python3.12/site-packages"
_RECORDS = {f"{_SITE}/{name.replace('-', '_')}-{version}.dist-info/RECORD" for name, version in WHEELS.items()}
_PATCHED = (f"{_SITE}/{aiperf_source.PARSER_FILE}", f"{_SITE}/{aiperf_source.METRIC_FILE}")
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
    """Stands in for ``pip install`` into this interpreter (lm-eval), and records argv. git runs for real, so a fetch
    reads the fixture checkout; the client environment is the fake build and wheel index of :class:`FakeClient`."""

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


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    return FakeClient(tmp_path, monkeypatch)


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> _FakeInstall:
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


def test_staging_copies_the_client_the_task_the_agentx_aggregator_and_the_client_environment_and_nothing_else(
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
        BUILD_FILE,
        STAGED_LOCK,
        *_RECORDS,
        *_PATCHED,
    }
    assert (root / _CLIENT).read_text() == _CLIENT_TEXT
    assert not (root / "infx/bench_serving/__pycache__").exists()


def test_the_normalised_manifest_is_the_pinned_build_the_lock_wheels_and_this_interpreters_lm_eval(
    staged, client: FakeClient, monkeypatch
) -> None:
    cache, commit, _ = staged
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: LM_EVAL_VERSION}.get)

    identity = inference_suite.normalised_manifest(cache)

    assert client.downloads == [client.build.url]
    assert identity.inferencex == commit
    assert (identity.interpreter_version, identity.interpreter_sha256) == (client.build.version, client.build.sha256)
    assert {package.name: package.wheel_sha256 for package in identity.packages} == {
        **client.lock_shas(),
        "lm-eval": LM_EVAL_WHEEL_SHA256,
    }


@pytest.mark.parametrize("installed", [None, "0.4.11"])
def test_the_normalised_manifest_refuses_an_lm_eval_off_its_pin(staged, monkeypatch, installed) -> None:
    cache, _, _ = staged
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: installed}.get)

    with pytest.raises(StagingError, match=re.escape(LM_EVAL_REQUIREMENT)):
        inference_suite.normalised_manifest(cache)


def test_a_client_environment_that_cannot_be_installed_leaves_no_manifest(tmp_path, monkeypatch, install, client):
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})
    next(client.index.glob("aiohttp-*")).write_bytes(b"another aiohttp")

    with pytest.raises(StagingError, match="downloading the wheels the AgentX client lock pins"):
        stage(repo, tmp_path / "tools")
    assert not (tmp_path / "tools" / MANIFEST_FILE).exists()


def test_staging_patches_the_harness_to_export_error_osl_and_the_manifest_names_the_patched_bytes(staged) -> None:
    cache, _, digest = staged
    root = inferencex_root(cache)
    parser = (root / _PATCHED[0]).read_text()
    assert "token_counts=record.token_counts or TokenCounts()," in parser
    assert 'tag = "error_osl"' in (root / _PATCHED[1]).read_text()
    files = StagingManifest.from_document(json.loads((cache / MANIFEST_FILE).read_text())).tools[INFERENCEX_TOOL].files
    for relative in _PATCHED:
        assert files[relative] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    record = (root / _SITE / "agentx_harness-1.0.13.dist-info" / "RECORD").read_bytes().decode()
    for relative in (aiperf_source.PARSER_FILE, aiperf_source.METRIC_FILE):
        line = aiperf_source.record_line(relative, (root / _SITE / relative).read_bytes())
        assert line in record, "the RECORD the manifest hashes lists the patched bytes"

    (root / _PATCHED[0]).write_text(aiperf_source.PARSER)

    assert f"{root / _PATCHED[0]} (changed since staging)" in verify(cache, digest).missing


def test_a_harness_that_cannot_be_patched_fails_the_staging_and_leaves_no_manifest(
    tmp_path, monkeypatch, install, client: FakeClient
):
    moved = aiperf_source.PARSER.replace("# Parsed record was invalid, return as error record", "# the branch moved")
    (client.harness / aiperf_source.PARSER_FILE).write_text(moved)
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})

    with pytest.raises(StagingError, match="cannot be patched.*inference_result_parser.py does not hold the branch"):
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


@pytest.mark.parametrize("required", [_CLIENT, AGENTX_AGGREGATOR_FILE, AGENTX_PYTHON, *_PATCHED])
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


def test_check_exits_one_naming_staging_missing_then_zero_with_the_digest_once_staged(
    staged, tmp_path, monkeypatch, capsys
) -> None:
    cache, _, digest = staged
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: LM_EVAL_VERSION}.get)

    assert staging.main(["check", "--tool-cache", str(tmp_path / "empty")]) == 1
    assert capsys.readouterr().err.startswith(f"{FailedStep.STAGING_MISSING.value}: ")
    assert staging.main(["check", "--tool-cache", str(cache)]) == 0
    assert capsys.readouterr().out.strip() == digest


@pytest.mark.parametrize("installed", [None, "0.4.11"])
def test_check_refuses_an_intact_cache_whose_interpreter_lacks_the_pinned_lm_eval(
    staged, monkeypatch, capsys, installed
) -> None:
    cache, _, _ = staged
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: installed}.get)

    assert staging.main(["check", "--tool-cache", str(cache)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"{FailedStep.STAGING_MISSING.value}: ")
    assert f"{sys.executable} has {LM_EVAL_DISTRIBUTION} {installed}, not {LM_EVAL_REQUIREMENT}" in err


@pytest.mark.parametrize(("installed", "installs"), [(None, True), ("0.4.11", True), (LM_EVAL_VERSION, False)])
def test_stage_without_a_source_installs_lm_eval_beside_an_intact_cache_and_stages_nothing(
    staged, install, monkeypatch, capsys, installed, installs
) -> None:
    cache, _, digest = staged
    install.calls.clear()
    manifest = (cache / MANIFEST_FILE).stat().st_mtime_ns
    monkeypatch.setattr(staging, "_distribution_version", {LM_EVAL_DISTRIBUTION: installed}.get)

    assert staging.main(["stage", "--tool-cache", str(cache)]) == 0
    assert capsys.readouterr().out.strip() == digest
    assert install.calls == ([_lm_eval_install()] if installs else [])
    assert (cache / MANIFEST_FILE).stat().st_mtime_ns == manifest


def test_stage_without_a_source_refuses_a_cache_that_does_not_verify_and_installs_nothing(
    staged, install, tmp_path, capsys
) -> None:
    cache, _, _ = staged
    install.calls.clear()
    task = inferencex_root(cache) / GSM8K_TASK
    task.write_text("edited\n")

    for tool_cache, named in ((tmp_path / "empty", MANIFEST_FILE), (cache, f"{task} (changed since staging)")):
        assert staging.main(["stage", "--tool-cache", str(tool_cache)]) == 1
        err = capsys.readouterr().err
        assert err.startswith(f"{FailedStep.STAGING_MISSING.value}: ") and named in err
    assert install.calls == []


def test_stage_from_a_source_installs_lm_eval_at_its_pin(staged, install, monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(staging, "_distribution_version", {}.get)
    install.calls.clear()

    assert staging.main(["stage", "--source", str(tmp_path / "InferenceX"), "--tool-cache", str(tmp_path / "t")]) == 0
    assert _lm_eval_install() in install.calls
    assert verify(tmp_path / "t", capsys.readouterr().out.strip()).failed_step is None


def test_the_manifest_digest_a_host_names_is_the_one_stage_printed_and_none_without_a_manifest(
    staged, tmp_path
) -> None:
    cache, _, digest = staged

    assert inference_suite.manifest_sha256(cache) == digest
    assert inference_suite.manifest_sha256(tmp_path / "empty") is None


def test_the_verified_digest_is_that_of_an_intact_cache_and_none_for_a_damaged_or_empty_one(staged, tmp_path) -> None:
    cache, _, digest = staged

    assert inference_suite.verified_manifest_sha256(cache) == digest
    (inferencex_root(cache) / GSM8K_TASK).write_text("edited\n")
    assert inference_suite.verified_manifest_sha256(cache) is None
    assert inference_suite.verified_manifest_sha256(tmp_path / "empty") is None


def test_the_package_names_the_inferencex_pin_a_host_keys_its_staging_by() -> None:
    assert inference_suite.INFERENCEX_CLIENT_COMMIT == PINNED_TOOLS[INFERENCEX_TOOL]


def test_pinned_tools_name_the_client_the_base_launch_records() -> None:
    assert dict(PINNED_TOOLS) == {INFERENCEX_TOOL: INFERENCEX_CLIENT_COMMIT}


def _cache_at_the_real_pin(cache: Path) -> str:
    root = tool_root(cache, INFERENCEX_TOOL, INFERENCEX_CLIENT_COMMIT)
    files = {}
    staged = ((GSM8K_TASK, "task: gsm8k\n"), (_CLIENT, _CLIENT_TEXT), (AGENTX_AGGREGATOR_FILE, ""), (AGENTX_PYTHON, ""))
    staged += tuple((relative, "") for relative in _PATCHED)
    for relative, text in staged:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
        files[relative] = hashlib.sha256(text.encode()).hexdigest()
    data = StagingManifest({INFERENCEX_TOOL: StagedTool(INFERENCEX_CLIENT_COMMIT, files)}).to_bytes()
    (cache / MANIFEST_FILE).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


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


@pytest.fixture
def publishing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: _FakeInstall) -> tuple[Path, Path]:
    """A fixture checkout pinned and the directory stagings are published in."""
    repo, commit = _checkout(tmp_path)
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: commit})
    return repo, tmp_path / "runner-cache" / "inference-suite"


def _publishing_dirs(stagings: Path) -> list[str]:
    return [path.name for path in stagings.iterdir() if path.name.startswith(staging.PUBLISHING_PREFIX)]


def test_publish_renames_a_verified_staging_into_its_key_and_leaves_no_private_directory(publishing) -> None:
    repo, stagings = publishing

    published = staging.publish(stagings, repo)

    assert published == stagings / staging.staging_key()
    assert verify(published, staging.manifest_sha256(published)).failed_step is None
    assert stat.S_IMODE(stagings.stat().st_mode) == 0o775 & ~current_umask(), "not shared storage: the umask decides"
    assert _publishing_dirs(stagings) == []
    assert (inferencex_root(published) / AGENTX_PYTHON).is_file()


def test_below_shared_storage_the_stagings_directory_is_the_groups(publishing, tmp_path: Path) -> None:
    repo, _stagings = publishing
    cache = tmp_path / "shared-cache"
    cache.mkdir()
    cache.chmod(0o2775)
    if not cache.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    stagings = cache / "inference-suite"

    staging.publish(stagings, repo)

    assert stat.S_IMODE(stagings.stat().st_mode) == 0o2775


def test_publish_uses_a_published_staging_as_it_is(publishing, client: FakeClient) -> None:
    repo, stagings = publishing
    published = staging.publish(stagings, repo)
    manifest_mtime = (published / MANIFEST_FILE).stat().st_mtime_ns

    assert staging.publish(stagings, repo) == published
    assert client.downloads == [client.build.url]
    assert (published / MANIFEST_FILE).stat().st_mtime_ns == manifest_mtime


def test_publish_refuses_a_published_staging_that_no_longer_verifies_and_leaves_it(publishing) -> None:
    repo, stagings = publishing
    published = staging.publish(stagings, repo)
    task = inferencex_root(published) / GSM8K_TASK
    task.write_text("edited\n")

    with pytest.raises(StagingError, match="does not verify, and is never rewritten"):
        staging.publish(stagings, repo)
    assert task.read_text() == "edited\n"


def test_a_change_to_any_input_gives_a_new_key(publishing, client: FakeClient, monkeypatch) -> None:
    key = staging.staging_key()
    client.lock.write_text(client.lock.read_text() + "# another lock\n")
    assert staging.staging_key() != key
    monkeypatch.setattr(staging, "PINNED_TOOLS", {INFERENCEX_TOOL: "e" * 40})
    assert len({key, staging.staging_key()}) == 2
    moved = staging.staging_key()
    monkeypatch.setattr(staging.aiperf_patch, "PATCH_SHA256", "f" * 64)
    assert staging.staging_key() != moved, "a changed patch of the client is staged anew"


def _race(monkeypatch: pytest.MonkeyPatch, other_stager) -> None:
    """Run ``other_stager`` to completion while the first stager has staged but not yet renamed."""
    real = staging._fsync_tree
    raced: list[Path] = []

    def fsync_then_lose(root: Path) -> None:
        real(root)
        if not raced:
            raced.append(root)
            other_stager()

    monkeypatch.setattr(staging, "_fsync_tree", fsync_then_lose)


def test_two_stagers_publish_one_staging_and_the_loser_removes_only_its_own_directory(publishing, monkeypatch) -> None:
    repo, stagings = publishing
    winner: list[Path] = []
    _race(monkeypatch, lambda: winner.append(staging.publish(stagings, repo)))

    loser = staging.publish(stagings, repo)

    assert loser == winner[0] == stagings / staging.staging_key()
    assert _publishing_dirs(stagings) == []
    assert verify(loser, staging.manifest_sha256(loser)).failed_step is None


def test_a_loser_refuses_a_winner_that_is_other_software_naming_the_packages(
    publishing, client: FakeClient, monkeypatch, tmp_path
) -> None:
    repo, stagings = publishing
    monkeypatch.setattr(staging, "staging_key", lambda: "k" * 64)
    wheel = client.index / "datasets-4.8.0-py3-none-any.whl"
    wheel.write_bytes(b"datasets 4.8.0")
    other_lock = tmp_path / "other.lock"
    other_lock.write_text(
        re.sub(
            r"datasets==\S+ --hash=sha256:\w+",
            f"datasets==4.8.0 --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
            client.lock.read_text(),
        )
    )

    def other_host() -> None:
        with monkeypatch.context() as other:
            other.setattr(staging.client_env, "AGENTX_LOCK", other_lock)
            staging.publish(stagings, repo)

    _race(monkeypatch, other_host)

    with pytest.raises(StagingError, match=r"not the one staged here: datasets 4\.8\.0 .* 4\.7\.0"):
        staging.publish(stagings, repo)
    assert _publishing_dirs(stagings) == []


def test_publish_and_its_readers_skip_a_private_staging_directory(publishing) -> None:
    repo, stagings = publishing
    stray = stagings / f"{staging.PUBLISHING_PREFIX}{staging.staging_key()}-another-host"
    stray.mkdir(parents=True)
    (stray / MANIFEST_FILE).write_text("half written")

    published = staging.publish(stagings, repo)

    assert published.name == staging.staging_key()
    assert (stray / MANIFEST_FILE).read_text() == "half written"


def test_the_publish_command_prints_the_published_directory_or_refuses_with_staging_missing(publishing, capsys):
    repo, stagings = publishing

    assert cli.main(["publish", "--stagings", str(stagings), "--source", str(repo)]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == str(stagings / staging.staging_key())
    (inferencex_root(stagings / staging.staging_key()) / GSM8K_TASK).unlink()
    assert cli.main(["publish", "--stagings", str(stagings), "--source", str(repo)]) == cli.EXIT_REFUSED
    assert capsys.readouterr().err.startswith(f"{FailedStep.STAGING_MISSING.value}: ")
