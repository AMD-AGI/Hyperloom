# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for ``forge_submit.submit_auto`` -- the ``--auto`` sibling entry.

``submit_auto`` is the subprocess boundary the nomination handler crosses: it
builds the ``forge-loop --auto`` argv, runs the child in an isolated group under
an absolute deadline, and returns forge's raw envelope UNCHANGED so every sibling
patch survives. The integration test at
``inference_optimizer/tests/test_forge_nomination_dispatch.py`` monkeypatches this
function wholesale, so nothing else exercises its ~160-line body. These pin it.

Only ``subprocess.Popen`` (and the GPU/backend env shims that would otherwise
probe hardware) are faked; the argv build, the two result-parse channels, and the
timeout/failure envelope shapes all run for real.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


_REAL_POPEN = subprocess.Popen


def _patch_forge_launch(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    """Route only the ``forge-loop`` launch to ``fake``; leave git real.

    Workspace staging is not mocked here, so it runs real ``git``. forge_submit
    holds the subprocess module itself, so patching ``Popen`` on it patches the
    global one -- a blanket fake would swallow those git calls too.
    """

    def _dispatch(command, **kwargs):
        if any("forge-loop" == str(part) for part in command):
            return fake(command, **kwargs)
        return _REAL_POPEN(command, **kwargs)

    monkeypatch.setattr(forge_submit.subprocess, "Popen", _dispatch)


def _neutralize_env_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the backend-env shim from touching real hardware/config.

    ``gpu_target``/``gpu_type`` are passed explicitly to every call below, so the
    resolve helpers never probe; only the kernel-backend apply needs silencing.
    """
    monkeypatch.setattr(forge_submit, "_apply_kernel_backend_env", lambda _env: None)


class _FakeProcess:
    """A forge child that returns a canned ``__FORGE_RESULT__`` stdout blob."""

    pid = 54321

    def __init__(self, *, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def communicate(self, timeout=None):  # noqa: ARG002 - signature parity
        return self._stdout, self._stderr


def _kernel_repo(tmp_path: Path, *, name: str = "framework") -> tuple[Path, Path]:
    """A real git checkout holding one tracked kernel source.

    Staging is not mocked in this file, so the brief has to name a tree that
    ``git worktree add`` can actually branch from and whose source is tracked.
    """
    repo = tmp_path / name
    (repo / "pkg").mkdir(parents=True)
    source = repo / "pkg" / "attention.py"
    source.write_text("def paged_attention_v1():\n    return None\n", encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-qm", "seed"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return repo, source


def _write_request(tmp_path: Path, *, rows: list[dict] | None = None) -> Path:
    """A nomination request plus the manifest it points at.

    Defaults to a single eligible row in a real git checkout, which is the
    single-target case the contract requires to run end to end.
    """
    if rows is None:
        _repo, source = _kernel_repo(tmp_path)
        rows = [
            {
                "kernel_name": "paged_attention_v1",
                "gpu_pct": 30.0,
                "source_file": str(source),
                "kernel_repo": str(source.parent.parent),
                "reason_class": "resolved",
                "attempts": 0,
                "rejected": False,
            }
        ]
    manifest = tmp_path / "forge_candidate_manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "hot_kernels": rows}), encoding="utf-8")
    request = tmp_path / "forge_nomination_input.json"
    request.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "lane": "rewrite",
                "candidates_path": str(manifest),
                "trace_path": str(tmp_path / "decode.trace.json"),
                "lane_budget_sec": 9000,
                "max_kernels": 1,
            }
        ),
        encoding="utf-8",
    )
    return request


def test_argv_carries_auto_nomination_and_omits_named_kernel_flags(tmp_path, monkeypatch):
    """The ``--auto`` argv hands forge the brief, never a named kernel.

    ``--kernel``/``--source-files`` are what make the named path a named path;
    forge derives the target from the nomination, so their presence here would
    mean a candidate leaked into a run that has none. ``--driver`` is different:
    a fresh campaign refuses to start without one and --auto supplies no kernel
    to derive it from, so the placeholder is passed deliberately.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_dir = tmp_path / "attempt"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        payload = {"patches": [{"kernel_name": "k001_kernel"}], "nomination": {"selected": 1}}
        return _FakeProcess(returncode=0, stdout=f"__FORGE_RESULT__{json.dumps(payload)}__FORGE_RESULT__")

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=output_dir,
        timeout_s=1800,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    # The envelope is returned UNCHANGED -- patches[] and nomination survive.
    assert result == {"patches": [{"kernel_name": "k001_kernel"}], "nomination": {"selected": 1}}

    command = captured["command"]
    assert command[:5] == [sys.executable, "-m", "kernelforge.cli", "forge-loop", "--auto"]
    # Forge reads the STAGED brief, whose candidate paths point into the staged
    # workspace; the caller's original brief points at the live tree.
    staged_request = Path(command[command.index("--nomination-input") + 1])
    assert staged_request != request.resolve()
    assert staged_request.name == "forge_nomination_input.staged.json"
    # Named-path flags are absent -- this is the whole point of the sibling entry.
    for named in ("--kernel", "--source-files"):
        assert named not in command, named
    # The workspace is the staged worktree, never the caller's session dir: the
    # live install tree must not be edited by a campaign.
    staged_workspace = Path(command[command.index("--workspace") + 1])
    assert staged_workspace == output_dir / "worktree"
    assert staged_workspace != workspace
    # A driver exists and lives inside that workspace, which forge requires.
    driver = Path(command[command.index("--driver") + 1])
    assert driver.is_file()
    assert driver.parent == staged_workspace
    assert command[command.index("--experiment-id") + 1] == forge_submit._FORGE_EXPERIMENT_ID
    assert command[command.index("--gpu-target") + 1] == "gfx950"
    # Isolated process group, and the child runs from the tree it may edit.
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["cwd"] == str(staged_workspace)


def test_result_json_sidecar_wins_over_stdout(tmp_path, monkeypatch):
    """The ``--result-json`` sidecar is the primary channel; stdout is fallback.

    When both exist and disagree, the on-disk sidecar is authoritative -- it is
    what forge commits last, so a truncated stdout blob must never override it.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_dir = tmp_path / "attempt"

    sidecar_payload = {"patches": [{"kernel_name": "from_sidecar"}], "improved": True}
    stdout_payload = {"patches": [{"kernel_name": "from_stdout"}], "improved": False}

    def fake_popen(command, **_kwargs):
        # forge writes its result-json before exiting; find the path from argv.
        result_json = Path(command[command.index("--result-json") + 1])
        result_json.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        return _FakeProcess(
            returncode=0,
            stdout=f"__FORGE_RESULT__{json.dumps(stdout_payload)}__FORGE_RESULT__",
        )

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=output_dir,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )
    assert result == sidecar_payload


def test_nonzero_exit_without_result_is_a_failed_envelope(tmp_path, monkeypatch):
    """A child that crashes with no parseable result yields status=failed.

    The handler treats this as a surfaced failure, not a clean empty nomination,
    so ``patches`` must be empty and the child's reason must ride in ``error``.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def fake_popen(_command, **_kwargs):
        return _FakeProcess(returncode=7, stdout="", stderr="Error: nomination rejected\n")

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )
    assert result["status"] == "failed"
    assert result["patches"] == []
    assert "rc=7" in result["error"]
    assert "nomination rejected" in result["error"]


def test_timeout_is_hard_killed_and_reported_as_timeout(tmp_path, monkeypatch):
    """A run that blows the deadline is force-terminated and marked timeout.

    The envelope's ``status`` must be ``timeout`` (not ``failed``) so the handler
    can distinguish a doomed run from a rejected one.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    terminated = {"n": 0}

    class TimeoutProcess:
        pid = 54321
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(["forge-loop"], timeout)

    def fake_terminate(_proc):
        terminated["n"] += 1
        return "partial stdout", "partial stderr"

    _patch_forge_launch(monkeypatch, lambda *_a, **_k: TimeoutProcess())
    monkeypatch.setattr(forge_submit, "_terminate_forge_process", fake_terminate)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        timeout_s=10,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )
    assert terminated["n"] == 1
    assert result["status"] == "timeout"
    assert result["patches"] == []
    assert "10" in result["error"]


def test_stale_result_sidecar_is_cleared_before_launch(tmp_path, monkeypatch):
    """A previous run's ``forge_cli_result.json`` must never be read as this run's.

    ``submit_auto`` unlinks the sidecar up front; a child that then writes no
    result must fall through to the failed envelope rather than salvaging the
    stale one.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_dir = tmp_path / "attempt"
    (output_dir / "forge_experiments").mkdir(parents=True)
    stale = output_dir / "forge_cli_result.json"
    stale.write_text(json.dumps({"patches": [{"kernel_name": "STALE"}]}), encoding="utf-8")

    def fake_popen(_command, **_kwargs):
        # Child produces nothing new: no sidecar rewrite, no __FORGE_RESULT__.
        return _FakeProcess(returncode=1, stdout="crashed before writing a result\n")

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=output_dir,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )
    assert result["status"] == "failed"
    assert result["patches"] == []
    assert not stale.exists()


def test_a_nonzero_exit_is_failed_even_with_a_parseable_sidecar(tmp_path, monkeypatch):
    """forge's task-preparation failure writes a sidecar and then exits nonzero.

    That sidecar carries neither ``status`` nor ``patches``, so returning it
    verbatim reads as a clean empty nomination and the phase latches as if a
    pass had run. Only exit code 0 makes an empty ``patches`` a valid answer.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def fake_popen(command, **_kwargs):
        result_json = Path(command[command.index("--result-json") + 1])
        result_json.write_text(json.dumps({"improved": False, "error": "task prep failed"}), encoding="utf-8")
        return _FakeProcess(returncode=2, stdout="", stderr="Error: could not prepare task\n")

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    assert result["status"] == "failed"
    assert result["patches"] == []
    assert "rc=2" in result["error"]
    # The sidecar may only enrich the error, never decide the outcome.
    assert "task prep failed" in result["error"]


def test_a_failed_run_reports_no_patches_even_when_the_sidecar_lists_some(tmp_path, monkeypatch):
    """A patch from a run whose own tooling broke must not reach the queue."""
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def fake_popen(command, **_kwargs):
        result_json = Path(command[command.index("--result-json") + 1])
        result_json.write_text(
            json.dumps({"patches": [{"kernel_name": "half_written"}], "status": "complete"}),
            encoding="utf-8",
        )
        return _FakeProcess(returncode=1, stdout="", stderr="boom\n")

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    assert result["status"] == "failed"
    assert result["patches"] == []


def test_a_timeout_outranks_a_sidecar_that_claims_success(tmp_path, monkeypatch):
    """A hard-killed run keeps the timeout status the handler routes on.

    forge can commit a sidecar and then hang; the deadline is still the truth
    about that run, so a ``complete`` claim on disk must not override it.
    """
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    class TimeoutProcess:
        pid = 54321
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(["forge-loop"], timeout)

    def fake_popen(command, **_kwargs):
        result_json = Path(command[command.index("--result-json") + 1])
        result_json.write_text(
            json.dumps({"status": "complete", "patches": [{"kernel_name": "mid_flight"}]}),
            encoding="utf-8",
        )
        return TimeoutProcess()

    _patch_forge_launch(monkeypatch, fake_popen)
    monkeypatch.setattr(forge_submit, "_terminate_forge_process", lambda _proc: ("", ""))

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        timeout_s=10,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    assert result["status"] == "timeout"
    assert result["patches"] == []


def test_the_nomination_summary_survives_a_failed_run(tmp_path, monkeypatch):
    """Counts are triage data, so they ride along even on a failure."""
    _neutralize_env_probes(monkeypatch)
    request = _write_request(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    summary = {"candidates_seen": 9, "resolved": 4, "selected": 1}

    def fake_popen(command, **_kwargs):
        result_json = Path(command[command.index("--result-json") + 1])
        result_json.write_text(json.dumps({"nomination": summary}), encoding="utf-8")
        return _FakeProcess(returncode=3, stdout="", stderr="late crash\n")

    _patch_forge_launch(monkeypatch, fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    assert result["status"] == "failed"
    assert result["nomination"] == summary


def test_a_single_target_brief_reaches_forge_with_everything_a_campaign_needs(tmp_path, monkeypatch):
    """The contract requires one nominated target to run the whole way through.

    A fresh campaign refuses to start without both a kernel and a driver, and
    resolves both against the workspace. --auto names no kernel and no driver, so
    without staging the run dies inside forge on a bare ValueError. Everything
    the campaign gate checks must therefore be true of the argv handed over.
    """
    _neutralize_env_probes(monkeypatch)
    repo, source = _kernel_repo(tmp_path)
    request = _write_request(
        tmp_path,
        rows=[
            {
                "kernel_name": "paged_attention_v1",
                "gpu_pct": 42.0,
                "source_file": str(source),
                "kernel_repo": str(repo),
                "reason_class": "resolved",
                "attempts": 0,
                "rejected": False,
            }
        ],
    )
    output_dir = tmp_path / "attempt"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProcess(returncode=0, stdout='__FORGE_RESULT__{"patches": []}__FORGE_RESULT__')

    _patch_forge_launch(monkeypatch, fake_popen)

    forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=output_dir,
        timeout_s=1800,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    command = captured["command"]
    staged_workspace = Path(command[command.index("--workspace") + 1])
    driver = Path(command[command.index("--driver") + 1])
    staged_brief = json.loads(Path(command[command.index("--nomination-input") + 1]).read_text(encoding="utf-8"))
    staged_rows = json.loads(Path(staged_brief["candidates_path"]).read_text(encoding="utf-8"))["hot_kernels"]

    # The driver the campaign gate demands exists inside the workspace.
    assert driver.is_file()
    assert driver.parent == staged_workspace
    # The nominatable kernel now resolves INSIDE that workspace, which is the
    # containment check every candidate path previously failed.
    (row,) = staged_rows
    nominated = Path(row["source_file"])
    assert nominated.is_file()
    assert nominated.is_relative_to(staged_workspace)
    assert nominated.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    # The live install tree is untouched: forge edits the worktree, not it.
    assert staged_workspace != repo
    assert not staged_workspace.is_relative_to(repo)


def test_a_row_outside_the_staged_tree_is_not_offered(tmp_path, monkeypatch):
    """One worktree cannot hold two trees, so off-tree rows must be withheld.

    Leaving them in the staged brief would offer the nominator targets that can
    only fail the containment check.
    """
    _neutralize_env_probes(monkeypatch)
    repo, source = _kernel_repo(tmp_path, name="vllm")
    other_repo, other_source = _kernel_repo(tmp_path, name="aiter")
    request = _write_request(
        tmp_path,
        rows=[
            {
                "kernel_name": "hot",
                "gpu_pct": 40.0,
                "source_file": str(source),
                "kernel_repo": str(repo),
                "rejected": False,
            },
            {
                "kernel_name": "other_tree",
                "gpu_pct": 30.0,
                "source_file": str(other_source),
                "kernel_repo": str(other_repo),
                "rejected": False,
            },
        ],
    )
    output_dir = tmp_path / "attempt"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProcess(returncode=0, stdout='__FORGE_RESULT__{"patches": []}__FORGE_RESULT__')

    _patch_forge_launch(monkeypatch, fake_popen)

    forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=output_dir,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    command = captured["command"]
    staged_brief = json.loads(Path(command[command.index("--nomination-input") + 1]).read_text(encoding="utf-8"))
    staged_rows = json.loads(Path(staged_brief["candidates_path"]).read_text(encoding="utf-8"))["hot_kernels"]

    assert [row["kernel_name"] for row in staged_rows] == ["hot"]


def test_an_unresolved_row_is_carried_into_the_staged_brief(tmp_path, monkeypatch):
    """Unresolved rows are the reason the whole list is handed over at all."""
    _neutralize_env_probes(monkeypatch)
    repo, source = _kernel_repo(tmp_path)
    request = _write_request(
        tmp_path,
        rows=[
            {"kernel_name": "hot", "gpu_pct": 40.0, "source_file": str(source), "kernel_repo": str(repo)},
            {"kernel_name": "unlocated", "gpu_pct": 35.0, "source_file": "", "reason_class": "symbol_not_found"},
        ],
    )
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProcess(returncode=0, stdout='__FORGE_RESULT__{"patches": []}__FORGE_RESULT__')

    _patch_forge_launch(monkeypatch, fake_popen)

    forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    command = captured["command"]
    staged_brief = json.loads(Path(command[command.index("--nomination-input") + 1]).read_text(encoding="utf-8"))
    staged_rows = json.loads(Path(staged_brief["candidates_path"]).read_text(encoding="utf-8"))["hot_kernels"]

    assert sorted(row["kernel_name"] for row in staged_rows) == ["hot", "unlocated"]


def test_a_brief_with_no_resolved_source_is_refused_before_launch(tmp_path, monkeypatch):
    """No stageable tree means no workspace can make any nomination runnable."""
    _neutralize_env_probes(monkeypatch)
    request = _write_request(
        tmp_path,
        rows=[{"kernel_name": "unlocated", "gpu_pct": 40.0, "source_file": "", "reason_class": "unknown"}],
    )

    def _boom(_command, **_kwargs):
        raise AssertionError("forge must not launch without a stageable workspace")

    _patch_forge_launch(monkeypatch, _boom)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    assert result["status"] == "failed"
    assert result["patches"] == []
    assert "staging failed" in result["error"]
    assert "no candidate with a resolved source file" in result["error"]


def test_a_rejected_row_cannot_decide_the_staged_tree(tmp_path, monkeypatch):
    """A row this session already rejected can never be picked, so it must not
    choose which tree gets staged -- otherwise the eligible rows are withheld."""
    _neutralize_env_probes(monkeypatch)
    repo, source = _kernel_repo(tmp_path, name="live")
    _dead_repo, dead_source = _kernel_repo(tmp_path, name="rejected_tree")
    request = _write_request(
        tmp_path,
        rows=[
            {
                "kernel_name": "already_rejected",
                "gpu_pct": 90.0,
                "source_file": str(dead_source),
                "kernel_repo": str(_dead_repo),
                "rejected": True,
            },
            {
                "kernel_name": "still_eligible",
                "gpu_pct": 10.0,
                "source_file": str(source),
                "kernel_repo": str(repo),
                "rejected": False,
            },
        ],
    )
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProcess(returncode=0, stdout='__FORGE_RESULT__{"patches": []}__FORGE_RESULT__')

    _patch_forge_launch(monkeypatch, fake_popen)

    forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    command = captured["command"]
    staged_brief = json.loads(Path(command[command.index("--nomination-input") + 1]).read_text(encoding="utf-8"))
    staged_rows = json.loads(Path(staged_brief["candidates_path"]).read_text(encoding="utf-8"))["hot_kernels"]

    assert [row["kernel_name"] for row in staged_rows] == ["still_eligible"]


def test_a_source_named_through_a_symlink_is_still_staged(tmp_path, monkeypatch):
    """A path reached via a symlinked root is genuinely inside the tree.

    ``link/pkg/attention.py`` shares no string prefix with the real repo root, so
    comparing unresolved paths drops the row as off-tree and the brief arrives
    empty. Both sides must be resolved before the paths are rewritten.
    """
    _neutralize_env_probes(monkeypatch)
    repo, source = _kernel_repo(tmp_path, name="real")
    linked = tmp_path / "link"
    linked.symlink_to(repo, target_is_directory=True)
    request = _write_request(
        tmp_path,
        rows=[
            {
                "kernel_name": "through_a_symlink",
                "gpu_pct": 40.0,
                # Named through the link, while the tree itself is the real dir.
                "source_file": str(linked / "pkg" / "attention.py"),
                "kernel_repo": str(linked),
                "rejected": False,
            }
        ],
    )
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProcess(returncode=0, stdout='__FORGE_RESULT__{"patches": []}__FORGE_RESULT__')

    _patch_forge_launch(monkeypatch, fake_popen)

    forge_submit.submit_auto(
        nomination_input=str(request),
        output_dir=tmp_path / "attempt",
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    command = captured["command"]
    staged_workspace = Path(command[command.index("--workspace") + 1])
    staged_brief = json.loads(Path(command[command.index("--nomination-input") + 1]).read_text(encoding="utf-8"))
    staged_rows = json.loads(Path(staged_brief["candidates_path"]).read_text(encoding="utf-8"))["hot_kernels"]

    (row,) = staged_rows
    assert row["kernel_name"] == "through_a_symlink"
    nominated = Path(row["source_file"])
    assert nominated.is_relative_to(staged_workspace)
    assert nominated.is_file()
