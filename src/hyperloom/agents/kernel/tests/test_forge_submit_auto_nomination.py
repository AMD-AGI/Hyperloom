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


def _write_request(tmp_path: Path) -> Path:
    request = tmp_path / "forge_nomination_input.json"
    request.write_text(json.dumps({"protocol_version": 1, "lane": "rewrite"}), encoding="utf-8")
    return request


def test_argv_carries_auto_nomination_and_omits_named_kernel_flags(tmp_path, monkeypatch):
    """The ``--auto`` argv hands forge the request, never a named kernel.

    ``--kernel``/``--driver``/``--source-files`` are exactly what makes the named
    path a named path; forge derives the target from the nomination, so their
    presence here would mean a candidate leaked into a run that has none.
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

    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        workspace=str(workspace),
        output_dir=output_dir,
        timeout_s=1800,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )

    # The envelope is returned UNCHANGED -- patches[] and nomination survive.
    assert result == {"patches": [{"kernel_name": "k001_kernel"}], "nomination": {"selected": 1}}

    command = captured["command"]
    assert command[:5] == [sys.executable, "-m", "kernelforge.cli", "forge-loop", "--auto"]
    # The request rides through, resolved to an absolute path.
    assert command[command.index("--nomination-input") + 1] == str(request.resolve())
    # Named-path flags are absent -- this is the whole point of the sibling entry.
    for named in ("--kernel", "--driver", "--source-files"):
        assert named not in command, named
    # The contract flags the handler relies on are present.
    assert command[command.index("--workspace") + 1] == str(workspace)
    assert command[command.index("--experiment-id") + 1] == forge_submit._FORGE_EXPERIMENT_ID
    assert command[command.index("--gpu-target") + 1] == "gfx950"
    # Isolated process group + child cwd, mirroring _run_loop_via_cli.
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["cwd"] == str(workspace)


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

    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        workspace=str(workspace),
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

    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        workspace=str(workspace),
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

    monkeypatch.setattr(forge_submit.subprocess, "Popen", lambda *_a, **_k: TimeoutProcess())
    monkeypatch.setattr(forge_submit, "_terminate_forge_process", fake_terminate)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        workspace=str(workspace),
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

    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    result = forge_submit.submit_auto(
        nomination_input=str(request),
        workspace=str(workspace),
        output_dir=output_dir,
        gpu_target="gfx950",
        gpu_type="mi355x",
    )
    assert result["status"] == "failed"
    assert result["patches"] == []
    assert not stale.exists()
