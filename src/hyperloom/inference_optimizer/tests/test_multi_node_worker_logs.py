# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-worker sglang server logs: the terminal-state probes, and the post-mortem sweep a failed restart leaves."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_worker_logs as worker_logs

_WEDGE_LINE = "WARNING Health check failed: detokenizer not responding\n"


def _write_server_log(shared: Path, ip: str, body: str, rank: int = 0) -> Path:
    """Write one worker's shared-FS server log the way the launcher names it."""
    path = shared / ("mn_infera_server_" + ip + "_r" + str(rank) + ".log")
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/shared/mn-logs", "/shared/mn-logs"),
        ("  /shared/mn-logs  ", "/shared/mn-logs"),
        ("relative/mn-logs", ""),
        ("$NOT_EXPANDED/mn-logs", ""),
        ("", ""),
    ],
)
def test_shared_log_dir_is_only_usable_when_it_resolved_to_an_absolute_path(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    """An unexpanded ``$VAR`` or a relative dir names nothing readable, so the probes must skip rather than guess."""
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", raw)

    assert worker_logs.shared_server_log_dir() == expected


def test_a_persistent_detokenizer_stall_reads_as_wedged(tmp_path: Path) -> None:
    """Weights load but the detokenizer never heartbeats, so /health can only time out."""
    _write_server_log(tmp_path, "10.0.0.1", _WEDGE_LINE * 20)

    assert worker_logs.worker_detokenizer_wedged(str(tmp_path), "10.0.0.1") is True


def test_a_few_detokenizer_warnings_are_not_a_wedge(tmp_path: Path) -> None:
    """The marker repeats about every 2s; a handful is a blip the wait should keep waiting through."""
    _write_server_log(tmp_path, "10.0.0.1", _WEDGE_LINE * 3)

    assert worker_logs.worker_detokenizer_wedged(str(tmp_path), "10.0.0.1") is False


def test_an_unreadable_log_is_not_a_wedge(tmp_path: Path) -> None:
    """No log is no evidence: the gate keeps waiting instead of failing the round."""
    assert worker_logs.worker_detokenizer_wedged(str(tmp_path), "10.0.0.9") is False
    assert worker_logs.worker_detokenizer_wedged("", "10.0.0.9") is False


@pytest.mark.parametrize(
    "line,fragment",
    [
        ("sglang: error: unrecognized arguments: --nope\n", "unrecognized arguments"),
        ("RuntimeError: sglang subprocess exited with code -9 before reporting ready\n", "before reporting ready"),
        ("torch.OutOfMemoryError: HIP out of memory\n", "out of memory"),
    ],
)
def test_a_terminal_startup_failure_is_reported_with_its_reason(tmp_path: Path, line: str, fragment: str) -> None:
    """These never recover, so the wait must name the cause instead of burning its whole budget."""
    _write_server_log(tmp_path, "10.0.0.2", "loading weights\n" + line)

    reason = worker_logs.worker_startup_crashed(str(tmp_path), "10.0.0.2")

    assert reason is not None
    assert fragment in reason
    assert len(reason) <= 200


def test_a_healthy_startup_log_reports_no_crash(tmp_path: Path) -> None:
    _write_server_log(tmp_path, "10.0.0.2", "loading weights\nserver started\n")

    assert worker_logs.worker_startup_crashed(str(tmp_path), "10.0.0.2") is None
    assert worker_logs.worker_startup_crashed("", "10.0.0.2") is None


def test_the_shared_tail_reads_the_newest_rank_and_names_its_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relaunch writes a new rank file; the post-mortem must carry the current one, not the first on disk."""
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))
    old = _write_server_log(tmp_path, "10.0.0.3", "first launch\n", rank=0)
    new = _write_server_log(tmp_path, "10.0.0.3", "second launch\n", rank=1)
    import os

    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    tail = worker_logs._shared_worker_log_tail("10.0.0.3")

    assert tail is not None
    assert "second launch" in tail
    assert tail.startswith("# source: ")


def test_no_shared_log_for_a_worker_reads_as_nothing_to_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))

    assert worker_logs._shared_worker_log_tail("10.0.0.4") is None


def test_ssh_access_needs_both_a_key_and_known_hosts() -> None:
    """Half a credential cannot reach a pod, so the sweep falls back to the shared FS alone."""
    assert worker_logs._ssh_access({"ssh_key_path": "/k"}) is None
    assert worker_logs._ssh_access({"ssh_known_hosts": "/kh"}) is None

    access = worker_logs._ssh_access({"ssh_key_path": "/k", "ssh_known_hosts": "/kh", "ssh_port": 2244})

    assert access is not None
    assert access.default_port == 2244


def _session_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A session dir to collect into, a shared log dir to read from, and a two-pod state."""
    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(tmp_path))
    return {
        "prefill_pods": [{"podIP": "10.0.1.1", "podId": "prefill-0"}, {"podId": "no-ip"}],
        "decode_pods": [{"podIP": "10.0.1.2", "podId": "decode-0"}],
    }


def test_a_failed_restart_leaves_one_post_mortem_per_reachable_pod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pods are torn down after the failure, so the log has to be on disk before that happens."""
    state = _session_state(tmp_path, monkeypatch)
    _write_server_log(tmp_path, "10.0.1.1", "prefill log\n")
    _write_server_log(tmp_path, "10.0.1.2", "decode log\n")

    worker_logs.collect_worker_server_logs(state, "serving probe never passed")

    collected = sorted(p.name for p in (tmp_path / "session" / "server_logs").iterdir())
    assert len(collected) == 2
    assert [name for name in collected if name.endswith("_prefill_prefill-0.log")]
    assert [name for name in collected if name.endswith("_decode_decode-0.log")]
    body = next((tmp_path / "session" / "server_logs").glob("*_prefill_prefill-0.log")).read_text(encoding="utf-8")
    assert "collected on restart failure (shared-fs): serving probe never passed" in body
    assert "prefill log" in body


def test_a_pod_with_no_log_anywhere_is_skipped_rather_than_written_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty file would read as "the server said nothing", which is a different finding."""
    state = _session_state(tmp_path, monkeypatch)
    _write_server_log(tmp_path, "10.0.1.1", "prefill log\n")

    worker_logs.collect_worker_server_logs(state, "boom")

    assert len(list((tmp_path / "session" / "server_logs").iterdir())) == 1


def test_the_sweep_falls_back_to_an_ssh_tail_when_the_shared_fs_has_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An Infera run without a shared mount still has to leave a post-mortem behind."""
    state = _session_state(tmp_path, monkeypatch)
    state["ssh_key_path"] = "/key"
    state["ssh_known_hosts"] = "/known_hosts"
    import hyperloom.inference_optimizer.multi_node as mn

    monkeypatch.setattr(mn, "SshTarget", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(
        mn,
        "ssh_run",
        lambda target, command, timeout: types.SimpleNamespace(stdout="remote log\n", stderr="warn\n"),
        raising=False,
    )

    worker_logs.collect_worker_server_logs(state, "boom")

    bodies = [p.read_text(encoding="utf-8") for p in (tmp_path / "session" / "server_logs").iterdir()]
    assert len(bodies) == 2
    assert all("remote log" in body for body in bodies)
    assert all("--- ssh stderr ---" in body for body in bodies)


def test_a_pod_whose_ssh_tail_fails_does_not_stop_the_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One unreachable pod must not cost the post-mortem of the others."""
    state = _session_state(tmp_path, monkeypatch)
    state["ssh_key_path"] = "/key"
    state["ssh_known_hosts"] = "/known_hosts"
    _write_server_log(tmp_path, "10.0.1.1", "prefill log\n")
    import hyperloom.inference_optimizer.multi_node as mn

    def _boom(target, command, timeout):
        raise OSError("unreachable")

    monkeypatch.setattr(mn, "SshTarget", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(mn, "ssh_run", _boom, raising=False)

    worker_logs.collect_worker_server_logs(state, "boom")

    assert len(list((tmp_path / "session" / "server_logs").iterdir())) == 1


def test_without_a_session_directory_there_is_nowhere_to_collect_into(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SESSION_DIR", raising=False)

    worker_logs.collect_worker_server_logs({"prefill_pods": [{"podIP": "10.0.1.1"}]}, "boom")

    assert worker_logs._post_mortem_dir() == ""
