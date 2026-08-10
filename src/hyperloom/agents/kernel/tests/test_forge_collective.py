###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for the forge-loop wrapper that drives collective-kernel optimisation."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import forge_collective as fc  # noqa: E402


_CANDIDATE = {
    "device_kernel_name": "all_reduce_cross_device",
    "source_file": "/repo/csrc/include/custom_all_reduce.cuh",
    "source_line": 811,
    "source_function": "fused_ar_rms",
    "kernel_contract": {"kind": "collective", "collective_op": "all_reduce", "world_size": 4},
    "input_shapes": [{"shape": "(4096, 7168)"}],
    "input_dtypes": ["bf16"],
    "gpu_pct": 10.0,
}


def _payload(tmp_path: Path, **extra) -> dict:
    item = {
        "output_dir": str(tmp_path),
        "candidate": _CANDIDATE,
        "source_file": _CANDIDATE["source_file"],
        "kernel_repo": "/repo",
        "tp": 4,
        "target_functions": [_CANDIDATE["source_function"]],
    }
    item.update(extra)
    return item


def _rig(tmp_path: Path) -> dict:
    from collective_driver_generator import generate_collective_driver

    return generate_collective_driver(_CANDIDATE, tmp_path, tp=4)


# --- Command construction -----------------------------------------------------


def test_cmd_invokes_forge_loop_as_a_module(tmp_path):
    """The wrapper must invoke the importable KernelForge module."""
    cmd = fc._build_cmd(
        _payload(tmp_path),
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[:1] == [sys.executable]
    assert cmd[1:4] == ["-m", "kernel_agents.cli", "forge-loop"]


def test_an_explicit_cli_override_is_honoured(tmp_path):
    """An operator pinning a real console script must still win."""
    payload = dict(_payload(tmp_path), cli="/usr/local/bin/kernel-agents")
    cmd = fc._build_cmd(
        payload,
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[:2] == ["/usr/local/bin/kernel-agents", "forge-loop"]


def test_cmd_carries_rank_count_and_generated_rig(tmp_path):
    rig = _rig(tmp_path)
    cmd = fc._build_cmd(
        _payload(tmp_path),
        rig,
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    # Wrapping the launcher alone profiles a process that runs no kernel.
    assert "--nproc-per-node" in cmd
    assert cmd[cmd.index("--nproc-per-node") + 1] == "4"
    assert cmd[cmd.index("--driver") + 1] == rig["driver"]
    assert cmd[cmd.index("--program-md-file") + 1] == rig["program"]
    assert cmd[cmd.index("--task-type") + 1] == "repository"
    assert cmd[cmd.index("--deadline-unix") + 1] == "9999999999"


def test_cmd_defaults_target_the_noise_floor(tmp_path):
    """A collective's real speedup often sits near the noise floor."""
    cmd = fc._build_cmd(
        _payload(tmp_path),
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[cmd.index("--bench-repeat") + 1] == "3"
    assert cmd[cmd.index("--calibrate-noise-floor") + 1] == "5"
    assert cmd[cmd.index("--snr-threshold") + 1] == str(fc.DEFAULT_SNR_THRESHOLD)


def test_cmd_preserves_explicit_zero_snr_threshold(tmp_path):
    """A configured threshold must not be replaced by a default."""
    cmd = fc._build_cmd(
        _payload(tmp_path, snr_threshold=0),
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[cmd.index("--snr-threshold") + 1] == "0"


@pytest.mark.parametrize(
    "field",
    ["bench_repeat", "calibrate_noise_floor"],
)
def test_cmd_rejects_zero_iteration_controls(tmp_path, field):
    """Iteration controls must be positive integers."""
    with pytest.raises(ValueError, match="positive integer"):
        fc._build_cmd(
            _payload(tmp_path, **{field: 0}),
            _rig(tmp_path),
            tmp_path,
            deadline_unix=9_999_999_999,
        )


def test_cmd_requires_source_and_repo(tmp_path):
    rig = _rig(tmp_path)
    for missing in ("source_file", "kernel_repo"):
        payload = _payload(tmp_path)
        payload[missing] = ""
        payload["candidate"] = {**_CANDIDATE, "source_file": "", "kernel_repo": ""}
        try:
            fc._build_cmd(
                payload,
                rig,
                tmp_path,
                deadline_unix=9_999_999_999,
            )
        except ValueError as exc:
            assert missing in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"missing {missing} was accepted")


def test_target_functions_list_is_joined(tmp_path):
    cmd = fc._build_cmd(
        _payload(tmp_path, target_functions=["a::b", "c"]),
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[cmd.index("--target-functions") + 1] == "a::b,c"


# --- Result normalisation -----------------------------------------------------


def _write_forge_result(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "forge_result.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_exported_patch(tmp_path: Path, name: str = "a.cuh") -> Path:
    """Write a patch in the wrapper's trusted export directory."""
    patch = tmp_path / "optimized_versions" / "forge.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(
        f"diff --git a/{name} b/{name}\n",
        encoding="utf-8",
    )
    return patch


def test_kept_result_maps_to_keep_and_requires_e2e(tmp_path):
    rig = _rig(tmp_path)
    _write_forge_result(
        tmp_path,
        {"improved": True, "mean_case_speedup": 1.35},
    )
    patch = _write_exported_patch(tmp_path)
    out = fc._normalize_result(
        str(tmp_path),
        0,
        rig,
        patch_path=str(patch),
        changed_files=["a.cuh"],
        best_commit="abc123",
    )
    assert out["decision"] == "KEEP"
    assert out["micro_decision"] == "candidate"
    assert out["kernel_speedup"] == 1.35
    assert out["artifact_files"] == ["a.cuh"]
    # Kernel parity alone never authorises an integrate.
    assert out["requires_e2e_validation"] is True


def test_not_kept_result_maps_to_revert(tmp_path):
    _write_forge_result(tmp_path, {"improved": False})
    out = fc._normalize_result(str(tmp_path), 0, _rig(tmp_path))
    assert out["decision"] == "REVERT"
    assert out["micro_decision"] == "no_improvement"
    assert out["requires_e2e_validation"] is False


def test_production_result_fields_map_to_keep(tmp_path):
    _write_forge_result(
        tmp_path,
        {
            "improved": True,
            "mean_case_speedup": 2.0,
            "iteration_count": 7,
        },
    )
    patch = _write_exported_patch(tmp_path, "k.cuh")
    out = fc._normalize_result(
        str(tmp_path),
        0,
        _rig(tmp_path),
        source_file="/repo/k.cuh",
        kernel_repo="/repo",
        patch_path=str(patch),
        best_commit="abc123",
    )
    assert out["kept"] is True
    assert out["kernel_speedup"] == 2.0
    assert out["source_file"] == "/repo/k.cuh"
    assert out["kernel_repo"] == "/repo"
    assert out["iterations"] == 7


def test_improvement_without_exportable_patch_is_rejected(tmp_path):
    """A microbenchmark win cannot enter E2E integration without a patch."""
    _write_forge_result(tmp_path, {"improved": True, "mean_case_speedup": 1.2})
    out = fc._normalize_result(str(tmp_path), 0, _rig(tmp_path))
    assert out["status"] == "failed"
    assert out["decision"] == "REVERT"
    assert out["error_class"] == "unverified_collective_improvement"


def test_kernel_forge_error_payload_is_not_no_improvement(tmp_path):
    """Preparation errors must remain failures even when no candidate was kept."""
    _write_forge_result(
        tmp_path,
        {
            "error": "task_preparation_failed",
            "detail": "driver preflight failed",
        },
    )
    out = fc._normalize_result(str(tmp_path), 2, _rig(tmp_path))
    assert out["status"] == "failed"
    assert out["error_class"] == "task_preparation_failed"
    assert out["error"] == "driver preflight failed"


def test_missing_result_file_is_reported_not_raised(tmp_path):
    out = fc._normalize_result(str(tmp_path), 1, _rig(tmp_path))
    assert out["status"] == "failed"
    assert "no forge_result.json" in out["error"]


def test_corrupt_result_file_is_reported(tmp_path):
    (tmp_path / "forge_result.json").write_text("{not json", encoding="utf-8")
    out = fc._normalize_result(str(tmp_path), 0, _rig(tmp_path))
    assert out["status"] == "failed"
    assert "invalid forge result" in out["error"]


def test_result_carries_collective_metadata(tmp_path):
    rig = _rig(tmp_path)
    _write_forge_result(tmp_path, {"improved": False})
    out = fc._normalize_result(str(tmp_path), 0, rig)
    assert out["collective_op"] == "all_reduce"
    assert out["world_size"] == "4"
    assert out["engine"] == "forge_collective"


def test_timeout_result_is_a_plain_revert(tmp_path):
    exc = subprocess.TimeoutExpired(["kernel-agents"], 10)
    out = fc._timeout_result(str(tmp_path), 10, exc)
    assert out["decision"] == "REVERT"
    assert out["error_class"] == "subprocess_timeout"


# --- End-to-end wrapper contract ---------------------------------------------


def _git(repo: Path, *args: str) -> str:
    """Run one checked git command in a temporary repository."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a clean repository containing one collective source."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.com")
    source = repo / "kernel.cuh"
    source.write_text("__global__ void kernel() { int x = 0; }\n", encoding="utf-8")
    _git(repo, "add", "kernel.cuh")
    _git(repo, "commit", "-m", "baseline")
    return repo, source


def test_main_exports_patch_then_restores_live_repo(tmp_path, monkeypatch):
    """A KEEP must retain its patch while returning the live repo to baseline."""
    repo, source = _make_repo(tmp_path)
    user_cache = repo / "user-cache.txt"
    user_cache.write_text("keep\n", encoding="utf-8")
    stale_campaign = repo / "forge_experiments"
    stale_campaign.mkdir()
    (stale_campaign / "campaign_config.json").write_text(
        '{"stale": true}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    payload = _payload(
        output_dir,
        source_file=str(source),
        kernel_repo=str(repo),
    )
    payload["candidate"] = {
        **_CANDIDATE,
        "source_file": str(source),
        "kernel_repo": str(repo),
    }
    monkeypatch.setattr(fc, "_needs_inplace", lambda _repo: True)

    def _fake_run(cmd, timeout):
        """Commit a synthetic Forge win in the prepared temporary branch."""
        workspace = Path(cmd[cmd.index("--workspace") + 1])
        kernel = Path(cmd[cmd.index("--kernel") + 1])
        result_path = Path(cmd[cmd.index("--result-json") + 1])
        kernel.write_text("__global__ void kernel() { int x = 1; }\n", encoding="utf-8")
        (workspace / "generated.tmp").write_text("remove\n", encoding="utf-8")
        _git(workspace, "add", "kernel.cuh")
        _git(workspace, "commit", "-m", "optimize")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        result_path.write_text(
            json.dumps(
                {
                    "improved": True,
                    "mean_case_speedup": 1.1,
                    "iteration_count": 2,
                    "best_commit": best_commit,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(fc, "_run_with_tree_timeout", _fake_run)
    input_json = tmp_path / "in.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    rc = fc.main(["--input-json", str(input_json)])
    assert rc == 0
    written = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert written["decision"] == "KEEP"
    assert written["source_file"] == str(source)
    assert written["kernel_repo"] == str(repo)
    assert Path(written["patch"]).is_file()
    assert "x = 1" in (output_dir / "optimized_versions" / "forge.patch").read_text()
    assert source.read_text(encoding="utf-8") == "__global__ void kernel() { int x = 0; }\n"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert user_cache.read_text(encoding="utf-8") == "keep\n"
    assert not (repo / "generated.tmp").exists()
    assert _git(repo, "status", "--porcelain") == "?? user-cache.txt"
    assert (output_dir / "prior_forge_experiments" / "campaign_config.json").is_file()
    assert _git(repo, "config", "--local", "--get", "user.name") == "test"
    assert _git(repo, "config", "--local", "--get", "user.email") == "test@example.com"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_stale_inplace_branch_recovers_from_durable_journal(tmp_path, monkeypatch):
    """A hard-crashed temporary branch must recover its exact clean baseline."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    payload = _payload(
        output_dir,
        source_file=str(source),
        kernel_repo=str(repo),
    )
    payload["candidate"] = {
        **_CANDIDATE,
        "source_file": str(source),
        "kernel_repo": str(repo),
    }
    monkeypatch.setattr(fc, "_needs_inplace", lambda _repo: True)
    context = fc._prepare_collective_workspace(payload, output_dir)
    assert context is not None
    source.write_text("__global__ void kernel() { int x = 9; }\n", encoding="utf-8")
    _git(repo, "add", "kernel.cuh")
    _git(repo, "commit", "-m", "interrupted optimize")
    context["restore"]["lock_fd"].close()

    assert fc._recover_stale_inplace(str(repo)) is True

    assert source.read_text(encoding="utf-8") == "__global__ void kernel() { int x = 0; }\n"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "config", "--local", "--get", "user.name") == "test"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_obsolete_precheckout_journal_does_not_block_repo(tmp_path):
    """A journal must not restore over later user repository changes."""
    repo, source = _make_repo(tmp_path)
    original_head = _git(repo, "rev-parse", "HEAD")
    fc._write_restore_journal(
        str(repo),
        {
            "repo": str(repo),
            "orig_branch": "main",
            "orig_head": original_head,
            "branch": "forge/collective-obsolete",
            "source_file": str(source),
            "backup": source.read_bytes(),
            "relpath": "kernel.cuh",
            "base_commit": original_head,
            "config_snapshot": fc._config_snapshot(str(repo)),
            "baseline_untracked": [],
        },
    )
    source.write_text(
        "__global__ void kernel() { int x = 7; }\n",
        encoding="utf-8",
    )
    _git(repo, "add", "kernel.cuh")
    _git(repo, "commit", "-m", "user update")
    updated_head = _git(repo, "rev-parse", "HEAD")

    assert fc._recover_stale_inplace(str(repo)) is True

    assert _git(repo, "rev-parse", "HEAD") == updated_head
    assert "x = 7" in source.read_text(encoding="utf-8")
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_timeout_salvages_authoritative_best_result(tmp_path, monkeypatch):
    """A validated published best must survive timeout before result-json write."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"
    payload = _payload(
        output_dir,
        source_file=str(source),
        kernel_repo=str(repo),
    )
    payload["candidate"] = {
        **_CANDIDATE,
        "source_file": str(source),
        "kernel_repo": str(repo),
    }
    monkeypatch.setattr(fc, "_needs_inplace", lambda _repo: True)

    def _timeout_after_keep(cmd, timeout):
        """Publish a validated best and then simulate wrapper timeout."""
        workspace = Path(cmd[cmd.index("--workspace") + 1])
        kernel = Path(cmd[cmd.index("--kernel") + 1])
        kernel.write_text(
            "__global__ void kernel() { int x = 2; }\n",
            encoding="utf-8",
        )
        _git(workspace, "add", "kernel.cuh")
        _git(workspace, "commit", "-m", "validated optimize")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        campaign = workspace / "forge_experiments"
        campaign.mkdir()
        (campaign / "best_result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "correctness_passed": True,
                    "commit_hash": best_commit,
                    "baseline_wall_ms": 10.0,
                    "best_wall_ms": 8.0,
                    "mean_case_speedup": 1.25,
                    "search_start_mean_case_speedup": 1.0,
                }
            ),
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(fc, "_run_with_tree_timeout", _timeout_after_keep)
    input_json = tmp_path / "in.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    rc = fc.main(["--input-json", str(input_json)])

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 124
    assert written["decision"] == "KEEP"
    assert written["salvaged"] is True
    assert Path(written["patch"]).is_file()
    assert source.read_text(encoding="utf-8") == "__global__ void kernel() { int x = 0; }\n"


def test_timeout_keeps_classification_with_partial_result(tmp_path, monkeypatch):
    """A truncated result at deadline remains a timeout verdict."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"
    payload = _payload(
        output_dir,
        source_file=str(source),
        kernel_repo=str(repo),
    )
    payload["candidate"] = {
        **_CANDIDATE,
        "source_file": str(source),
        "kernel_repo": str(repo),
    }
    monkeypatch.setattr(fc, "_needs_inplace", lambda _repo: True)

    def _timeout_with_partial_result(cmd, timeout):
        """Write an interrupted result and capture the internal deadline."""
        deadline = int(cmd[cmd.index("--deadline-unix") + 1])
        assert deadline <= int(time.time()) + timeout - 25
        (output_dir / "forge_result.json").write_text(
            '{"improved":',
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(
        fc,
        "_run_with_tree_timeout",
        _timeout_with_partial_result,
    )
    input_json = tmp_path / "in.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    rc = fc.main(["--input-json", str(input_json)])

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 124
    assert written["decision"] == "REVERT"
    assert written["error_class"] == "subprocess_timeout"


def test_active_campaign_is_never_moved(tmp_path):
    """Campaign archival must fail closed while KernelForge owns its lock."""
    workspace = tmp_path / "repo"
    campaign = workspace / "forge_experiments"
    campaign.mkdir(parents=True)
    lock_path = campaign / "workspace.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active Forge campaign"):
            fc._preserve_campaign(
                str(workspace),
                tmp_path / "output",
                "prior_forge_experiments",
            )
    assert campaign.is_dir()


def test_main_reports_structured_failure_on_bad_input(tmp_path, capsys):
    bad = tmp_path / "in.json"
    bad.write_text(json.dumps({"candidate": {}}), encoding="utf-8")  # no output_dir
    rc = fc.main(["--input-json", str(bad)])
    assert rc == 2
    output = capsys.readouterr().out
    payload = json.loads(
        output.split(fc.RESULT_BEGIN, 1)[1].split(fc.RESULT_END, 1)[0]
    )
    assert payload["status"] == "failed"
    assert payload["engine"] == "forge_collective"
