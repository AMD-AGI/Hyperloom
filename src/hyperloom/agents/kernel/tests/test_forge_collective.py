###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for the forge-loop wrapper that drives collective-kernel optimisation."""

from __future__ import annotations

import base64
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import forge_collective as fc  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Restore ``os.environ`` after every test.

    ``_inject_author_gateway_env`` mutates ``os.environ`` in place by design and
    ``monkeypatch`` does not revert keys a function writes directly, so without
    this snapshot the injected auth aliases pollute later auth/endpoint tests in
    a full-suite run.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


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

_AUTHOR_ENV_KEYS = (
    "OPENAI_BASE_URL",
    "SAFE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "IS_SANDBOX",
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "API_TIMEOUT_MS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "DISABLE_AUTOUPDATER",
)


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


def _clear_author_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop inherited author environment so the derivation path is deterministic."""
    for key in _AUTHOR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


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
    assert cmd[1:4] == ["-m", "kernelforge.cli", "forge-loop"]


def test_an_explicit_cli_override_is_honoured(tmp_path):
    """An operator pinning a real console script must still win."""
    payload = dict(_payload(tmp_path), cli="/usr/local/bin/kernelforge")
    cmd = fc._build_cmd(
        payload,
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[:2] == ["/usr/local/bin/kernelforge", "forge-loop"]


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
    ["bench_repeat"],
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


def test_cmd_carries_kb_identity(tmp_path):
    """forge-loop needs these to pick the kernel backend and place the KB page."""
    payload = _payload(
        tmp_path,
        source_files=["/repo/custom_all_reduce.cuh"],
        operator_name="cross_device_reduce_1stage",
        framework="sglang",
        experience_id="attempt-7",
    )
    cmd = fc._build_cmd(
        payload,
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[cmd.index("--kernel-backend") + 1] == fc.COLLECTIVE_KERNEL_BACKEND
    assert cmd[cmd.index("--framework") + 1] == "sglang"
    assert cmd[cmd.index("--operator-name") + 1] == "cross_device_reduce_1stage"
    assert cmd[cmd.index("--source-files") + 1] == "/repo/custom_all_reduce.cuh"
    assert cmd[cmd.index("--experiment-id") + 1] == fc.EXPERIMENT_ID
    assert cmd[cmd.index("--experience-id") + 1] == "attempt-7"
    # Both were rejected upstream: one is a hidden legacy alias, the other a
    # documented no-op.
    assert "--workload-key" not in cmd
    assert "--max-iters" not in cmd


def test_cmd_pins_codex_backend_and_disables_claude_fallback(tmp_path):
    """OpenAI-only collective must not silently fall back to unauthenticated Claude."""
    cmd = fc._build_cmd(
        _payload(
            tmp_path,
            agent_backend="codex",
            llm_model="gpt-forge-only",
        ),
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[cmd.index("--agent-backend") + 1] == "codex"
    assert cmd[cmd.index("--agent-fallback-provider") + 1] == "none"
    assert cmd[cmd.index("--model") + 1] == "gpt-forge-only"


def test_cmd_pins_claude_backend_and_model(tmp_path):
    cmd = fc._build_cmd(
        _payload(
            tmp_path,
            agent_backend="claude",
            llm_model="claude-forge-only",
        ),
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )
    assert cmd[cmd.index("--agent-backend") + 1] == "claude"
    assert "--agent-fallback-provider" not in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-forge-only"


def test_cmd_resumes_an_interrupted_campaign(tmp_path):
    """A surviving run_state.json is the only state forge-loop resumes from."""
    repo = tmp_path / "repo"
    (repo / "forge_experiments").mkdir(parents=True)
    (repo / "forge_experiments" / "run_state.json").write_text("{}", encoding="utf-8")
    payload = _payload(tmp_path, kernel_repo=str(repo))

    cmd = fc._build_cmd(
        payload,
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )

    assert "--resume" in cmd
    # forge-loop owns the campaign configuration once saved and rejects these
    # alongside --resume.
    for rejected in (
        "--kernel",
        "--driver",
        "--program-md-file",
        "--source-files",
        "--operator-name",
    ):
        assert rejected not in cmd
    assert cmd[cmd.index("--workspace") + 1] == str(repo)


def test_cmd_starts_fresh_without_campaign_state(tmp_path):
    """Campaign artifacts without run_state.json must not trigger --resume."""
    repo = tmp_path / "repo"
    (repo / "forge_experiments").mkdir(parents=True)
    payload = _payload(tmp_path, kernel_repo=str(repo))

    cmd = fc._build_cmd(
        payload,
        _rig(tmp_path),
        tmp_path,
        deadline_unix=9_999_999_999,
    )

    assert "--resume" not in cmd
    assert "--kernel" in cmd


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("invalid", id="string"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(True, id="boolean"),
    ],
)
def test_cmd_rejects_invalid_snr_threshold(tmp_path, value):
    """SNR thresholds must be finite real numbers."""
    with pytest.raises(ValueError, match="snr_threshold must be finite"):
        fc._build_cmd(
            _payload(tmp_path, snr_threshold=value),
            _rig(tmp_path),
            tmp_path,
            deadline_unix=9_999_999_999,
        )


@pytest.mark.parametrize(
    "target_functions",
    [
        pytest.param([], id="empty-list"),
        pytest.param("kernel", id="string"),
        pytest.param(["  "], id="blank-item"),
        pytest.param([1], id="non-string-item"),
    ],
)
def test_cmd_rejects_invalid_target_functions(tmp_path, target_functions):
    """Target functions must be a populated list of names."""
    with pytest.raises(ValueError, match="target_functions"):
        fc._build_cmd(
            _payload(tmp_path, target_functions=target_functions),
            _rig(tmp_path),
            tmp_path,
            deadline_unix=9_999_999_999,
        )


@pytest.mark.parametrize(
    "deadline",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param(1.5, id="float"),
    ],
)
def test_cmd_rejects_invalid_deadline(tmp_path, deadline):
    """The internal deadline must be a positive integer epoch."""
    with pytest.raises(ValueError, match="deadline_unix must be a positive integer"):
        fc._build_cmd(
            _payload(tmp_path),
            _rig(tmp_path),
            tmp_path,
            deadline_unix=deadline,
        )


# --- Timeout validation -------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(1.5, id="float"),
        pytest.param(True, id="boolean"),
        pytest.param("invalid", id="string"),
        pytest.param(object(), id="object"),
    ],
)
def test_timeout_rejects_non_integer_values(raw):
    """Wrapper timeouts must be integer-compatible values."""
    with pytest.raises(ValueError, match="invalid collective timeout"):
        fc._timeout_sec({"timeout": raw})


@pytest.mark.parametrize("raw", [0, -5])
def test_timeout_rejects_non_positive_values(raw):
    """Wrapper timeouts must reserve positive execution time."""
    with pytest.raises(ValueError, match="collective timeout must be positive"):
        fc._timeout_sec({"timeout": raw})


def test_timeout_uses_secondary_argument():
    """The legacy timeout_sec argument remains a supported fallback."""
    assert fc._timeout_sec({"timeout_sec": "42"}) == 42


def test_timeout_uses_environment_fallback(monkeypatch):
    """The environment supplies a timeout when payload fields are absent."""
    monkeypatch.setenv("FORGE_COLLECTIVE_TIMEOUT", "77")
    assert fc._timeout_sec({}) == 77


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(1.5, id="float"),
        pytest.param(True, id="boolean"),
        pytest.param("invalid", id="string"),
    ],
)
def test_campaign_timeout_rejects_invalid_finalize_grace(raw):
    """Finalize grace must be an integer-compatible duration."""
    with pytest.raises(ValueError, match="invalid finalize_grace_sec"):
        fc._campaign_timeout_sec(
            {"timeout": 3600, "finalize_grace_sec": raw},
            0,
        )


def test_campaign_timeout_rejects_negative_finalize_grace():
    """Finalize grace cannot consume negative wrapper time."""
    with pytest.raises(ValueError, match="finalize_grace_sec cannot be negative"):
        fc._campaign_timeout_sec(
            {"timeout": 3600, "finalize_grace_sec": -1},
            0,
        )


@pytest.mark.parametrize(
    "elapsed",
    [
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(True, id="boolean"),
        pytest.param("invalid", id="string"),
    ],
)
def test_campaign_timeout_rejects_invalid_elapsed_time(elapsed):
    """Elapsed time must be finite and non-negative."""
    with pytest.raises(ValueError, match="elapsed_sec must be finite and non-negative"):
        fc._campaign_timeout_sec({"timeout": 3600}, elapsed)


def test_campaign_timeout_rejects_exhausted_budget():
    """A campaign must retain the minimum executable budget."""
    with pytest.raises(ValueError, match="leaves no campaign time"):
        fc._campaign_timeout_sec(
            {"timeout": 359, "finalize_grace_sec": 300},
            0,
        )


def test_campaign_timeout_reserves_finalize_and_elapsed_time():
    """Campaign time excludes elapsed wrapper work and finalize grace."""
    assert (
        fc._campaign_timeout_sec(
            {"timeout": 1000, "finalize_grace_sec": 100},
            20.9,
        )
        == 880
    )


@pytest.mark.parametrize(
    "timeout",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param(1.5, id="float"),
    ],
)
def test_tree_timeout_rejects_invalid_duration(timeout):
    """Process-tree timeouts must be positive integers."""
    with pytest.raises(ValueError, match="timeout_sec must be a positive integer"):
        fc._run_with_tree_timeout([sys.executable, "-c", "pass"], timeout)


def test_tree_timeout_returns_completed_process():
    """A quick child returns its exit code and captured streams."""
    proc = fc._run_with_tree_timeout(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout'); sys.stderr.write('stderr')",
        ],
        5,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "stdout"
    assert proc.stderr == "stderr"


def test_tree_timeout_terminates_slow_process():
    """A slow child is terminated within the one-second test deadline."""
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        fc._run_with_tree_timeout(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            1,
        )
    assert raised.value.timeout == 1


# --- Input, environment, and output helpers ----------------------------------


def test_gateway_env_is_derived_from_openai_settings(monkeypatch):
    """Missing author settings are derived without leaking process state."""
    _clear_author_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "token-1")

    def _root_euid() -> int:
        """Simulate execution as root."""
        return 0

    monkeypatch.setattr(fc.os, "geteuid", _root_euid, raising=False)

    fc._inject_author_gateway_env()

    assert fc.os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert fc.os.environ["ANTHROPIC_API_KEY"] == "token-1"
    assert fc.os.environ["ANTHROPIC_AUTH_TOKEN"] == "token-1"
    assert fc.os.environ["IS_SANDBOX"] == "1"
    assert fc.os.environ["GIT_AUTHOR_NAME"] == "forge-bot"
    assert fc.os.environ["GIT_AUTHOR_EMAIL"] == "forge-bot@local"
    assert fc.os.environ["GIT_COMMITTER_NAME"] == "forge-bot"
    assert fc.os.environ["GIT_COMMITTER_EMAIL"] == "forge-bot@local"
    assert fc.os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert fc.os.environ["DISABLE_AUTOUPDATER"] == "1"


def test_gateway_env_preserves_operator_settings(monkeypatch):
    """Existing credentials and identity remain authoritative."""
    _clear_author_env(monkeypatch)
    values = {
        "OPENAI_BASE_URL": "https://gateway.example/v1",
        "OPENAI_API_KEY": "derived-token",
        "ANTHROPIC_BASE_URL": "https://anthropic.example",
        "ANTHROPIC_API_KEY": "api-token",
        "ANTHROPIC_AUTH_TOKEN": "auth-token",
        "IS_SANDBOX": "0",
        "GIT_AUTHOR_NAME": "author",
        "GIT_AUTHOR_EMAIL": "author@example.com",
        "GIT_COMMITTER_NAME": "committer",
        "GIT_COMMITTER_EMAIL": "committer@example.com",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0",
        "DISABLE_AUTOUPDATER": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    def _user_euid() -> int:
        """Simulate execution as a non-root user."""
        return 1000

    monkeypatch.setattr(fc.os, "geteuid", _user_euid, raising=False)

    fc._inject_author_gateway_env()

    assert {key: fc.os.environ[key] for key in values} == values


def test_a_legacy_key_does_not_outrank_the_openai_key(monkeypatch):
    """The configured key wins over ``SAFE_API_KEY``, which the installers purge."""
    _clear_author_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("SAFE_API_KEY", "stale-legacy-key")
    monkeypatch.setenv("OPENAI_API_KEY", "live-key")

    fc._inject_author_gateway_env()

    assert fc.os.environ["ANTHROPIC_API_KEY"] == "live-key"
    assert fc.os.environ["ANTHROPIC_AUTH_TOKEN"] == "live-key"


def test_a_legacy_key_alone_authenticates_nothing(monkeypatch):
    """``SAFE_API_KEY`` alone derives no credential.

    The Git identity assertions pin that the rest of the seeding still runs, so
    this cannot pass by the function doing nothing at all.
    """
    _clear_author_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("SAFE_API_KEY", "stale-legacy-key")

    fc._inject_author_gateway_env()

    assert fc.os.environ.get("ANTHROPIC_API_KEY") is None
    assert fc.os.environ.get("ANTHROPIC_AUTH_TOKEN") is None
    assert fc.os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert fc.os.environ["GIT_AUTHOR_NAME"] == "forge-bot"


def test_load_input_json_requires_path():
    """Input loading rejects a missing command-line path."""
    with pytest.raises(ValueError, match="--input-json is required"):
        fc._load_input_json("")


def test_load_input_json_requires_object(tmp_path):
    """Input JSON must contain an object at its root."""
    path = tmp_path / "input.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        fc._load_input_json(str(path))


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        pytest.param("out", "err", id="text"),
        pytest.param(b"out", b"err", id="bytes"),
    ],
)
def test_relay_writes_text_and_bytes(capsys, stdout, stderr):
    """Captured text and bytes are relayed to their matching streams."""
    fc._relay(stdout, stderr)
    captured = capsys.readouterr()
    assert captured.out == "out"
    assert captured.err == "err"


def test_persist_logs_skips_missing_output_directory(tmp_path):
    """Log persistence is a no-op before the output directory exists."""
    fc._persist_logs(str(tmp_path / "missing"), "stdout", "stderr")
    assert not (tmp_path / "missing").exists()


def test_persist_logs_writes_available_streams_and_warns(tmp_path, capsys):
    """One failed log write does not prevent the other stream persisting."""
    (tmp_path / "forge_loop_stdout.log").mkdir()

    fc._persist_logs(str(tmp_path), "stdout", "stderr")

    captured = capsys.readouterr()
    assert "failed to persist" in captured.err
    assert (tmp_path / "forge_loop_stderr.log").read_text() == "stderr"


def test_emit_reports_result_write_failure(tmp_path, capsys):
    """Emission still prints the result when result.json cannot be written."""
    (tmp_path / "result.json").mkdir()

    fc._emit({"status": "failed"}, str(tmp_path))

    captured = capsys.readouterr()
    assert "failed to persist result.json" in captured.err
    assert fc.RESULT_BEGIN in captured.out


# --- Exported patch validation ------------------------------------------------


def test_exported_patch_outside_output_directory_is_rejected(tmp_path):
    """A patch outside optimized_versions is never trusted."""
    patch = tmp_path / "stray.patch"
    patch.write_text("diff --git a/a b/a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the output directory"):
        fc._validated_exported_patch(str(tmp_path), str(patch))


def test_missing_exported_patch_is_rejected(tmp_path):
    """A declared export must resolve to an existing regular file."""
    patch = tmp_path / "optimized_versions" / "forge.patch"
    patch.parent.mkdir()
    with pytest.raises(ValueError, match="exported patch does not exist"):
        fc._validated_exported_patch(str(tmp_path), str(patch))


def test_undecodable_exported_patch_is_rejected(tmp_path):
    """Exported patches must be valid UTF-8 text."""
    patch = tmp_path / "optimized_versions" / "forge.patch"
    patch.parent.mkdir()
    patch.write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(ValueError, match="cannot read exported patch"):
        fc._validated_exported_patch(str(tmp_path), str(patch))


def test_non_diff_exported_patch_is_rejected(tmp_path):
    """Exported patch contents must begin with a Git diff header."""
    patch = tmp_path / "optimized_versions" / "forge.patch"
    patch.parent.mkdir()
    patch.write_text("not a diff\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid content"):
        fc._validated_exported_patch(str(tmp_path), str(patch))


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


def _normalize_payload(
    tmp_path: Path,
    payload: object,
    rc: int = 0,
    **kwargs: object,
) -> dict:
    """Normalize one directly supplied Forge result payload."""
    return fc._normalize_result(
        str(tmp_path),
        rc,
        _rig(tmp_path),
        result_payload=payload,
        **kwargs,
    )


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


def test_non_mapping_result_payload_is_rejected(tmp_path):
    """Injected Forge results must be mappings."""
    out = _normalize_payload(tmp_path, ["not", "a", "mapping"])
    assert out["error_class"] == "invalid_forge_result"
    assert "must be a mapping" in out["error"]


def test_empty_result_payload_is_rejected(tmp_path):
    """An empty Forge result cannot represent a completed campaign."""
    out = _normalize_payload(tmp_path, {})
    assert out["error_class"] == "empty_forge_result"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(123, id="integer"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param([], id="list"),
    ],
)
def test_invalid_result_error_field_is_rejected(tmp_path, value):
    """Forge error fields must contain a non-empty string."""
    out = _normalize_payload(tmp_path, {"error": value})
    assert out["error_class"] == "invalid_forge_result"
    assert out["error"] == "forge result has invalid error"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"improved": "yes"}, id="string"),
        pytest.param({"improved": 1}, id="integer"),
        pytest.param({"detail": "missing flag"}, id="missing"),
    ],
)
def test_invalid_result_improved_field_is_rejected(tmp_path, payload):
    """Successful result envelopes require a boolean improved field."""
    out = _normalize_payload(tmp_path, payload)
    assert out["error_class"] == "invalid_forge_result"
    assert out["error"] == "forge result requires boolean improved when no error"


@pytest.mark.parametrize(
    "speedup",
    [
        pytest.param("1.2", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param([], id="list"),
    ],
)
def test_non_numeric_speedup_is_rejected(tmp_path, speedup):
    """Mean speedup must be a real numeric value when present."""
    out = _normalize_payload(
        tmp_path,
        {"improved": False, "mean_case_speedup": speedup},
    )
    assert out["error_class"] == "invalid_forge_result"
    assert out["error"] == "forge result has invalid mean_case_speedup"


@pytest.mark.parametrize(
    "speedup",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
    ],
)
def test_non_positive_or_non_finite_speedup_is_rejected(tmp_path, speedup):
    """Mean speedup must be finite and greater than zero."""
    out = _normalize_payload(
        tmp_path,
        {"improved": False, "mean_case_speedup": speedup},
    )
    assert out["error_class"] == "invalid_forge_result"
    assert out["error"] == "forge result has invalid mean_case_speedup"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"improved": True}, id="missing"),
        pytest.param(
            {"improved": True, "mean_case_speedup": 1.0},
            id="equal",
        ),
        pytest.param(
            {"improved": True, "mean_case_speedup": 0.9},
            id="slower",
        ),
    ],
)
def test_improved_result_requires_speedup_above_one(tmp_path, payload):
    """A requested keep must report a measurable speedup."""
    out = _normalize_payload(tmp_path, payload)
    assert out["error_class"] == "invalid_forge_result"
    assert "greater than one" in out["error"]


def test_not_improved_result_rejects_speedup_above_one(tmp_path):
    """The improved flag and reported speedup must agree."""
    out = _normalize_payload(
        tmp_path,
        {"improved": False, "mean_case_speedup": 1.4},
    )
    assert out["error_class"] == "invalid_forge_result"
    assert out["error"] == "forge result has inconsistent improvement fields"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"improved": False, "iteration_count": "7"},
            id="string",
        ),
        pytest.param(
            {"improved": False, "iteration_count": True},
            id="boolean",
        ),
        pytest.param(
            {"improved": False, "iteration_count": 1.0},
            id="float",
        ),
        pytest.param(
            {"improved": False, "iteration_count": -1},
            id="negative",
        ),
        pytest.param(
            {
                "improved": False,
                "source": "best_result.json",
                "iteration": -3,
            },
            id="published-best",
        ),
    ],
)
def test_invalid_iteration_fields_are_rejected(tmp_path, payload):
    """Iteration metadata must be a non-negative integer."""
    out = _normalize_payload(tmp_path, payload)
    assert out["error_class"] == "invalid_forge_result"
    assert "invalid iteration" in out["error"]


def test_invalid_exported_patch_is_classified(tmp_path):
    """Patch validation failures retain their dedicated error class."""
    patch = tmp_path / "stray.patch"
    patch.write_text("diff --git a/a b/a\n", encoding="utf-8")
    out = _normalize_payload(
        tmp_path,
        {"improved": True, "mean_case_speedup": 1.2},
        patch_path=str(patch),
        best_commit="abc123",
    )
    assert out["error_class"] == "invalid_exported_patch"


def test_nonzero_exit_without_forge_error_is_loop_failure(tmp_path):
    """A nonzero process exit without a validated best remains a failure."""
    out = _normalize_payload(tmp_path, {"improved": False}, rc=3)
    assert out["error_class"] == "forge_loop_failed"
    assert "rc=3" in out["error"]


def test_nonzero_exit_with_validated_best_is_salvaged(tmp_path):
    """A validated patch remains keepable after a late process failure."""
    patch = _write_exported_patch(tmp_path)
    out = _normalize_payload(
        tmp_path,
        {"improved": True, "mean_case_speedup": 1.2},
        rc=124,
        patch_path=str(patch),
        changed_files=["a.cuh"],
        best_commit="abc123",
    )
    assert out["status"] == "ok"
    assert out["decision"] == "KEEP"
    assert out["salvaged"] is True


def test_forge_result_file_requires_object(tmp_path):
    """The persisted Forge result root must be a JSON object."""
    (tmp_path / "forge_result.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        fc._load_forge_result(str(tmp_path))


def test_timeout_result_is_a_plain_revert(tmp_path):
    exc = subprocess.TimeoutExpired(["kernelforge"], 10)
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


def _repository_payload(
    output_dir: Path,
    source: Path,
    repo: Path,
    **extra: object,
) -> dict:
    """Build a wrapper payload rooted in one temporary repository."""
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
    payload.update(extra)
    return payload


def _write_journal_payload(repo: Path, payload: object) -> Path:
    """Write raw or JSON restore-journal content for recovery tests."""
    journal = repo / ".git" / "hyperloom_collective_restore.json"
    text = payload if isinstance(payload, str) else json.dumps(payload)
    journal.write_text(text, encoding="utf-8")
    return journal


def _run_main(tmp_path: Path, payload: dict) -> int:
    """Invoke the wrapper with one temporary input JSON file."""
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    return fc.main(["--input-json", str(input_json)])


# --- Repository helper failures ----------------------------------------------


def test_config_snapshot_records_absent_identity(tmp_path):
    """Missing local Git identity is represented by None values."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
    )
    assert fc._config_snapshot(str(repo)) == {
        "user.name": None,
        "user.email": None,
    }


def test_config_snapshot_reports_git_failure(tmp_path, monkeypatch):
    """Unexpected Git config failures are not treated as absent keys."""

    def _fail_git(_repo: str, *_args: str):
        """Return an unexpected Git failure."""
        return subprocess.CompletedProcess([], 128, "", "fatal")

    monkeypatch.setattr(fc, "_git", _fail_git)
    with pytest.raises(RuntimeError, match="could not read local Git config"):
        fc._config_snapshot(str(tmp_path))


def test_restore_config_clears_absent_baseline_keys(tmp_path):
    """Identity keys absent at baseline are removed from local config."""
    repo, _ = _make_repo(tmp_path)

    fc._restore_config(
        str(repo),
        {"user.name": None, "user.email": None},
    )

    for key in ("user.name", "user.email"):
        proc = subprocess.run(
            ["git", "-C", str(repo), "config", "--local", "--get", key],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1


def test_restore_config_reports_failed_unset(tmp_path, monkeypatch):
    """Failure to clear a baseline-absent key is reported."""

    def _fail_git(_repo: str, *_args: str):
        """Return a failed Git unset operation."""
        return subprocess.CompletedProcess([], 128, "", "fatal")

    monkeypatch.setattr(fc, "_git", _fail_git)
    with pytest.raises(RuntimeError, match="could not clear local Git config"):
        fc._restore_config(str(tmp_path), {"user.name": None})


def test_restore_config_reports_failed_set(tmp_path, monkeypatch):
    """Failure to restore a baseline identity value is reported."""

    def _fail_git(_repo: str, *_args: str):
        """Return a failed Git set operation."""
        return subprocess.CompletedProcess([], 1, "", "fatal")

    monkeypatch.setattr(fc, "_git", _fail_git)
    with pytest.raises(RuntimeError, match="could not restore local Git config"):
        fc._restore_config(str(tmp_path), {"user.name": "test"})


def test_tracked_baseline_patch_reports_git_failure(tmp_path, monkeypatch):
    """Tracked baseline capture fails closed on a Git error."""

    def _fail_git(_repo: str, *_args: str):
        """Return a failed Git diff operation."""
        return subprocess.CompletedProcess([], 128, "", "fatal")

    monkeypatch.setattr(fc, "_git", _fail_git)
    with pytest.raises(RuntimeError, match="cannot capture tracked repository baseline"):
        fc._tracked_baseline_patch(str(tmp_path))


def test_restore_journal_rejects_non_bytes_tracked_patch(tmp_path):
    """Durable journals accept tracked patches only as bytes."""
    with pytest.raises(RuntimeError, match="tracked baseline patch must be bytes"):
        fc._write_restore_journal(
            str(tmp_path),
            {"baseline_tracked_patch": "not-bytes"},
        )


def test_recorded_tree_rejects_invalid_untracked_baseline(tmp_path):
    """Journaled untracked paths must be a list of non-empty strings."""
    with pytest.raises(RuntimeError, match="invalid untracked baseline"):
        fc._recorded_tree_matches(
            str(tmp_path),
            {"baseline_untracked": [""]},
        )


def test_recorded_tree_falls_back_to_porcelain_status(tmp_path):
    """Legacy journals without a checksum use tracked porcelain status."""
    repo, source = _make_repo(tmp_path)
    payload = {
        "orig_branch": "main",
        "orig_head": _git(repo, "rev-parse", "HEAD"),
        "baseline_untracked": [],
    }

    assert fc._recorded_tree_matches(str(repo), payload) is True

    source.write_text("dirty\n", encoding="utf-8")
    assert fc._recorded_tree_matches(str(repo), payload) is False


def test_verify_restored_repo_detects_drift(tmp_path):
    """Restore verification rejects a tree that misses its recorded HEAD."""
    repo, _ = _make_repo(tmp_path)
    with pytest.raises(RuntimeError, match="did not restore its recorded baseline"):
        fc._verify_restored_repo(
            str(repo),
            {
                "orig_branch": "main",
                "orig_head": "0" * 40,
                "baseline_untracked": [],
            },
        )


def test_preserve_campaign_avoids_existing_destination(tmp_path):
    """Campaign preservation never overwrites an existing destination."""
    workspace = tmp_path / "workspace"
    campaign = workspace / "forge_experiments"
    campaign.mkdir(parents=True)
    (campaign / "campaign.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "prior_forge_experiments").mkdir()

    fc._preserve_campaign(
        str(workspace),
        output_dir,
        "prior_forge_experiments",
    )

    preserved = [path for path in output_dir.iterdir() if path.name.startswith("prior_forge_experiments_")]
    assert len(preserved) == 1
    assert (preserved[0] / "campaign.json").is_file()


def test_verified_worktree_removal_detects_surviving_directory(
    tmp_path,
    monkeypatch,
):
    """Verified cleanup fails when the worktree directory survives."""
    workspace = tmp_path / "worktree"
    workspace.mkdir()

    def _skip_remove(*_args, **_kwargs) -> None:
        """Leave the synthetic worktree directory in place."""
        return None

    monkeypatch.setattr(fc, "_remove_worktree", _skip_remove)
    with pytest.raises(RuntimeError, match="worktree still exists"):
        fc._remove_verified_worktree(
            str(tmp_path / "repo"),
            str(tmp_path / "repo" / "kernel.cuh"),
            str(workspace),
            "forge/test",
        )


@pytest.mark.parametrize(
    "returncode,match",
    [
        pytest.param(0, "branch still exists", id="surviving-branch"),
        pytest.param(128, "cannot verify", id="unverifiable-branch"),
    ],
)
def test_verified_worktree_removal_checks_branch_result(
    tmp_path,
    monkeypatch,
    returncode,
    match,
):
    """Verified cleanup rejects surviving or unverifiable branches."""

    def _skip_remove(*_args, **_kwargs) -> None:
        """Treat the absent synthetic worktree as removed."""
        return None

    def _branch_result(_repo: str, *_args: str):
        """Return the requested branch verification status."""
        return subprocess.CompletedProcess([], returncode, "", "fatal")

    monkeypatch.setattr(fc, "_remove_worktree", _skip_remove)
    monkeypatch.setattr(fc, "_git", _branch_result)

    with pytest.raises(RuntimeError, match=match):
        fc._remove_verified_worktree(
            str(tmp_path / "repo"),
            str(tmp_path / "repo" / "kernel.cuh"),
            str(tmp_path / "missing-worktree"),
            "forge/test",
        )


def test_export_rejects_best_outside_campaign_ancestry(tmp_path):
    """A best commit must descend from the recorded campaign baseline."""
    repo, source = _make_repo(tmp_path)
    base_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--orphan", "unrelated")
    (repo / "other.txt").write_text("unrelated\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-m", "unrelated")
    unrelated_commit = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(RuntimeError, match="not descended from this campaign baseline"):
        fc._export_collective_result(
            {
                "workspace": str(repo),
                "base_commit": base_commit,
                "prepared_source": str(source),
                "source_file": str(source),
            },
            str(tmp_path / "output"),
            {
                "improved": True,
                "best_commit": unrelated_commit,
            },
        )


def test_export_requires_non_empty_patch(tmp_path, monkeypatch):
    """A validated best without patch content cannot be exported."""
    repo, source = _make_repo(tmp_path)
    base_commit = _git(repo, "rev-parse", "HEAD")
    output_dir = tmp_path / "output"
    patch = output_dir / "optimized_versions" / "forge.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("", encoding="utf-8")

    def _empty_export(*_args, **_kwargs):
        """Return no changed artifacts and leave the patch empty."""
        return "", []

    monkeypatch.setattr(fc, "_export_best_artifacts", _empty_export)
    with pytest.raises(RuntimeError, match="produced no exportable patch"):
        fc._export_collective_result(
            {
                "workspace": str(repo),
                "base_commit": base_commit,
                "prepared_source": str(source),
                "source_file": str(source),
            },
            str(output_dir),
            {
                "improved": True,
                "best_commit": base_commit,
            },
        )


def test_main_exports_patch_then_restores_live_repo(tmp_path, monkeypatch):
    """A KEEP must retain its patch while returning the live repo to baseline."""
    _clear_author_env(monkeypatch)
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


def test_inplace_workspace_preserves_dirty_tracked_baseline(
    tmp_path,
    monkeypatch,
):
    """Profiler patches present before Forge must survive exact restoration."""
    repo, source = _make_repo(tmp_path)
    profiler = repo / "profiler.py"
    removed = repo / "legacy.py"
    profiler.write_text("enabled = False\n", encoding="utf-8")
    removed.write_text("legacy = True\n", encoding="utf-8")
    _git(repo, "add", "profiler.py", "legacy.py")
    _git(repo, "commit", "-m", "add profiler files")

    profiler.write_text("enabled = True\n", encoding="utf-8")
    removed.unlink()
    user_cache = repo / "user-cache.txt"
    user_cache.write_text("keep\n", encoding="utf-8")
    baseline_status = _git(repo, "status", "--porcelain")

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    payload = _payload(
        output_dir,
        source_file=str(source),
        kernel_repo=str(repo),
    )
    monkeypatch.setattr(fc, "_needs_inplace", lambda _repo: True)

    context = fc._prepare_collective_workspace(payload, output_dir)

    assert context is not None
    source.write_text("__global__ void kernel() { int x = 4; }\n", encoding="utf-8")
    (repo / "generated.tmp").write_text("remove\n", encoding="utf-8")
    _git(repo, "add", "kernel.cuh")
    _git(repo, "commit", "-m", "forge edit")
    fc._restore_collective_workspace(context)

    assert source.read_text(encoding="utf-8") == "__global__ void kernel() { int x = 0; }\n"
    assert profiler.read_text(encoding="utf-8") == "enabled = True\n"
    assert not removed.exists()
    assert user_cache.read_text(encoding="utf-8") == "keep\n"
    assert not (repo / "generated.tmp").exists()
    assert _git(repo, "status", "--porcelain") == baseline_status
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
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


def test_precommit_crash_replays_dirty_tracked_baseline(tmp_path):
    """A crash before the baseline commit must not discard profiler patches."""
    repo, source = _make_repo(tmp_path)
    profiler = repo / "profiler.py"
    removed = repo / "legacy.py"
    profiler.write_text("enabled = False\n", encoding="utf-8")
    removed.write_text("legacy = True\n", encoding="utf-8")
    _git(repo, "add", "profiler.py", "legacy.py")
    _git(repo, "commit", "-m", "add profiler files")

    profiler.write_text("enabled = True\n", encoding="utf-8")
    removed.unlink()
    baseline_status = _git(repo, "status", "--porcelain")
    original_head = _git(repo, "rev-parse", "HEAD")
    branch = "forge/collective-precommit"
    fc._write_restore_journal(
        str(repo),
        {
            "repo": str(repo),
            "orig_branch": "main",
            "orig_head": original_head,
            "branch": branch,
            "source_file": str(source),
            "backup": source.read_bytes(),
            "relpath": "kernel.cuh",
            "base_commit": original_head,
            "config_snapshot": fc._config_snapshot(str(repo)),
            "baseline_untracked": [],
            "baseline_tracked_patch": fc._tracked_baseline_patch(str(repo)),
            "baseline_in_base_commit": False,
        },
    )
    _git(repo, "checkout", "-b", branch)
    source.write_text("__global__ void kernel() { int x = 9; }\n", encoding="utf-8")

    assert fc._recover_stale_inplace(str(repo)) is True

    assert source.read_text(encoding="utf-8") == "__global__ void kernel() { int x = 0; }\n"
    assert profiler.read_text(encoding="utf-8") == "enabled = True\n"
    assert not removed.exists()
    assert _git(repo, "status", "--porcelain") == baseline_status
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
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


def test_interrupted_restore_is_replayed_from_an_ordinary_branch(tmp_path):
    """A half-finished restore looks like an ordinary branch and must still recover.

    ``_restore_inplace`` returns HEAD to the original branch and drops the temp
    branch in a ``finally`` that runs even when the baseline replay raised, so a
    crash there leaves the agent's edits in the tree under a normal branch name.
    Judging staleness by branch name alone would discard the only record that
    can put the user's repository back.
    """
    repo, source = _make_repo(tmp_path)
    original_head = _git(repo, "rev-parse", "HEAD")
    baseline = source.read_bytes()
    fc._write_restore_journal(
        str(repo),
        {
            "repo": str(repo),
            "orig_branch": "main",
            "orig_head": original_head,
            "branch": "forge/collective-interrupted",
            "source_file": str(source),
            "backup": baseline,
            "relpath": "kernel.cuh",
            "base_commit": original_head,
            "config_snapshot": fc._config_snapshot(str(repo)),
            "baseline_untracked": [],
            "baseline_in_base_commit": True,
        },
    )
    # The agent's edit survives in the working tree; HEAD never moved.
    source.write_text("__global__ void kernel() { int agent = 1; }\n", encoding="utf-8")

    assert fc._recover_stale_inplace(str(repo)) is True

    assert source.read_bytes() == baseline
    assert _git(repo, "rev-parse", "HEAD") == original_head
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


# --- Recovery validation -----------------------------------------------------


def test_stale_recovery_skips_held_repository_lock(tmp_path):
    """Recovery yields when another campaign owns the repository lock."""
    repo, _ = _make_repo(tmp_path)
    lock_path = repo / ".git" / "forge_inplace.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert fc._recover_stale_inplace(str(repo)) is False


def test_stale_recovery_rejects_unreadable_head(tmp_path):
    """Recovery fails closed when the current branch cannot be read."""
    repo = tmp_path / "empty-repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(RuntimeError, match="cannot read current branch"):
        fc._recover_stale_inplace(str(repo))


def test_stale_branch_without_journal_is_rejected(tmp_path):
    """A stale Forge branch requires durable restore metadata."""
    repo, _ = _make_repo(tmp_path)
    _git(repo, "checkout", "-b", "forge/abandoned")
    with pytest.raises(RuntimeError, match="has no restore journal"):
        fc._recover_stale_inplace(str(repo))


@pytest.mark.parametrize(
    "payload,match",
    [
        pytest.param(
            "{not json",
            "invalid collective restore journal",
            id="corrupt-json",
        ),
        pytest.param(
            [1, 2],
            "must be an object",
            id="non-object",
        ),
        pytest.param(
            {"branch": "main"},
            "has no source backup",
            id="missing-backup",
        ),
        pytest.param(
            {"branch": "main", "backup_b64": "!!!"},
            "invalid source backup",
            id="corrupt-backup",
        ),
        pytest.param(
            {
                "branch": "main",
                "backup_b64": "",
                "baseline_tracked_patch_b64": 7,
            },
            "invalid tracked baseline",
            id="tracked-type",
        ),
        pytest.param(
            {
                "branch": "main",
                "backup_b64": "",
                "baseline_tracked_patch_b64": "!!!",
            },
            "invalid tracked baseline",
            id="corrupt-tracked-patch",
        ),
        pytest.param(
            {
                "branch": "main",
                "backup_b64": "",
                "baseline_tracked_patch_b64": base64.b64encode(b"patch").decode("ascii"),
                "baseline_tracked_sha256": "0" * 64,
            },
            "checksum mismatch",
            id="tracked-checksum",
        ),
    ],
)
def test_stale_recovery_rejects_invalid_journal(tmp_path, payload, match):
    """Malformed restore journals fail before repository mutation."""
    repo, _ = _make_repo(tmp_path)
    _write_journal_payload(repo, payload)
    with pytest.raises(RuntimeError, match=match):
        fc._recover_stale_inplace(str(repo))


def test_stale_recovery_rejects_branch_mismatch(tmp_path):
    """A stale branch must match the journaled Forge branch."""
    repo, _ = _make_repo(tmp_path)
    _git(repo, "checkout", "-b", "forge/actual")
    _write_journal_payload(repo, {"branch": "forge/other"})
    with pytest.raises(RuntimeError, match="does not match"):
        fc._recover_stale_inplace(str(repo))


def test_matching_clean_journal_is_consumed(tmp_path):
    """A clean pre-checkout journal is verified and removed."""
    repo, source = _make_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    fc._write_restore_journal(
        str(repo),
        {
            "repo": str(repo),
            "orig_branch": "main",
            "orig_head": head,
            "branch": "forge/not-checked-out",
            "source_file": str(source),
            "backup": source.read_bytes(),
            "relpath": "kernel.cuh",
            "base_commit": head,
            "config_snapshot": {
                "user.name": "test",
                "user.email": "test@example.com",
            },
            "baseline_untracked": [],
            "baseline_tracked_patch": fc._tracked_baseline_patch(str(repo)),
        },
    )

    assert fc._recover_stale_inplace(str(repo)) is True

    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "config", "--local", "--get", "user.name") == "test"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


# --- Workspace preparation and restoration ----------------------------------


@pytest.mark.parametrize("field", ["source_file", "kernel_repo"])
def test_prepare_workspace_requires_source_and_repo(tmp_path, field):
    """Workspace preparation requires both repository paths."""
    payload = {
        "source_file": "/repo/kernel.cuh",
        "kernel_repo": "/repo",
    }
    payload[field] = ""
    with pytest.raises(ValueError, match=f"{field} is required"):
        fc._prepare_collective_workspace(payload, tmp_path)


def test_prepare_workspace_skips_when_recovery_is_busy(tmp_path, monkeypatch):
    """In-place preparation yields when stale recovery cannot lock the repo."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    def _busy_recovery(_repo: str) -> bool:
        """Simulate recovery lock contention."""
        return False

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_recover_stale_inplace", _busy_recovery)

    assert (
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )
        is None
    )


def test_prepare_workspace_skips_when_repo_lock_is_unavailable(
    tmp_path,
    monkeypatch,
):
    """In-place preparation yields when its second lock acquisition fails."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    def _recovery_succeeds(_repo: str) -> bool:
        """Simulate a completed stale-recovery check."""
        return True

    def _lock_unavailable(_repo: str) -> None:
        """Simulate repository lock contention."""
        return None

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_recover_stale_inplace", _recovery_succeeds)
    monkeypatch.setattr(fc, "_acquire_repo_lock", _lock_unavailable)

    assert (
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )
        is None
    )


def test_prepare_workspace_rejects_unresolvable_baseline(
    tmp_path,
    monkeypatch,
):
    """In-place preparation fails when Git cannot resolve HEAD."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    real_git = fc._git

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    def _recovery_succeeds(_repo: str) -> bool:
        """Bypass recovery so revision lookup reaches preparation."""
        return True

    def _fail_revision_lookup(repo_path: str, *args: str):
        """Fail revision lookups while preserving other Git operations."""
        if args[:1] == ("rev-parse",):
            return subprocess.CompletedProcess([], 128, "", "fatal")
        return real_git(repo_path, *args)

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_recover_stale_inplace", _recovery_succeeds)
    monkeypatch.setattr(fc, "_git", _fail_revision_lookup)

    with pytest.raises(RuntimeError, match="cannot resolve repository baseline"):
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )


def test_prepare_workspace_recovers_after_inplace_exception(
    tmp_path,
    monkeypatch,
):
    """A failed in-place helper consumes its provisional journal."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    def _raise_prepare(
        _source: str,
        _repo: str,
        _branch: str,
        *,
        lock_fd,
    ):
        """Release the test lock before simulating helper failure."""
        fc._release_repo_lock(lock_fd)
        raise RuntimeError("prepare exploded")

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_prepare_inplace", _raise_prepare)

    with pytest.raises(RuntimeError, match="prepare exploded"):
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )

    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_prepare_workspace_recovers_when_inplace_is_unavailable(
    tmp_path,
    monkeypatch,
):
    """An unavailable in-place helper restores the provisional baseline."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    def _return_unavailable(
        _source: str,
        _repo: str,
        _branch: str,
        *,
        lock_fd,
    ) -> None:
        """Release the test lock and report unavailable preparation."""
        fc._release_repo_lock(lock_fd)
        return None

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_prepare_inplace", _return_unavailable)

    context = fc._prepare_collective_workspace(
        _repository_payload(output_dir, source, repo),
        output_dir,
    )

    assert context is None
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_prepare_workspace_rolls_back_failed_final_journal(
    tmp_path,
    monkeypatch,
):
    """A failed final journal write restores the exact dirty baseline."""
    repo, source = _make_repo(tmp_path)
    profiler = repo / "profiler.py"
    profiler.write_text("enabled = False\n", encoding="utf-8")
    _git(repo, "add", "profiler.py")
    _git(repo, "commit", "-m", "add profiler")
    profiler.write_text("enabled = True\n", encoding="utf-8")
    baseline_status = _git(repo, "status", "--porcelain")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    real_write = fc._write_restore_journal
    calls = 0

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    def _fail_second_journal(repo_path: str, restore: dict) -> None:
        """Fail only the final durable-journal update."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("journal write exploded")
        real_write(repo_path, restore)

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_write_restore_journal", _fail_second_journal)

    with pytest.raises(RuntimeError, match="journal write exploded"):
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )

    assert profiler.read_text(encoding="utf-8") == "enabled = True\n"
    assert _git(repo, "status", "--porcelain") == baseline_status
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_prepare_workspace_creates_and_removes_worktree(
    tmp_path,
    monkeypatch,
):
    """The isolated path creates a disposable Git worktree."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _uses_worktree(_repo: str) -> bool:
        """Force the isolated worktree path."""
        return False

    monkeypatch.setattr(fc, "_needs_inplace", _uses_worktree)

    context = fc._prepare_collective_workspace(
        _repository_payload(output_dir, source, repo),
        output_dir,
    )

    assert context is not None
    assert context["inplace"] is False
    workspace = Path(context["workspace"])
    assert workspace.is_dir()
    fc._restore_collective_workspace(context)
    assert not workspace.exists()


def test_prepare_workspace_cleans_failed_worktree(tmp_path, monkeypatch):
    """Worktree setup exceptions still run verified cleanup."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    expected_snapshot = {"user.name": "baseline"}
    restored = []

    def _uses_worktree(_repo: str) -> bool:
        """Force the isolated worktree path."""
        return False

    def _snapshot(_repo: str) -> dict:
        """Return a deterministic Git configuration snapshot."""
        return expected_snapshot

    def _raise_worktree(*_args, **_kwargs):
        """Simulate a worktree creation failure."""
        raise RuntimeError("worktree exploded")

    def _record_restore(repo_path: str, snapshot: dict) -> None:
        """Record the Git configuration restoration request."""
        restored.append((repo_path, snapshot))

    monkeypatch.setattr(fc, "_needs_inplace", _uses_worktree)
    monkeypatch.setattr(fc, "_config_snapshot", _snapshot)
    monkeypatch.setattr(fc, "_prepare_worktree", _raise_worktree)
    monkeypatch.setattr(fc, "_restore_config", _record_restore)

    with pytest.raises(RuntimeError, match="worktree exploded"):
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )

    assert restored == [(str(repo), expected_snapshot)]


def test_prepare_workspace_returns_none_for_unavailable_worktree(
    tmp_path,
    monkeypatch,
):
    """An unavailable isolated worktree produces a clean skip."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _uses_worktree(_repo: str) -> bool:
        """Force the isolated worktree path."""
        return False

    def _return_unavailable(*_args, **_kwargs) -> None:
        """Report that no isolated worktree could be prepared."""
        return None

    monkeypatch.setattr(fc, "_needs_inplace", _uses_worktree)
    monkeypatch.setattr(fc, "_prepare_worktree", _return_unavailable)

    assert (
        fc._prepare_collective_workspace(
            _repository_payload(output_dir, source, repo),
            output_dir,
        )
        is None
    )


def _prepare_inplace_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict]:
    """Prepare one real in-place context for restoration tests."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place preparation path."""
        return True

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    context = fc._prepare_collective_workspace(
        _repository_payload(output_dir, source, repo),
        output_dir,
    )
    assert context is not None
    return repo, source, context


def _synthetic_inplace_context(tmp_path: Path) -> tuple[Path, dict]:
    """Build a minimal in-place context without repository side effects."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "kernel.cuh"
    source.write_bytes(b"changed")
    return source, {
        "inplace": True,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "kernel_repo": str(repo),
        "config_snapshot": {},
        "restore": {
            "source_file": str(source),
            "backup": b"baseline",
        },
    }


def test_restore_workspace_reraises_campaign_preservation_failure(
    tmp_path,
    monkeypatch,
):
    """Preservation failure is raised only after successful restoration."""
    repo, _, context = _prepare_inplace_context(tmp_path, monkeypatch)

    def _raise_preserve(*_args, **_kwargs):
        """Simulate an unreadable campaign directory."""
        raise RuntimeError("campaign locked")

    monkeypatch.setattr(fc, "_preserve_campaign", _raise_preserve)

    with pytest.raises(RuntimeError, match="campaign locked"):
        fc._restore_collective_workspace(context)

    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (repo / ".git" / "hyperloom_collective_restore.json").exists()


def test_restore_workspace_detects_unrecovered_source(tmp_path, monkeypatch):
    """Silent in-place restore drift is detected before journal removal."""
    _, context = _synthetic_inplace_context(tmp_path)

    def _skip_restore(_restore: object) -> None:
        """Leave the synthetic source unchanged."""
        return None

    monkeypatch.setattr(fc, "_restore_inplace", _skip_restore)

    with pytest.raises(RuntimeError, match="did not recover"):
        fc._restore_collective_workspace(context)


def test_restore_workspace_wraps_dual_failure(tmp_path, monkeypatch):
    """Restore failure retains preceding campaign-preservation context."""
    _, context = _synthetic_inplace_context(tmp_path)

    def _raise_preserve(*_args, **_kwargs):
        """Simulate campaign-preservation failure."""
        raise RuntimeError("campaign locked")

    def _raise_restore(_restore: object) -> None:
        """Simulate repository restoration failure."""
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(fc, "_preserve_campaign", _raise_preserve)
    monkeypatch.setattr(fc, "_restore_inplace", _raise_restore)

    with pytest.raises(
        RuntimeError,
        match="workspace restore failed after campaign preservation failed",
    ):
        fc._restore_collective_workspace(context)


def test_restore_workspace_propagates_plain_restore_failure(
    tmp_path,
    monkeypatch,
):
    """A lone repository restoration failure is propagated unchanged."""
    _, context = _synthetic_inplace_context(tmp_path)

    def _raise_restore(_restore: object) -> None:
        """Simulate repository restoration failure."""
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(fc, "_restore_inplace", _raise_restore)

    with pytest.raises(RuntimeError, match="restore exploded"):
        fc._restore_collective_workspace(context)


def test_worktree_restore_restores_config_after_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    """Worktree cleanup failure cannot skip Git identity restoration."""
    restored = []
    context = {
        "inplace": False,
        "kernel_repo": str(tmp_path / "repo"),
        "source_file": str(tmp_path / "repo" / "kernel.cuh"),
        "workspace": str(tmp_path / "worktree"),
        "branch": "forge/test",
        "config_snapshot": {"user.name": "test"},
    }

    def _raise_cleanup(*_args, **_kwargs):
        """Simulate verified worktree cleanup failure."""
        raise RuntimeError("cleanup exploded")

    def _record_restore(repo: str, snapshot: dict) -> None:
        """Record the Git configuration restoration request."""
        restored.append((repo, snapshot))

    monkeypatch.setattr(fc, "_remove_verified_worktree", _raise_cleanup)
    monkeypatch.setattr(fc, "_restore_config", _record_restore)

    with pytest.raises(RuntimeError, match="cleanup exploded"):
        fc._restore_collective_workspace(context)

    assert restored == [
        (
            str(tmp_path / "repo"),
            {"user.name": "test"},
        )
    ]


def test_timeout_salvages_authoritative_best_result(tmp_path, monkeypatch):
    """A validated published best must survive timeout before result-json write."""
    _clear_author_env(monkeypatch)
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
                    "schema_version": 2,
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
    _clear_author_env(monkeypatch)
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


def test_main_skips_when_workspace_is_unavailable(tmp_path, monkeypatch):
    """Repository lock contention maps to a structured skipped result."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"

    def _return_unavailable(_payload: dict, _output_dir: Path) -> None:
        """Report unavailable workspace preparation."""
        return None

    monkeypatch.setattr(
        fc,
        "_prepare_collective_workspace",
        _return_unavailable,
    )

    rc = _run_main(
        tmp_path,
        _repository_payload(output_dir, source, repo),
    )

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 0
    assert written["status"] == "skipped"
    assert written["error_class"] == "collective_workspace_unavailable"


def test_main_rejects_non_mapping_candidate(tmp_path, monkeypatch):
    """Candidate validation failures retain their exception classification."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"
    payload = _repository_payload(output_dir, source, repo)
    payload["candidate"] = ["not", "a", "mapping"]

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place workspace path."""
        return True

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)

    rc = _run_main(tmp_path, payload)

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 2
    assert written["error_class"] == "ValueError"
    assert "candidate must be a mapping" in written["error"]
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_main_propagates_corrupt_result_without_salvage(
    tmp_path,
    monkeypatch,
):
    """A corrupt result without a published best remains invalid."""
    _clear_author_env(monkeypatch)
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place workspace path."""
        return True

    def _write_corrupt_result(cmd: list[str], _timeout: int):
        """Write invalid JSON before reporting process success."""
        result_path = Path(cmd[cmd.index("--result-json") + 1])
        result_path.write_text("{not json", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(
        fc,
        "_run_with_tree_timeout",
        _write_corrupt_result,
    )

    rc = _run_main(
        tmp_path,
        _repository_payload(output_dir, source, repo),
    )

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 2
    assert written["error_class"] == "ValueError"
    assert "invalid forge result" in written["error"]


def test_main_classifies_export_failure(tmp_path, monkeypatch):
    """A validated win with failed export is forced to revert."""
    _clear_author_env(monkeypatch)
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place workspace path."""
        return True

    def _write_validated_win(cmd: list[str], _timeout: int):
        """Commit a valid result for the export failure path."""
        workspace = Path(cmd[cmd.index("--workspace") + 1])
        kernel = Path(cmd[cmd.index("--kernel") + 1])
        result_path = Path(cmd[cmd.index("--result-json") + 1])
        kernel.write_text(
            "__global__ void kernel() { int x = 3; }\n",
            encoding="utf-8",
        )
        _git(workspace, "add", "kernel.cuh")
        _git(workspace, "commit", "-m", "optimize")
        result_path.write_text(
            json.dumps(
                {
                    "improved": True,
                    "mean_case_speedup": 1.2,
                    "best_commit": _git(workspace, "rev-parse", "HEAD"),
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _raise_export(*_args, **_kwargs):
        """Simulate failure while exporting validated artifacts."""
        raise RuntimeError("export exploded")

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_run_with_tree_timeout", _write_validated_win)
    monkeypatch.setattr(fc, "_export_best_artifacts", _raise_export)

    _run_main(
        tmp_path,
        _repository_payload(output_dir, source, repo),
    )

    written = json.loads((output_dir / "result.json").read_text())
    assert written["status"] == "failed"
    assert written["decision"] == "REVERT"
    assert written["kept"] is False
    assert written["error_class"] == "collective_export_failed"


def test_main_attaches_valid_partial_result_to_timeout(
    tmp_path,
    monkeypatch,
):
    """A parseable interrupted result is attached to timeout metadata."""
    _clear_author_env(monkeypatch)
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place workspace path."""
        return True

    def _write_partial_then_timeout(cmd: list[str], timeout: int):
        """Write progress metadata before simulating campaign timeout."""
        result_path = Path(cmd[cmd.index("--result-json") + 1])
        result_path.write_text(
            json.dumps({"improved": False, "iteration_count": 4}),
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(
        fc,
        "_run_with_tree_timeout",
        _write_partial_then_timeout,
    )

    rc = _run_main(
        tmp_path,
        _repository_payload(output_dir, source, repo),
    )

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 124
    assert written["error_class"] == "subprocess_timeout"
    assert written["partial_forge_result"]["iteration_count"] == 4
    assert (output_dir / "forge_loop_stdout.log").read_text() == "partial stdout"
    assert (output_dir / "forge_loop_stderr.log").read_text() == "partial stderr"


def test_main_classifies_workspace_restore_failure(
    tmp_path,
    monkeypatch,
):
    """A final repository restore failure replaces the prior result."""
    _clear_author_env(monkeypatch)
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "run"

    def _requires_inplace(_repo: str) -> bool:
        """Force the in-place workspace path."""
        return True

    def _write_no_improvement(cmd: list[str], _timeout: int):
        """Write a completed no-improvement result."""
        result_path = Path(cmd[cmd.index("--result-json") + 1])
        result_path.write_text(
            json.dumps({"improved": False}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    real_restore = fc._restore_collective_workspace

    def _restore_then_raise(context: dict) -> None:
        """Restore safely before simulating a reported restore failure."""
        real_restore(context)
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(fc, "_needs_inplace", _requires_inplace)
    monkeypatch.setattr(fc, "_run_with_tree_timeout", _write_no_improvement)
    monkeypatch.setattr(
        fc,
        "_restore_collective_workspace",
        _restore_then_raise,
    )

    rc = _run_main(
        tmp_path,
        _repository_payload(output_dir, source, repo),
    )

    written = json.loads((output_dir / "result.json").read_text())
    assert rc == 2
    assert written["error_class"] == "collective_workspace_restore_failed"
    assert written["pre_restore_result"]["decision"] == "REVERT"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


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
    payload = json.loads(output.split(fc.RESULT_BEGIN, 1)[1].split(fc.RESULT_END, 1)[0])
    assert payload["status"] == "failed"
    assert payload["engine"] == "forge_collective"


# --- Bandwidth carried back into the session -----------------------------------


def test_bandwidth_travels_on_the_forge_result(tmp_path):
    """forge-loop consumes the driver's stdout, so the loop result is the channel."""
    payload = {"allreduce_bf16_1024x6144": {"bytes": 12582912.0, "busbw_gbps": 375.5}}
    result = fc._normalize_result(
        str(tmp_path),
        0,
        {},
        result_payload={"improved": True, "case_bandwidth": payload},
    )

    assert result["bandwidth"] == payload


def test_a_loop_result_without_bandwidth_reports_none(tmp_path):
    """An older forge-loop still produces a verdict, just no bandwidth."""
    result = fc._normalize_result(str(tmp_path), 0, {}, result_payload={"improved": True})

    assert "bandwidth" not in result
