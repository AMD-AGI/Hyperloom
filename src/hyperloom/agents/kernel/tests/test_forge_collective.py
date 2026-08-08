###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for the forge-loop wrapper that drives collective-kernel optimisation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
}


def _payload(tmp_path: Path, **extra) -> dict:
    item = {
        "output_dir": str(tmp_path),
        "candidate": _CANDIDATE,
        "source_file": _CANDIDATE["source_file"],
        "kernel_repo": "/repo",
        "tp": 4,
    }
    item.update(extra)
    return item


def _rig(tmp_path: Path) -> dict:
    from collective_driver_generator import generate_collective_driver

    return generate_collective_driver(_CANDIDATE, tmp_path, tp=4)


# --- Command construction -----------------------------------------------------


def test_cmd_invokes_forge_loop_as_a_module(tmp_path):
    """The ``kernel-agents`` console script is usually not installed.

    kernel_agents is resolved from $FORGE_PATH on PYTHONPATH, so the bare name
    raised FileNotFoundError and the collective lane died before its first
    iteration. forge_submit already shells out via ``-m``.
    """
    cmd = fc._build_cmd(_payload(tmp_path), _rig(tmp_path), tmp_path)
    assert cmd[:1] == [sys.executable]
    assert cmd[1:4] == ["-m", "kernel_agents.cli", "forge-loop"]


def test_an_explicit_cli_override_is_honoured(tmp_path):
    """An operator pinning a real console script must still win."""
    payload = dict(_payload(tmp_path), cli="/usr/local/bin/kernel-agents")
    cmd = fc._build_cmd(payload, _rig(tmp_path), tmp_path)
    assert cmd[:2] == ["/usr/local/bin/kernel-agents", "forge-loop"]


def test_cmd_carries_rank_count_and_generated_rig(tmp_path):
    rig = _rig(tmp_path)
    cmd = fc._build_cmd(_payload(tmp_path), rig, tmp_path)
    # Wrapping the launcher alone profiles a process that runs no kernel.
    assert "--nproc-per-node" in cmd
    assert cmd[cmd.index("--nproc-per-node") + 1] == "4"
    assert cmd[cmd.index("--driver") + 1] == rig["driver"]
    assert cmd[cmd.index("--program-md-file") + 1] == rig["program"]
    assert cmd[cmd.index("--task-type") + 1] == "repository"


def test_cmd_defaults_target_the_noise_floor(tmp_path):
    """A collective's real speedup often sits near the noise floor."""
    cmd = fc._build_cmd(_payload(tmp_path), _rig(tmp_path), tmp_path)
    assert cmd[cmd.index("--bench-repeat") + 1] == "3"
    assert cmd[cmd.index("--calibrate-noise-floor") + 1] == "5"
    assert cmd[cmd.index("--snr-threshold") + 1] == str(fc.DEFAULT_SNR_THRESHOLD)


def test_cmd_requires_source_and_repo(tmp_path):
    rig = _rig(tmp_path)
    for missing in ("source_file", "kernel_repo"):
        payload = _payload(tmp_path)
        payload[missing] = ""
        payload["candidate"] = {**_CANDIDATE, "source_file": "", "kernel_repo": ""}
        try:
            fc._build_cmd(payload, rig, tmp_path)
        except ValueError as exc:
            assert missing in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"missing {missing} was accepted")


def test_target_functions_list_is_joined(tmp_path):
    cmd = fc._build_cmd(_payload(tmp_path, target_functions=["a::b", "c"]), _rig(tmp_path), tmp_path)
    assert cmd[cmd.index("--target-functions") + 1] == "a::b,c"


# --- Result normalisation -----------------------------------------------------


def _write_forge_result(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "forge_result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_kept_result_maps_to_keep_and_requires_e2e(tmp_path):
    rig = _rig(tmp_path)
    _write_forge_result(tmp_path, {"kept": True, "speedup": 1.35, "changed_files": ["a.cuh"]})
    out = fc._normalize_result(str(tmp_path), 0, rig)
    assert out["decision"] == "KEEP"
    assert out["micro_decision"] == "candidate"
    assert out["kernel_speedup"] == 1.35
    assert out["artifact_files"] == ["a.cuh"]
    # Kernel parity alone never authorises an integrate.
    assert out["requires_e2e_validation"] is True


def test_not_kept_result_maps_to_revert(tmp_path):
    _write_forge_result(tmp_path, {"kept": False})
    out = fc._normalize_result(str(tmp_path), 0, _rig(tmp_path))
    assert out["decision"] == "REVERT"
    assert out["micro_decision"] == "no_improvement"
    assert out["requires_e2e_validation"] is False


def test_alternate_field_names_are_tolerated(tmp_path):
    """forge-loop's schema is not frozen, so reading must be defensive."""
    _write_forge_result(tmp_path, {"improved": True, "best_speedup": 2.0, "kernel": "/repo/k.cuh"})
    out = fc._normalize_result(str(tmp_path), 0, _rig(tmp_path))
    assert out["kept"] is True
    assert out["kernel_speedup"] == 2.0
    assert out["source_file"] == "/repo/k.cuh"


def test_missing_result_file_is_reported_not_raised(tmp_path):
    out = fc._normalize_result(str(tmp_path), 1, _rig(tmp_path))
    assert out["status"] == "failed"
    assert "no forge_result.json" in out["error"]


def test_corrupt_result_file_is_reported(tmp_path):
    (tmp_path / "forge_result.json").write_text("{not json", encoding="utf-8")
    out = fc._normalize_result(str(tmp_path), 0, _rig(tmp_path))
    assert out["status"] == "failed"
    assert "parse error" in out["error"]


def test_result_carries_collective_metadata(tmp_path):
    rig = _rig(tmp_path)
    _write_forge_result(tmp_path, {"kept": False})
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


def test_main_emits_sentinel_and_result_json(tmp_path, monkeypatch):
    _write_forge_result(tmp_path, {"kept": True, "speedup": 1.1})
    monkeypatch.setattr(
        fc,
        "_run_with_tree_timeout",
        lambda cmd, timeout: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    input_json = tmp_path / "in.json"
    input_json.write_text(json.dumps(_payload(tmp_path)), encoding="utf-8")

    rc = fc.main(["--input-json", str(input_json)])
    assert rc == 0
    written = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert written["decision"] == "KEEP"
    # The rig must have been generated as a side effect.
    assert (tmp_path / "driver.py").is_file()
    assert (tmp_path / "program.md").is_file()


def test_main_reports_structured_failure_on_bad_input(tmp_path, capsys):
    bad = tmp_path / "in.json"
    bad.write_text(json.dumps({"candidate": {}}), encoding="utf-8")  # no output_dir
    rc = fc.main(["--input-json", str(bad)])
    assert rc == 2
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "failed"
    assert payload["engine"] == "forge_collective"
