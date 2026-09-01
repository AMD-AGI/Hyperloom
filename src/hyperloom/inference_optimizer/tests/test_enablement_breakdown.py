# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement observability in session_breakdown.json."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_enablement


def _state(**kw):
    return dict(kw)


def test_collect_enablement_returns_empty_when_nothing():
    assert collect_enablement(Path("/tmp"), _state(), []) == {}


def test_collect_enablement_eval_origin_surfaced():
    out = collect_enablement(
        Path("/tmp"),
        _state(
            enablement_origin="eval",
            enablement_baseline_eval_kind="accuracy_below_floor",
            enablement_observed_accuracy=0.12,
            enablement_accuracy_floor=0.3,
            enablement_observed_task="gsm8k",
            enablement_observed_metric="exact_match",
            enablement_eval_contract_fingerprint="fp1",
            enablement_validation_pending=True,
            enablement_probe_config_path="/tmp/runs/materialized.yaml",
        ),
        [],
    )
    assert out["origin"] == "eval"
    assert out["trigger_kind"] == "accuracy_below_floor"
    assert out["observed_accuracy"] == 0.12
    assert out["accuracy_floor"] == 0.3
    assert out["eval_contract_fingerprint"] == "fp1"
    assert out["validation_pending"] is True
    assert out["probe_config_path"]


def test_collect_enablement_build_manifest_surfaced():
    manifest = [
        {
            "ok": True,
            "attempt_root": "/s/enablement/builds/t1",
            "failure_class": "ok",
            "failure_summary": "",
            "installed_versions": {
                "torch": "2.10.0+git8514f05",
                "aiter_ref": "v0.1.0",
                "aiter_sha": "abc1234",
                "arch": "gfx950",
            },
            "build_probes": ["import aiter: ok"],
            "build_log_path": "/s/enablement/builds/t1/build.log",
        }
    ]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert "build_attempts" in out
    assert out["build_attempt_count"] == 1
    ba = out["build_attempts"][0]
    assert ba["ok"] is True
    assert ba["failure_class"] == "ok"
    assert ba["ref"] == "v0.1.0"
    assert ba["gpu_arch"] == "gfx950"
    assert ba["installed_versions"]["torch"] == "2.10.0+git8514f05"
    assert ba["attempt_root"] == "/s/enablement/builds/t1"


def test_collect_enablement_last_build_failure():
    lbf = {"failure_class": "timeout", "failure_summary": "ran out of time"}
    out = collect_enablement(
        Path("/tmp"),
        _state(enablement_last_build_failure=lbf),
        [],
    )
    assert out["last_build_failure"]["failure_class"] == "timeout"
    assert "ran out of time" in out["last_build_failure"]["failure_summary"]


def test_collect_enablement_routing_sentinels_excluded():
    """Routing sentinels (entries with no 'ok' key) must not appear as build_attempts."""
    manifest = [
        {"task_id": "t1", "routed": True},  # routing sentinel, no 'ok'
        {
            "ok": False,
            "failure_class": "compile_error",
            "failure_summary": "x",
            "attempt_root": "/a",
            "installed_versions": {},
            "build_probes": [],
        },
    ]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert out["build_attempt_count"] == 1
    assert out["build_attempts"][0]["failure_class"] == "compile_error"


def test_collect_enablement_multiple_attempts():
    manifest = [
        {
            "ok": False,
            "failure_class": "timeout",
            "failure_summary": "t1",
            "attempt_root": "/a1",
            "installed_versions": {},
            "build_probes": [],
            "action": {"component": "aiter", "ref": "v0.1.0", "gpu_arch": "gfx950"},
        },
        {
            "ok": True,
            "failure_class": "ok",
            "failure_summary": "",
            "attempt_root": "/a2",
            "installed_versions": {"aiter_ref": "v0.2.0", "arch": "gfx950"},
            "build_probes": ["import aiter: ok"],
            "action": {"component": "aiter"},
        },
    ]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert out["build_attempt_count"] == 2
    assert out["build_attempts"][0]["failure_class"] == "timeout"
    assert out["build_attempts"][1]["ok"] is True


def test_collect_enablement_vllm_ref_surfaced():
    manifest = [
        {
            "ok": True,
            "failure_class": "ok",
            "failure_summary": "",
            "attempt_root": "/a",
            "installed_versions": {"vllm_ref": "v0.19.0", "arch": "gfx950"},
            "build_probes": ["import vllm: ok"],
            "build_log_path": "/a/build.log",
        }
    ]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert out["build_attempts"][0]["ref"] == "v0.19.0"


def test_collect_enablement_combined_rung3_and_rung5():
    """Attempt runtimes and build_manifest fields co-exist."""
    out = collect_enablement(
        Path("/tmp"),
        _state(
            enablement_attempt_runtimes=[
                {
                    "framework": "vllm",
                    "acquisition_method": "wheel",
                    "repo_url": "",
                    "ref": "",
                    "index_url": "",
                }
            ],
            enablement_build_manifest=[
                {
                    "ok": False,
                    "failure_class": "symbol_missing",
                    "failure_summary": "fp4_moe missing",
                    "attempt_root": "/b",
                    "installed_versions": {},
                    "build_probes": [],
                }
            ],
            enablement_last_build_failure={"failure_class": "symbol_missing", "failure_summary": "fp4_moe missing"},
        ),
        [],
    )
    assert "attempt_runtimes" in out
    assert "build_attempts" in out
    assert "last_build_failure" in out
    assert out["last_build_failure"]["failure_class"] == "symbol_missing"


def test_collect_enablement_history_preserved_after_success():
    """After success, enablement_origin is cleared to '' but trigger history must still be surfaced."""
    out = collect_enablement(
        Path("/tmp"),
        _state(
            # After success, origin is cleared but eval_kind is preserved.
            enablement_origin="",
            enablement_succeeded=True,
            enablement_baseline_eval_kind="accuracy_below_floor",
            enablement_observed_accuracy=0.12,
            enablement_accuracy_floor=0.3,
            enablement_eval_contract_fingerprint="fp-done",
            enablement_validation_pending=False,
            enablement_probe_config_path="/tmp/runs/probe.yaml",
            enablement_accepted_config_path="/tmp/runs/accepted.yaml",
        ),
        [],
    )
    # The block should still be non-empty despite origin being cleared.
    assert out
    assert out.get("trigger_kind") == "accuracy_below_floor"
    assert out.get("observed_accuracy") == 0.12
    assert out.get("eval_contract_fingerprint") == "fp-done"
    assert out.get("succeeded") is True
    assert out.get("validation_pending") is False


def test_collect_enablement_succeeded_flag():
    """succeeded field is exposed correctly in the breakdown."""
    pending_out = collect_enablement(
        Path("/tmp"),
        _state(
            enablement_origin="eval",
            enablement_baseline_eval_kind="accuracy_unavailable",
            enablement_succeeded=False,
            enablement_validation_pending=True,
        ),
        [],
    )
    assert pending_out.get("succeeded") is False
    assert pending_out.get("validation_pending") is True

    done_out = collect_enablement(
        Path("/tmp"),
        _state(
            enablement_origin="",
            enablement_baseline_eval_kind="accuracy_unavailable",
            enablement_succeeded=True,
            enablement_validation_pending=False,
        ),
        [],
    )
    assert done_out.get("succeeded") is True


def test_collect_enablement_boot_origin_round_surfaced():
    """A boot-origin round repaired by a source patch provisions no runtime and
    builds nothing; it must still be visible."""
    out = collect_enablement(
        Path("/tmp"),
        _state(
            enablement_mode="launch",
            enablement_attempts=1,
            enablement_inflight_task_id="spec-1",
            enablement_succeeded=True,
            enablement_last_specialist_task_id="spec-1",
            enablement_launch_log="EngineCore failed to start.\nTraceback (most recent call last):",
            enablement_attempt_runtimes=[],
            enablement_build_manifest=[],
        ),
        [],
    )
    assert out
    assert out["mode"] == "launch"
    assert out["engaged"] is True
    assert out["dispatched"] is True
    assert out["origin"] == "boot"
    assert out["attempts"] == 1
    assert out["succeeded"] is True
    assert out["last_specialist_task_id"] == "spec-1"
    assert "EngineCore failed to start." in out["launch_log_excerpt"]


def test_collect_enablement_opt_out_recorded_but_armed_idle_hidden():
    """``all`` is the default, so an armed lane that never fired is not worth a
    block; an explicit opt-out is, since it explains why nothing self-healed."""
    off_out = collect_enablement(Path("/tmp"), _state(enablement_mode="off"), [])
    assert off_out["mode"] == "off"
    assert off_out["engaged"] is False
    assert off_out["attempts"] == 0
    assert collect_enablement(Path("/tmp"), _state(enablement_mode="all"), []) == {}
    # A session predating the flag loads with the SharedState default.
    assert collect_enablement(Path("/tmp"), _state(), []) == {}


def test_collect_enablement_kept_patches_relativized_and_log_bounded():
    out = collect_enablement(
        Path("/tmp/sess"),
        _state(
            enablement_mode="all",
            enablement_attempts=2,
            enablement_stall_streak=1,
            enablement_kept_patches=["/tmp/sess/patches/001_fix.patch"],
            enablement_kept_stack_action={"kind": "runtime_candidate", "framework": "vllm"},
            enablement_candidate_refs=["PR:901"],
            enablement_setup_commands=["pip install -e ."],
            enablement_human_review_logged=["log-a", "log-b"],
            enablement_launch_log="x" * 5000,
        ),
        [],
    )
    assert out["kept_patches"] == ["patches/001_fix.patch"]
    assert out["kept_stack_action"]["kind"] == "runtime_candidate"
    assert out["candidate_refs"] == ["PR:901"]
    assert out["setup_commands"] == ["pip install -e ."]
    assert out["human_review_count"] == 2
    assert out["stall_streak"] == 1
    assert len(out["launch_log_excerpt"]) == 2000


def test_collect_enablement_framework_root_surfaced():
    """kept_patches are unreadable without the tree they apply to."""
    out = collect_enablement(
        Path("/tmp/sess"),
        _state(
            enablement_kept_patches=["/tmp/sess/patches/001.patch"],
            enablement_framework_root="/sgl-workspace/sglang",
        ),
        [],
    )
    assert out["framework_root"] == "/sgl-workspace/sglang"
    assert "framework_root" not in collect_enablement(Path("/tmp/sess"), _state(enablement_attempts=1), [])


def test_collect_enablement_setting_script_field(tmp_path):
    script = tmp_path / "reports" / "enablement" / "enablement_setting.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    out = collect_enablement(
        tmp_path,
        _state(enablement_attempts=1),
        [],
    )
    assert out.get("setting_script") == "reports/enablement/enablement_setting.sh"


def test_collect_enablement_setting_script_absent_when_no_file(tmp_path):
    out = collect_enablement(
        tmp_path,
        _state(enablement_attempts=1),
        [],
    )
    assert "setting_script" not in out


# ---- kept_artifacts in collect_enablement ----------------------------------


def test_collect_enablement_kept_artifacts_included():
    """Well-formed kept_artifacts entries appear in the output."""
    out = collect_enablement(
        Path("/tmp/sess"),
        _state(
            enablement_engaged=True,
            enablement_attempts=1,
            enablement_kept_artifacts=[
                {
                    "target": "/sgl-workspace/sglang/srt/foo.py",
                    "rel_target": "srt/foo.py",
                    "kind": "python_source",
                    "existed": True,
                    "backup": "/tmp/sess/runs/integrate_patch/t1/artifact_backups/000_foo.py.bak",
                    "source": "/tmp/sess/runs/specialist/t1/worktree/artifacts/foo.py",
                }
            ],
        ),
        [],
    )
    arts = out.get("kept_artifacts", [])
    assert len(arts) == 1
    assert arts[0]["target"] == "/sgl-workspace/sglang/srt/foo.py"
    assert arts[0]["rel_target"] == "srt/foo.py"
    assert arts[0]["kind"] == "python_source"


def test_collect_enablement_kept_artifacts_tolerates_dirty_entries():
    """Non-dict and no-target entries must not raise; they are silently filtered."""
    out = collect_enablement(
        Path("/tmp/sess"),
        _state(
            enablement_engaged=True,
            enablement_attempts=1,
            enablement_kept_artifacts=[
                "a plain string",
                None,
                42,
                {"rel_target": "no_target.py", "kind": "k"},  # missing target
                {"target": "/sgl-workspace/sglang/srt/real.py", "rel_target": "srt/real.py", "kind": ""},
            ],
        ),
        [],
    )
    arts = out.get("kept_artifacts", [])
    assert len(arts) == 1
    assert arts[0]["target"] == "/sgl-workspace/sglang/srt/real.py"
