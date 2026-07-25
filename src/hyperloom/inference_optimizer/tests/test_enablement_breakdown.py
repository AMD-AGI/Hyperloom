# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Targeted-build observability in session_breakdown.json."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_enablement


def _state(**kw):
    return dict(kw)


def test_collect_enablement_returns_empty_when_nothing():
    assert collect_enablement(Path("/tmp"), _state(), []) == {}


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
            "built_artifacts": ["/s/enablement/builds/t1/module_aiter_core.so"],
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
        {"task_id": "t1", "routed": True},          # routing sentinel, no 'ok'
        {"ok": False, "failure_class": "compile_error", "failure_summary": "x",
         "attempt_root": "/a", "installed_versions": {}, "built_artifacts": []},
    ]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert out["build_attempt_count"] == 1
    assert out["build_attempts"][0]["failure_class"] == "compile_error"


def test_collect_enablement_multiple_attempts():
    manifest = [
        {"ok": False, "failure_class": "timeout", "failure_summary": "t1",
         "attempt_root": "/a1", "installed_versions": {}, "built_artifacts": [],
         "action": {"component": "aiter", "ref": "v0.1.0", "gpu_arch": "gfx950"}},
        {"ok": True, "failure_class": "ok", "failure_summary": "",
         "attempt_root": "/a2", "installed_versions": {"aiter_ref": "v0.2.0", "arch": "gfx950"},
         "built_artifacts": ["/a2/lib.so"], "action": {"component": "aiter"}},
    ]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert out["build_attempt_count"] == 2
    assert out["build_attempts"][0]["failure_class"] == "timeout"
    assert out["build_attempts"][1]["ok"] is True


def test_collect_enablement_vllm_ref_surfaced():
    manifest = [{
        "ok": True, "failure_class": "ok", "failure_summary": "",
        "attempt_root": "/a",
        "installed_versions": {"vllm_ref": "v0.19.0", "arch": "gfx950"},
        "built_artifacts": ["/a/vllm/_C.so"],
        "build_log_path": "/a/build.log",
    }]
    out = collect_enablement(Path("/tmp"), _state(enablement_build_manifest=manifest), [])
    assert out["build_attempts"][0]["ref"] == "v0.19.0"


def test_collect_enablement_combined_rung3_and_rung5():
    """Both attempt-runtime (stack_actions) and build_manifest fields co-exist."""
    out = collect_enablement(
        Path("/tmp"),
        _state(
            enablement_stack_actions=[{
                "kind": "runtime_candidate", "framework": "vllm", "capability": "deepseek_v4",
                "acquisition_method": "wheel", "repo_url": "", "ref": "", "index_url": "", "reason": "",
            }],
            enablement_build_manifest=[{
                "ok": False, "failure_class": "symbol_missing", "failure_summary": "fp4_moe missing",
                "attempt_root": "/b", "installed_versions": {}, "built_artifacts": [],
            }],
            enablement_last_build_failure={"failure_class": "symbol_missing", "failure_summary": "fp4_moe missing"},
        ),
        [],
    )
    assert "stack_actions" in out
    assert "build_attempts" in out
    assert "last_build_failure" in out
    assert out["last_build_failure"]["failure_class"] == "symbol_missing"
