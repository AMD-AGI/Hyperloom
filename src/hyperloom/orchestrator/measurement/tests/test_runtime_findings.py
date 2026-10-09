# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors.baseline import _attach_baseline_launch_evidence
from hyperloom.orchestrator.measurement.runtime_findings import (
    correctness_fix_refusal,
    persist_runtime_findings,
    render_runtime_findings,
    scan_server_log,
)


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "server.log"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _detected(report: dict) -> list[tuple[str, str, int]]:
    return [(f["rule_id"], f["subject"], f["count"]) for f in report["findings"] if f["status"] == "detected"]


def _statuses(report: dict) -> dict[str, str]:
    return {f["rule_id"]: f["status"] for f in report["findings"] if f["status"] != "detected"}


VLLM_LOG = (
    "(APIServer pid=11) WARNING 10-09 12:00:00 [interface.py:1461] "
    "Unknown vLLM environment variable detected: VLLM_MOE_N_SPLIT_SCHEDULE\n"
    "(EngineCore_DP0 pid=12) WARNING 10-09 12:00:01 [compilation.py:1183] fuse_rope_kvcache is enabled, "
    "but splitting_ops is None and Inductor graph partition is not enabled.Disabling fuse_rope_kvcache."
    "Please either set splitting_ops to an empty list []or set use_inductor_graph_partition to True "
    "to enable RoPE+KV cache fusion.\n"
    "(EngineCore_DP1 pid=13) WARNING 10-09 12:00:01 [compilation.py:1183] fuse_rope_kvcache is enabled, "
    "but splitting_ops is None and Inductor graph partition is not enabled.Disabling fuse_rope_kvcache.\n"
    "[aiter] shape is M:64, N:7168, K:2048, not found tuned config in /tmp/bf16_tuned_gemm.csv, will use default\n"
    "[aiter] shape is M:128, N:7168, K:2048, not found tuned config in /tmp/bf16_tuned_gemm.csv, will use default\n"
    "[aiter] shape is M:64, N:7168, K:2048, not found tuned config in /tmp/bf16_tuned_gemm.csv, will use default\n"
)


def test_vllm_log_reports_each_rule_once_per_subject(tmp_path):
    report = scan_server_log(_write(tmp_path, VLLM_LOG), "vllm")

    assert _detected(report) == [
        ("aiter.tuned_miss", "aiter_tuned_config", 2),
        ("feature_disabled", "fuse_rope_kvcache", 2),
        ("vllm.unknown_env", "VLLM_MOE_N_SPLIT_SCHEDULE", 1),
    ]
    assert _statuses(report) == {
        "capability_disabled": "not_detected",
        "engine_adjusted": "not_detected",
        "runtime.traceback": "not_detected",
    }


def test_traceback_subject_is_exception_class_behind_process_prefix(tmp_path):
    log = (
        "(EngineCore_DP0 pid=12) ERROR 10-09 12:00:05 [core.py:900] EngineCore encountered an issue.\n"
        "(EngineCore_DP0 pid=12) Traceback (most recent call last):\n"
        '(EngineCore_DP0 pid=12)   File "/opt/vllm/v1/engine/core.py", line 890, in run\n'
        "(EngineCore_DP0 pid=12)     raise RuntimeError(msg)\n"
        "(EngineCore_DP0 pid=12) RuntimeError: HIP error: invalid device function\n"
    )
    report = scan_server_log(_write(tmp_path, log), "vllm")

    assert _detected(report) == [("runtime.traceback", "RuntimeError", 1)]
    traceback = next(f for f in report["findings"] if f["rule_id"] == "runtime.traceback")
    assert traceback["category"] == "correctness"
    assert traceback["evidence"] == "RuntimeError: HIP error: invalid device function"


def test_sglang_capability_and_adjusted_settings(tmp_path):
    log = (
        "[2026-10-09 12:00:00] chunked prefill size is adjusted from 16384 to 8192\n"
        "[2026-10-09 12:00:01] aiter_mla_supported() returned False\n"
    )
    report = scan_server_log(_write(tmp_path, log), "sglang")

    assert _detected(report) == [
        ("capability_disabled", "aiter_mla_supported", 1),
        ("engine_adjusted", "chunked_prefill_size", 1),
    ]
    assert "vllm.unknown_env" not in _statuses(report)


def test_atom_skips_rules_it_cannot_observe(tmp_path):
    report = scan_server_log(_write(tmp_path, "server ready\n"), "atom")

    assert _statuses(report) == {
        "feature_disabled": "not_detected",
        "capability_disabled": "not_detected",
        "aiter.tuned_miss": "not_detected",
        "runtime.traceback": "not_detected",
    }


def test_missing_log_marks_every_rule_unknown(tmp_path):
    report = scan_server_log(None, "custom")

    assert [(f["rule_id"], f["status"], f["reason"]) for f in report["findings"]] == [
        ("feature_disabled", "unknown", "no_server_log"),
        ("capability_disabled", "unknown", "no_server_log"),
        ("aiter.tuned_miss", "unknown", "no_server_log"),
        ("runtime.traceback", "unknown", "no_server_log"),
    ]


def test_unreadable_log_marks_every_rule_unknown(tmp_path):
    report = scan_server_log(str(tmp_path / "absent.log"), "vllm")

    assert {f["reason"] for f in report["findings"]} == {"unreadable: FileNotFoundError"}
    assert {f["status"] for f in report["findings"]} == {"unknown"}


def test_evidence_is_flattened_and_defanged(tmp_path):
    log = "WARNING Disabling cascade attention <script>```\n"
    report = scan_server_log(_write(tmp_path, log), "sglang")

    finding = next(f for f in report["findings"] if f["rule_id"] == "feature_disabled")
    assert finding["subject"] == "cascade attention \u2039script\u203a`\u200b``"
    assert finding["evidence"] == "WARNING Disabling cascade attention \u2039script\u203a`\u200b``"


def test_render_lists_detected_then_clear_then_unknown(tmp_path):
    log = _write(tmp_path, VLLM_LOG)
    slot = tmp_path / "slot"
    persist_runtime_findings(scan_server_log(log, "vllm"), slot=slot)

    out = render_runtime_findings({"launch_evidence_path": str(slot / "launch_evidence.json")})

    assert out.splitlines() == [
        f"runtime findings for {log} [vllm]",
        "- detected [perf_path] aiter.tuned_miss aiter_tuned_config x2: [aiter] shape is M:64, N:7168, K:2048, "
        "not found tuned config in /tmp/bf16_tuned_gemm.csv, will use default",
        "- detected [perf_path] feature_disabled fuse_rope_kvcache x2: (EngineCore_DP0 pid=12) WARNING 10-09 "
        "12:00:01 [compilation.py:1183] fuse_rope_kvcache is enabled, but splitting_ops is None and Inductor "
        "graph partition is not enabled.Disabling fuse_rope_kvcache.Please either set splitting_ops to an empty "
        "list []or set use_inductor_graph_partition to True to enabl",
        "- detected [correctness] vllm.unknown_env VLLM_MOE_N_SPLIT_SCHEDULE x1: (APIServer pid=11) WARNING "
        "10-09 12:00:00 [interface.py:1461] Unknown vLLM environment variable detected: VLLM_MOE_N_SPLIT_SCHEDULE",
        "- not_detected: capability_disabled, engine_adjusted, runtime.traceback",
    ]


def test_render_reports_missing_slot_and_file(tmp_path):
    assert render_runtime_findings({}) == "(no measurement slot recorded for the current best)"
    missing = tmp_path / "slot" / "launch_evidence.json"
    assert render_runtime_findings({"launch_evidence_path": str(missing)}) == (
        f"(no runtime findings written at {tmp_path / 'slot' / 'runtime_findings.json'})"
    )


def test_render_lists_unknown_reason(tmp_path):
    persist_runtime_findings(scan_server_log(None, "custom"), slot=tmp_path)

    out = render_runtime_findings({"launch_evidence_path": str(tmp_path / "launch_evidence.json")})

    assert out.splitlines() == [
        "runtime findings for (no server log) [custom]",
        "- unknown: feature_disabled (no_server_log)",
        "- unknown: capability_disabled (no_server_log)",
        "- unknown: aiter.tuned_miss (no_server_log)",
        "- unknown: runtime.traceback (no_server_log)",
    ]


def test_baseline_writes_runtime_findings(tmp_path):
    (tmp_path / "config.yaml").write_text("benchmark:\n  framework: vllm\n", encoding="utf-8")
    (tmp_path / "server.log").write_text("Unknown vLLM environment variable detected: VLLM_FOO\n", encoding="utf-8")
    result: dict = {}

    _attach_baseline_launch_evidence(
        result, config_path=tmp_path / "config.yaml", output_dir=tmp_path, framework="vllm"
    )

    out = render_runtime_findings(result)
    assert (
        out.splitlines()[1]
        == "- detected [correctness] vllm.unknown_env VLLM_FOO x1: Unknown vLLM environment variable detected: VLLM_FOO"
    )


def _report(*findings: tuple[str, str, str, str]) -> dict:
    return {
        "findings": [
            {"rule_id": r, "category": c, "status": s, "subject": subj, "count": 1, "evidence": "", "reason": ""}
            for r, c, s, subj in findings
        ]
    }


def test_correctness_fix_refusal_cases():
    before = _report(
        ("runtime.traceback", "correctness", "detected", "RuntimeError"),
        ("feature_disabled", "perf_path", "detected", "fuse_rope_kvcache"),
    )
    resolved = _report(("runtime.traceback", "correctness", "not_detected", ""))
    other_error = _report(("runtime.traceback", "correctness", "detected", "ValueError"))
    still = _report(("runtime.traceback", "correctness", "detected", "RuntimeError"))
    blind = _report(("runtime.traceback", "correctness", "unknown", ""))

    assert correctness_fix_refusal(before, resolved, "runtime.traceback:RuntimeError") == ""
    assert correctness_fix_refusal(before, other_error, "runtime.traceback:RuntimeError") == ""
    assert correctness_fix_refusal(before, still, "runtime.traceback:RuntimeError") == (
        "runtime.traceback:RuntimeError is still detected on the candidate run"
    )
    assert correctness_fix_refusal(before, blind, "runtime.traceback:RuntimeError") == (
        "runtime.traceback was not observable on the candidate run"
    )
    assert correctness_fix_refusal(before, resolved, "feature_disabled:fuse_rope_kvcache") == (
        "feature_disabled:fuse_rope_kvcache is perf_path, not correctness"
    )
    assert correctness_fix_refusal(before, resolved, "runtime.traceback") == (
        "resolves_finding 'runtime.traceback' is not rule_id:subject"
    )
    assert correctness_fix_refusal(None, resolved, "runtime.traceback:RuntimeError") == (
        "no runtime findings for the current best"
    )
    assert correctness_fix_refusal(before, None, "runtime.traceback:RuntimeError") == (
        "no runtime findings for the candidate run"
    )


def test_persist_writes_slot_file(tmp_path):
    report = scan_server_log(None, "custom")
    path = persist_runtime_findings(report, slot=tmp_path / "variant_00_x")

    assert path == str(tmp_path / "variant_00_x" / "runtime_findings.json")
    assert Path(path).read_text(encoding="utf-8").startswith("{")
