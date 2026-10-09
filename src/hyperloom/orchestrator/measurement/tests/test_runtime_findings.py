# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.measurement.runtime_findings import persist_runtime_findings, scan_server_log


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


def test_persist_writes_slot_file(tmp_path):
    report = scan_server_log(None, "custom")
    path = persist_runtime_findings(report, slot=tmp_path / "variant_00_x")

    assert path == str(tmp_path / "variant_00_x" / "runtime_findings.json")
    assert Path(path).read_text(encoding="utf-8").startswith("{")
