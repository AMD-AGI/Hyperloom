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
    report = scan_server_log(_write(tmp_path, VLLM_LOG), "vllm", declared_env=("VLLM_MOE_N_SPLIT_SCHEDULE",))

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


def test_unknown_env_reports_only_declared_names(tmp_path):
    log = "".join(
        f"(APIServer pid=11) WARNING 09-22 14:58:51 [envs.py:1734] Unknown vLLM environment variable detected: {name}\n"
        for name in ("VLLM_PYTHON", "VLLM_VENV_ROOT", "VLLM_ATTENTION_BACKEND")
    )
    report = scan_server_log(_write(tmp_path, log), "vllm", declared_env={"VLLM_ATTENTION_BACKEND", "TP"})

    assert _detected(report) == [("vllm.unknown_env", "VLLM_ATTENTION_BACKEND", 1)]


def test_unknown_env_without_declared_names_is_not_detected(tmp_path):
    log = "Unknown vLLM environment variable detected: VLLM_PYTHON\n"
    report = scan_server_log(_write(tmp_path, log), "vllm")

    assert _detected(report) == []
    assert _statuses(report)["vllm.unknown_env"] == "not_detected"


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


def test_traceback_subject_behind_vllm_logger_prefix(tmp_path):
    prefix = "(APIServer pid=29434) ERROR 09-22 15:43:00 [async_llm.py:819]"
    log = (
        f"{prefix} AsyncLLM output_handler failed.\n"
        f"{prefix} Traceback (most recent call last):\n"
        f'{prefix}   File "/opt/hyperloom/vllm/vllm/v1/engine/core_client.py", line 1105, in get_output_async\n'
        f"{prefix}     raise self._format_exception(outputs) from None\n"
        f"{prefix} vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.\n"
        "(APIServer pid=29434) INFO:     Shutting down\n"
    )
    report = scan_server_log(_write(tmp_path, log), "vllm")

    assert _detected(report) == [("runtime.traceback", "vllm.v1.engine.exceptions.EngineDeadError", 1)]
    traceback = next(f for f in report["findings"] if f["rule_id"] == "runtime.traceback")
    assert traceback["evidence"] == "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue."


def test_traceback_subject_behind_torch_rank_prefix(tmp_path):
    prefix = "[rank0]:W0914 19:04:11.578000 74827 torch/_inductor/codecache.py:639] [0/0]"
    log = (
        f"{prefix} Failed to pickle cache key\n"
        f"{prefix} Traceback (most recent call last):\n"
        f'{prefix}   File "/opt/venv/lib/python3.12/site-packages/torch/_inductor/codecache.py", line 629, in dumps\n'
        f"{prefix}     self.dump(obj)\n"
        f"{prefix} RuntimeError: <pybind11 object> is not pickleable\n"
    )
    report = scan_server_log(_write(tmp_path, log), "sglang")

    assert _detected(report) == [("runtime.traceback", "RuntimeError", 1)]


def test_shutdown_tracebacks_are_not_findings(tmp_path):
    log = (
        "[2026-09-14 15:48:56] ERROR:    Exception in ASGI application\n"
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run\n'
        "    return runner.run(main)\n"
        "SystemExit: 0\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.12/asyncio/tasks.py", line 520, in wait_for\n'
        "    return await fut\n"
        "asyncio.exceptions.CancelledError\n"
        "[2026-09-14 15:48:56] ERROR:    Exception in ASGI application\n"
    )
    report = scan_server_log(_write(tmp_path, log), "sglang")

    assert _detected(report) == []
    assert _statuses(report)["runtime.traceback"] == "not_detected"


def test_traceback_without_exception_line_keeps_empty_subject(tmp_path):
    log = (
        "Traceback (most recent call last):\n"
        '  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 10, in run\n'
        "[2026-09-14 15:48:56] Scheduler exited unexpectedly\n"
    )
    report = scan_server_log(_write(tmp_path, log), "sglang")

    assert _detected(report) == [("runtime.traceback", "", 1)]


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


SGLANG_COMM_LOG = (
    "[2026-09-10 14:31:35 TP3] [AR] All-reduce call path: NCCL (custom AR disabled)\n"
    "[2026-09-10 14:31:35 TP4] [AR] All-reduce call path: NCCL (custom AR disabled)\n"
    "[2026-09-11 05:05:26 TP4] multimem all-gather disabled "
    "(module 'torch.distributed._symmetric_memory' has no attribute 'set_signal_pad_size')\n"
)


def test_sglang_reports_communication_fallbacks(tmp_path):
    report = scan_server_log(_write(tmp_path, SGLANG_COMM_LOG), "sglang")

    assert _detected(report) == [
        ("comm.custom_ar_disabled", "custom_all_reduce", 2),
        ("comm.multimem_allgather_disabled", "multimem_all_gather", 1),
    ]
    finding = next(f for f in report["findings"] if f["rule_id"] == "comm.multimem_allgather_disabled")
    assert finding["category"] == "perf_path"
    assert finding["evidence"] == (
        "[2026-09-11 05:05:26 TP4] multimem all-gather disabled "
        "(module 'torch.distributed._symmetric_memory' has no attribute 'set_signal_pad_size')"
    )


def test_communication_fallback_rules_are_sglang_only(tmp_path):
    report = scan_server_log(_write(tmp_path, SGLANG_COMM_LOG), "vllm")

    assert _detected(report) == []
    assert not {f["rule_id"] for f in report["findings"]} & {
        "comm.custom_ar_disabled",
        "comm.multimem_allgather_disabled",
    }


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
    persist_runtime_findings(scan_server_log(log, "vllm", declared_env=("VLLM_MOE_N_SPLIT_SCHEDULE",)), slot=slot)

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
    (tmp_path / "config.yaml").write_text(
        "benchmark:\n  framework: vllm\n  envs:\n    VLLM_FOO: '1'\n    VLLM_DIR: /opt/vllm\n", encoding="utf-8"
    )
    (tmp_path / "server.log").write_text(
        "Unknown vLLM environment variable detected: VLLM_FOO\n"
        "Unknown vLLM environment variable detected: VLLM_PYTHON\n"
        "Unknown vLLM environment variable detected: VLLM_DIR\n",
        encoding="utf-8",
    )
    result: dict = {}

    _attach_baseline_launch_evidence(
        result, config_path=tmp_path / "config.yaml", output_dir=tmp_path, framework="vllm"
    )

    out = render_runtime_findings(result)
    assert out.splitlines()[1:] == [
        "- detected [correctness] vllm.unknown_env VLLM_FOO x1: Unknown vLLM environment variable detected: VLLM_FOO",
        "- not_detected: feature_disabled, capability_disabled, engine_adjusted, aiter.tuned_miss, runtime.traceback",
    ]


def _report(*findings: tuple[str, str, str, str]) -> dict:
    return {
        "findings": [
            {"rule_id": r, "category": c, "status": s, "subject": subj, "count": 1, "evidence": "", "reason": ""}
            for r, c, s, subj in findings
        ]
    }


def test_correctness_fix_refusal_cases():
    env = "vllm.unknown_env"
    before = _report(
        (env, "correctness", "detected", "VLLM_FOO"),
        ("runtime.traceback", "correctness", "detected", "RuntimeError"),
        ("feature_disabled", "perf_path", "detected", "fuse_rope_kvcache"),
    )
    resolved = _report((env, "correctness", "not_detected", ""))
    other_env = _report((env, "correctness", "detected", "VLLM_BAR"))
    still = _report((env, "correctness", "detected", "VLLM_FOO"))
    blind = _report((env, "correctness", "unknown", ""))

    assert correctness_fix_refusal(before, resolved, f"{env}:VLLM_FOO") == ""
    assert correctness_fix_refusal(before, other_env, f"{env}:VLLM_FOO") == ""
    assert correctness_fix_refusal(before, still, f"{env}:VLLM_FOO") == (
        f"{env}:VLLM_FOO is still detected on the candidate run"
    )
    assert correctness_fix_refusal(before, blind, f"{env}:VLLM_FOO") == (
        f"{env} was not observable on the candidate run"
    )
    assert correctness_fix_refusal(before, resolved, f"{env}") == f"resolves_finding {env!r} is not rule_id:subject"
    assert correctness_fix_refusal(None, resolved, f"{env}:VLLM_FOO") == "no runtime findings for the current best"
    assert correctness_fix_refusal(before, None, f"{env}:VLLM_FOO") == "no runtime findings for the candidate run"
    assert correctness_fix_refusal(before, resolved, f"{env}:VLLM_ABSENT") == (
        f"{env}:VLLM_ABSENT is not detected on the current best"
    )


def test_fallback_rules_can_justify_a_correctness_fix():
    rules = ("feature_disabled", "capability_disabled", "comm.custom_ar_disabled", "comm.multimem_allgather_disabled")
    before = _report(*((rule, "perf_path", "detected", "x") for rule in rules))
    resolved = _report(*((rule, "perf_path", "not_detected", "") for rule in rules))

    assert [correctness_fix_refusal(before, resolved, f"{rule}:x") for rule in rules] == ["", "", "", ""]


def test_unlisted_rules_cannot_justify_a_correctness_fix():
    rules = ("runtime.traceback", "aiter.tuned_miss", "engine_adjusted")
    before = _report(*((rule, "correctness", "detected", "x") for rule in rules))
    resolved = _report(*((rule, "correctness", "not_detected", "") for rule in rules))

    assert [correctness_fix_refusal(before, resolved, f"{rule}:x") for rule in rules] == [
        "runtime.traceback cannot justify a correctness fix",
        "aiter.tuned_miss cannot justify a correctness fix",
        "engine_adjusted cannot justify a correctness fix",
    ]


def test_persist_writes_slot_file(tmp_path):
    report = scan_server_log(None, "custom")
    path = persist_runtime_findings(report, slot=tmp_path / "variant_00_x")

    assert path == str(tmp_path / "variant_00_x" / "runtime_findings.json")
    assert Path(path).read_text(encoding="utf-8").startswith("{")
