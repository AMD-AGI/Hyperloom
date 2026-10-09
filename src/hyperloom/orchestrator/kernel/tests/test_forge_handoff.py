# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from kernelforge.kernel_rewrite_controller.handoff import read_handoff
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
    _additional_directories,
)

from hyperloom.inference_optimizer.session.session_paths import (
    forge_cycle_dir,
    next_forge_attempt_dir,
)
from hyperloom.orchestrator.kernel.forge_handoff import write_forge_handoff


class _State(SimpleNamespace):
    def current_profile_workload_context(self) -> dict:
        return dict(self.profile_context)


def _state(**overrides) -> _State:
    values = {
        "macro_cycle": 3,
        "model_name": "example/model",
        "model_path": "/models/example",
        "model_class": "decoder",
        "precision": "fp8",
        "tp": 8,
        "ep": 2,
        "isl": 1024,
        "osl": 256,
        "conc": 64,
        "max_model_len": 4096,
        "framework": "sglang",
        "framework_version": "0.5.0",
        "framework_repo_path": "",
        "baseline_config_path": "",
        "current_best": {},
        "last_profile_trace": "",
        "last_trace_analyze": {},
        "profile_context": {
            "framework": "sglang",
            "precision": "fp8",
            "model_path": "/models/example",
            "tp": 8,
            "isl": 1024,
            "osl": 256,
            "conc": 64,
            "max_model_len": 4096,
            "server_args": "--tp 8",
            "extra_envs": {"PROFILE_ENV": "enabled"},
            "unset_envs": ["STALE_SETTING"],
        },
    }
    values.update(overrides)
    return _State(**values)


def test_write_forge_handoff_records_context_and_absolute_evidence_paths(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    raw_trace = artifacts / "trace.json"
    analysis_md = artifacts / "analysis.md"
    candidates = artifacts / "kernel_candidates.json"
    source_resolution = artifacts / "kernel_source_resolution.json"
    for path in (raw_trace, analysis_md, candidates, source_resolution):
        path.write_text("{}\n", encoding="utf-8")

    state = _state(
        last_profile_trace=str(raw_trace),
        last_trace_analyze={
            "trace_input": str(raw_trace),
            "analysis_md_path": str(analysis_md),
            "candidates_path": str(candidates),
            "trace_health_warnings": [{"code": "partial_trace", "message": "one source was unavailable"}],
        },
    )
    handoff_dir = write_forge_handoff(
        session_dir,
        state,
        env_spec={
            "config": {
                "server_launch_flags": "--tp 8 --mem-fraction-static 0.9",
                "extra_server_args": "--enable-torch-compile",
                "extra_envs": {
                    "SAFE_SETTING": "1",
                    "SERVICE_API_KEY": "must-not-be-written",
                },
            },
            "launch_recipe": str(tmp_path / "recipe.yaml"),
        },
    )

    assert handoff_dir == session_dir / "kernel-agent" / "forge" / "cycle-3" / "handoff"
    workload = (handoff_dir / "workload.md").read_text(encoding="utf-8")
    serving = (handoff_dir / "serving-context.md").read_text(encoding="utf-8")
    evidence = (handoff_dir / "trace-evidence.md").read_text(encoding="utf-8")

    assert "example/model" in workload
    assert "Tensor parallelism:** `8`" in workload
    assert "--tp 8 --mem-fraction-static 0.9" in serving
    assert "--enable-torch-compile" in serving
    assert "PROFILE_ENV=enabled" in serving
    assert "SAFE_SETTING=1" in serving
    assert "SERVICE_API_KEY" not in serving
    assert str(raw_trace.resolve()) in evidence
    assert str(analysis_md.resolve()) in evidence
    assert str(candidates.resolve()) in evidence
    assert str(source_resolution.resolve()) in evidence
    assert "partial_trace" in evidence
    assert not (handoff_dir / raw_trace.name).exists()


def test_trace_evidence_resolution_filename_has_one_owner(tmp_path: Path) -> None:
    """The handoff derives the artifact name from the single contract constant, not a literal."""
    from hyperloom.common.kernel_source_contract import SOURCE_RESOLUTION_FILENAME
    from hyperloom.orchestrator.kernel.forge_handoff import build_trace_evidence_md

    candidates = tmp_path / "run" / "kernel_candidates.json"
    candidates.parent.mkdir(parents=True)
    candidates.write_text("{}\n", encoding="utf-8")
    state = _state(last_trace_analyze={"candidates_path": str(candidates)})

    evidence = build_trace_evidence_md(state)
    assert str((candidates.parent / SOURCE_RESOLUTION_FILENAME)) in evidence


def test_trace_evidence_ends_with_current_best_runtime_findings(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel.forge_handoff import build_trace_evidence_md
    from hyperloom.orchestrator.measurement.runtime_findings import persist_runtime_findings, scan_server_log

    log = tmp_path / "server.log"
    log.write_text("WARNING Disabling fuse_rope_kvcache.\n", encoding="utf-8")
    persist_runtime_findings(scan_server_log(str(log), "sglang"), slot=tmp_path / "slot")
    state = _state(current_best_measurement={"launch_evidence_path": str(tmp_path / "slot" / "launch_evidence.json")})

    evidence = build_trace_evidence_md(state)

    assert _section(evidence, "Runtime Findings") == (
        "```text\n"
        f"runtime findings for {log} [sglang]\n"
        "- detected [perf_path] feature_disabled fuse_rope_kvcache x1: WARNING Disabling fuse_rope_kvcache.\n"
        "- not_detected: capability_disabled, comm.custom_ar_disabled, comm.multimem_allgather_disabled, "
        "engine_adjusted, aiter.tuned_miss, runtime.traceback\n"
        "```\n"
    )


def test_trace_evidence_runtime_findings_not_available_without_current_best() -> None:
    from hyperloom.orchestrator.kernel.forge_handoff import build_trace_evidence_md

    evidence = build_trace_evidence_md(_state(current_best_measurement={}))

    assert _section(evidence, "Runtime Findings") == "```text\nnot available\n```\n"
    assert _section(evidence, "Hot GEMMs Missing Tuned Config") == "```text\nnot available\n```\n"
    assert _section(evidence, "Exposed Memcpy") == "```text\nnot available\n```\n"


def test_trace_evidence_reports_exposed_memcpy_from_gpu_timeline(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel.forge_handoff import build_trace_evidence_md

    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("| Compute % | 96.0% |\n", encoding="utf-8")
    (tmp_path / "perf_report_csvs").mkdir()
    (tmp_path / "perf_report_csvs" / "gpu_timeline.csv").write_text(
        "type,time ms,percent\ncomputation_time,1295.08,96.05\nexposed_memcpy_time,40.5,3.0012\n", encoding="utf-8"
    )

    evidence = build_trace_evidence_md(_state(last_trace_analyze={"analysis_md_path": str(analysis_md)}))

    assert _section(evidence, "Exposed Memcpy") == "```text\nexposed_memcpy=3.0% of GPU time\n```\n"


def _section(evidence: str, title: str) -> str:
    """The body of one ``## title`` section, up to the next section."""
    return evidence.split(f"## {title}\n\n", 1)[1].split("\n## ", 1)[0].rstrip("\n") + "\n"


_MISS_LOG = "[aiter] shape is M:64, N:3072, K:7168, not found tuned config in /tmp/a8w8.csv, will use default config!\n"


def _gemm_row(kernel_id: str, name: str, a: str, b: str) -> dict:
    return {
        "kernel_id": kernel_id,
        "name": name,
        "gpu_pct": 12.5,
        "input_shapes": [{"shape": a, "call_num": 10}, {"shape": b, "call_num": 10}],
    }


def _tuned_miss_state(tmp_path: Path, log_text: str | None) -> _State:
    import json

    from hyperloom.orchestrator.measurement.runtime_findings import persist_runtime_findings, scan_server_log

    tmp_path.mkdir(parents=True, exist_ok=True)
    log = None
    if log_text is not None:
        log = tmp_path / "server.log"
        log.write_text(log_text, encoding="utf-8")
    persist_runtime_findings(scan_server_log(str(log) if log else None, "sglang"), slot=tmp_path / "slot")
    candidates = tmp_path / "kernel_candidates.json"
    rows = [
        _gemm_row("k1", "gemm_a8w8_blockscale", "(64,7168) fp8", "(3072,7168) fp8"),
        _gemm_row("k2", "gemm_a8w8_blockscale", "(128,7168) fp8", "(3072,7168) fp8"),
        _gemm_row("k3", "fused_moe_kernel", "(64,7168) fp8", "(3072,7168) fp8"),
    ]
    candidates.write_text(json.dumps({"hot_kernels": rows}), encoding="utf-8")
    return _state(
        last_trace_analyze={"candidates_path": str(candidates)},
        current_best_measurement={"launch_evidence_path": str(tmp_path / "slot" / "launch_evidence.json")},
    )


def test_trace_evidence_names_hot_gemms_whose_shapes_missed_tuned_config(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel.forge_handoff import build_trace_evidence_md

    evidence = build_trace_evidence_md(_tuned_miss_state(tmp_path, _MISS_LOG))

    assert _section(evidence, "Hot GEMMs Missing Tuned Config") == (
        "```text\n"
        "- k1 gemm_a8w8_blockscale gpu_pct=12.5: M=64 N=3072 K=7168\n"
        "Rows not listed: unknown; a tuned-config hit is only logged under AITER_LOG_TUNED_CONFIG.\n"
        "```\n"
    )


def test_trace_evidence_tuned_miss_states_without_a_join(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel.forge_handoff import build_trace_evidence_md

    clean = build_trace_evidence_md(_tuned_miss_state(tmp_path / "clean", "server ready\n"))
    blind = build_trace_evidence_md(_tuned_miss_state(tmp_path / "blind", None))

    assert _section(clean, "Hot GEMMs Missing Tuned Config") == (
        "```text\nnone: the current-best server log reports no tuned-config miss\n```\n"
    )
    assert _section(blind, "Hot GEMMs Missing Tuned Config") == "```text\nunknown: no_server_log\n```\n"


def test_opportunity_rules_defer_fallback_paths_and_name_the_bound() -> None:
    from kernelforge.kernel_rewrite_controller.opportunity_agent import _system_prompt

    prompt = _system_prompt()

    assert "12. Read the Runtime Findings section of trace-evidence.md" in prompt
    assert "do not\n    publish a rewrite of the fallback implementation" in prompt
    assert "The Hot GEMMs Missing Tuned\n    Config section names the exact kernel_ids" in prompt
    assert "Exposed Memcpy section" in prompt


def test_write_forge_handoff_survives_missing_trace_artifacts(tmp_path: Path) -> None:
    missing_candidates = tmp_path / "missing" / "kernel_candidates.json"
    state = _state(
        last_trace_analyze={
            "candidates_path": str(missing_candidates),
        },
    )

    handoff_dir = write_forge_handoff(tmp_path / "session", state)

    assert (handoff_dir / "workload.md").is_file()
    assert (handoff_dir / "serving-context.md").is_file()
    evidence = (handoff_dir / "trace-evidence.md").read_text(encoding="utf-8")
    assert f"`{missing_candidates.resolve()}` (missing)" in evidence
    assert "Profile raw trace:** not provided" in evidence
    assert "Kernel source resolution:" in evidence


def test_handoff_exposes_configured_git_source_roots_without_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    framework_repo = tmp_path / "sglang"
    framework_repo.mkdir()
    (framework_repo / ".git").mkdir()
    nested_source = framework_repo / "python" / "sglang"
    nested_source.mkdir(parents=True)
    aiter_repo = tmp_path / "aiter"
    aiter_repo.mkdir()
    (aiter_repo / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    non_git = tmp_path / "site-packages"
    non_git.mkdir()
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        os.pathsep.join((str(nested_source), str(aiter_repo), str(non_git))),
    )
    state = _state(framework_repo_path=str(framework_repo))

    handoff_dir = write_forge_handoff(tmp_path / "session", state)

    serving = (handoff_dir / "serving-context.md").read_text(encoding="utf-8")
    assert serving.count(str(framework_repo.resolve())) == 1
    assert str(aiter_repo.resolve()) in serving
    assert str(non_git.resolve()) not in serving
    additional = _additional_directories(read_handoff(handoff_dir))
    assert str(framework_repo.resolve()) in additional
    assert str(aiter_repo.resolve()) in additional


def test_each_kernel_entry_gets_its_own_attempt_directory(tmp_path: Path) -> None:
    """A second KERNEL entry needs an output root the controller has not used."""
    session = tmp_path / "session"

    first = next_forge_attempt_dir(session, 3)
    first.mkdir(parents=True)
    second = next_forge_attempt_dir(session, 3)

    assert first == forge_cycle_dir(session, 3) / "attempt-0"
    assert second == forge_cycle_dir(session, 3) / "attempt-1"
    assert next_forge_attempt_dir(session, 4) == forge_cycle_dir(session, 4) / "attempt-0"


def test_a_foreign_directory_does_not_disturb_attempt_numbering(tmp_path: Path) -> None:
    session = tmp_path / "session"
    cycle = forge_cycle_dir(session, 0)
    (cycle / "attempt-0").mkdir(parents=True)
    (cycle / "handoff").mkdir()
    (cycle / "attempt-not-a-number").mkdir()

    assert next_forge_attempt_dir(session, 0) == cycle / "attempt-1"


def test_the_handoff_can_be_written_beside_the_attempt_that_consumes_it(tmp_path: Path) -> None:
    session = tmp_path / "session"
    attempt = next_forge_attempt_dir(session, 3)

    written = write_forge_handoff(session, _state(), handoff_dir=attempt / "handoff")

    assert written == attempt / "handoff"
    assert (written / "workload.md").is_file()
    # The per-cycle default location is untouched, so nothing else has to move.
    assert not (forge_cycle_dir(session, 3) / "handoff").exists()
