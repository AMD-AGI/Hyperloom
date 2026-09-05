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
    """A second KERNEL entry needs an output root the controller has not used.

    ``macro_cycle`` cannot supply one: it only advances in EXPLORE, which never
    runs under ``--no-framework-agent``, so keying on the cycle alone leaves every
    re-entry refused before it starts.
    """
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
