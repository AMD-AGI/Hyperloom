# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The packaged ``kernelforge-v1`` mapping over the document ``kernelforge export-runs`` writes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb.collect import collect, load_mapping

MAPPING = "kernelforge-v1"
CAMPAIGN = "5f1c0a9e2b7d4c38a6e0f1b2c3d4e5f6"


def _file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    orchestration = tmp_path / "orchestration" / "iter_001"
    return {
        "log": _file(tmp_path / "forge-loop.log", "Starting autonomous iteration loop\n"),
        "plan": _file(orchestration / "optimization_plan.md", "Summary: Fuse the passes.\n\n# Fuse\nOne sweep.\n"),
        "reasoning_log": _file(
            orchestration / "optimization_plan.reasoning.md", "Summary: Two passes cost a read.\n\n## Evidence\n"
        ),
        "kernel": _file(tmp_path / "candidates" / "iter_001" / "softmax_kernel.py", "BLOCK = 128\n"),
        "lesson": _file(tmp_path / "lessons" / "iter_001.md", "One sweep saved a global read.\n"),
    }


def _session_unit(files: dict[str, Path], **fields: Any) -> dict[str, Any]:
    """A single-session iteration unit, shaped as the normalizer writes it."""

    unit = {
        "unit_id": "forge_loop_triton_softmax/iteration-1",
        "kind": "iteration",
        "run_id": CAMPAIGN,
        "seq": 1,
        "identity": {"kernel": "softmax_kernel", "backend": "triton", "gpu": "mi300x"},
        "provenance": {
            "llm_backend": "claude",
            "llm_model": "Claude-Opus-5",
            "commit": "abc1234",
            "candidate_origin": "session",
        },
        "log_files": [str(path) for path in files.values()],
        "objective": "kernel_mean_case_speedup@v1",
        "started_at": "2026-10-15T10:00:00",
        "completed_at": "2026-10-15T10:04:30",
        "best_mean_case_speedup": 1.0,
        "decision": "KEEP",
        "mean_case_speedup": 1.18,
        "plan_file": str(files["plan"]),
        "plan": "Small rows are bound by launch latency. This plan fuses the max and sum passes.",
        "source_ms": 0.052,
        "approach": "Fuse the max and sum passes into one sweep",
        "source_code": str(files["kernel"]),
        "lesson_file": str(files["lesson"]),
        "lesson": "One sweep saved a global read.",
        "reasoning": "Two passes over each row cost a second read. Fusing them halves the traffic.",
        "reasoning_file": ".claude/projects/-runs-forge-loop-triton-softmax/session.jsonl",
        "reasoning_log": str(files["reasoning_log"]),
        "alternatives": None,
    }
    unit.update(fields)
    return unit


def test_the_mapping_produces_the_packaged_kernelforge_declaration() -> None:
    mapping = load_mapping(MAPPING)
    assert mapping.declaration.schema_ref.startswith("schema:sha256:")
    assert (mapping.producer.name, mapping.producer.version) == ("kernelforge", "10")
    reasoning_log = mapping.declaration.field("rationale", "reasoning_log")
    assert reasoning_log is not None
    assert (reasoning_log.kind.value, reasoning_log.required) == ("file", False)


def test_a_session_unit_projects_a_complete_experience(files: dict[str, Path]) -> None:
    report = collect(MAPPING, {"runs": [_session_unit(files)]}, dry_run=True).to_dict()

    assert report["skipped"] == [] and report["counts"]["collected"] == 1
    experience = report["collected"][0]["experience"]
    assert {key: experience[key] for key in ("run_id", "seq", "created_at", "completed_at", "objective")} == {
        "run_id": CAMPAIGN,
        "seq": 1,
        "created_at": "2026-10-15T10:00:00Z",
        "completed_at": "2026-10-15T10:04:30Z",
        "objective": "kernel_mean_case_speedup@v1",
    }
    assert experience["identity"] == {"kernel": "softmax_kernel", "backend": "triton", "gpu": "mi300x"}
    assert experience["baseline"] == {"current_best_mean_case_speedup": 1.0, "source_ms": 0.052}
    assert experience["rationale"] == {
        "reasoning": "Two passes over each row cost a second read. Fusing them halves the traffic.",
        "plan": "Small rows are bound by launch latency. This plan fuses the max and sum passes.",
        "plan_file": _ref(files["plan"]),
        "reasoning_file": ".claude/projects/-runs-forge-loop-triton-softmax/session.jsonl",
        "reasoning_log": _ref(files["reasoning_log"]),
    }
    assert experience["change"] == {
        "approach": "Fuse the max and sum passes into one sweep",
        "source_code": _ref(files["kernel"]),
    }
    assert experience["outcome"] == {"decision": "KEEP", "mean_case_speedup": 1.18}
    assert experience["reflection"] == {
        "lesson": "One sweep saved a global read.",
        "lesson_file": _ref(files["lesson"]),
    }
    assert experience["provenance"]["source_ref"] == "forge_loop_triton_softmax/iteration-1"
    assert experience["provenance"]["extra"]["candidate_origin"] == "session"
    assert experience["raw_logs"] == [_ref(path) for path in files.values()]


def test_a_port_unit_projects_with_its_port_decision(files: dict[str, Path]) -> None:
    port = {
        "unit_id": "sess-gemm/port",
        "kind": "port",
        "run_id": "run_20261001_120000_gemm-forge",
        "seq": 0,
        "identity": {"kernel": "gemm_a8w8_n1024_k4096", "gpu": "mi355x"},
        "provenance": {"pipeline": "forge-rewrite-by-flydsl", "candidate_origin": "port"},
        "log_files": [str(files["log"])],
        "objective": "kernel_mean_case_speedup@v1",
        "started_at": None,
        "approach": "op=gemm_a8w8 shapes=4",
        "source_ms": 1.5,
        "decision": "PORT OK",
        "mean_case_speedup": 1.25,
        "completed_at": "2026-10-01T12:00:12.000",
        "reasoning": "Ported the operator to FlyDSL tile by tile.",
    }

    [row] = collect(MAPPING, {"runs": [port]}, dry_run=True).collected

    assert row.experience.outcome == {"decision": "PORT OK", "mean_case_speedup": 1.25}
    assert row.experience.created_at == row.experience.completed_at


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"read_error": "run_state.json names no campaign_id"}, "The logs could not be read as a run"),
        (
            {"stop": "budget-refused"},
            "No decision for this forge-loop iteration: the session log shows the forge-loop refused",
        ),
        ({"stop": "crashed"}, "No decision for this forge-loop iteration: the forge-loop crashed"),
        (
            {"stop": "cut-off"},
            "No decision for this forge-loop iteration: the session log shows forge-rewrite terminating",
        ),
        ({"stop": "log-ended"}, "No decision for this forge-loop iteration: the session log ends during it"),
        (
            {"stop": "no-decision"},
            'No decision for this forge-loop iteration: the session log holds its "--- Iteration N"',
        ),
        ({"kind": "port", "seq": 0}, "No PORT result"),
        ({"reasoning": None}, "No reasoning, which every Experience requires"),
        ({"log_files": []}, "No log files were found for this run."),
    ],
)
def test_units_the_mapping_requires_more_of_are_skipped_with_its_reason(
    files: dict[str, Path], fields: dict[str, Any], reason: str
) -> None:
    unit = _session_unit(files, **fields)
    if "stop" in fields or fields.get("kind") == "port":
        del unit["completed_at"]

    report = collect(MAPPING, {"runs": [unit]}, dry_run=True).to_dict()

    assert report["collected"] == []
    assert report["skipped"][0]["reason"].startswith(reason)
