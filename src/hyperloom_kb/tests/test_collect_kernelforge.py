# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The packaged ``kernelforge-v1`` mapping over the document ``kernelforge export-runs`` writes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import repeat_group_key
from hyperloom_kb.collect import collect, load_mapping
from hyperloom_kb.schema import CATEGORIES, TEXT_MAX_BYTES, FieldKind, FieldRole

MAPPING = "kernelforge-v1"
CAMPAIGN = "5f1c0a9e2b7d4c38a6e0f1b2c3d4e5f6"
TOKEN_PLAN = "Summary: per token: one scale."
REASONING = "Summary: Two passes over each row cost a second read.\n\n## Evidence\nFusing them halves the traffic."
ANALYSIS_REPORT = "# Analysis report\n\nEach row is read twice; both passes are bound by memory bandwidth."
GENERATION = "analysis/ce146831869e122424b2a67038ae466565882a3d/generation-001"


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
        "plan": _file(orchestration / "optimization_plan.md", "Summary: Fuse the passes.\n\n# Fuse\nOne sweep.\n"),
        "kernel": _file(tmp_path / "candidates" / "iter_001" / "softmax_kernel.py", "BLOCK = 128\n"),
        "diff": _file(tmp_path / "candidates" / "iter_001" / "change.diff", "--- a/softmax_kernel.py\n"),
        "raw_log": _file(tmp_path / "raw-logs-not-collected.log", ""),
    }


def _session_unit(files: dict[str, Path], **fields: Any) -> dict[str, Any]:
    """A single-session iteration unit, shaped as the normalizer writes it."""

    unit = {
        "unit_id": "forge_loop_triton_softmax/iteration-1",
        "run_id": CAMPAIGN,
        "seq": 1,
        "identity": {"operator": "_softmax_kernel", "kernel": "softmax", "backend": "triton", "gpu": "mi300x"},
        "provenance": {
            "candidate_origin": "session",
            "meta": {
                "commit_hash": "abc1234",
                "wall_ms": 0.044,
                "mean_case_speedup": 1.18,
                "bench": {"measurements": [{"median_ms": 0.044, "all_times_ms": [0.043, 0.044, 0.045]}]},
                "vgpr": None,
            },
            "analysis_generation": GENERATION,
        },
        "started_at": "2026-10-15T10:00:00",
        "completed_at": "2026-10-15T10:04:30",
        "best_wall_ms_before": 0.052,
        "decision": "KEEP",
        "candidate_wall_ms": 0.044,
        "plan_file": str(files["plan"]),
        "plan": "Small rows are bound by launch latency. This plan fuses the max and sum passes.",
        "source_ms": 0.052,
        "source_code": str(files["kernel"]),
        "change_diff_file": str(files["diff"]),
        "analysis_report": ANALYSIS_REPORT,
        "reasoning": f"{REASONING}\n",
    }
    unit.update(fields)
    return unit


def test_the_mapping_produces_the_packaged_kernelforge_declaration() -> None:
    mapping = load_mapping(MAPPING)
    declaration = mapping.declaration
    assert declaration.schema_ref.startswith("schema:sha256:")
    assert (mapping.producer.name, mapping.producer.version) == ("kernelforge", "10")
    [objective] = declaration.objectives
    assert {key: objective.to_dict()[key] for key in ("id", "unit", "direction")} == {
        "id": "kernel_time_ms@v1",
        "unit": "ms",
        "direction": "lower_is_better",
    }
    measured = [declaration.role_field(category, FieldRole.MEASUREMENT) for category in ("baseline", "outcome")]
    assert [field.name for field in measured if field is not None] == ["best_wall_ms_before", "candidate_wall_ms"]
    assert [(field.name, field.kind.value, field.required) for field in declaration.identity] == [
        ("operator", "string", False),
        ("kernel", "string", False),
        ("framework", "string", False),
        ("backend", "string", False),
        ("backend_version", "version", False),
        ("gpu", "string", False),
        ("dtype", "string", False),
        ("driver_sha256", "string", False),
    ]
    linked = [
        (category, field.name)
        for category in CATEGORIES
        for field in declaration.fields(category)
        if field.kind is FieldKind.FILE
    ]
    assert linked == [("rationale", "plan_file"), ("change", "source_code"), ("change", "change_diff_file")]
    assert [(field.name, field.kind.value, field.required) for field in declaration.fields("reflection")] == [
        ("analysis_report", "text", False),
    ]


def test_a_session_unit_projects_a_complete_experience(files: dict[str, Path]) -> None:
    unit = _session_unit(files)

    report = collect(MAPPING, {"runs": [unit], "raw_logs": [str(files["raw_log"])]}, dry_run=True).to_dict()

    assert report["skipped"] == [] and report["counts"]["collected"] == 1
    experience = report["collected"][0]["experience"]
    assert {key: experience[key] for key in ("run_id", "seq", "created_at", "completed_at", "objective")} == {
        "run_id": CAMPAIGN,
        "seq": 1,
        "created_at": "2026-10-15T10:00:00Z",
        "completed_at": "2026-10-15T10:04:30Z",
        "objective": "kernel_time_ms@v1",
    }
    assert experience["identity"] == {
        "operator": "_softmax_kernel",
        "kernel": "softmax",
        "backend": "triton",
        "gpu": "mi300x",
    }
    assert experience["baseline"] == {"best_wall_ms_before": 0.052, "source_ms": 0.052}
    assert experience["rationale"] == {
        "reasoning": REASONING,
        "plan": "Small rows are bound by launch latency. This plan fuses the max and sum passes.",
        "plan_file": _ref(files["plan"]),
    }
    assert experience["change"] == {"source_code": _ref(files["kernel"]), "change_diff_file": _ref(files["diff"])}
    assert experience["outcome"] == {"decision": "KEEP", "candidate_wall_ms": 0.044}
    assert experience["reflection"] == {"analysis_report": ANALYSIS_REPORT}
    assert experience["provenance"]["source_ref"] == "forge_loop_triton_softmax/iteration-1"
    assert experience["provenance"]["extra"] == {
        "candidate_origin": "session",
        "analysis_generation": GENERATION,
        "meta": {"commit_hash": "abc1234", "wall_ms": 0.044, "mean_case_speedup": 1.18},
    }
    assert (
        experience["raw_logs"]
        == [_ref(files["raw_log"])]
        == [
            {
                "name": "raw-logs-not-collected.log",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "bytes": 0,
            }
        ]
    )


def test_a_unit_without_an_analysis_report_projects_none(files: dict[str, Path]) -> None:
    report = collect(MAPPING, {"runs": [_session_unit(files, analysis_report=None)]}, dry_run=True).to_dict()

    [row] = report["collected"]
    assert "analysis_report" not in row["experience"]["reflection"]


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"read_error": "run_state.json names no campaign_id"}, "The logs could not be read as a run"),
        ({"reasoning": None}, "No reasoning, which every Experience requires"),
    ],
)
def test_units_the_mapping_requires_more_of_are_skipped_with_its_reason(
    files: dict[str, Path], fields: dict[str, Any], reason: str
) -> None:
    report = collect(MAPPING, {"runs": [_session_unit(files, **fields)]}, dry_run=True).to_dict()

    assert report["collected"] == []
    assert report["skipped"][0]["reason"].startswith(reason)


def test_reasoning_longer_than_a_text_field_holds_skips_the_candidate(files: dict[str, Path]) -> None:
    unit = _session_unit(files, reasoning="x" * (TEXT_MAX_BYTES + 1))

    report = collect(MAPPING, {"runs": [unit]}, dry_run=True).to_dict()

    assert report["collected"] == []
    assert report["skipped"][0]["reason"].startswith("schema validation failed: rationale.reasoning holds")


def test_a_per_token_plan_and_its_file_are_collected(files: dict[str, Path]) -> None:
    files["plan"].write_text(f"{TOKEN_PLAN}\n", encoding="utf-8")
    unit = _session_unit(files, plan=TOKEN_PLAN)

    report = collect(MAPPING, {"runs": [unit], "raw_logs": [str(files["raw_log"])]}, dry_run=True).to_dict()

    assert report["skipped"] == [] and report["counts"]["collected"] == 1
    assert report["collected"][0]["experience"]["rationale"]["plan"] == TOKEN_PLAN


def test_a_token_named_case_and_plan_prose_in_meta_are_not_published(files: dict[str, Path]) -> None:
    unit = _session_unit(files)
    unit["provenance"]["meta"].update(
        {
            "bench": {"case_times": {"tokens_128_e8": 1.2}},
            "plan": TOKEN_PLAN,
            "rationale": '{"max_tokens": 4}',
            "kept": False,
            "blockage_epoch": 0,
            "commit_hash": "",
        }
    )

    report = collect(MAPPING, {"runs": [unit], "raw_logs": [str(files["raw_log"])]}, dry_run=True).to_dict()

    assert report["skipped"] == [] and report["counts"]["collected"] == 1
    meta = report["collected"][0]["experience"]["provenance"]["extra"]["meta"]
    assert meta["mean_case_speedup"] == 1.18
    assert meta["kept"] is False and meta["blockage_epoch"] == 0
    assert "commit_hash" not in meta
    for name in ("plan", "rationale", "bench"):
        assert name not in meta


def test_different_driver_hashes_do_not_share_a_repeat_group(files: dict[str, Path]) -> None:
    """The hash is in the identity, so it is in the repeat group: the same bytes group, different bytes do not."""
    digest_a = "a" * 64
    digest_b = "b" * 64
    identity = {"operator": "_softmax_kernel", "kernel": "softmax", "backend": "triton", "gpu": "mi300x"}

    def unit(seq: int, digest: str) -> dict[str, Any]:
        return _session_unit(
            files,
            identity={**identity, "driver_sha256": digest},
            unit_id=f"forge_loop_triton_softmax/iteration-{seq}",
            seq=seq,
        )

    report = collect(MAPPING, {"runs": [unit(1, digest_a), unit(2, digest_a), unit(3, digest_b)]}, dry_run=True)
    declaration = load_mapping(MAPPING).declaration
    keys = [repeat_group_key(declaration, row.experience) for row in report.collected]

    assert report.skipped == () and len(keys) == 3
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert [row.experience.identity["driver_sha256"] for row in report.collected] == [digest_a, digest_a, digest_b]
