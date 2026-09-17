# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""The analyst session: what it is asked, and what it is allowed to do."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kernelforge.roofline_ceiling.analyst import (
    CeilingAnalysisError,
    build_request,
    load_role,
    run_ceiling_analysis,
)
from kernelforge.roofline_ceiling.contract import Hardware
from kernelforge.roofline_ceiling.evidence import EvidenceBundle
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_DATASHEET, PEAK_SOURCE_EMPIRICAL


class _Backend:
    """A backend that replays canned answers and records the specs it was given."""

    name = "fake"

    def __init__(self, *answers: str):
        self._answers = list(answers)
        self.specs: list = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        text = self._answers.pop(0) if self._answers else ""
        return type("Result", (), {"text": text, "end_reason": "agent_stopped"})()


def _bundle(tmp_path, peak_source: str = PEAK_SOURCE_EMPIRICAL) -> EvidenceBundle:
    artifacts = tmp_path / "evidence"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "trace").mkdir(exist_ok=True)
    (artifacts / "trace" / "kernel_stats.csv").write_text("name,count\n", encoding="utf-8")
    return EvidenceBundle(
        hardware=Hardware(
            arch="gfx950",
            peak_flops={"bf16_mfma": 1.686e15},
            bandwidth={"hbm": 6.24e12, "mall": 8.49e12},
            peak_source=peak_source,
            dispatch_floor_s=3.0e-6,
        ),
        artifacts_dir=artifacts,
        observed_ms={"c0": 40.0},
        notes=("kernel trace unavailable: nothing",),
    )


_GOOD = json.dumps(
    {
        "cases": [{"case_id": "c0", "t_ideal_ms": 12.8, "bound": "memory"}],
        "confidence": "high",
        "analysis_md": "# Performance ceiling analysis\n\nCase `c0`: 8e10 bytes / 6.24 TB/s = 12.8 ms.",
    }
)


def _analyse(backend, tmp_path, **overrides):
    kwargs = {
        "canonical_id": "roofline-ceiling:op:gfx950",
        "workdir": str(tmp_path),
        "kernel_files": ["kernel.py"],
        "driver_script": "driver.py",
        "performance_command": ["bash", "-c", "python3 driver.py"],
        "case_ids": ["c0"],
        "case_params": {},
        "evidence": _bundle(tmp_path),
    }
    kwargs.update(overrides)
    return asyncio.run(run_ceiling_analysis(backend, **kwargs))


def test_the_role_document_ships_with_the_package():
    role = load_role()

    assert "Performance Ceiling Analyst" in role
    assert "You do **not** own the hardware figures" in role


def test_the_role_document_hands_the_composition_to_the_analyst():
    """It offers a default rule and tells the analyst when to leave it."""
    role = load_role()

    assert "You own the whole estimate" in role
    assert "Depart from the default" in role
    assert "occupancy" in role.lower()
    assert "partial overlap between stages" in role


def test_the_request_states_the_measured_figures_rather_than_leaving_them_to_recall(tmp_path):
    request = json.loads(
        build_request(
            kernel_files=["kernel.py"],
            driver_script="driver.py",
            performance_command=["bash", "-c", "run"],
            case_ids=["c0"],
            case_params={"tokens": 1},
            evidence=_bundle(tmp_path),
        )
    )

    hardware = request["hardware"]
    assert hardware["peak_flops_by_instruction_path"] == {"bf16_mfma": 1.686e15}
    assert hardware["dispatch_floor_s"] == 3.0e-6
    assert "units" in hardware
    assert request["scored_case_ids"] == ["c0"]
    assert "output_schema" in request


def test_the_request_offers_every_memory_level_that_was_measured(tmp_path):
    """Pinning the analyst to HBM mis-bounds a cache-resident working set."""
    request = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path),
        )
    )

    levels = request["hardware"]["bandwidth_bytes_per_s_by_memory_level"]
    assert levels == {"hbm": 6.24e12, "mall": 8.49e12}
    assert "was not measured on this box" in request["hardware"]["note"]


def test_the_request_says_whether_the_peaks_were_measured_or_read_off_a_datasheet(tmp_path):
    measured = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path, PEAK_SOURCE_EMPIRICAL),
        )
    )
    datasheet = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path, PEAK_SOURCE_DATASHEET),
        )
    )

    assert "roof-only" in measured["hardware"]["peak_source_meaning"]
    assert "not an achievable target" in datasheet["hardware"]["peak_source_meaning"]


def test_the_request_hands_over_the_evidence_it_collected(tmp_path):
    request = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path),
        )
    )

    assert "trace/kernel_stats.csv" in request["evidence_files"]
    assert request["observed_ms_under_profiler"] == {"c0": 40.0}
    assert "must not be used to back-solve" in request["observed_ms_meaning"]


def test_the_analyst_session_can_only_read(tmp_path):
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    policy = backend.specs[0].tool_policy
    assert backend.specs[0].writable is False
    assert (policy.read, policy.search) == (True, True)
    assert (policy.write, policy.shell) == (False, False)


def test_the_analyst_is_granted_the_evidence_directory(tmp_path):
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    assert str(tmp_path / "evidence") in backend.specs[0].additional_directories


def test_a_well_formed_answer_becomes_a_report(tmp_path):
    report = _analyse(_Backend(_GOOD), tmp_path)

    assert list(report.ideal_ms()) == ["c0"]
    assert report.confidence == "high"


def test_a_fenced_answer_is_still_read(tmp_path):
    report = _analyse(_Backend(f"Here it is:\n```json\n{_GOOD}\n```"), tmp_path)

    assert report.ideal_ms()["c0"] > 0


def test_a_malformed_answer_is_repaired_once(tmp_path):
    backend = _Backend("not json at all", _GOOD)

    report = _analyse(backend, tmp_path)

    assert len(backend.specs) == 2
    assert "validation_error" in backend.specs[1].user_prompt
    assert report.ideal_ms()["c0"] > 0


def test_a_second_malformed_answer_is_not_chased_further(tmp_path):
    backend = _Backend("nope", "still nope")

    with pytest.raises(CeilingAnalysisError, match="no valid ceiling model"):
        _analyse(backend, tmp_path)

    assert len(backend.specs) == 2


def test_an_answer_missing_a_scored_case_is_rejected_rather_than_published(tmp_path):
    with pytest.raises(CeilingAnalysisError, match="no ceiling for scored case"):
        _analyse(_Backend(_GOOD, _GOOD), tmp_path, case_ids=["c0", "c1"])


def test_a_silent_session_is_an_error_not_an_empty_ceiling(tmp_path):
    with pytest.raises(CeilingAnalysisError, match="no text"):
        _analyse(_Backend(""), tmp_path)


def test_the_role_document_is_the_system_prompt(tmp_path):
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    assert backend.specs[0].system_prompt == load_role()
    assert backend.specs[0].role == "ceiling analyst"


def test_the_shipped_role_lives_where_the_package_data_glob_reaches_it():
    from kernelforge.resources import packaged_data_root

    assert (packaged_data_root() / "roofline_ceiling" / "ceiling_analyst.md").is_file()
    assert Path(packaged_data_root()).name == "data"
