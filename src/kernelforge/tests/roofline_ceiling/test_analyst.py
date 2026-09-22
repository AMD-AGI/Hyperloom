# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""The analyst session: what it is asked, where it may write, and how it is read."""

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
from kernelforge.roofline_ceiling.device_profile import DeviceIdentity
from kernelforge.roofline_ceiling.evidence import EvidenceBundle
from kernelforge.roofline_ceiling.report import REPORT_FILENAME

_GOOD = {"cases": {"c0": 12.8}, "mean_ideal_ms": 12.8}


class _Backend:
    """A backend that writes canned files and records the specs it was given."""

    name = "fake"

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self.specs: list = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        if self._payloads:
            payload = self._payloads.pop(0)
            if payload is not None:
                target = Path(self._output_dir(spec)) / REPORT_FILENAME
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        return type("Result", (), {"text": "done", "end_reason": "agent_stopped"})()

    @staticmethod
    def _output_dir(spec) -> str:
        return spec.additional_directories[0]


def _bundle(tmp_path) -> EvidenceBundle:
    artifacts = tmp_path / "evidence"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "trace").mkdir(exist_ok=True)
    (artifacts / "trace" / "kernel_stats.csv").write_text("name,count\n", encoding="utf-8")
    return EvidenceBundle(
        identity=DeviceIdentity(
            arch="gfx950",
            device_name="AMD Instinct MI355X",
            compute_partition="SPX",
            memory_partition="NPS1",
        ),
        artifacts_dir=artifacts,
        observed_ms={"c0": 40.0},
        notes=("kernel trace unavailable: nothing",),
    )


def _analyse(backend, tmp_path, **overrides):
    kwargs = {
        "workdir": str(tmp_path),
        "output_dir": tmp_path / "out",
        "kernel_files": ["kernel.py"],
        "driver_script": "driver.py",
        "performance_command": ["bash", "-c", "python3 driver.py"],
        "case_ids": ["c0"],
        "case_params": {},
        "evidence": _bundle(tmp_path),
    }
    kwargs.update(overrides)
    return asyncio.run(run_ceiling_analysis(backend, **kwargs))


# --- the role document ---------------------------------------------------------


def test_the_role_document_ships_with_the_package():
    role = load_role()

    assert "Performance Ceiling Analyst" in role
    assert "Step 0 — establish this machine's roofs" in role


def test_the_role_document_names_both_files_as_the_deliverable():
    role = load_role()

    assert "performance_ceiling.json" in role
    assert "performance_ceiling_analysis.md" in role


def test_the_role_document_hands_the_composition_to_the_analyst():
    """It offers a default rule and tells the analyst when to leave it."""
    role = load_role()

    assert "You own the whole estimate" in role
    assert "Depart from the default" in role
    assert "occupancy" in role.lower()


# --- the request ---------------------------------------------------------------


def test_the_request_names_the_machine_rather_than_supplying_its_roofs(tmp_path):
    """The analyst measures the peaks itself; handing it any would pre-empt that."""
    request = json.loads(
        build_request(
            kernel_files=["kernel.py"],
            driver_script="driver.py",
            performance_command=["bash", "-c", "run"],
            case_ids=["c0"],
            case_params={"tokens": 1},
            evidence=_bundle(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert request["machine"]["arch"] == "gfx950"
    assert request["machine"]["compute_partition"] == "SPX"
    assert "peak_flops" not in request["machine"]
    assert request["scored_case_ids"] == ["c0"]


def test_the_request_spells_out_both_files_and_where_they_go(tmp_path):
    request = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert request["output_dir"] == str(tmp_path / "out")
    assert set(request["output_files"]) == {"performance_ceiling.json", "performance_ceiling_analysis.md"}
    assert "equal-weight" in request["output_files"]["performance_ceiling.json"]


def test_the_request_hands_over_the_evidence_it_collected(tmp_path):
    request = json.loads(
        build_request(
            kernel_files=[],
            driver_script="",
            performance_command=[],
            case_ids=["c0"],
            case_params={},
            evidence=_bundle(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert "trace/kernel_stats.csv" in request["evidence_files"]
    assert request["observed_ms"] == {"c0": 40.0}
    assert "back-solved" in request["observed_ms_meaning"]
    assert "none may exceed it" in request["observed_ms_meaning"]


# --- the session ---------------------------------------------------------------


def test_the_session_can_run_the_profiler_it_needs(tmp_path):
    """Measuring the roofs takes a shell, and installing the tool takes more."""
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    policy = backend.specs[0].tool_policy
    assert backend.specs[0].writable is True
    assert (policy.read, policy.search, policy.write, policy.shell) == (True, True, True, True)


def test_the_session_may_write_only_where_it_was_told(tmp_path):
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    assert backend.specs[0].additional_directories == [
        str(tmp_path / "out"),
        str(tmp_path / "evidence"),
    ]


def test_the_kernel_under_optimization_stays_out_of_reach(tmp_path):
    """Two lines: the hook refuses the edit, the guard restores anything a shell touched."""
    backend = _Backend(_GOOD)

    _analyse(backend, tmp_path)

    assert backend.specs[0].protected_globs == ["*"]
    assert backend.specs[0].hooks.pre_tool_use


def _deny_reason(hooks, tool_name: str, file_path: str):
    hook = hooks.pre_tool_use[0]
    verdict = asyncio.run(
        hook.callback({"tool_name": tool_name, "tool_input": {"file_path": file_path}}, None, None)
    )
    return (verdict.get("hookSpecificOutput") or {}).get("permissionDecisionReason")


def test_the_hook_refuses_an_edit_outside_the_output_directories(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    reason = _deny_reason(backend.specs[0].hooks, "Write", str(tmp_path / "kernel.py"))

    assert reason is not None
    assert "read-only" in reason


def test_the_hook_allows_the_files_the_analyst_is_asked_for(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    assert _deny_reason(backend.specs[0].hooks, "Write", str(tmp_path / "out" / REPORT_FILENAME)) is None
    assert _deny_reason(backend.specs[0].hooks, "Write", str(tmp_path / "evidence" / "roofs.csv")) is None


def test_the_hook_leaves_tools_that_do_not_write_alone(tmp_path):
    backend = _Backend(_GOOD)
    _analyse(backend, tmp_path)

    assert _deny_reason(backend.specs[0].hooks, "Read", str(tmp_path / "kernel.py")) is None


# --- reading the answer back ---------------------------------------------------


def test_a_file_the_analyst_wrote_becomes_the_report(tmp_path):
    report = _analyse(_Backend(_GOOD), tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}


def test_an_unreadable_file_is_handed_back_once_with_the_reason(tmp_path):
    backend = _Backend({"cases": {"c0": "fast"}}, _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
    assert len(backend.specs) == 2
    assert "finite positive number" in backend.specs[1].user_prompt


def test_a_session_that_never_writes_the_file_is_told_so(tmp_path):
    backend = _Backend(None, _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
    assert "not written" in backend.specs[1].user_prompt


def test_two_unreadable_attempts_end_the_run_rather_than_a_third(tmp_path):
    backend = _Backend({"cases": {}}, {"cases": {}})

    with pytest.raises(CeilingAnalysisError, match="no readable performance_ceiling.json"):
        _analyse(backend, tmp_path)

    assert len(backend.specs) == 2


def test_malformed_json_on_disk_is_a_repairable_failure(tmp_path):
    backend = _Backend("{not json", _GOOD)

    report = _analyse(backend, tmp_path)

    assert report.ideal_ms() == {"c0": 12.8}
