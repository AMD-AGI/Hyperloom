# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a custom single-file round adds to the scriptable workload the suite runs.

The boundary is enforced before any budget is spent, the export counters are armed and
delivered together, and an AOTI profiling leg asks Inductor for per-launch scopes. A
round of any other scriptable framework is left exactly as the suite built it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._suite.custom_workload import (
    PROFILE_TRACE_DIR_ENV,
    custom_workload,
    enforce_custom_scope,
)
from hyperloom.orchestrator.measurement.aoti_evidence import EVIDENCE_DIR_ENV, evidence_dir
from inference_suite.contracts.config import ScriptableWorkload
from inference_suite.contracts.layout import RunLayout

_KERNEL_PROFILE_ENV = "TORCHINDUCTOR_CPP_ENABLE_KERNEL_PROFILE"


def _declare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **scope: object) -> Path:
    """Write an optimization-scope declaration the hooks will read, and point them at it."""
    model = tmp_path / "model.py"
    model.write_text("def hyperloom_optimize(m):\n    return m\n", encoding="utf-8")
    declaration = {"schema_version": 2, "file": str(model), **scope}
    (tmp_path / "optimization_scope.json").write_text(json.dumps(declaration), encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    return model


def _aoti(enabled: bool = True) -> dict[str, object]:
    return {"aoti": {"enabled": enabled}}


def test_another_scriptable_framework_is_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, **_aoti())
    workload = ScriptableWorkload(params={"XDIT_NUM_STEPS": 28})
    assert custom_workload(workload, "xdit", tmp_path, profile=True) == workload


def test_export_counters_are_armed_and_delivered_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, **_aoti())
    run_dir = tmp_path / "run_0"
    result = custom_workload(ScriptableWorkload(), "custom", run_dir, profile=False)
    assert result.params[EVIDENCE_DIR_ENV] == str(evidence_dir(RunLayout(run_dir).root))
    # Armed without delivered leaves the gate shut for the whole campaign: the shim only
    # runs if Python imports the bundled sitecustomize.
    assert "sitecustomize.py" in {
        path.name for path in Path(str(result.params["PYTHONPATH"]).split(os.pathsep)[0]).iterdir()
    }


def test_delivery_keeps_the_import_paths_the_point_already_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, **_aoti())
    result = custom_workload(
        ScriptableWorkload(params={"PYTHONPATH": "/operator/site"}), "custom", tmp_path, profile=False
    )
    assert str(result.params["PYTHONPATH"]).split(os.pathsep)[-1] == "/operator/site"


def test_export_not_declared_arms_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, **_aoti(enabled=False))
    result = custom_workload(ScriptableWorkload(), "custom", tmp_path, profile=True)
    assert EVIDENCE_DIR_ENV not in result.params
    assert _KERNEL_PROFILE_ENV not in result.params


def test_kernel_profile_is_asked_for_on_profiling_legs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, **_aoti())
    assert (
        custom_workload(ScriptableWorkload(), "custom", tmp_path, profile=False).params.get(_KERNEL_PROFILE_ENV) is None
    )
    assert custom_workload(ScriptableWorkload(), "custom", tmp_path, profile=True).params[_KERNEL_PROFILE_ENV] == "1"


def test_a_profiling_leg_is_told_where_its_trace_goes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite names the two framework spellings itself; a custom harness reads neither."""
    _declare(tmp_path, monkeypatch)
    run_dir = tmp_path / "run_0"
    result = custom_workload(ScriptableWorkload(), "custom", run_dir, profile=True)
    assert result.params[PROFILE_TRACE_DIR_ENV] == str(RunLayout(run_dir).profile_torch_trace)


def test_an_unreadable_scope_costs_the_gate_its_evidence_but_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "optimization_scope.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    result = custom_workload(ScriptableWorkload(), "custom", tmp_path, profile=True)
    assert EVIDENCE_DIR_ENV not in result.params
    assert _KERNEL_PROFILE_ENV not in result.params


def test_scope_is_enforced_against_the_exported_repo_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _declare(tmp_path, monkeypatch)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(model.parent))
    enforce_custom_scope("custom")
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="permits changes only to"):
        enforce_custom_scope("custom")


def test_a_malformed_export_block_fails_the_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing else reports it: every other caller of aoti_config swallows the error."""
    model = _declare(tmp_path, monkeypatch, aoti={"enabled": "yes"})
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(model.parent))
    with pytest.raises(ValueError, match="aoti.enabled must be a boolean"):
        enforce_custom_scope("custom")


def test_no_declaration_enforces_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    enforce_custom_scope("custom")
