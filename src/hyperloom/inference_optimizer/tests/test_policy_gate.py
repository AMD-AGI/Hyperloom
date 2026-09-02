# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for policy gate.py pure helpers + PolicyGate path/freeform helpers:
presence checks, GPU-count probing, lane ceilings, path allowlists, and the
free-form task-description guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.policy import gate as pol
from hyperloom.orchestrator.policy.gate import (
    PolicyDenied,
    PolicyGate,
    _delegate_field_present,
    _value_is_present,
    detect_gpu_count,
    gpu_specialist_ceiling,
    research_lane_ceiling,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry


# -- _value_is_present -----------------------------------------------------
def test_value_is_present() -> None:
    assert _value_is_present(None) is False
    assert _value_is_present("") is False
    assert _value_is_present("   ") is False
    assert _value_is_present("x") is True
    assert _value_is_present([]) is False
    assert _value_is_present([1]) is True
    assert _value_is_present({}) is False
    assert _value_is_present({"a": 1}) is True
    assert _value_is_present(0) is True


def test_delegate_field_present() -> None:
    assert _delegate_field_present({"reason": "r"}, "reason") is True
    assert _delegate_field_present({"params": {"reason": "r"}}, "reason") is True
    assert _delegate_field_present({"params": {}}, "reason") is False
    assert _delegate_field_present({}, "reason") is False


# -- detect_gpu_count ------------------------------------------------------
def test_detect_gpu_count_env_mask(monkeypatch) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2")
    assert detect_gpu_count() == 3


def test_detect_gpu_count_rocr_mask_wins(monkeypatch) -> None:
    # ROCR is canonical on ROCm, checked before HIP/CUDA.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    assert detect_gpu_count() == 4


def test_detect_gpu_count_empty_mask_returns_zero(monkeypatch) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    assert detect_gpu_count() == 0


def test_detect_gpu_count_rocm_smi_fallback(monkeypatch) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class _CP:
        returncode = 0
        stdout = "GPU[0]\t: foo\nGPU[1]\t: bar\nother line\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _CP())
    assert detect_gpu_count() == 2


def test_detect_gpu_count_rocm_smi_missing(monkeypatch) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert detect_gpu_count() == 0


# -- research_lane_ceiling / gpu_specialist_ceiling -----------------------
def test_research_lane_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(pol, "detect_gpu_count", lambda: 4)
    assert research_lane_ceiling() == 8
    monkeypatch.setattr(pol, "detect_gpu_count", lambda: 0)
    assert research_lane_ceiling() == pol.RESEARCH_LANE_CEILING_FALLBACK


def test_gpu_specialist_ceiling_shared_state() -> None:
    class _SS:
        gpu_specialist_capacity = 3

    assert gpu_specialist_ceiling(_SS()) == 3


def test_gpu_specialist_ceiling_shared_state_invalid() -> None:
    class _SS:
        gpu_specialist_capacity = "bad"

    assert gpu_specialist_ceiling(_SS()) == 0


def test_gpu_specialist_ceiling_env(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "5")
    assert gpu_specialist_ceiling(None) == 5
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "bad")
    assert gpu_specialist_ceiling(None) == 0


# -- PolicyGate path helpers ----------------------------------------------
def _gate(session_dir: Path | None = None) -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry(), session_dir=session_dir)


def test_path_under_session_disabled_when_no_session_dir() -> None:
    assert _gate(None)._path_under_session("/anything") is True


def test_path_under_session_inside_and_escape(tmp_path: Path) -> None:
    g = _gate(tmp_path)
    assert g._path_under_session(str(tmp_path / "a" / "b.txt")) is True
    assert g._path_under_session(str(tmp_path)) is True
    assert g._path_under_session("/etc/passwd") is False


def test_path_in_source_allowlist(monkeypatch) -> None:
    g = _gate(None)
    monkeypatch.setattr(pol, "resolve_source_file_allowlist", lambda: ("/srv/sglang/",))
    assert g._path_in_source_allowlist("/srv/sglang/foo.py") is True
    assert g._path_in_source_allowlist("/srv/sglang/sub/foo.py") is True
    assert g._path_in_source_allowlist("/other/foo.py") is False
    # Traversal and shared-prefix boundary must NOT slip past.
    assert g._path_in_source_allowlist("/srv/sglang/../etc/passwd") is False
    assert g._path_in_source_allowlist("/srv/sglangX/foo.py") is False


def test_path_in_trace_allowlist(monkeypatch) -> None:
    g = _gate(None)
    monkeypatch.setattr(pol, "_trace_path_allowlist", lambda: ("/shared/profile/",))
    assert g._path_in_trace_allowlist("/shared/profile/run.json.gz") is True
    assert g._path_in_trace_allowlist("/elsewhere/run.json.gz") is False
    assert g._path_in_trace_allowlist("/shared/profile/../secret") is False
    assert g._path_in_trace_allowlist("/shared/profileX/run.json.gz") is False


# -- _check_freeform_task_description -------------------------------------
def test_freeform_description_empty() -> None:
    with pytest.raises(PolicyDenied) as exc:
        PolicyGate._check_freeform_task_description("", where="task[0]")
    assert exc.value.rule == "specialist_freeform_empty_description"


def test_freeform_description_too_long() -> None:
    big = "x" * (pol.SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS + 1)
    with pytest.raises(PolicyDenied) as exc:
        PolicyGate._check_freeform_task_description(big, where="task[0]")
    assert exc.value.rule == "specialist_freeform_description_too_long"


def test_freeform_description_valid() -> None:
    PolicyGate._check_freeform_task_description(
        "Investigate the MoE kernel launch overhead and propose tuning.",
        where="task[0]",
    )


def test_freeform_description_destructive_text_allowed() -> None:
    PolicyGate._check_freeform_task_description(
        "killall -9 python",
        where="task[0]",
    )


# -- Coordinator-internal denial message ----------------------------------


def test_phase_semantics_prompt_names_every_internal_action() -> None:
    """The orchestration prompt must name the actions PolicyGate will deny.

    Telling the model that ``framework`` is Coordinator-managed while the
    runtime denies ``framework_agent`` invites a proposal that costs a tick
    and gets rejected as phase_incompatible.
    """
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        COORDINATOR_INTERNAL_ACTIONS,
    )
    from hyperloom.orchestrator.prompts.prompt_builder import _section_phase_semantics

    rendered = "\n".join(_section_phase_semantics(kernel_enabled=True))

    missing = sorted(a for a in COORDINATOR_INTERNAL_ACTIONS if a not in rendered)
    assert not missing, f"Coordinator-internal actions absent from the prompt: {missing}"
