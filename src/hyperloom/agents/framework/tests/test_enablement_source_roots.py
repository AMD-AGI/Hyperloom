# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for §6.2: source-root/version injection into the enablement mandate."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hyperloom.agents.framework.enablement import FailureSignature, MISSING_MODEL_ARCH
from hyperloom.agents.framework.enablement_ops import (
    _FRAMEWORK_ROOT_HINT,
    _ROCM_HIP_ROOT_HINT,
    _resolve_actual_root_hints,
    build_mandate,
)
from hyperloom.agents.framework.enablement import EnablementRequest


def _req(framework: str = "vllm") -> EnablementRequest:
    return EnablementRequest(
        framework=framework,
        model="deepseek-v4",
        repo_url="",
        launch_log="model arch not supported",
    )


def _sig() -> FailureSignature:
    return FailureSignature(kind=MISSING_MODEL_ARCH, confidence=0.9)


# ---------------------------------------------------------------------------
# _resolve_actual_root_hints
# ---------------------------------------------------------------------------

def test_returns_real_roots_when_probe_finds_something():
    with patch(
        "hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env",
        return_value="/sgl-workspace/vllm:/opt/rocm",
    ), patch(
        "hyperloom.orchestrator.framework.paths.summarise_framework_root_discovery",
        return_value="vllm=ok",
    ):
        hints = _resolve_actual_root_hints("vllm")
    assert any("/sgl-workspace/vllm" in h for h in hints)
    assert any("vllm=ok" in h for h in hints)


def test_falls_back_to_generic_when_probe_empty():
    with patch(
        "hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env",
        return_value="",
    ):
        hints = _resolve_actual_root_hints("vllm")
    assert _FRAMEWORK_ROOT_HINT in hints
    assert _ROCM_HIP_ROOT_HINT in hints


def test_falls_back_on_probe_exception():
    with patch(
        "hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env",
        side_effect=RuntimeError("no roots"),
    ):
        hints = _resolve_actual_root_hints("vllm")
    assert _FRAMEWORK_ROOT_HINT in hints


def test_version_appended_when_package_installed():
    with patch(
        "hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env",
        return_value="/sgl-workspace/vllm",
    ), patch(
        "hyperloom.orchestrator.framework.paths.summarise_framework_root_discovery",
        return_value="vllm=ok",
    ), patch(
        "hyperloom.agents.framework.enablement_ops._resolve_package_version",
        return_value="0.9.1+rocm",
    ):
        hints = _resolve_actual_root_hints("vllm")
    assert any("0.9.1+rocm" in h for h in hints)


# ---------------------------------------------------------------------------
# build_mandate root_hints propagation
# ---------------------------------------------------------------------------

def test_build_mandate_uses_resolved_roots_in_task_description():
    with patch(
        "hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env",
        return_value="/sgl-workspace/vllm:/opt/rocm",
    ), patch(
        "hyperloom.orchestrator.framework.paths.summarise_framework_root_discovery",
        return_value="vllm=ok",
    ):
        mandate = build_mandate(_req(), signature=_sig())
    assert "/sgl-workspace/vllm" in mandate.task_description
    assert any("/sgl-workspace/vllm" in h for h in mandate.allowed_root_hints)


def test_build_mandate_explicit_root_hints_override_discovery():
    """Caller-supplied root_hints bypass _resolve_actual_root_hints."""
    mandate = build_mandate(
        _req(),
        signature=_sig(),
        root_hints=["/custom/root"],
    )
    assert "/custom/root" in mandate.allowed_root_hints
    assert "/custom/root" in mandate.task_description


def test_build_mandate_falls_back_gracefully_when_no_roots():
    with patch(
        "hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env",
        return_value="",
    ):
        mandate = build_mandate(_req(), signature=_sig())
    assert _FRAMEWORK_ROOT_HINT in mandate.allowed_root_hints
    assert _FRAMEWORK_ROOT_HINT in mandate.task_description
