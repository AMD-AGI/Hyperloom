# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the YAML-layer runtime override (§6.3).

Verifies that runtime_override lands in materialized YAML benchmark.envs
and that os.environ is never mutated.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    _build_variant_yaml,
    apply_runtime_override,
)


def _base_yaml(tmp_path: Path, framework: str = "vllm") -> Path:
    cfg = {"benchmark": {"framework": framework, "model": "/model", "envs": {}}}
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Unit tests for apply_runtime_override
# ---------------------------------------------------------------------------

def test_path_prefix_prepended():
    envs: dict[str, str] = {}
    apply_runtime_override(envs, {"path_prefix": "/attempt/bin"})
    assert envs["PATH"].startswith("/attempt/bin")


def test_path_prefix_not_duplicated():
    envs = {"PATH": "/attempt/bin:/usr/bin"}
    apply_runtime_override(envs, {"path_prefix": "/attempt/bin"})
    assert envs["PATH"].count("/attempt/bin") == 1


def test_pythonpath_prefix_prepended(tmp_path):
    envs: dict[str, str] = {}
    apply_runtime_override(envs, {"pythonpath_prefix": str(tmp_path)})
    assert str(tmp_path) in envs.get("PYTHONPATH", "")


def test_pythonpath_unsafe_colon_dropped():
    envs: dict[str, str] = {}
    apply_runtime_override(envs, {"pythonpath_prefix": "/a:/b"})
    assert "PYTHONPATH" not in envs


def test_framework_keys_written():
    envs: dict[str, str] = {}
    apply_runtime_override(envs, {
        "framework_bin": "/attempt/bin/vllm",
        "framework_python": "/attempt/bin/python",
        "framework_venv_root": "/attempt/venv",
    })
    assert envs["HYPERLOOM_FRAMEWORK_BIN"] == "/attempt/bin/vllm"
    assert envs["HYPERLOOM_FRAMEWORK_PYTHON"] == "/attempt/bin/python"
    assert envs["HYPERLOOM_FRAMEWORK_VENV_ROOT"] == "/attempt/venv"


def test_empty_override_is_noop():
    envs: dict[str, str] = {"FOO": "bar"}
    apply_runtime_override(envs, {})
    assert envs == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# Integration test: override lands in materialized YAML, NOT os.environ
# ---------------------------------------------------------------------------

def test_runtime_override_in_yaml_not_process_env(tmp_path):
    """The attempt framework_bin must appear in materialized YAML benchmark.envs,
    and os.environ must be unchanged."""
    base_yaml = _base_yaml(tmp_path)
    variant = GridVariant(name="test-rt")
    variant.runtime_override = {
        "path_prefix": "/attempt/bin",
        "framework_bin": "/attempt/bin/vllm",
        "framework_python": "/attempt/bin/python",
    }

    env_before = dict(os.environ)

    out_yaml = _build_variant_yaml(
        base_yaml_path=base_yaml,
        base_extra_args="",
        variant=variant,
        output_subdir=tmp_path / "out",
    )

    # os.environ must be unchanged
    assert dict(os.environ) == env_before, "runtime_override mutated os.environ"

    with out_yaml.open(encoding="utf-8") as f:
        materialized = yaml.safe_load(f)

    envs = materialized["benchmark"]["envs"]
    assert envs.get("HYPERLOOM_FRAMEWORK_BIN") == "/attempt/bin/vllm"
    assert envs.get("HYPERLOOM_FRAMEWORK_PYTHON") == "/attempt/bin/python"
    assert "/attempt/bin" in envs.get("PATH", "")


def test_no_runtime_override_field_is_noop(tmp_path):
    """A variant with no runtime_override attribute must not raise."""
    base_yaml = _base_yaml(tmp_path)
    variant = GridVariant(name="plain")
    out_yaml = _build_variant_yaml(
        base_yaml_path=base_yaml,
        base_extra_args="",
        variant=variant,
        output_subdir=tmp_path / "out2",
    )
    with out_yaml.open(encoding="utf-8") as f:
        materialized = yaml.safe_load(f)
    envs = materialized["benchmark"]["envs"]
    assert "HYPERLOOM_FRAMEWORK_BIN" not in envs
