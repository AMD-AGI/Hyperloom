# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the YAML-layer runtime override.

Verifies that runtime_override lands in materialized YAML benchmark.envs
and that os.environ is never mutated.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path


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


# ---------------------------------------------------------------------------
# compiled-artifact runtime prefixes
# ---------------------------------------------------------------------------

def test_pythonpath_prefixes_multi_entry_ordered_prepended():
    envs = {"PYTHONPATH": "/opt/venv"}
    apply_runtime_override(envs, {"pythonpath_prefixes": ["/a/pkg", "/b/pkg"]})
    assert envs["PYTHONPATH"] == "/a/pkg:/b/pkg:/opt/venv"


def test_pythonpath_prefixes_drops_unsafe_entry():
    envs = {}
    apply_runtime_override(envs, {"pythonpath_prefixes": ["/good", "../evil"]})
    assert envs.get("PYTHONPATH") == "/good"


def test_ld_library_path_prefix_prepends_and_preserves_rocm():
    envs = {"LD_LIBRARY_PATH": "/opt/rocm/lib"}
    apply_runtime_override(envs, {"ld_library_path_prefix": ["/attempt/lib"]})
    assert envs["LD_LIBRARY_PATH"] == "/attempt/lib:/opt/rocm/lib"


def test_ld_library_path_prefix_multi_entry_order():
    envs = {}
    apply_runtime_override(envs, {"ld_library_path_prefix": ["/a", "/b"]})
    assert envs["LD_LIBRARY_PATH"] == "/a:/b"


def test_runtime_env_merged():
    envs = {}
    apply_runtime_override(envs, {"runtime_env": {"INFERENCE_OPTIMIZER_AITER_JIT_DIR": "/j", "AITER_REBUILD": "1"}})
    assert envs["INFERENCE_OPTIMIZER_AITER_JIT_DIR"] == "/j"
    assert envs["AITER_REBUILD"] == "1"


def test_runtime_env_rejects_reserved_and_blocked_keys():
    envs = {}
    apply_runtime_override(envs, {"runtime_env": {"PATH": "/evil", "LD_PRELOAD": "/x", "1BAD": "y", "OK_KEY": "v"}})
    assert envs == {"OK_KEY": "v"}


def test_entrypoint_bin_dir_prepended_to_path():
    envs = {"PATH": "/usr/bin"}
    apply_runtime_override(envs, {"entrypoint_bin_dir": "/attempt/bin"})
    assert envs["PATH"] == "/attempt/bin:/usr/bin"


def test_entrypoint_bin_dir_unsafe_dropped():
    envs = {"PATH": "/usr/bin"}
    apply_runtime_override(envs, {"entrypoint_bin_dir": "/a:/b"})
    assert envs["PATH"] == "/usr/bin"


def test_runtime_python_exe_writes_hyperloom_framework_python():
    envs: dict[str, str] = {}
    apply_runtime_override(envs, {"runtime_python_exe": "/venv/bin/python3.11"})
    assert envs["HYPERLOOM_FRAMEWORK_PYTHON"] == "/venv/bin/python3.11"


def test_runtime_python_exe_overrides_framework_python():
    """runtime_python_exe must win when both keys are present."""
    envs: dict[str, str] = {}
    apply_runtime_override(envs, {
        "framework_python": "/old/bin/python",
        "runtime_python_exe": "/venv/bin/python3.11",
    })
    assert envs["HYPERLOOM_FRAMEWORK_PYTHON"] == "/venv/bin/python3.11"


def test_extended_framework_runtime_lands_in_yaml(tmp_path):
    """An extended FrameworkRuntime.to_runtime_override lands end-to-end in YAML."""
    from hyperloom.orchestrator.framework.stack_actions import FrameworkRuntime

    base_yaml = _base_yaml(tmp_path)
    (tmp_path / "pkg").mkdir()
    rt = FrameworkRuntime(
        bin_path="/attempt/bin",
        pythonpath_prefixes=(str(tmp_path / "pkg"),),
        ld_library_path_prefix=("/attempt/lib",),
        runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(tmp_path / "jit")},
        entrypoint_bin_dir="/attempt/console",
    )
    variant = GridVariant(name="rt5")
    variant.runtime_override = rt.to_runtime_override()

    out_yaml = _build_variant_yaml(
        base_yaml_path=base_yaml,
        base_extra_args="",
        variant=variant,
        output_subdir=tmp_path / "out5",
    )
    with out_yaml.open(encoding="utf-8") as f:
        envs = yaml.safe_load(f)["benchmark"]["envs"]
    assert str(tmp_path / "pkg") in envs["PYTHONPATH"]
    assert "/attempt/lib" in envs["LD_LIBRARY_PATH"]
    assert envs["INFERENCE_OPTIMIZER_AITER_JIT_DIR"] == str(tmp_path / "jit")
    assert "/attempt/console" in envs["PATH"]


# ---------------------------------------------------------------------------
# Fingerprint: runtime_override participates but stays back-compatible
# ---------------------------------------------------------------------------

def test_fingerprint_unchanged_for_empty_override():
    plain = GridVariant(name="a", extra_server_args="--x 1")
    with_empty = GridVariant(name="b", extra_server_args="--x 1")
    with_empty.runtime_override = {}
    assert plain.fingerprint == with_empty.fingerprint


def test_fingerprint_changes_with_runtime_override():
    base = GridVariant(name="a", extra_server_args="--x 1")
    overridden = GridVariant(name="a", extra_server_args="--x 1")
    overridden.runtime_override = {"pythonpath_prefixes": ["/a/pkg"]}
    assert base.fingerprint != overridden.fingerprint


def test_fingerprint_order_independent_for_override():
    v1 = GridVariant(name="a")
    v1.runtime_override = {"pythonpath_prefixes": ["/a", "/b"], "runtime_env": {"X": "1", "Y": "2"}}
    v2 = GridVariant(name="a")
    v2.runtime_override = {"runtime_env": {"Y": "2", "X": "1"}, "pythonpath_prefixes": ["/b", "/a"]}
    assert v1.fingerprint == v2.fingerprint
