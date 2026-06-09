# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-opt dispatch shape / path validation.

A candidate may only dispatch with a non-empty trace-anchored shape and
existing source/workspace paths; ``dry_run`` and an escape flag bypass the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh


def _candidate(**over):
    base = {
        "kernel_id": "k001",
        "name": "rmsnorm",
        "shapes": [{"input_shape": "(8,4096) bf16"}],
        "shape_provenance": "torch_trace",
    }
    base.update(over)
    return base


def test_empty_shape_is_rejected(tmp_path: Path):
    payload = {"kernel_id": "k001", "candidate": _candidate(shapes=[])}
    res = krh._validate_kernel_shape_and_paths(payload, session_dir=tmp_path)
    assert res is not None
    assert res["error_class"] == "empty_kernel_shape"
    assert res["status"] == "failed"


def test_non_empty_shape_passes(tmp_path: Path):
    payload = {"kernel_id": "k001", "candidate": _candidate()}
    assert krh._validate_kernel_shape_and_paths(
        payload, session_dir=tmp_path,
    ) is None


def test_untrusted_provenance_is_rejected(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "candidate": _candidate(shape_provenance="config_derived"),
    }
    res = krh._validate_kernel_shape_and_paths(payload, session_dir=tmp_path)
    assert res is not None
    assert res["error_class"] == "untrusted_shape_provenance"


def test_missing_source_path_is_rejected(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "source_file": "/does/not/exist/kernel.py",
        "candidate": _candidate(),
    }
    res = krh._validate_kernel_shape_and_paths(payload, session_dir=tmp_path)
    assert res is not None
    assert res["error_class"] == "missing_source_path"


def test_existing_source_path_passes(tmp_path: Path):
    src = tmp_path / "kernel.py"
    src.write_text("# kernel\n", encoding="utf-8")
    payload = {
        "kernel_id": "k001",
        "source_file": str(src),
        "candidate": _candidate(),
    }
    assert krh._validate_kernel_shape_and_paths(
        payload, session_dir=tmp_path,
    ) is None


def test_workload_config_provenance_passes(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "candidate": _candidate(
            shape_provenance="workload_config",
            shapes=["prefill token-count M up to >=4096"],
        ),
    }
    assert krh._validate_kernel_shape_and_paths(
        payload, session_dir=tmp_path,
    ) is None
    payload = {
        "kernel_id": "k001",
        "allow_empty_kernel_shape": True,
        "candidate": _candidate(shapes=[]),
    }
    assert krh._validate_kernel_shape_and_paths(
        payload, session_dir=tmp_path,
    ) is None


def test_escape_env_allows_empty_shape(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", "1")
    payload = {"kernel_id": "k001", "candidate": _candidate(shapes=[])}
    assert krh._validate_kernel_shape_and_paths(
        payload, session_dir=tmp_path,
    ) is None


def test_dry_run_bypasses_validation(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "dry_run": True,
        "source_file": "/does/not/exist/kernel.py",
        "candidate": _candidate(shapes=[]),
    }
    assert krh._validate_kernel_shape_and_paths(
        payload, session_dir=tmp_path,
    ) is None


@pytest.mark.asyncio
async def test_run_optimization_single_rejects_empty_shape(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "source_file": "/sgl-workspace/sglang/kernels/x.py",
        "candidate": {
            "kernel_id": "k001",
            "name": "rmsnorm",
            "reusable_native_kernel": True,
            "source_file": "/sgl-workspace/sglang/kernels/x.py",
            "shapes": [],
        },
        "_single_kernel": True,
    }
    res = await krh._run_optimization_single(payload, session_dir=tmp_path)
    assert res["status"] == "failed"
    assert res["error_class"] == "empty_kernel_shape"


# shape_provenance stamping in tracelens finalization
def test_finalize_candidates_stamps_trace_provenance():
    import importlib.util
    import sys

    tools_dir = Path("kernel-agent/tools").resolve()
    sys.path.insert(0, str(tools_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "tracelens_analysis_for_test",
            str(tools_dir / "tracelens_analysis.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(tools_dir))

    rows = [
        {"name": "with_shape", "duration_us": 10.0, "shapes": [{"a": 1}]},
        {"name": "no_shape", "duration_us": 5.0, "shapes": []},
    ]
    out = mod._finalize_candidates(rows)
    by_name = {r["name"]: r for r in out}
    assert by_name["with_shape"]["shape_provenance"] == "torch_trace"
    assert "shape_provenance" not in by_name["no_shape"]
