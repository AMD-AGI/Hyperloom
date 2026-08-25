# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-opt dispatch shape-provenance / path validation.

A shapeless candidate dispatches: the backend's driver preparation recovers the
operand dims the trace never recorded. Provenance is validated only for a shape
that is present, and source/workspace paths must exist; ``dry_run`` bypasses all
of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh


def _candidate(**over):
    base = {
        "kernel_id": "k001",
        "name": "rmsnorm",
        "shapes": [{"input_shape": "(8,4096) bf16"}],
        "shape_provenance": "torch_trace",
    }
    base.update(over)
    return base


def test_empty_shape_dispatches(tmp_path: Path):
    # A graph-launched kernel records no cpu_op parent, so the trace carries no
    # argument dims for it. Refusing the dispatch does not produce a measured
    # shape, it only makes the hottest kernels of a captured model permanently
    # unoptimizable.
    payload = {"kernel_id": "k001", "candidate": _candidate(shapes=[])}
    assert (
        krh._validate_kernel_shape_and_paths(
            payload,
            session_dir=tmp_path,
        )
        is None
    )


@pytest.mark.parametrize("malformed", [{"input_shape": "(8,4096)"}, "(8,4096) bf16", 42])
def test_a_malformed_shape_is_refused_not_read_as_absent(tmp_path: Path, malformed):
    """Absent dims are dispatchable; a broken producer is not.

    All three of these are truthy, so a bare truthiness test admitted them and
    the type error surfaced in driver preparation -- charged to this kernel's
    retry quota and reported as an optimization failure. Reading them as
    shapeless would be worse than refusing: an empty ``shapes`` is evidence the
    trace could not record and the backend recovers it from source, so a
    producer that broke would be indistinguishable from a graph-launched kernel
    and stay broken.
    """
    payload = {"kernel_id": "k001", "candidate": _candidate(shapes=malformed)}
    out = krh._validate_kernel_shape_and_paths(payload, session_dir=tmp_path)
    assert out is not None
    assert out["error_class"] == "malformed_kernel_shapes"


@pytest.mark.parametrize("provenance", ["unresolved", "launch_grid", "tile_name"])
def test_empty_shape_dispatches_whatever_provenance_says(
    tmp_path: Path,
    provenance: str,
):
    # On a shapeless row the marker names why the dims are absent, so it must
    # not be read as an untrusted operand dim -- that would close the removed
    # empty-shape gate from the provenance side.
    payload = {
        "kernel_id": "k001",
        "candidate": _candidate(shapes=[], shape_provenance=provenance),
    }
    assert (
        krh._validate_kernel_shape_and_paths(
            payload,
            session_dir=tmp_path,
        )
        is None
    )


def test_non_empty_shape_passes(tmp_path: Path):
    payload = {"kernel_id": "k001", "candidate": _candidate()}
    assert (
        krh._validate_kernel_shape_and_paths(
            payload,
            session_dir=tmp_path,
        )
        is None
    )


def test_untrusted_provenance_is_rejected(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "candidate": _candidate(shape_provenance="config_derived"),
    }
    res = krh._validate_kernel_shape_and_paths(payload, session_dir=tmp_path)
    assert res is not None
    assert res["error_class"] == "untrusted_shape_provenance"


def test_capture_backfill_provenance_passes(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "candidate": _candidate(shape_provenance="capture_backfill"),
    }
    assert (
        krh._validate_kernel_shape_and_paths(
            payload,
            session_dir=tmp_path,
        )
        is None
    )


@pytest.mark.parametrize("provenance", ["launch_grid", "tile_name"])
def test_geometry_provenance_is_rejected(tmp_path: Path, provenance: str):
    # launch_grid / tile_name are coarse geometry, not operand dims: the gate
    # must reject them even though the candidate carries a non-empty shape.
    payload = {
        "kernel_id": "k001",
        "candidate": _candidate(shape_provenance=provenance),
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
    assert (
        krh._validate_kernel_shape_and_paths(
            payload,
            session_dir=tmp_path,
        )
        is None
    )


def test_dry_run_bypasses_validation(tmp_path: Path):
    payload = {
        "kernel_id": "k001",
        "dry_run": True,
        "source_file": "/does/not/exist/kernel.py",
        "candidate": _candidate(shapes=[]),
    }
    assert (
        krh._validate_kernel_shape_and_paths(
            payload,
            session_dir=tmp_path,
        )
        is None
    )


@pytest.mark.asyncio
async def test_run_optimization_single_passes_the_shape_gate(tmp_path: Path):
    # The shapeless candidate reaches the path check instead of being turned
    # away for its missing dims, so the failure reported is the one the payload
    # actually has.
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
    assert res["error_class"] == "missing_source_path"


def test_finalize_candidates_stamps_trace_provenance():
    import importlib.util
    import sys

    tools_dir = Path("src/hyperloom/agents/kernel/tools").resolve()
    sys.path.insert(0, str(tools_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "tracelens_analysis_for_test",
            str(tools_dir / "tracelens_analysis.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so self-referential dataclass annotations resolve under py3.10.
        sys.modules[spec.name] = mod
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
