# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cross-layer contract: a bypass candidate's trace shapes must be consumable by
BOTH the orchestrator kernel-opt gate AND the GEAK harness (no crash, not the
synthetic default sweep).

This guards the bypass<->downstream shape-format contract that lets bypass
replace TraceLens end-to-end: bypass emits ``input_shapes``/``shapes`` as
``[{"call_num", "shape": "(dims) dtype<br>..."}]`` — the same form the harness
(``_build_configs`` / ``_parse_shape_string``) and TraceLens candidates use. A
regression to raw Kineto dims (``[[m, k], ...]``) crashes ``_build_configs`` with
``AttributeError: 'list' object has no attribute 'get'``.

Source-file resolution (op_to_source / trace kernel_file, incl. the /tmp
inductor exclusion) is orthogonal and covered by test_bypass_source_resolver.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "kernel-agent" / "tools"
sys.path.insert(0, str(_TOOLS))

import _bypass_report as report  # noqa: E402
import harness_generator as hg  # noqa: E402

from inference_optimizer.orchestrator import kernel_request_handlers as krh  # noqa: E402


def _analyze(kernels):
    total = sum(k["gpu_time_us"] for k in kernels) or 1.0
    for k in kernels:
        k.setdefault("gpu_pct", round(k["gpu_time_us"] / total * 100.0, 4))
        k.setdefault("count", 1)
        k.setdefault("op_name", "")
    return {"status": "ok", "aggregation_scope": "full_trace", "kernels": kernels, "ops": []}


def _candidate_with_shapes() -> dict:
    """A bypass candidate carrying real trace shapes (rms_norm: 2D + 1D operand)."""
    kernels = [{
        "name": "triton_rms_norm", "op_name": "aten::rms_norm",
        "gpu_time_us": 100.0, "count": 8,
        "op_shapes": [[4096, 2560], [2560]], "op_dtypes": ["c10::BFloat16", "c10::BFloat16"],
    }]
    cands = report.build_candidates(_analyze(kernels), framework="vllm", target_platform="MI300X")
    return cands["hot_kernels"][0]


def test_bypass_candidate_shapes_are_contract_shaped():
    cand = _candidate_with_shapes()
    # Contract form: list of {call_num, shape:"(dims) dtype<br>..."} dicts — NOT
    # raw Kineto dim lists (those crash the harness).
    for field in ("shapes", "input_shapes"):
        assert isinstance(cand[field], list) and cand[field]
        entry = cand[field][0]
        assert isinstance(entry, dict) and "shape" in entry and "call_num" in entry
    assert cand["input_shapes"][0]["shape"] == "(4096,2560) bf16<br>(2560,) bf16"
    assert cand["shape_provenance"] == "torch_trace"


def test_bypass_candidate_feeds_real_shapes_to_geak_harness():
    # _build_configs must parse the real trace shapes (no AttributeError) and NOT
    # fall back to the synthetic default sweep, so GEAK benchmarks the real dims.
    cand = _candidate_with_shapes()
    built = hg._build_configs(cand)
    assert built != hg._default_configs()
    assert "4096" in built[0] or "2560" in built[0]


def test_bypass_candidate_passes_orchestrator_gate(tmp_path: Path):
    # The kernel-opt gate accepts a bypass candidate with contract shapes,
    # trusted provenance, and an existing source. (Source resolution itself is
    # tested elsewhere; here we inject a real path to isolate the shape gate.)
    cand = _candidate_with_shapes()
    src = tmp_path / "rmsnorm_kernel.py"
    src.write_text("# editable kernel\n", encoding="utf-8")
    cand["source_file"] = str(src)
    cand["reusable_native_kernel"] = True
    payload = {"candidate": cand, "kernel_id": cand["kernel_id"], "workspace_path": str(tmp_path)}
    assert krh._validate_kernel_shape_and_paths(payload, session_dir=tmp_path) is None


def test_raw_kineto_dims_would_crash_harness_regression_guard():
    # Documents WHY the contract format matters: the pre-fix raw-dims form
    # (candidate["input_shapes"] = [[[m, k], ...]]) crashes _build_configs.
    import pytest

    raw = {"input_shapes": [[[4096, 2560], [2560]]]}
    with pytest.raises(AttributeError):
        hg._build_configs(raw)
