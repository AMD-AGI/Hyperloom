"""PR-K — Prompt rendering carries the source-promotion notice.

When :mod:`tracelens_analysis` promotes a candidate from a python
``@compile_ops`` wrapper to the device source ``.cu`` (see PR-K's
``upgrade_aiter_compile_ops_launcher``), the candidate dict carries:

* ``source_file`` set to the device source path (rewrite target);
* ``launcher_source_file`` set to the original wrapper path;
* ``source_promoted_from_launcher = True`` (audit flag).

:func:`kernel_optimization.build_prompt` must surface this information
so the LLM does not silently rewrite the python wrapper (zero runtime
effect, REVERT @-0% E2E). The contract pinned by these tests:

* When promoted, the prompt body contains a ``SOURCE ATTRIBUTION NOTE``
  block naming both the launcher and the device source, and tells the
  LLM in three numbered hard rules NOT to modify the wrapper.
* When NOT promoted, the prompt body must NOT contain the notice (no
  byte-level bloat or contradictory guidance for vendor / Triton kernels
  whose trace source already pointed at the right file).
* :func:`build_kernel_metadata` carries ``launcher_source_file`` and
  ``source_promoted_from_launcher`` so non-text consumers (GEAK
  task_parser JSON path) see the same fields.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_optimization as ko  # noqa: E402


def _build_args(**overrides) -> Namespace:
    base = {
        "source_file": "",
        "kernel_id": "k001",
        "num_gpus": 1,
        "budget_minutes": 60,
        "target_platform": "MI355X",
        "geak_cost_limit": 5.0,
        "oob_max_turns": 8,
        "dry_run": False,
        "extra_sglang_args": "",
    }
    base.update(overrides)
    return Namespace(**base)


def _candidate(**overrides) -> dict:
    base = {
        "name": "aiter::ck_moe_stage1",
        "kernel_id": "k001",
        "source_file": "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu",
        "source_type": "hip_cpp",
        "kernel_repo": "/sgl-workspace/aiter",
        "gpu_pct": 25.0,
        "shapes": [[128, 4096]],
        "duration_us": 1234.5,
        "call_count": 100,
        "is_multigpu": False,
        "num_gpus_recommended": 1,
        "benchmark_files": [],
    }
    base.update(overrides)
    return base


def test_build_prompt_renders_promotion_block_for_promoted_candidate() -> None:
    """A promoted candidate's prompt must contain the SOURCE ATTRIBUTION
    NOTE block with both the launcher and device source paths plus the
    three hard rules."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        source_promoted_from_launcher=True,
    )
    args = _build_args()
    prompt = ko.build_prompt(cand, args)

    assert "SOURCE ATTRIBUTION NOTE" in prompt
    assert "/sgl-workspace/aiter/aiter/ops/moe_op.py" in prompt
    assert "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu" in prompt
    # Three numbered hard rules must all be present so the LLM can't
    # silently fall through one.
    assert "1. DO NOT modify the Python wrapper" in prompt
    assert "2. The device source may be a CODEGEN ENTRY" in prompt
    assert "3. Preserve function names, signatures" in prompt
    # The notice must explicitly call out the bypass mechanism so the
    # LLM understands WHY wrapper patches don't work (not just that they
    # shouldn't be made).
    assert "@compile_ops" in prompt
    assert "ZERO runtime effect" in prompt


def test_build_prompt_no_promotion_block_for_un_promoted_candidate() -> None:
    """Vendor / Triton / direct-device kernels traced to the right file
    on the first pass must NOT carry the notice — it would either
    contradict the actual trace attribution (if launcher is empty) or
    duplicate kernel_url uselessly (if launcher == source_file)."""
    cand = _candidate(
        # No launcher_source_file, no source_promoted_from_launcher flag.
    )
    args = _build_args()
    prompt = ko.build_prompt(cand, args)
    assert "SOURCE ATTRIBUTION NOTE" not in prompt
    assert "@compile_ops" not in prompt


def test_build_prompt_no_promotion_block_when_flag_false_with_launcher() -> None:
    """Defensive: even if a stray ``launcher_source_file`` field leaked
    through without the explicit promotion flag (e.g. partial state from
    an interrupted finalize), the notice must NOT render. The flag is
    the contract gate, not the field's mere presence."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        # Missing source_promoted_from_launcher → no notice.
    )
    args = _build_args()
    prompt = ko.build_prompt(cand, args)
    assert "SOURCE ATTRIBUTION NOTE" not in prompt


def test_build_kernel_metadata_carries_launcher_fields_when_promoted() -> None:
    """The structured JSON metadata block (parsed by GEAK's task_parser)
    must surface the same launcher / promoted-from fields so non-prose
    consumers see the attribution."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        source_promoted_from_launcher=True,
    )
    args = _build_args()
    md = ko.build_kernel_metadata(cand, args)
    assert md["kernel_path"].endswith("gemm_moe_ck2stages.cu")
    assert md["launcher_source_file"] == "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    assert md["source_promoted_from_launcher"] is True


def test_build_kernel_metadata_default_launcher_fields_when_not_promoted() -> None:
    """Un-promoted candidates carry empty / False values — never None
    (downstream JSON parsers expect string + bool, not null)."""
    cand = _candidate()
    args = _build_args()
    md = ko.build_kernel_metadata(cand, args)
    assert md["launcher_source_file"] == ""
    assert md["source_promoted_from_launcher"] is False


def test_build_kernel_metadata_metadata_is_json_serializable_when_promoted() -> None:
    """The metadata block is dropped into the prompt as ``json.dumps(...)``
    — a non-serializable value here would render as ``<not serializable>``
    and silently corrupt GEAK's task_parser. Round-trip the metadata
    through json to pin the contract."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        source_promoted_from_launcher=True,
    )
    args = _build_args()
    md = ko.build_kernel_metadata(cand, args)
    # Should NOT raise.
    encoded = json.dumps(md, sort_keys=True)
    parsed = json.loads(encoded)
    assert parsed["launcher_source_file"] == "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    assert parsed["source_promoted_from_launcher"] is True
