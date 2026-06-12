# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-K — aiter @compile_ops launcher → device source promotion.

Promotes the aiter ``aiter/ops/`` Python wrapper to the device ``.cu``
before the candidate reaches the LLM; narrow allowlist, no-op otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_TLA_PATH = Path(__file__).resolve().parent / "tracelens_analysis.py"


@pytest.fixture(scope="module")
def tla() -> types.ModuleType:
    """Load tracelens_analysis.py without running its full CLI bootstrap."""
    spec = importlib.util.spec_from_file_location(
        "_tracelens_analysis_under_test", _TLA_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def aiter_repo(tmp_path: Path) -> Path:
    """Build a synthetic aiter editable-checkout tree with the device sources promotion should find."""
    repo = tmp_path / "aiter_editable" / "aiter"  # mimic /sgl-workspace/aiter/
    (repo / "aiter" / "ops").mkdir(parents=True)
    (repo / "aiter" / "ops" / "moe_op.py").write_text(
        '"""@compile_ops wrapper around module_moe_ck2stages."""\n'
    )
    (repo / "aiter" / "ops" / "rmsnorm.py").write_text('# Triton wrapper.\n')

    (repo / "csrc" / "ck_gemm_moe_2stages_codegen").mkdir(parents=True)
    (repo / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_ck2stages.cu").write_text(
        '// codegen entry for module_moe_ck2stages\n'
    )
    (repo / "csrc" / "kernels").mkdir()
    (repo / "csrc" / "kernels" / "topk_softmax_kernels.cu").write_text("// topk\n")
    (repo / "csrc" / "kernels" / "topk_softmax_kernels_group.cu").write_text("// topk_group\n")
    (repo / "csrc" / "kernels" / "moe_align_block_size_kernels.cu").write_text("// align\n")
    (repo / "csrc" / "kernels" / "moe_fused_gate.cu").write_text("// gate\n")

    (repo / ".git").mkdir()
    return repo


def test_promotes_ck_moe_stage1_wrapper_to_codegen_cu(
    tla, aiter_repo: Path,
) -> None:
    wrapper = aiter_repo / "aiter" / "ops" / "moe_op.py"
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(wrapper), "aiter::ck_moe_stage1", str(aiter_repo),
    )
    assert out == str(aiter_repo / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_ck2stages.cu")


def test_promotes_ck_moe_stage2_wrapper_to_codegen_cu(
    tla, aiter_repo: Path,
) -> None:
    wrapper = aiter_repo / "aiter" / "ops" / "moe_op.py"
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(wrapper), "aiter::ck_moe_stage2", str(aiter_repo),
    )
    assert out.endswith("gemm_moe_ck2stages.cu")


@pytest.mark.parametrize("kernel_name,expected_basename", [
    ("aiter::topk_softmax_group", "topk_softmax_kernels_group.cu"),
    ("aiter::topk_softmax", "topk_softmax_kernels.cu"),
    ("aiter::moe_align_block_size", "moe_align_block_size_kernels.cu"),
    ("aiter::moe_fused_gate", "moe_fused_gate.cu"),
])
def test_promotes_named_compile_ops_to_kernels_cu(
    tla, aiter_repo: Path, kernel_name: str, expected_basename: str,
) -> None:
    wrapper = aiter_repo / "aiter" / "ops" / "moe_op.py"
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(wrapper), kernel_name, str(aiter_repo),
    )
    assert Path(out).name == expected_basename


def test_promotion_noop_when_source_not_aiter_ops(tla, aiter_repo: Path) -> None:
    """Non-aiter sources (sglang/vllm) are never promoted by this rule."""
    fake = aiter_repo / "fake_sglang.py"
    fake.write_text("# not aiter\n")
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(fake), "aiter::ck_moe_stage1", str(aiter_repo),
    )
    assert out == str(fake)


def test_promotion_noop_when_source_not_python(tla, aiter_repo: Path) -> None:
    """A .cu / .cuh source already IS device source — no promotion needed."""
    cu = aiter_repo / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_ck2stages.cu"
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(cu), "aiter::ck_moe_stage1", str(aiter_repo),
    )
    assert out == str(cu)


def test_promotion_noop_when_kernel_name_unmatched(
    tla, aiter_repo: Path,
) -> None:
    """rmsnorm has no @compile_ops codegen — wrapper is the right target."""
    wrapper = aiter_repo / "aiter" / "ops" / "rmsnorm.py"
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(wrapper), "aiter::rmsnorm2d_fwd", str(aiter_repo),
    )
    assert out == str(wrapper)


def test_promotion_noop_when_target_cu_missing(
    tla, aiter_repo: Path, tmp_path: Path, monkeypatch,
) -> None:
    """If the expected ``.cu`` isn't on disk, refuse to fabricate a path — return the wrapper unchanged."""
    # Delete the ck_moe codegen .cu to simulate a version mismatch.
    (aiter_repo / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_ck2stages.cu").unlink()
    # Empty fallback so the promoter can't silently succeed via the real /sgl-workspace/aiter.
    empty_fallback = tmp_path / "no_aiter_here"
    empty_fallback.mkdir()
    monkeypatch.setattr(tla, "_AITER_FALLBACK_REPO", str(empty_fallback))
    wrapper = aiter_repo / "aiter" / "ops" / "moe_op.py"
    out = tla.upgrade_aiter_compile_ops_launcher(
        str(wrapper), "aiter::ck_moe_stage1", str(aiter_repo),
    )
    assert out == str(wrapper)


def test_promotion_falls_back_to_sgl_workspace_aiter_for_wheel_install(
    tla, tmp_path: Path, monkeypatch,
) -> None:
    """A wheel-install wrapper has no co-located ``csrc/``; the promoter falls back to the editable checkout at the canonical sandbox path."""
    # Wheel-install layout (no csrc/ here).
    wheel_root = tmp_path / "wheel_dist" / "aiter"
    (wheel_root / "ops").mkdir(parents=True)
    wrapper = wheel_root / "ops" / "moe_op.py"
    wrapper.write_text("# wheel wrapper\n")

    # Editable checkout that DOES have csrc/.
    editable = tmp_path / "sgl-workspace" / "aiter"
    (editable / "csrc" / "ck_gemm_moe_2stages_codegen").mkdir(parents=True)
    cu = editable / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_ck2stages.cu"
    cu.write_text("// codegen\n")
    (editable / ".git").mkdir()

    monkeypatch.setattr(tla, "_AITER_FALLBACK_REPO", str(editable))

    out = tla.upgrade_aiter_compile_ops_launcher(
        str(wrapper), "aiter::ck_moe_stage1", "",  # no kernel_repo from caller
    )
    assert out == str(cu)


def test_finalize_candidates_records_launcher_source_file_on_promotion(
    tla, aiter_repo: Path,
) -> None:
    """End-to-end: wrapper-sourced candidate gets promoted source_file + launcher_source_file."""
    wrapper = str(aiter_repo / "aiter" / "ops" / "moe_op.py")
    candidates = [{
        "name": "aiter::ck_moe_stage1",
        "duration_us": 5000.0,
        "call_count": 100,
        "source_file": wrapper,
        "shapes": [[128, 4096]],
    }]
    out = tla._finalize_candidates(candidates, total_dur=10000.0)
    cand = out[0]
    assert cand["source_file"].endswith("gemm_moe_ck2stages.cu")
    assert cand["launcher_source_file"] == wrapper
    assert cand["source_promoted_from_launcher"] is True
    assert cand["source_type"] == "hip_cpp"


def test_finalize_candidates_no_launcher_field_when_no_promotion(
    tla, aiter_repo: Path,
) -> None:
    """When promotion does not fire (rmsnorm), finalize must NOT add ``launcher_source_file`` (else duplicate kernel_url)."""
    wrapper = str(aiter_repo / "aiter" / "ops" / "rmsnorm.py")
    candidates = [{
        "name": "aiter::rmsnorm2d_fwd",
        "duration_us": 1000.0,
        "call_count": 50,
        "source_file": wrapper,
        "shapes": [[128, 4096]],
    }]
    out = tla._finalize_candidates(candidates, total_dur=2000.0)
    cand = out[0]
    assert cand["source_file"] == wrapper
    assert "launcher_source_file" not in cand
    assert cand.get("source_promoted_from_launcher") is not True
