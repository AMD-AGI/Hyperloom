# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-K — aiter JIT cache invalidation around .cu/.cuh rebuilds.

A csrc patch would be masked by a stale per-instance ``jit/build/`` .so
(Qwen3-30B-A3B false REVERT @-2.66% E2E), so apply moves it aside before
rebuild and revert moves it back; aiter-only no-op for sglang/vllm.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent / "apply_kernel_patch.py"


@pytest.fixture(scope="module")
def apply_tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_apply_kernel_patch_under_test", _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "target_path,expected",
    [
        ("/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu", True),
        ("/sgl-workspace/aiter/csrc/include/foo.cuh", True),
        ("/usr/local/lib/python3.12/dist-packages/aiter/csrc/include/bar.hpp", True),
        ("/opt/venv/lib/python3.10/site-packages/aiter/csrc/baz.cpp", True),
        # Negative — sglang/vllm csrc, aiter python source, unrelated paths.
        ("/sgl-workspace/sglang/sgl-kernel/csrc/foo.cu", False),
        ("/sgl-workspace/vllm/csrc/bar.cu", False),
        ("/sgl-workspace/aiter/aiter/ops/moe_op.py", False),
        ("/tmp/foo/aiter.cu", False),
        # Edge: leading-slash anchor in ``/aiter/csrc/`` must not match ``myaiter``.
        ("/sgl-workspace/myaiter/csrc/foo.cu", False),
    ],
)
def test_target_is_in_aiter_csrc(apply_tool, target_path: str, expected: bool) -> None:
    assert apply_tool._target_is_in_aiter_csrc(Path(target_path)) is expected


def test_invalidate_skips_when_target_not_aiter_csrc(
    apply_tool, tmp_path: Path,
) -> None:
    """sglang / vllm / non-csrc targets must NOT touch jit/build/."""
    sglang_target = tmp_path / "fake_sglang" / "sgl-kernel" / "csrc" / "x.cu"
    sglang_target.parent.mkdir(parents=True)
    sglang_target.write_text("// fake")

    backup = tmp_path / "backup"
    out = apply_tool._invalidate_aiter_jit_build(sglang_target, backup)
    assert out["status"] == "skipped"
    assert "aiter/csrc" in out["reason"]


def test_invalidate_skips_when_jit_build_missing(
    apply_tool, tmp_path: Path,
) -> None:
    """aiter csrc target but no jit/build/ on disk → skipped."""
    target = tmp_path / "aiter_root" / "aiter" / "csrc" / "foo.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// fake")

    backup = tmp_path / "backup"
    fake_jit_build = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    out = apply_tool._invalidate_aiter_jit_build(
        target, backup, jit_build_dir_override=fake_jit_build,
    )
    assert out["status"] == "skipped"
    assert "does not exist" in out["reason"]


def test_invalidate_skips_when_jit_build_empty(
    apply_tool, tmp_path: Path,
) -> None:
    """aiter csrc target + empty jit/build/ → skipped (nothing to move)."""
    target = tmp_path / "aiter_root" / "aiter" / "csrc" / "foo.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// fake")

    backup = tmp_path / "backup"
    fake_jit_build = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    fake_jit_build.mkdir(parents=True)

    out = apply_tool._invalidate_aiter_jit_build(
        target, backup, jit_build_dir_override=fake_jit_build,
    )
    assert out["status"] == "skipped"
    assert "empty" in out["reason"]
    assert fake_jit_build.is_dir(), "empty dir should not have been moved"


def test_invalidate_moves_jit_build_when_populated(
    apply_tool, tmp_path: Path,
) -> None:
    """Populated jit/build/ is MOVED (not copied) into backup_dir/jit_build."""
    target = tmp_path / "aiter_root" / "aiter" / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_ck2stages.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// fake codegen entry")

    fake_jit_build = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    fake_jit_build.mkdir(parents=True)
    (fake_jit_build / "module_moe_ck2stages_b16_b16_silu_no").mkdir()
    (fake_jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "module.so").write_bytes(b"\x7fELF\x02fake_so")
    (fake_jit_build / "module_moe_topksoftmax_asm").mkdir()
    (fake_jit_build / "module_moe_topksoftmax_asm" / "module.so").write_bytes(b"\x7fELF\x02fake_so")

    backup = tmp_path / "backup"
    out = apply_tool._invalidate_aiter_jit_build(
        target, backup, jit_build_dir_override=fake_jit_build,
    )

    assert out["status"] == "ok"
    assert out["src"] == str(fake_jit_build)
    assert out["backup_path"] == str(backup / "jit_build")
    assert "moved_at" in out

    assert not fake_jit_build.exists(), \
        "jit/build/ must be moved aside, not copied"
    assert (backup / "jit_build" / "module_moe_ck2stages_b16_b16_silu_no" / "module.so").is_file()
    assert (backup / "jit_build" / "module_moe_topksoftmax_asm" / "module.so").is_file()


def test_invalidate_refuses_to_clobber_existing_backup(
    apply_tool, tmp_path: Path,
) -> None:
    """A pre-existing backup_dir/jit_build/ must fail the apply, not overwrite the prior backup."""
    target = tmp_path / "aiter_root" / "aiter" / "csrc" / "foo.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// fake")

    fake_jit_build = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    fake_jit_build.mkdir(parents=True)
    (fake_jit_build / "module_x").mkdir()
    (fake_jit_build / "module_x" / "x.so").write_bytes(b"so")

    backup = tmp_path / "backup"
    backup.mkdir()
    # Prior backup left over from earlier failed apply.
    (backup / "jit_build").mkdir()
    (backup / "jit_build" / "leftover.txt").write_text("stale")

    out = apply_tool._invalidate_aiter_jit_build(
        target, backup, jit_build_dir_override=fake_jit_build,
    )
    assert out["status"] == "failed"
    assert "already exists" in out["error"]
    assert (fake_jit_build / "module_x" / "x.so").is_file()


def test_restore_moves_backup_back(apply_tool, tmp_path: Path) -> None:
    """Round-trip: invalidate then restore returns to pre-apply state."""
    target = tmp_path / "aiter_root" / "aiter" / "csrc" / "foo.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// fake")

    fake_jit_build = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    fake_jit_build.mkdir(parents=True)
    (fake_jit_build / "module_a").mkdir()
    (fake_jit_build / "module_a" / "a.so").write_bytes(b"so_a")

    backup = tmp_path / "backup"
    invalidate = apply_tool._invalidate_aiter_jit_build(
        target, backup, jit_build_dir_override=fake_jit_build,
    )
    assert invalidate["status"] == "ok"
    assert not fake_jit_build.exists()

    restore = apply_tool._restore_aiter_jit_build(invalidate)
    assert restore["status"] == "ok"
    assert restore["restored_to"] == str(fake_jit_build)
    assert (fake_jit_build / "module_a" / "a.so").read_bytes() == b"so_a"
    assert not (backup / "jit_build").exists()


def test_restore_clears_regenerated_dir_first(
    apply_tool, tmp_path: Path,
) -> None:
    """Restore must wipe a regenerated jit/build/ before moving the backup back (else shutil.move nests)."""
    target = tmp_path / "aiter_root" / "aiter" / "csrc" / "foo.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// fake")

    fake_jit_build = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    fake_jit_build.mkdir(parents=True)
    (fake_jit_build / "module_old").mkdir()
    (fake_jit_build / "module_old" / "v0.so").write_bytes(b"v0")

    backup = tmp_path / "backup"
    invalidate = apply_tool._invalidate_aiter_jit_build(
        target, backup, jit_build_dir_override=fake_jit_build,
    )
    assert invalidate["status"] == "ok"

    # Simulate rebuild + first-import regenerating jit/build/ afresh.
    fake_jit_build.mkdir(parents=True)
    (fake_jit_build / "module_new").mkdir()
    (fake_jit_build / "module_new" / "v1.so").write_bytes(b"v1")

    restore = apply_tool._restore_aiter_jit_build(invalidate)
    assert restore["status"] == "ok"
    assert (fake_jit_build / "module_old" / "v0.so").read_bytes() == b"v0"
    assert not (fake_jit_build / "module_new").exists(), \
        "regenerated jit/build/ must be cleared before restore"


def test_restore_skips_when_no_backup(apply_tool) -> None:
    """A manifest with no jit_build_backup must produce a 'skipped' restore, not a crash."""
    out = apply_tool._restore_aiter_jit_build(
        {"status": "skipped", "reason": "target not under aiter/csrc/"}
    )
    assert out["status"] == "skipped"

    out = apply_tool._restore_aiter_jit_build({})
    assert out["status"] == "skipped"


def test_restore_skips_when_backup_path_missing(
    apply_tool, tmp_path: Path,
) -> None:
    """If the backup vanished between apply and revert, restore must skip cleanly."""
    fake_src = tmp_path / "aiter_root" / "aiter" / "jit" / "build"
    fake_backup = tmp_path / "backup" / "jit_build"
    out = apply_tool._restore_aiter_jit_build({
        "status": "ok",
        "src": str(fake_src),
        "backup_path": str(fake_backup),
        "moved_at": "2026-05-24T00:00:00Z",
    })
    assert out["status"] == "skipped"
    assert "missing" in out["reason"]


# End-to-end: apply_kernel_patch → revert_kernel_patch with jit invalidation.
def _write_aiter_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a synthetic aiter editable-checkout layout (csrc target + jit/build cache)."""
    repo = tmp_path / "sgl-workspace" / "aiter"
    csrc_dir = repo / "csrc" / "ck_gemm_moe_2stages_codegen"
    csrc_dir.mkdir(parents=True)
    target = csrc_dir / "gemm_moe_ck2stages.cu"
    target.write_text(
        '#include <cstdio>\n'
        'namespace aiter {\n'
        'void ck_moe_stage1(int x) { printf("v0\\n"); }\n'
        '}\n'
    )

    jit_build = repo / "aiter" / "jit" / "build"
    jit_build.mkdir(parents=True)
    (jit_build / "module_moe_ck2stages_b16_b16_silu_no").mkdir()
    (jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").write_bytes(b"v0")

    return repo, target, jit_build


def test_apply_then_revert_roundtrip_invalidates_and_restores_jit_cache(
    apply_tool, tmp_path: Path,
) -> None:
    """End-to-end apply → revert cycle (rebuild mocked): jit/build/ moved aside then both source and cache restored."""
    repo, target, jit_build = _write_aiter_repo(tmp_path)

    patch_file = tmp_path / "patched.cu"
    patch_file.write_text(
        '#include <cstdio>\n'
        'namespace aiter {\n'
        'void ck_moe_stage1(int x) { printf("v1\\n"); }\n'
        '}\n'
    )

    backup_root = tmp_path / "backups"

    # Make aiter resolve to the synthetic layout for _aiter_jit_build_dir().
    fake_spec = types.SimpleNamespace(submodule_search_locations=[str(repo / "aiter")])
    fake_rebuild_ok = {
        "status": "ok",
        "returncode": 0,
        "stdout_tail": "ok",
        "stderr_tail": "",
        "command": ["fake-rebuild"],
        "cwd": str(repo),
    }
    with patch.object(apply_tool.importlib.util, "find_spec",
                      return_value=fake_spec), \
         patch.object(apply_tool, "_run_rebuild",
                      return_value=fake_rebuild_ok):
        result = apply_tool.apply_kernel_patch(
            patch_path=patch_file,
            target_file=target,
            backup_root=backup_root,
            kernel_id="k001",
            allow_unknown_target=True,
        )

    assert result["status"] == "ok", result
    assert result["jit_build_backup"]["status"] == "ok", result["jit_build_backup"]
    assert result["jit_build_backup"]["src"] == str(jit_build)
    assert not jit_build.exists(), \
        "live jit/build/ must be moved aside before rebuild"
    backup_jit_build = Path(result["jit_build_backup"]["backup_path"])
    assert (backup_jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").is_file()
    assert "v1" in target.read_text()

    revert = apply_tool.revert_kernel_patch(result["manifest_path"])
    assert revert["status"] == "ok"
    assert "v0" in target.read_text()
    assert (jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").read_bytes() == b"v0"
    assert not backup_jit_build.exists(), "restore should consume the backup"


def test_skip_rebuild_does_not_invalidate_jit_build(
    apply_tool, tmp_path: Path,
) -> None:
    """``skip_rebuild=True`` means the operator handles rebuild, so apply must NOT touch jit/build/."""
    repo, target, jit_build = _write_aiter_repo(tmp_path)

    patch_file = tmp_path / "patched.cu"
    patch_file.write_text(
        '#include <cstdio>\n'
        'namespace aiter {\n'
        'void ck_moe_stage1(int x) { printf("v1\\n"); }\n'
        '}\n'
    )

    fake_spec = types.SimpleNamespace(submodule_search_locations=[str(repo / "aiter")])
    with patch.object(apply_tool.importlib.util, "find_spec",
                      return_value=fake_spec):
        result = apply_tool.apply_kernel_patch(
            patch_path=patch_file,
            target_file=target,
            backup_root=tmp_path / "backups",
            kernel_id="k001",
            skip_rebuild=True,
            allow_unknown_target=True,
        )

    assert result["status"] == "ok"
    assert result["jit_build_backup"]["status"] == "skipped"
    assert (jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").is_file()
