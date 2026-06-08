# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-K — aiter JIT cache invalidation around .cu/.cuh rebuilds.

Background
----------
``aiter`` ships ``@compile_ops("module_<name>", gen_func=...)`` decorators
that JIT-codegen + hipcc-compile per-instance ``.so`` files into
``<aiter>/jit/build/module_<name>_<sig>/``. ``setup.py develop`` (the
default rebuild for an aiter editable install) rebuilds the python package
+ statically-linked ``.so`` but does NOT touch ``jit/build/`` entries — so
a patch under ``aiter/csrc/ck_gemm_moe_2stages_codegen/`` would rebuild
the wheel yet the next ``import aiter.ops.moe_op`` would still load the
pre-patch ``module_moe_ck2stages_*.so`` from ``jit/build/``. Net effect on
historical Qwen3-30B-A3B-Base sessions: integrate measured the unchanged
kernel and emitted REVERT, masking patch effectiveness as -2.66% E2E.

The :mod:`apply_kernel_patch` tool now moves ``<aiter>/jit/build/`` aside
into the per-(kernel, target) backup_root BEFORE the rebuild step so the
post-rebuild first import re-codegens + re-compiles every module from
clean state. ``shutil.move`` is atomic on the same filesystem and zero-
copy. :func:`revert_kernel_patch` moves the backup back, removing any
regenerated ``jit/build/`` first so the pre-patch state is restored
bit-for-bit.

Scope: ONLY aiter is affected. ``sglang`` and ``vllm`` have no JIT codegen
layer (their ``.so`` are produced by setup.py at install time), so the
invalidation step is a no-op for those targets and the standard
``setup.py develop`` rebuild + ``cache_clear`` is sufficient.

Tests:

* ``_target_is_in_aiter_csrc`` recognizes editable + dist-packages layouts.
* ``_invalidate_aiter_jit_build`` skips on non-aiter targets, missing /
  empty jit/build, and refuses to clobber a pre-existing backup.
* On ok, the entire jit/build/ tree is moved aside (NOT copied) so the
  source location is empty / absent post-call.
* ``_restore_aiter_jit_build`` undoes the move, including the case where
  rebuild + first-import regenerated a fresh jit/build/ on top.
* End-to-end ``apply_kernel_patch`` against a synthetic aiter-csrc target
  produces a manifest with ``jit_build_backup`` and the live jit/build/
  is gone post-apply; ``revert_kernel_patch`` restores it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Tool import — apply_kernel_patch.py is a standalone shell tool, not a
# package. ``importlib.util`` loads it from the source tree so this test can
# run without ``HYPERLOOM_KERNEL_AGENT_ROOT`` env / install.sh side-effects.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# _target_is_in_aiter_csrc
# ---------------------------------------------------------------------------
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
        # Edge: ``aiter/csrc`` substring inside another package name must
        # not match — the leading-slash anchor in ``/aiter/csrc/`` prevents
        # collisions with ``myaiter`` / ``test_aiter`` / etc.
        ("/sgl-workspace/myaiter/csrc/foo.cu", False),
    ],
)
def test_target_is_in_aiter_csrc(apply_tool, target_path: str, expected: bool) -> None:
    assert apply_tool._target_is_in_aiter_csrc(Path(target_path)) is expected


# ---------------------------------------------------------------------------
# _invalidate_aiter_jit_build
# ---------------------------------------------------------------------------
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
    # Don't create fake_jit_build — should be skipped.
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
    # Populate with two synthetic compile_ops modules so the directory has
    # real content like a live aiter sandbox would.
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

    # Source is GONE (moved, not copied).
    assert not fake_jit_build.exists(), \
        "jit/build/ must be moved aside, not copied"
    # Backup contains the original tree intact.
    assert (backup / "jit_build" / "module_moe_ck2stages_b16_b16_silu_no" / "module.so").is_file()
    assert (backup / "jit_build" / "module_moe_topksoftmax_asm" / "module.so").is_file()


def test_invalidate_refuses_to_clobber_existing_backup(
    apply_tool, tmp_path: Path,
) -> None:
    """A pre-existing backup_dir/jit_build/ from a prior dirty run must
    fail the apply rather than silently overwriting the previous backup."""
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
    # Live jit/build/ unchanged — caller can recover by clearing the
    # collision before retrying.
    assert (fake_jit_build / "module_x" / "x.so").is_file()


# ---------------------------------------------------------------------------
# _restore_aiter_jit_build
# ---------------------------------------------------------------------------
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
    # Backup path is gone after restore (mv is one-way).
    assert not (backup / "jit_build").exists()


def test_restore_clears_regenerated_dir_first(
    apply_tool, tmp_path: Path,
) -> None:
    """After rebuild, the first import may have already regenerated
    jit/build/ on top of the original location. Restore must wipe that
    fresh dir before moving the backup back so the pre-patch state is
    restored bit-for-bit (otherwise shutil.move would fail or nest)."""
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

    # Simulate rebuild + first-import regenerating jit/build/ with fresh
    # patched-version compile artifacts.
    fake_jit_build.mkdir(parents=True)
    (fake_jit_build / "module_new").mkdir()
    (fake_jit_build / "module_new" / "v1.so").write_bytes(b"v1")

    restore = apply_tool._restore_aiter_jit_build(invalidate)
    assert restore["status"] == "ok"
    # Pre-patch state restored: only module_old, no module_new.
    assert (fake_jit_build / "module_old" / "v0.so").read_bytes() == b"v0"
    assert not (fake_jit_build / "module_new").exists(), \
        "regenerated jit/build/ must be cleared before restore"


def test_restore_skips_when_no_backup(apply_tool) -> None:
    """A manifest that never recorded a jit_build_backup (non-aiter apply,
    or skipped invalidation) must produce a 'skipped' restore, not a crash."""
    out = apply_tool._restore_aiter_jit_build(
        {"status": "skipped", "reason": "target not under aiter/csrc/"}
    )
    assert out["status"] == "skipped"

    out = apply_tool._restore_aiter_jit_build({})
    assert out["status"] == "skipped"


def test_restore_skips_when_backup_path_missing(
    apply_tool, tmp_path: Path,
) -> None:
    """If something deleted the backup between apply and revert, restore
    must skip cleanly (no crash) so the rest of revert can still run."""
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


# ---------------------------------------------------------------------------
# End-to-end: apply_kernel_patch → revert_kernel_patch with jit invalidation.
# ---------------------------------------------------------------------------
def _write_aiter_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a synthetic aiter editable-checkout layout:

        <repo>/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu  (target)
        <repo>/aiter/jit/build/module_moe_ck2stages_*/...              (jit cache)
    """
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
    """End-to-end ``apply_kernel_patch`` → ``revert_kernel_patch`` cycle.

    The rebuild step is mocked (``_run_rebuild`` → status=ok) so the test
    stays hermetic, but the rebuild path is exercised — that's the only
    branch where invalidation runs. Skipping rebuild entirely (the
    ``skip_rebuild=True`` shortcut) intentionally does NOT invalidate
    jit/build/ on the assumption that an operator who said "don't rebuild"
    has already handled cache state themselves.

    Verifies:
      * manifest + apply result carry ``jit_build_backup`` with status=ok
      * live jit/build/ is moved aside post-apply (NOT present at the
        original location)
      * the patched source landed at the target
      * revert restores BOTH source and jit/build/ to pre-apply state
    """
    repo, target, jit_build = _write_aiter_repo(tmp_path)

    patch_file = tmp_path / "patched.cu"
    patch_file.write_text(
        '#include <cstdio>\n'
        'namespace aiter {\n'
        'void ck_moe_stage1(int x) { printf("v1\\n"); }\n'
        '}\n'
    )

    backup_root = tmp_path / "backups"

    # Force the apply to think aiter is importable at our synthetic layout
    # so _aiter_jit_build_dir() returns repo/aiter/jit/build.
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
        # Allow the unknown target root since our synthetic /tmp/.../
        # sgl-workspace/aiter/ tree isn't in KNOWN_TARGET_ROOTS.
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
    # Live jit/build/ has been moved aside.
    assert not jit_build.exists(), \
        "live jit/build/ must be moved aside before rebuild"
    backup_jit_build = Path(result["jit_build_backup"]["backup_path"])
    assert (backup_jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").is_file()
    # The patched source landed at the target.
    assert "v1" in target.read_text()

    # Revert: restores both source and jit/build/.
    revert = apply_tool.revert_kernel_patch(result["manifest_path"])
    assert revert["status"] == "ok"
    assert "v0" in target.read_text()
    assert (jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").read_bytes() == b"v0"
    assert not backup_jit_build.exists(), "restore should consume the backup"


def test_skip_rebuild_does_not_invalidate_jit_build(
    apply_tool, tmp_path: Path,
) -> None:
    """Operator-supplied ``skip_rebuild=True`` means "I'll handle rebuild
    myself" — apply must NOT touch jit/build/ in that case (otherwise the
    operator's manual rebuild would lose its jit cache for unrelated
    modules they were not patching)."""
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
    # jit/build/ untouched.
    assert (jit_build / "module_moe_ck2stages_b16_b16_silu_no" / "kernel.so").is_file()
