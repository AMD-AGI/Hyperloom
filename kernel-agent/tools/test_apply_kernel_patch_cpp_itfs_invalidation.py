"""PR-K2 — aiter cpp_itfs RUNTIME-compiled cache invalidation (GH #458).

Background
----------
aiter ``csrc/cpp_itfs`` kernels (e.g. ``paged_attention`` ->
``csrc/cpp_itfs/pa/pa_kernels.cuh``) are NOT built by ``setup.py develop``;
they are runtime-compiled on first call by ``compile_template_op``
(``csrc/cpp_itfs/utils.py``) into
``$AITER_ROOT_DIR/build/<md_name>_<md5(params)>/lib.so`` (default
``$HOME/.aiter/build``). The cache folder name hashes kernel *parameters*,
NOT source content, so a pristine and a patched build of the same kernel
collide on the SAME directory; ``compile_template_op`` only rebuilds when
``lib.so`` is missing, so after :mod:`apply_kernel_patch` lands the new
``.cuh`` (and runs the no-op-for-this-class ``setup.py develop``), the next
server reuses the STALE pristine ``lib.so`` and integrate measures ~0% gain
on a genuinely-good kernel (observed -0.17% on a +2.5% pa kernel, RUN2).

These tests cover the fix:

* ``_target_is_in_aiter_cpp_itfs`` recognizes cpp_itfs editable + dist layouts
  and rejects non-cpp_itfs aiter csrc / sglang / vllm targets.
* ``_cpp_itfs_module_names`` scrapes ``MD_NAME`` from the drivers next to the
  patched source so invalidation can be scoped to the impacted module(s).
* ``_invalidate_aiter_cpp_itfs_cache`` moves only the matching
  ``<md_name>_*`` cache dirs aside (leaving unrelated modules intact), falls
  back to the whole build root when no MD_NAME is determinable, skips when
  the build dir is absent, and refuses to clobber a pre-existing backup.
* ``_restore_aiter_cpp_itfs_cache`` round-trips the move, clearing any dir
  the re-baseline regenerated first.
* ``verify_cpp_itfs_rebuilt`` is a no-op for non-cpp_itfs, reports ``stale``
  when no fresh ``lib.so`` landed, and ``ok`` when one did.
* End-to-end ``apply_kernel_patch`` against a synthetic cpp_itfs target moves
  the runtime cache aside + records ``cpp_itfs_cache_backup`` (and revert
  restores it), while a non-cpp_itfs (sglang) target leaves the cpp_itfs
  build root untouched.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent / "apply_kernel_patch.py"


@pytest.fixture(scope="module")
def apply_tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_apply_kernel_patch_cpp_itfs_under_test", _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _target_is_in_aiter_cpp_itfs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "target_path,expected",
    [
        ("/sgl-workspace/aiter/csrc/cpp_itfs/pa/pa_kernels.cuh", True),
        ("/sgl-workspace/aiter/csrc/cpp_itfs/pa/pa_ragged.cpp.jinja", True),
        ("/usr/local/lib/python3.12/dist-packages/aiter/csrc/cpp_itfs/mha/x.cuh", True),
        # Negative — aiter csrc but NOT cpp_itfs, plus sglang / vllm.
        ("/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm.cu", False),
        ("/sgl-workspace/aiter/csrc/include/foo.cuh", False),
        ("/sgl-workspace/sglang/sgl-kernel/csrc/cpp_itfs/foo.cu", False),
        ("/sgl-workspace/vllm/csrc/foo.cu", False),
    ],
)
def test_target_is_in_aiter_cpp_itfs(apply_tool, target_path: str, expected: bool) -> None:
    assert apply_tool._target_is_in_aiter_cpp_itfs(Path(target_path)) is expected


# ---------------------------------------------------------------------------
# _cpp_itfs_module_names
# ---------------------------------------------------------------------------
def test_cpp_itfs_module_names_scrapes_md_name(apply_tool, tmp_path: Path) -> None:
    pa_dir = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "pa"
    pa_dir.mkdir(parents=True)
    (pa_dir / "pa_kernels.cuh").write_text("// shared kernel header\n")
    (pa_dir / "pa_ragged.py").write_text('MD_NAME = "pa_ragged"\n')
    (pa_dir / "pa.py").write_text("MD_NAME = 'pa'\n")
    (pa_dir / "notes.py").write_text("X = 1  # no MD_NAME here\n")

    names = apply_tool._cpp_itfs_module_names(pa_dir / "pa_kernels.cuh")
    assert names == ["pa", "pa_ragged"]


def test_cpp_itfs_module_names_empty_when_no_driver(apply_tool, tmp_path: Path) -> None:
    d = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "mystery"
    d.mkdir(parents=True)
    (d / "kernel.cuh").write_text("// no python driver next to me\n")
    assert apply_tool._cpp_itfs_module_names(d / "kernel.cuh") == []


# ---------------------------------------------------------------------------
# _invalidate_aiter_cpp_itfs_cache
# ---------------------------------------------------------------------------
def _make_cache_dir(build_dir: Path, name: str, content: bytes = b"v0") -> Path:
    d = build_dir / name
    d.mkdir(parents=True)
    (d / "lib.so").write_bytes(content)
    return d


def test_invalidate_skips_non_cpp_itfs_target(apply_tool, tmp_path: Path) -> None:
    target = tmp_path / "aiter" / "csrc" / "ck_gemm" / "gemm.cu"
    target.parent.mkdir(parents=True)
    target.write_text("// non cpp_itfs aiter csrc\n")
    build_dir = tmp_path / "build"
    _make_cache_dir(build_dir, "gemm_abc")

    out = apply_tool._invalidate_aiter_cpp_itfs_cache(
        target, tmp_path / "backup", build_dir_override=build_dir,
    )
    assert out["status"] == "skipped"
    assert out["is_cpp_itfs"] is False
    # Untouched.
    assert (build_dir / "gemm_abc" / "lib.so").is_file()


def test_invalidate_scopes_to_module_md_names(apply_tool, tmp_path: Path) -> None:
    """Only the patched module's <md_name>_* dirs are moved aside; unrelated
    modules (e.g. gemm_*) stay put."""
    pa_dir = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "pa"
    pa_dir.mkdir(parents=True)
    target = pa_dir / "pa_kernels.cuh"
    target.write_text("// shared by pa + pa_ragged\n")
    (pa_dir / "pa_ragged.py").write_text('MD_NAME = "pa_ragged"\n')
    (pa_dir / "pa.py").write_text('MD_NAME = "pa"\n')

    build_dir = tmp_path / "build"
    _make_cache_dir(build_dir, "pa_ragged_HASHA")
    _make_cache_dir(build_dir, "pa_HASHB")
    _make_cache_dir(build_dir, "gemm_HASHC")  # unrelated module — must survive

    backup = tmp_path / "backup"
    out = apply_tool._invalidate_aiter_cpp_itfs_cache(
        target, backup, build_dir_override=build_dir,
    )

    assert out["status"] == "ok"
    assert out["is_cpp_itfs"] is True
    assert out["scope"] == "module"
    assert out["module_names"] == ["pa", "pa_ragged"]
    moved_srcs = {Path(m["src"]).name for m in out["moved"]}
    assert moved_srcs == {"pa_ragged_HASHA", "pa_HASHB"}
    # Affected module dirs moved aside; unrelated gemm_* untouched.
    assert not (build_dir / "pa_ragged_HASHA").exists()
    assert not (build_dir / "pa_HASHB").exists()
    assert (build_dir / "gemm_HASHC" / "lib.so").is_file()
    # Backup holds the originals.
    assert (backup / "cpp_itfs_cache" / "pa_ragged_HASHA" / "lib.so").is_file()
    assert (backup / "cpp_itfs_cache" / "pa_HASHB" / "lib.so").is_file()


def test_invalidate_falls_back_to_build_root_without_md_name(
    apply_tool, tmp_path: Path,
) -> None:
    d = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "mystery"
    d.mkdir(parents=True)
    target = d / "kernel.cuh"
    target.write_text("// no driver -> clear whole cpp_itfs build root\n")

    build_dir = tmp_path / "build"
    _make_cache_dir(build_dir, "alpha_1")
    _make_cache_dir(build_dir, "beta_2")

    out = apply_tool._invalidate_aiter_cpp_itfs_cache(
        target, tmp_path / "backup", build_dir_override=build_dir,
    )
    assert out["status"] == "ok"
    assert out["scope"] == "build_root"
    assert {Path(m["src"]).name for m in out["moved"]} == {"alpha_1", "beta_2"}
    assert not (build_dir / "alpha_1").exists()
    assert not (build_dir / "beta_2").exists()


def test_invalidate_skips_when_build_dir_missing(apply_tool, tmp_path: Path) -> None:
    pa_dir = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "pa"
    pa_dir.mkdir(parents=True)
    target = pa_dir / "pa_kernels.cuh"
    target.write_text("// fake\n")
    (pa_dir / "pa_ragged.py").write_text('MD_NAME = "pa_ragged"\n')

    out = apply_tool._invalidate_aiter_cpp_itfs_cache(
        target, tmp_path / "backup", build_dir_override=tmp_path / "nope",
    )
    assert out["status"] == "skipped"
    assert out["is_cpp_itfs"] is True
    assert "does not exist" in out["reason"]
    # The record still carries enough for the verify gate.
    assert out["module_names"] == ["pa_ragged"]
    assert "invalidated_unix" in out


def test_invalidate_refuses_to_clobber_existing_backup(
    apply_tool, tmp_path: Path,
) -> None:
    pa_dir = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "pa"
    pa_dir.mkdir(parents=True)
    target = pa_dir / "pa_kernels.cuh"
    target.write_text("// fake\n")
    (pa_dir / "pa_ragged.py").write_text('MD_NAME = "pa_ragged"\n')

    build_dir = tmp_path / "build"
    _make_cache_dir(build_dir, "pa_ragged_HASHA")

    backup = tmp_path / "backup"
    (backup / "cpp_itfs_cache" / "pa_ragged_HASHA").mkdir(parents=True)
    (backup / "cpp_itfs_cache" / "pa_ragged_HASHA" / "stale.txt").write_text("old")

    out = apply_tool._invalidate_aiter_cpp_itfs_cache(
        target, backup, build_dir_override=build_dir,
    )
    assert out["status"] == "failed"
    assert "already exists" in out["error"]
    # Live cache untouched so the caller can recover.
    assert (build_dir / "pa_ragged_HASHA" / "lib.so").is_file()


# ---------------------------------------------------------------------------
# _restore_aiter_cpp_itfs_cache
# ---------------------------------------------------------------------------
def test_restore_round_trip(apply_tool, tmp_path: Path) -> None:
    pa_dir = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "pa"
    pa_dir.mkdir(parents=True)
    target = pa_dir / "pa_kernels.cuh"
    target.write_text("// fake\n")
    (pa_dir / "pa_ragged.py").write_text('MD_NAME = "pa_ragged"\n')

    build_dir = tmp_path / "build"
    _make_cache_dir(build_dir, "pa_ragged_HASHA", b"v0")

    backup = tmp_path / "backup"
    invalidate = apply_tool._invalidate_aiter_cpp_itfs_cache(
        target, backup, build_dir_override=build_dir,
    )
    assert invalidate["status"] == "ok"
    assert not (build_dir / "pa_ragged_HASHA").exists()

    # Simulate the re-baseline regenerating a fresh (patched) dir on top.
    _make_cache_dir(build_dir, "pa_ragged_HASHA", b"v1")

    restore = apply_tool._restore_aiter_cpp_itfs_cache(invalidate)
    assert restore["status"] == "ok"
    # Pre-patch (v0) content restored; regenerated v1 cleared first.
    assert (build_dir / "pa_ragged_HASHA" / "lib.so").read_bytes() == b"v0"
    assert not (backup / "cpp_itfs_cache" / "pa_ragged_HASHA").exists()


def test_restore_skips_when_no_backup(apply_tool) -> None:
    assert apply_tool._restore_aiter_cpp_itfs_cache(
        {"status": "skipped", "is_cpp_itfs": False},
    )["status"] == "skipped"
    assert apply_tool._restore_aiter_cpp_itfs_cache({})["status"] == "skipped"


# ---------------------------------------------------------------------------
# verify_cpp_itfs_rebuilt
# ---------------------------------------------------------------------------
def test_verify_noop_for_non_cpp_itfs(apply_tool) -> None:
    out = apply_tool.verify_cpp_itfs_rebuilt({"is_cpp_itfs": False})
    assert out["verified"] is True
    assert out["status"] == "skipped"


def test_verify_stale_when_no_fresh_lib_so(apply_tool, tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    # A lib.so that predates the invalidation -> stale.
    old = _make_cache_dir(build_dir, "pa_ragged_HASHA")
    invalidated_unix = time.time()
    old_mtime = invalidated_unix - 1000.0
    os.utime(old / "lib.so", (old_mtime, old_mtime))

    out = apply_tool.verify_cpp_itfs_rebuilt({
        "is_cpp_itfs": True,
        "build_dir": str(build_dir),
        "module_names": ["pa_ragged"],
        "invalidated_unix": invalidated_unix,
    })
    assert out["verified"] is False
    assert out["status"] == "stale"


def test_verify_ok_when_fresh_lib_so(apply_tool, tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    invalidated_unix = time.time()
    fresh = _make_cache_dir(build_dir, "pa_ragged_HASHA")
    new_mtime = invalidated_unix + 5.0
    os.utime(fresh / "lib.so", (new_mtime, new_mtime))

    out = apply_tool.verify_cpp_itfs_rebuilt({
        "is_cpp_itfs": True,
        "build_dir": str(build_dir),
        "module_names": ["pa_ragged"],
        "invalidated_unix": invalidated_unix,
    })
    assert out["verified"] is True
    assert out["status"] == "ok"
    assert any("pa_ragged_HASHA" in p for p in out["fresh_lib_so"])


# ---------------------------------------------------------------------------
# End-to-end apply_kernel_patch → revert_kernel_patch.
# ---------------------------------------------------------------------------
_CUH_V0 = "#include <hip/hip_runtime.h>\n__global__ void pa_kernel_v0() {}\n"
_CUH_V1 = "#include <hip/hip_runtime.h>\n__global__ void pa_kernel_v1() {}\n"

_FAKE_REBUILD_OK = {
    "status": "ok",
    "returncode": 0,
    "stdout_tail": "ok",
    "stderr_tail": "",
    "command": ["fake-rebuild"],
    "cwd": "",
}


def test_apply_then_revert_invalidates_and_restores_cpp_itfs_cache(
    apply_tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "sgl-workspace" / "aiter"
    pa_dir = repo / "csrc" / "cpp_itfs" / "pa"
    pa_dir.mkdir(parents=True)
    target = pa_dir / "pa_kernels.cuh"
    target.write_text(_CUH_V0)
    (pa_dir / "pa_ragged.py").write_text('MD_NAME = "pa_ragged"\n')

    patch_file = tmp_path / "patched.cuh"
    patch_file.write_text(_CUH_V1)

    # Hermetic cpp_itfs runtime cache via $AITER_ROOT_DIR -> $.../build.
    aiter_root = tmp_path / "aiterroot"
    monkeypatch.setenv("AITER_ROOT_DIR", str(aiter_root))
    build_dir = aiter_root / "build"
    _make_cache_dir(build_dir, "pa_ragged_HASHA", b"stale-pristine")
    _make_cache_dir(build_dir, "gemm_HASHC", b"unrelated")

    # Make the orthogonal @compile_ops jit/build invalidation a clean skip
    # (and never touch the real /sgl-workspace/aiter) by stubbing the jit
    # build-dir resolver to "aiter not importable". Patching the narrow
    # helper avoids globally mocking importlib.find_spec, which would break
    # unrelated lazy imports during apply.
    with patch.object(apply_tool, "_aiter_jit_build_dir", return_value=None), \
         patch.object(apply_tool, "_run_rebuild", return_value=dict(_FAKE_REBUILD_OK)):
        result = apply_tool.apply_kernel_patch(
            patch_path=patch_file,
            target_file=target,
            backup_root=tmp_path / "backups",
            kernel_id="k_pa",
            allow_unknown_target=True,
        )

    assert result["status"] == "ok", result
    backup = result["cpp_itfs_cache_backup"]
    assert backup["status"] == "ok"
    assert backup["is_cpp_itfs"] is True
    assert backup["module_names"] == ["pa_ragged"]
    # Patched source landed.
    assert "pa_kernel_v1" in target.read_text()
    # Stale pristine cache moved aside; unrelated module survived.
    assert not (build_dir / "pa_ragged_HASHA").exists()
    assert (build_dir / "gemm_HASHC" / "lib.so").is_file()

    # Simulate the re-baseline server recompiling the patched kernel.
    _make_cache_dir(build_dir, "pa_ragged_HASHA", b"fresh-patched")

    revert = apply_tool.revert_kernel_patch(result["manifest_path"])
    assert revert["status"] == "ok"
    # Source + runtime cache restored to pristine.
    assert "pa_kernel_v0" in target.read_text()
    assert (build_dir / "pa_ragged_HASHA" / "lib.so").read_bytes() == b"stale-pristine"


def test_non_cpp_itfs_apply_leaves_cpp_itfs_cache_untouched(
    apply_tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-cpp_itfs (sglang) target must NOT touch $HOME/.aiter/build."""
    target = tmp_path / "sgl-workspace" / "sglang" / "sgl-kernel" / "csrc" / "x.cu"
    target.parent.mkdir(parents=True)
    target.write_text(_CUH_V0)
    patch_file = tmp_path / "patched.cu"
    patch_file.write_text(_CUH_V1)

    aiter_root = tmp_path / "aiterroot"
    monkeypatch.setenv("AITER_ROOT_DIR", str(aiter_root))
    build_dir = aiter_root / "build"
    _make_cache_dir(build_dir, "pa_ragged_HASHA", b"must-survive")

    with patch.object(apply_tool, "_aiter_jit_build_dir", return_value=None), \
         patch.object(apply_tool, "_run_rebuild", return_value=dict(_FAKE_REBUILD_OK)):
        result = apply_tool.apply_kernel_patch(
            patch_path=patch_file,
            target_file=target,
            backup_root=tmp_path / "backups",
            kernel_id="k_sgl",
            allow_unknown_target=True,
        )

    assert result["status"] == "ok", result
    assert result["cpp_itfs_cache_backup"]["is_cpp_itfs"] is False
    # cpp_itfs runtime cache completely untouched for a non-cpp_itfs target.
    assert (build_dir / "pa_ragged_HASHA" / "lib.so").read_bytes() == b"must-survive"
