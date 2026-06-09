"""PART B2 — target-type -> toolchain cache-invalidation REGISTRY.

Covers the single dispatch in ``apply_kernel_patch.py`` that generalizes the
GH #458 (#459) aiter cpp_itfs invalidation + the pre-existing aiter
@compile_ops jit/build invalidation into a registry keyed by toolchain, plus
the two NEW toolchains:

* **Triton** -- ``$TRITON_CACHE_DIR`` (default ``~/.triton/cache``) move-aside
  so the integrate re-baseline recompiles the patched ``@triton.jit`` kernel
  (THE robustness fix for the editable Triton fused_moe win).
* **torch inductor** -- ``$TORCHINDUCTOR_CACHE_DIR`` / ``~/.cache/torch/
  inductor`` / ``/tmp/torchinductor_*`` move-aside.

Each toolchain entry is checked for: correct detection, invalidate/restore
round-trip, fresh-build verification, and a strict no-op for unhandled
targets. The aiter entries delegate to the existing functions so the #459
behaviour is preserved (those names are exercised by
``test_apply_kernel_patch_cpp_itfs_invalidation.py``); here we pin the
registry wiring + the new Triton/inductor entries + the integrate dispatch
helpers (``rebuild_env_for_apply_result`` / ``verify_rebuilt_for_apply_result``).
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
        "_apply_kernel_patch_registry_under_test", _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mk_cache(d: Path, *files: str, content: bytes = b"v0") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_bytes(content)
    return d


# ---------------------------------------------------------------------------
# Registry shape + toolchain_for classification.
# ---------------------------------------------------------------------------
def test_registry_has_all_four_toolchains(apply_tool) -> None:
    names = [e.name for e in apply_tool.CACHE_INVALIDATION_REGISTRY]
    assert names == ["aiter_compile_ops", "aiter_cpp_itfs", "triton", "torch_inductor"]
    keys = {e.manifest_key for e in apply_tool.CACHE_INVALIDATION_REGISTRY}
    assert keys == {
        "jit_build_backup", "cpp_itfs_cache_backup",
        "triton_cache_backup", "inductor_cache_backup",
    }


@pytest.mark.parametrize(
    "target,expected",
    [
        ("/sgl-workspace/aiter/csrc/cpp_itfs/pa/pa_kernels.cuh", "aiter_cpp_itfs"),
        ("/sgl-workspace/aiter/csrc/ck_gemm/gemm.cu", "aiter_compile_ops"),
        ("/x/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py", "triton"),
        ("/tmp/torchinductor_root/abc/kernel.py", "torch_inductor"),
        ("/sgl-workspace/sglang/x.cu", None),
        ("/sgl-workspace/sglang/plain_helper.py", None),
    ],
)
def test_toolchain_for_classifies(apply_tool, target, expected) -> None:
    assert apply_tool.toolchain_for(target) == expected


# ---------------------------------------------------------------------------
# Triton detection.
# ---------------------------------------------------------------------------
def test_triton_detection_by_source_markers(apply_tool, tmp_path) -> None:
    f = tmp_path / "kernel.py"
    f.write_text("import triton\nimport triton.language as tl\n@triton.jit\ndef k():\n    tl.load(p)\n")
    assert apply_tool._target_is_triton(f) is True


def test_triton_detection_by_path(apply_tool, tmp_path) -> None:
    d = tmp_path / "fused_moe_triton"
    d.mkdir()
    f = d / "fused_moe.py"
    f.write_text("# no explicit markers, but path says triton\n")
    assert apply_tool._target_is_triton(f) is True


def test_triton_rejects_non_py_and_plain_py(apply_tool, tmp_path) -> None:
    cu = tmp_path / "x.cu"
    cu.write_text("__global__ void k() {}\n")
    assert apply_tool._target_is_triton(cu) is False
    plain = tmp_path / "plain.py"
    plain.write_text("import numpy as np\ndef f():\n    return 1\n")
    assert apply_tool._target_is_triton(plain) is False


def test_triton_cache_dir_honours_env(apply_tool, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "tc"))
    assert apply_tool._triton_cache_dir() == tmp_path / "tc"
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert apply_tool._triton_cache_dir() == tmp_path / "home" / ".triton" / "cache"


# ---------------------------------------------------------------------------
# Triton invalidate / restore round-trip + verify.
# ---------------------------------------------------------------------------
def test_triton_invalidate_skips_non_triton(apply_tool, tmp_path) -> None:
    cu = tmp_path / "x.cu"
    cu.write_text("__global__ void k() {}\n")
    out = apply_tool._invalidate_triton_cache(
        cu, tmp_path / "backup", cache_dir_override=tmp_path / "tc",
    )
    assert out["status"] == "skipped"
    assert out["is_triton"] is False


def test_triton_invalidate_moves_cache_aside_and_restores(apply_tool, tmp_path) -> None:
    target = tmp_path / "fused_moe.py"
    target.write_text("import triton\n@triton.jit\ndef k():\n    pass\n")
    cache = tmp_path / "tritoncache"
    _mk_cache(cache / "HASHA", "kernel.hsaco", content=b"v0")

    out = apply_tool._invalidate_triton_cache(
        target, tmp_path / "backup", cache_dir_override=cache,
    )
    assert out["status"] == "ok"
    assert out["is_triton"] is True
    assert not cache.exists()  # moved aside
    assert (tmp_path / "backup" / "triton_cache" / "tritoncache" / "HASHA" / "kernel.hsaco").is_file()

    # re-baseline recompiles into a fresh cache dir
    _mk_cache(cache / "HASHB", "new.hsaco", content=b"v1")
    restore = apply_tool._restore_triton_cache(out)
    assert restore["status"] == "ok"
    assert (cache / "HASHA" / "kernel.hsaco").read_bytes() == b"v0"
    assert not (cache / "HASHB").exists()  # regenerated cleared on restore


def test_triton_invalidate_skips_when_cache_absent(apply_tool, tmp_path) -> None:
    target = tmp_path / "fused_moe.py"
    target.write_text("import triton\n@triton.jit\ndef k():\n    pass\n")
    out = apply_tool._invalidate_triton_cache(
        target, tmp_path / "backup", cache_dir_override=tmp_path / "nope",
    )
    assert out["status"] == "skipped"
    assert out["is_triton"] is True
    assert "does not exist" in out["reason"]


def test_triton_verify_noop_stale_and_ok(apply_tool, tmp_path) -> None:
    # non-triton -> verified True (no-op)
    assert apply_tool.verify_triton_rebuilt({"is_triton": False})["verified"] is True
    # moved a non-empty cache but nothing fresh -> stale
    cache = tmp_path / "tc"
    _mk_cache(cache, "old.hsaco")
    since = time.time()
    old = since - 1000.0
    os.utime(cache / "old.hsaco", (old, old))
    rec = {
        "is_triton": True, "status": "ok", "invalidated_unix": since,
        "moved": [{"src": str(cache), "backup_path": str(tmp_path / "b")}],
    }
    assert apply_tool.verify_triton_rebuilt(rec)["verified"] is False
    # a fresh artifact appears -> verified
    fresh = cache / "fresh.hsaco"
    fresh.write_bytes(b"x")
    os.utime(fresh, (since + 5, since + 5))
    assert apply_tool.verify_triton_rebuilt(rec)["verified"] is True


# ---------------------------------------------------------------------------
# torch inductor detection + invalidate/restore/verify.
# ---------------------------------------------------------------------------
def test_inductor_detection(apply_tool, tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text("import torch\nm = torch.compile(net)\n")
    assert apply_tool._target_is_inductor(f) is True
    assert apply_tool._target_is_inductor(Path("/tmp/torchinductor_root/x/k.py")) is True
    plain = tmp_path / "plain.py"
    plain.write_text("def f():\n    return 1\n")
    assert apply_tool._target_is_inductor(plain) is False


def test_inductor_cache_dirs_env_and_default(apply_tool, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "ic"))
    assert apply_tool._inductor_cache_dirs() == [tmp_path / "ic"]
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USER", "alice")
    dirs = apply_tool._inductor_cache_dirs()
    assert tmp_path / "home" / ".cache" / "torch" / "inductor" in dirs
    assert Path("/tmp/torchinductor_alice") in dirs


def test_inductor_invalidate_moves_and_restores(apply_tool, tmp_path) -> None:
    target = tmp_path / "compiled_model.py"
    target.write_text("import torch\ntorch.compile(x)\n")
    c1 = tmp_path / "ind1"
    _mk_cache(c1 / "frag", "out.py", content=b"v0")
    out = apply_tool._invalidate_torch_inductor_cache(
        target, tmp_path / "backup", cache_dirs_override=[c1],
    )
    assert out["status"] == "ok"
    assert out["is_inductor"] is True
    assert not c1.exists()
    restore = apply_tool._restore_torch_inductor_cache(out)
    assert restore["status"] == "ok"
    assert (c1 / "frag" / "out.py").read_bytes() == b"v0"


def test_inductor_invalidate_skips_non_inductor(apply_tool, tmp_path) -> None:
    plain = tmp_path / "plain.py"
    plain.write_text("def f():\n    return 1\n")
    out = apply_tool._invalidate_torch_inductor_cache(
        plain, tmp_path / "backup", cache_dirs_override=[tmp_path / "ic"],
    )
    assert out["status"] == "skipped"
    assert out["is_inductor"] is False


# ---------------------------------------------------------------------------
# @compile_ops + cpp_itfs registry entries delegate to the existing funcs.
# ---------------------------------------------------------------------------
def test_compile_ops_entry_delegates(apply_tool) -> None:
    entry = next(e for e in apply_tool.CACHE_INVALIDATION_REGISTRY if e.name == "aiter_compile_ops")
    assert entry.invalidate is apply_tool._invalidate_aiter_jit_build
    assert entry.restore is apply_tool._restore_aiter_jit_build
    assert entry.requires_compiled is True
    assert entry.gates_keep is False
    assert entry.rebuild_env == {}


def test_cpp_itfs_entry_delegates_and_preserves_names(apply_tool) -> None:
    entry = next(e for e in apply_tool.CACHE_INVALIDATION_REGISTRY if e.name == "aiter_cpp_itfs")
    assert entry.invalidate is apply_tool._invalidate_aiter_cpp_itfs_cache
    assert entry.restore is apply_tool._restore_aiter_cpp_itfs_cache
    assert entry.verify is apply_tool.verify_cpp_itfs_rebuilt
    assert entry.requires_compiled is True
    assert entry.gates_keep is True
    assert entry.rebuild_env == {"AITER_REBUILD": "1"}
    # #459 public names stay importable from apply_kernel_patch.
    for name in (
        "_target_is_in_aiter_cpp_itfs", "_cpp_itfs_module_names",
        "_invalidate_aiter_cpp_itfs_cache", "_restore_aiter_cpp_itfs_cache",
        "verify_cpp_itfs_rebuilt",
    ):
        assert callable(getattr(apply_tool, name))


# ---------------------------------------------------------------------------
# integrate dispatch helpers.
# ---------------------------------------------------------------------------
def test_rebuild_env_for_apply_result(apply_tool) -> None:
    assert apply_tool.rebuild_env_for_apply_result(
        {"cpp_itfs_cache_backup": {"is_cpp_itfs": True}},
    ) == {"AITER_REBUILD": "1"}
    assert apply_tool.rebuild_env_for_apply_result(
        {"cpp_itfs_cache_backup": {"is_cpp_itfs": False}},
    ) == {}
    # triton move-aside needs no env
    assert apply_tool.rebuild_env_for_apply_result(
        {"triton_cache_backup": {"is_triton": True, "status": "ok"}},
    ) == {}
    assert apply_tool.rebuild_env_for_apply_result({}) == {}


def test_verify_rebuilt_for_apply_result_is_noop_off_paths(apply_tool) -> None:
    out = apply_tool.verify_rebuilt_for_apply_result({
        "cpp_itfs_cache_backup": {"is_cpp_itfs": False},
        "triton_cache_backup": {"is_triton": False},
    })
    assert out["verified"] is True
    assert out["status"] == "skipped"


def test_verify_rebuilt_for_apply_result_gates_triton(apply_tool, tmp_path) -> None:
    cache = tmp_path / "tc"
    _mk_cache(cache, "old.hsaco")
    since = time.time()
    old = since - 500.0
    os.utime(cache / "old.hsaco", (old, old))
    apply_result = {
        "triton_cache_backup": {
            "is_triton": True, "status": "ok", "invalidated_unix": since,
            "moved": [{"src": str(cache), "backup_path": str(tmp_path / "b")}],
        },
    }
    out = apply_tool.verify_rebuilt_for_apply_result(apply_result)
    assert out["verified"] is False
    assert "triton" in out["per_toolchain"]


# ---------------------------------------------------------------------------
# End-to-end apply -> revert for a Triton target (THE fused_moe robustness).
# ---------------------------------------------------------------------------
_TRITON_V0 = "import triton\nimport triton.language as tl\n\n@triton.jit\ndef fused_moe_kernel():\n    pass\n"
_TRITON_V1 = "import triton\nimport triton.language as tl\n\n@triton.jit\ndef fused_moe_kernel():\n    return 1\n"
_FAKE_REBUILD_OK = {"status": "ok", "returncode": 0, "stdout_tail": "ok", "stderr_tail": "", "command": ["x"], "cwd": ""}


def test_triton_apply_then_revert_e2e(apply_tool, tmp_path, monkeypatch) -> None:
    target = tmp_path / "sglang" / "fused_moe_triton" / "fused_moe.py"
    target.parent.mkdir(parents=True)
    target.write_text(_TRITON_V0)
    patch_file = tmp_path / "patched.py"
    patch_file.write_text(_TRITON_V1)

    cache = tmp_path / "tritoncache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache))
    _mk_cache(cache / "HASHA", "kernel.hsaco", content=b"stale")

    with patch.object(apply_tool, "_run_rebuild", return_value=dict(_FAKE_REBUILD_OK)):
        result = apply_tool.apply_kernel_patch(
            patch_path=patch_file, target_file=target,
            backup_root=tmp_path / "backups", kernel_id="k_fmoe",
            allow_unknown_target=True,
        )
    assert result["status"] == "ok", result
    assert result["toolchain"] == "triton"
    tb = result["triton_cache_backup"]
    assert tb["is_triton"] is True and tb["status"] == "ok"
    assert "return 1" in target.read_text()
    assert not cache.exists()  # stale triton cache moved aside
    # cpp_itfs / jit / inductor untouched for a .py triton target
    assert result["cpp_itfs_cache_backup"]["is_cpp_itfs"] is False
    assert result["inductor_cache_backup"]["is_inductor"] is False

    # re-baseline recompiles
    _mk_cache(cache / "HASHB", "fresh.hsaco", content=b"fresh")
    revert = apply_tool.revert_kernel_patch(result["manifest_path"])
    assert revert["status"] == "ok"
    assert "return 1" not in target.read_text()  # source restored to v0
    assert (cache / "HASHA" / "kernel.hsaco").read_bytes() == b"stale"


def test_unhandled_py_target_is_full_noop(apply_tool, tmp_path, monkeypatch) -> None:
    """A plain (non-triton, non-inductor) .py target leaves every toolchain
    cache untouched -- the registry is a strict no-op off its paths."""
    target = tmp_path / "sglang" / "plain_layer.py"
    target.parent.mkdir(parents=True)
    target.write_text("import numpy as np\n\ndef forward(x):\n    return x\n")
    patch_file = tmp_path / "patched.py"
    patch_file.write_text("import numpy as np\n\ndef forward(x):\n    return x + 1\n")

    cache = tmp_path / "tritoncache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache))
    _mk_cache(cache / "HASHA", "kernel.hsaco", content=b"must-survive")

    result = apply_tool.apply_kernel_patch(
        patch_path=patch_file, target_file=target,
        backup_root=tmp_path / "backups", kernel_id="k_plain",
        allow_unknown_target=True,
    )
    assert result["status"] == "ok", result
    assert result["toolchain"] is None
    assert result["triton_cache_backup"]["is_triton"] is False
    assert result["cpp_itfs_cache_backup"]["is_cpp_itfs"] is False
    assert result["inductor_cache_backup"]["is_inductor"] is False
    # triton cache completely untouched
    assert (cache / "HASHA" / "kernel.hsaco").read_bytes() == b"must-survive"
