###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass benchmark/test-file resolver (content-grep)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bypass_benchmark_resolver import find_benchmark_files, repo_root_from_source  # noqa: E402


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "aiter"
    (repo / "op_tests").mkdir(parents=True)
    (repo / "csrc" / "kernels").mkdir(parents=True)
    # name contains the op keyword.
    (repo / "op_tests" / "test_rmsnorm2d.py").write_text("def test():\n    rmsnorm(x, w)\n", encoding="utf-8")
    # CONTENT contains the op but the file NAME does not (the precision case a
    # name-only match would miss): silu_and_mul lives in test_activation.py.
    (repo / "op_tests" / "test_activation.py").write_text("def test():\n    silu_and_mul(out, x)\n", encoding="utf-8")
    # multi-GPU harness -> must be demoted below the single-GPU one.
    (repo / "op_tests" / "test_rmsnorm_multigpu.py").write_text("def test():\n    rmsnorm(x)  # multi_gpu\n", encoding="utf-8")
    # mentions the keyword but is NOT a test/bench file -> excluded.
    (repo / "op_tests" / "util_rmsnorm.py").write_text("def rmsnorm(x):\n    return x\n", encoding="utf-8")
    (repo / "csrc" / "kernels" / "rmsnorm_quant_kernels.cu").write_text("// rmsnorm\n", encoding="utf-8")
    (repo / "csrc" / "kernels" / "act.cu").write_text("// activation\n", encoding="utf-8")
    return repo


def test_repo_root_walks_up_to_benchmark_dir(tmp_path):
    repo = _fake_repo(tmp_path)
    src = str(repo / "csrc" / "kernels" / "rmsnorm_quant_kernels.cu")
    assert repo_root_from_source(src) == str(repo.resolve())
    assert repo_root_from_source("") == ""


def test_finds_named_benchmark_and_demotes_multigpu(tmp_path):
    repo = _fake_repo(tmp_path)
    src = str(repo / "csrc" / "kernels" / "rmsnorm_quant_kernels.cu")
    files = find_benchmark_files("aiter::rmsnorm", src)
    names = [Path(f).name for f in files]
    assert "test_rmsnorm2d.py" in names
    # non-test file excluded despite containing the keyword.
    assert "util_rmsnorm.py" not in names
    # single-GPU harness ranked before the multi-GPU one.
    if "test_rmsnorm_multigpu.py" in names:
        assert names.index("test_rmsnorm2d.py") < names.index("test_rmsnorm_multigpu.py")


def test_content_match_finds_test_when_name_lacks_op(tmp_path):
    # Precision: silu_and_mul's benchmark is test_activation.py (name has no
    # 'silu'); content grep must still find it (a name-only match would miss).
    repo = _fake_repo(tmp_path)
    src = str(repo / "csrc" / "kernels" / "act.cu")
    files = find_benchmark_files("sgl_kernel::silu_and_mul", src)
    assert any(Path(f).name == "test_activation.py" for f in files)


def test_no_source_or_repo_returns_empty(tmp_path):
    assert find_benchmark_files("aiter::rmsnorm", "") == []
    # a source with no benchmark-dir ancestor resolves to no repo -> [].
    lonely = tmp_path / "lonely" / "x.cu"
    lonely.parent.mkdir(parents=True)
    lonely.write_text("// x\n", encoding="utf-8")
    assert find_benchmark_files("aiter::rmsnorm", str(lonely)) == []
