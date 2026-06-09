"""chaojhou review (point a) — tightened Triton / inductor cache detection.

These tests pin the REVIEW-improved detection (NOT the naive version): a
toolchain cache is only invalidated for files that actually OWN a compile-cache
entry, identified by the kernel-defining decorator / call (or a toolchain
source-path marker), NOT by a bare ``import triton`` / ``tl.load`` /
``torch.compile`` *substring*. Matching the bare substring would move the WHOLE
Triton (or inductor) cache aside for an unrelated edit -- and then make
integrate hard-gate KEEP on a "stale" verify that never had a patched kernel to
recompile -- so the over-broad behaviour is explicitly rejected here.

Run standalone (kernel-agent sandbox layout, no orchestrator on path)::

    cd kernel-agent/tools && python -m pytest \
        test_cache_invalidation_detection_tightening.py -q
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent / "apply_kernel_patch.py"


@pytest.fixture(scope="module")
def apply_tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_apply_kernel_patch_detect_tighten_under_test", _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Triton: bare ``import triton`` / ``tl.*`` must NOT be classified as a Triton
# kernel (the over-broad-detect negative required by review point a).
# ---------------------------------------------------------------------------
def test_bare_import_triton_without_jit_is_not_triton(apply_tool, tmp_path: Path) -> None:
    """A plain helper that ``import triton`` and references ``tl.load`` but
    defines NO ``@triton.jit`` kernel (and is not under a triton path) must
    NOT be detected as a Triton target."""
    f = tmp_path / "uses_triton_but_no_kernel.py"
    f.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "# references tl helpers but defines no compiled kernel decorator\n"
        "def describe(x):\n"
        "    return tl.load, triton.__version__\n"
    )
    assert apply_tool._target_is_triton(f) is False


def test_bare_import_triton_does_not_invalidate_cache(apply_tool, tmp_path: Path) -> None:
    """The whole-cache move-aside MUST NOT fire for a bare-``import triton``
    file: the Triton compile cache is left completely intact."""
    f = tmp_path / "uses_triton_but_no_kernel.py"
    f.write_text("import triton\nimport triton.language as tl\nx = tl.load\n")

    cache = tmp_path / "tritoncache"
    (cache / "HASHA").mkdir(parents=True)
    (cache / "HASHA" / "kernel.hsaco").write_bytes(b"must-survive")

    out = apply_tool._invalidate_triton_cache(
        f, tmp_path / "backup", cache_dir_override=cache,
    )
    assert out["status"] == "skipped"
    assert out["is_triton"] is False
    # Cache untouched -- not moved aside.
    assert (cache / "HASHA" / "kernel.hsaco").read_bytes() == b"must-survive"


def test_jit_decorator_is_triton_positive_control(apply_tool, tmp_path: Path) -> None:
    """Positive control: a real ``@triton.jit`` kernel IS detected, so the
    negative cases above are tightening, not blanket-disabling, detection."""
    f = tmp_path / "real_kernel.py"
    f.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
    )
    assert apply_tool._target_is_triton(f) is True


def test_autotuned_jit_kernel_is_triton(apply_tool, tmp_path: Path) -> None:
    """``@triton.autotune`` (which only ever decorates a @triton.jit kernel)
    is also recognized."""
    f = tmp_path / "autotuned.py"
    f.write_text(
        "import triton\n"
        "@triton.autotune(configs=[], key=['n'])\n"
        "@triton.jit\n"
        "def k():\n    pass\n"
    )
    assert apply_tool._target_is_triton(f) is True


# ---------------------------------------------------------------------------
# Inductor "same spirit": a bare ``torch.compile`` / ``torch._inductor`` prose
# mention (no actual call/decorator/import) must NOT be detected as inductor.
# ---------------------------------------------------------------------------
def test_bare_torch_compile_mention_is_not_inductor(apply_tool, tmp_path: Path) -> None:
    f = tmp_path / "mentions_only.py"
    f.write_text(
        "import torch\n"
        "# NOTE: torch.compile and torch._inductor are not used here yet\n"
        '"""We may adopt torch.compile later."""\n'
        "def forward(x):\n"
        "    return torch.relu(x)\n"
    )
    assert apply_tool._target_is_inductor(f) is False


def test_real_torch_compile_call_is_inductor(apply_tool, tmp_path: Path) -> None:
    f = tmp_path / "compiled.py"
    f.write_text("import torch\nmodel = torch.compile(net)\n")
    assert apply_tool._target_is_inductor(f) is True


def test_inductor_import_is_inductor(apply_tool, tmp_path: Path) -> None:
    f = tmp_path / "uses_inductor.py"
    f.write_text("from torch._inductor import config\nconfig.max_autotune = True\n")
    assert apply_tool._target_is_inductor(f) is True
