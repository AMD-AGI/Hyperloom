###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Guards that keep an unresolved-launcher sentinel from becoming a source_file.

TraceLens writes ``launcher_path = "Not found"`` for every Synthetic Op (a
device kernel with no cpu_op parent, e.g. a hand-written Triton kernel launched
straight from Python). That string used to survive parsing as a truthy
``source_file``, which skipped the grep fallback and made classify_patchability
reject the hottest kernels with "source not under a reusable framework root:
Not found". These tests pin the three guards that close that path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402
import tracelens_skill_runner as tsr  # noqa: E402


_HEADERS = [
    "operation",
    "args",
    "kernel path",
    "time (ms)",
    "%e2e",
    "count",
    "flops/byte",
    "efficiency",
    "bound",
]


def _row(kernel_path: str) -> dict | None:
    """Build one candidate from a data-table row carrying ``kernel_path``."""
    cells = [
        "hipModuleLaunchKernel->_mxfp8_grouped_gemm_kernel (Synthetic Op)",
        "",
        kernel_path,
        "15.671",
        "7.04",
        "114",
        "—",
        "—",
        "—",
    ]
    return tsr._row_to_candidate(
        _HEADERS,
        cells,
        category="other",
        rank=1,
        title="MXFP8 grouped GEMM",
        library="",
        impact={},
    )


# --- Guard 1: launcher placeholder normalisation ---------------------------


def test_not_found_sentinel_normalized_to_empty():
    cand = _row("Not found")
    assert cand is not None
    assert cand["source_file"] == ""
    assert cand["tracelens_launcher_path"] == ""


def test_not_found_sentinel_variants_normalized():
    for raw in ("Not found", "NOT FOUND", "not found", "not_found", "notfound", "  Not Found  "):
        cand = _row(raw)
        assert cand is not None, raw
        assert cand["source_file"] == "", raw


def test_legacy_dash_placeholders_still_normalized():
    for raw in ("-", "—", "–", "n/a", "unknown"):
        cand = _row(raw)
        assert cand is not None, raw
        assert cand["source_file"] == "", raw


def test_real_launcher_path_survives():
    real = "/sgl-workspace/sglang/python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py(124): _grouped_gemm_mxfp8"
    cand = _row(real)
    assert cand is not None
    assert cand["tracelens_launcher_path"] == real
    assert cand["source_file"] != ""


def test_not_found_is_in_shared_placeholder_set():
    assert "not found" in tsr._LAUNCHER_PATH_PLACEHOLDERS


# --- Guard 2: bare runtime-API names are not kernels ------------------------


def test_bare_runtime_api_names_detected():
    for name in (
        "hipGraphLaunch",
        "hipModuleLaunchKernel",
        "hipLaunchKernel",
        "cudaLaunchKernel",
        "cudaGraphLaunch",
    ):
        assert tl.is_runtime_api_name(name), name


def test_runtime_api_wrapping_a_kernel_is_not_blocked():
    """The wrapper prefix is stripped, so the real kernel must still resolve."""
    for name in (
        "hipModuleLaunchKernel->_mxfp8_grouped_gemm_kernel (Synthetic Op)",
        "hipGraphLaunch->_mxfp8_linear_kernel (Synthetic Op)",
    ):
        assert not tl.is_runtime_api_name(name), name


def test_grep_refuses_bare_runtime_api():
    for name in ("hipGraphLaunch", "hipModuleLaunchKernel", "hipLaunchKernel"):
        assert tl.locate_source_via_grep(name) == "", name


# --- Guard 3: a source_file must be path-shaped -----------------------------


def test_non_path_source_file_is_zeroed():
    """Any value lacking a source extension is rejected, not just 'Not found'.

    Covers the vendor labels TraceLens also emits in this field.
    """
    for sentinel in ("Not found", "N/A", "unknown", "TBD", "<unresolved>", "AITER (vendor)", "Triton (vendor)"):
        item = {"name": "k", "source_file": sentinel, "kernel_repo": "", "source_type": "python"}
        tl._stamp_candidate_metadata(item, None)
        assert item["source_file"] == "", sentinel
        assert item["source_file_rejected"] == sentinel
        assert item["source_resolution_method"] == "rejected_non_path_sentinel"


def test_path_shaped_but_absent_file_is_kept_and_flagged(tmp_path):
    """Keyed on shape, not presence: the analysis host need not own the
    serving container's filesystem."""
    missing = str(tmp_path / "does_not_exist.py")
    item = {"name": "k", "source_file": missing, "kernel_repo": "", "source_type": "python"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == missing
    assert item["source_file_missing_on_disk"] is True
    assert "source_file_rejected" not in item


def test_package_relative_path_survives():
    """TraceLens emits package-relative launchers such as sgl_kernel/moe.py."""
    item = {"name": "k", "source_file": "sgl_kernel/moe.py", "kernel_repo": "", "source_type": "python"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == "sgl_kernel/moe.py"
    assert "source_file_rejected" not in item


def test_existing_source_file_is_preserved_without_flags(tmp_path):
    real = tmp_path / "kernel.py"
    real.write_text("import triton\n", encoding="utf-8")
    item = {"name": "k", "source_file": str(real), "kernel_repo": "", "source_type": "python"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == str(real)
    assert "source_file_rejected" not in item
    assert "source_file_missing_on_disk" not in item


def test_empty_source_file_does_not_get_a_rejected_marker():
    item = {"name": "k", "source_file": "", "kernel_repo": "", "source_type": "unknown"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == ""
    assert "source_file_rejected" not in item


# --- Guard 4: trace-relative launcher paths are absolutized -----------------
#
# torch profiler records a frame path relative to the sys.path entry the module
# came from ("aiter/dist/x.py"). Patchability keys on an absolute framework
# root, so a relative path would be rejected for the wrong reason.


def test_relative_path_resolves_via_the_installed_package(monkeypatch):
    """The real case: the pinned checkout roots do not exist on this host.

    torch profiler records "vllm/models/x.py" relative to the sys.path entry the
    module came from. A pinned list cannot cover that -- the same package sits
    under /sgl-workspace in the serving image and under dist-packages on a wheel
    install -- so resolution has to locate the package at runtime.
    """
    # Pinned roots deliberately absent, exactly as on a wheel-install host.
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", ("/nonexistent/aiter/aiter",))
    tl._package_parent_dir.cache_clear()

    spec = importlib.util.find_spec("json")
    assert spec and spec.origin
    stdlib_dir = Path(spec.origin).parent

    got = tl.absolutize_launcher_path("json/decoder.py")
    assert got == str(stdlib_dir / "decoder.py")
    assert os.path.isfile(got)


def test_pinned_checkout_root_still_works_as_fallback(tmp_path, monkeypatch):
    """An editable checkout that this interpreter cannot import must still resolve."""
    pkg = tmp_path / "aiter" / "aiter" / "dist"
    pkg.mkdir(parents=True)
    (pkg / "custom_all_reduce.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", (str(tmp_path / "aiter" / "aiter"),))
    tl._package_parent_dir.cache_clear()

    got = tl.absolutize_launcher_path("aiter/dist/custom_all_reduce.py")
    assert got == str(pkg / "custom_all_reduce.py")


def test_non_identifier_head_is_not_probed_as_a_package(monkeypatch):
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", ())
    tl._package_parent_dir.cache_clear()
    assert tl.absolutize_launcher_path("not-an-identifier/mod.py") == "not-an-identifier/mod.py"


def test_absolute_launcher_path_is_returned_unchanged():
    assert tl.absolutize_launcher_path("/sgl-workspace/x.py") == "/sgl-workspace/x.py"


def test_unresolvable_relative_path_is_left_alone(monkeypatch, tmp_path):
    """Never fabricate: an unjoinable path stays as-is rather than becoming
    a plausible-looking path that does not exist."""
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", (str(tmp_path / "nope" / "nope"),))
    assert tl.absolutize_launcher_path("pkg/mod.py") == "pkg/mod.py"


def test_empty_path_is_safe():
    assert tl.absolutize_launcher_path("") == ""
