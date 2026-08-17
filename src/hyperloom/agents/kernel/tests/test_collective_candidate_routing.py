###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Routing rules for multi-GPU collective kernels.

Two populations must not be confused:

* **aiter / framework collectives** ship editable device source (``csrc/*.cu``),
  so they are legitimate optimization targets. TraceLens labels them
  ``AITER (vendor)`` in the source column, which is a placeholder rather than a
  path -- once that is rejected, the symbol resolves by name like any other
  kernel.
* **nccl / rccl collectives** are precompiled vendor binaries with no rewritable
  source. They must stay non-patchable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402
from _collective_names import kernel_name_implies_multigpu  # noqa: E402


# A real trace symbol: aiter's TP reduce-scatter, Itanium-mangled.
_AITER_RS = (
    "hipLaunchKernel->_ZN5aiter33reduce_scatter_cross_device_store"
    "IDF16bLi8EEEvPNS_8RankDataENS_11RankSignalsEjiiiP13__hip_stream_t (Synthetic Op)"
)
_AITER_SRC = "/sgl-workspace/aiter/csrc/kernels/all_reduce.cu"


def _candidate(name: str, source_file: str, **extra) -> dict:
    item = {
        "name": name,
        "source_file": source_file,
        "source_type": "hip_cpp",
        "kernel_repo": "/sgl-workspace/aiter",
        "is_multigpu": True,
    }
    item.update(extra)
    return item


# --- The vendor label is a placeholder, not a path ---------------------------


def test_aiter_vendor_label_is_not_a_source_path():
    assert not tl.looks_like_source_path("AITER (vendor)")
    assert not tl.looks_like_source_path("Triton (vendor)")
    assert not tl.looks_like_source_path("PyTorch Native (vendor)")
    assert not tl.looks_like_source_path("Tensile (vendor)")


def test_vendor_label_is_zeroed_so_name_resolution_can_run():
    item = _candidate(_AITER_RS, "AITER (vendor)")
    # Placeholder rejection lives in reject_non_path_source, called once at the
    # top of _finalize_candidates so the trace and grep tiers still get a turn.
    assert tl.reject_non_path_source(item) is True
    assert item["source_file"] == ""
    assert item["source_file_rejected"] == "AITER (vendor)"


# --- The mangled aiter symbol yields a usable search key ---------------------


def test_mangled_aiter_symbol_yields_the_bare_kernel_name():
    """Itanium demangling must surface the symbol, not the aiter namespace."""
    keywords = tl._candidate_keywords(_AITER_RS)
    assert keywords, "no search keyword extracted"
    assert keywords[0] == "reduce_scatter_cross_device_store"
    # The namespace token is blocklisted: it would match the whole repo.
    assert "aiter" not in keywords


def test_collective_symbol_is_not_mistaken_for_a_bare_launch_api():
    assert not tl.is_runtime_api_name(_AITER_RS)
    assert tl.is_runtime_api_name("hipLaunchKernel")


def test_collective_name_detection_agrees():
    assert kernel_name_implies_multigpu(_AITER_RS)


# --- aiter collectives are patchable once the source resolves ----------------


def test_aiter_collective_with_resolved_source_is_patchable():
    item = _candidate(_AITER_RS, _AITER_SRC)
    reusable, reason = tl.classify_patchability(item)
    assert reusable is True, reason
    assert "forge" in tl.recommend_backends({**item, "reusable_native_kernel": True})


def test_aiter_collective_survives_full_stamping():
    item = _candidate(_AITER_RS, _AITER_SRC)
    tl._stamp_candidate_metadata(item, None)
    assert item["reusable_native_kernel"] is True
    assert item["source_file"] == _AITER_SRC
    assert item["is_multigpu"] is True


# --- nccl / rccl stay non-patchable ------------------------------------------


def test_nccl_collective_stays_non_patchable():
    """A vendor binary has no rewritable source even with a path attached."""
    for name in (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
        "rccl_all_gather_kernel",
        "hipLaunchKernel->ncclDevKernel_ReduceScatter (Synthetic Op)",
    ):
        item = _candidate(name, "/sgl-workspace/rccl/src/collectives/all_reduce.cc")
        reusable, reason = tl.classify_patchability(item)
        assert reusable is False, f"{name} unexpectedly patchable"
        assert "non-patchable" in reason.lower()


def test_nccl_marker_check_precedes_source_resolution():
    """Even a perfectly good-looking source cannot rescue an nccl symbol."""
    item = _candidate("ncclDevKernel_AllReduce", _AITER_SRC)
    reusable, _reason = tl.classify_patchability(item)
    assert reusable is False
