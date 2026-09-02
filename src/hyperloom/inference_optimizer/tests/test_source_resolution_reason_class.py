# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Why a hot kernel is undispatchable, as a class rather than prose.

The routing gate reduces a rich verdict to one boolean plus a sentence, and the
sentence had no consumer. These pin the classification that keeps the verdict
actionable, and the split that matters: classes nothing can rescue, versus the
one class worth spending a search budget on.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.common import kernel_source_contract as ksc


def _classify(skip_reason: str, *, reusable: Any = False, source_file: str = "/repo/k.py") -> str:
    return ksc.classify_skip_reason(reusable=reusable, skip_reason=skip_reason, source_file=source_file)


def test_a_reusable_kernel_with_a_resolved_source_is_resolved() -> None:
    assert _classify("", reusable=True) == ksc.CLASS_RESOLVED


def test_a_reusable_kernel_without_a_source_is_not_resolved() -> None:
    """Reusable-in-principle is not dispatch-ready without a resolved source:
    such a row belongs in the search-budget bucket, not the located count, or the
    "how many hotspots did we rescue" metric over-reports resolved."""
    assert _classify("source file not resolved", reusable=True, source_file="") == ksc.CLASS_SOURCE_NOT_RESOLVED
    # Even with no verdict text, a reusable row with no source is unresolved.
    assert _classify("", reusable=True, source_file="") == ksc.CLASS_SOURCE_NOT_RESOLVED


def test_a_reusable_kernel_with_a_none_source_does_not_raise() -> None:
    """A missing (None) source is treated as unresolved, never a crash."""
    assert ksc.classify_skip_reason(reusable=True, skip_reason="", source_file=None) == ksc.CLASS_SOURCE_NOT_RESOLVED


@pytest.mark.parametrize(
    ("skip_reason", "expected"),
    [
        ("launch API, not a kernel (no rewritable body)", ksc.CLASS_LAUNCH_API_ONLY),
        ("source file not resolved", ksc.CLASS_SOURCE_NOT_RESOLVED),
        ("vendor binary (no rewritable source)", ksc.CLASS_VENDOR_BINARY),
        ("vendor backend library 'tensile' (precompiled binary, no rewritable source)", ksc.CLASS_VENDOR_BINARY),
        ("source: aiter_asm prebuilt assembly compute-core (.co loaded by ...)", ksc.CLASS_VENDOR_BINARY),
        ("vendor dispatch wrapper at /repo/x.cu", ksc.CLASS_DISPATCH_SHIM),
        ("torch dispatch shim (no rewritable kernel body): /repo/x.py", ksc.CLASS_DISPATCH_SHIM),
        ("non-patchable kernel name marker 'nccl' in 'ncclKernel'", ksc.CLASS_NON_PATCHABLE_NAME),
        ("source: no rewritable implementation for this operator", ksc.CLASS_NOT_REWRITABLE_VERDICT),
    ],
)
def test_each_real_skip_reason_maps_to_its_class(skip_reason: str, expected: str) -> None:
    """Every branch the routing gate can return is covered here by its own text."""
    assert _classify(skip_reason) == expected


def test_classification_is_case_insensitive() -> None:
    assert _classify("SOURCE FILE NOT RESOLVED") == ksc.CLASS_SOURCE_NOT_RESOLVED


def test_an_unmapped_reason_is_unknown_not_a_guess() -> None:
    assert _classify("something nobody wrote yet") == ksc.CLASS_UNKNOWN


def test_no_reason_and_no_source_file_reads_as_unresolved() -> None:
    assert _classify("", source_file="") == ksc.CLASS_SOURCE_NOT_RESOLVED


def test_no_reason_but_a_source_file_is_unknown() -> None:
    """A non-reusable row with a path and no explanation is not a rescue case."""
    assert _classify("") == ksc.CLASS_UNKNOWN


def test_vendor_prebuilt_beats_the_generic_source_prefix() -> None:
    """Marker order matters: the specific phrase must win over 'source:'."""
    assert _classify("source: aiter_asm prebuilt assembly compute-core") == ksc.CLASS_VENDOR_BINARY


def test_unsalvageable_set_excludes_the_one_recoverable_class() -> None:
    """The split a consumer acts on: skip these for free, spend only on the other."""
    assert ksc.CLASS_SOURCE_NOT_RESOLVED not in ksc.UNSALVAGEABLE_REASON_CLASSES
    assert ksc.CLASS_UNKNOWN not in ksc.UNSALVAGEABLE_REASON_CLASSES
    assert ksc.UNSALVAGEABLE_REASON_CLASSES <= ksc.KNOWN_REASON_CLASSES


def test_every_class_is_known() -> None:
    for value in (
        ksc.CLASS_RESOLVED,
        ksc.CLASS_LAUNCH_API_ONLY,
        ksc.CLASS_VENDOR_BINARY,
        ksc.CLASS_DISPATCH_SHIM,
        ksc.CLASS_NON_PATCHABLE_NAME,
        ksc.CLASS_NOT_REWRITABLE_VERDICT,
        ksc.CLASS_SOURCE_NOT_RESOLVED,
        ksc.CLASS_UNKNOWN,
    ):
        assert value in ksc.KNOWN_REASON_CLASSES


def test_make_entry_always_carries_a_class() -> None:
    entry = ksc.make_entry(kernel_id="k001", name="k", gpu_pct=1.0)
    assert entry["reason_class"] in ksc.KNOWN_REASON_CLASSES


def test_make_entry_honours_an_explicit_class() -> None:
    entry = ksc.make_entry(
        kernel_id="k001",
        name="k",
        gpu_pct=1.0,
        reason_class=ksc.CLASS_VENDOR_BINARY,
    )
    assert entry["reason_class"] == ksc.CLASS_VENDOR_BINARY


def test_document_validation_rejects_an_unknown_class() -> None:
    entry = ksc.make_entry(kernel_id="k001", name="k", gpu_pct=1.0)
    entry["reason_class"] = "invented"
    doc = ksc.make_document([entry], generated_by="test")
    assert any("unknown reason_class" in problem for problem in ksc.validate_document(doc))


def test_reason_class_is_a_required_entry_key() -> None:
    assert "reason_class" in ksc.REQUIRED_ENTRY_KEYS


def test_schema_minor_was_bumped_not_the_major() -> None:
    """Additive field: a 1.0.0 reader must still accept the document."""
    assert ksc.SOURCE_RESOLUTION_SCHEMA_VERSION.split(".")[0] == "1"
    assert ksc.SOURCE_RESOLUTION_SCHEMA_VERSION != "1.0.0"


def _entry(kernel_id: str, reason_class: str, gpu_pct: float) -> dict[str, Any]:
    entry = ksc.make_entry(kernel_id=kernel_id, name=kernel_id, gpu_pct=gpu_pct)
    entry["reason_class"] = reason_class
    return entry


def test_summary_counts_located_and_groups_the_rest() -> None:
    summary = ksc.summarize_resolution(
        [
            _entry("k001", ksc.CLASS_RESOLVED, 30.0),
            _entry("k002", ksc.CLASS_SOURCE_NOT_RESOLVED, 15.0),
            _entry("k003", ksc.CLASS_VENDOR_BINARY, 10.0),
            _entry("k004", ksc.CLASS_LAUNCH_API_ONLY, 5.0),
        ]
    )
    assert summary["total"] == 4
    assert summary["located"] == 1
    assert summary["by_class"][ksc.CLASS_SOURCE_NOT_RESOLVED] == 1
    assert summary["undispatchable_gpu_pct"] == pytest.approx(30.0)
    assert summary["recoverable"] == 1
    assert summary["unsalvageable"] == 2


def test_summary_ignores_non_finite_gpu_share() -> None:
    entry = _entry("k001", ksc.CLASS_SOURCE_NOT_RESOLVED, 0.0)
    entry["gpu_pct"] = float("inf")
    assert ksc.summarize_resolution([entry])["undispatchable_gpu_pct"] == 0.0


def test_summary_tolerates_junk_rows() -> None:
    summary = ksc.summarize_resolution([_entry("k001", ksc.CLASS_RESOLVED, 1.0), "junk", None])
    assert summary["total"] == 1


def test_summary_of_nothing_is_all_zero() -> None:
    summary = ksc.summarize_resolution([])
    assert summary["total"] == 0
    assert summary["located"] == 0
    assert summary["by_class"] == {}
    assert summary["undispatchable_gpu_pct"] == 0.0
