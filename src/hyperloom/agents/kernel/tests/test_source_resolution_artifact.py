###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the kernel source-resolution artifact.

The artifact exists so that "where does this kernel live, and how do we know"
has one versioned answer on disk instead of a scatter of candidate fields. That
only holds if the schema is enforced, so these pin the envelope and per-entry
keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402

from hyperloom.common import kernel_source_contract as ksc  # noqa: E402


# --- envelope and entry contract -------------------------------------------


def test_document_carries_a_major_versioned_envelope():
    doc = ksc.make_document([], generated_by="test")
    assert doc["schema_version"] == ksc.SOURCE_RESOLUTION_SCHEMA_VERSION
    assert ksc.validate_document(doc) == []


def test_entry_always_carries_every_required_key():
    """Consumers read these without defaulting, so absence is a contract break."""
    entry = ksc.make_entry(kernel_id="k001", name="k", gpu_pct=1.0)
    for key in ksc.REQUIRED_ENTRY_KEYS:
        assert key in entry, key


def test_validate_reports_every_problem_not_just_the_first():
    doc = {"schema_version": ksc.SOURCE_RESOLUTION_SCHEMA_VERSION, "entries": [{}]}
    problems = ksc.validate_document(doc)
    assert any("generated_by" in p for p in problems)
    assert sum("missing required key" in p for p in problems) >= len(ksc.REQUIRED_ENTRY_KEYS)


def test_validate_rejects_a_foreign_major_version():
    doc = ksc.make_document([], generated_by="test")
    doc["schema_version"] = "9.0.0"
    assert any("different major" in p for p in ksc.validate_document(doc))


def test_validate_catches_a_path_that_claims_to_be_unresolved():
    doc = ksc.make_document(
        [ksc.make_entry(kernel_id="k1", name="n", gpu_pct=1.0, source_file="/a/b.py")],
        generated_by="test",
    )
    assert any("unresolved" in p for p in ksc.validate_document(doc))


def test_validate_catches_a_path_that_claims_to_be_rejected():
    """A rejection method cannot simultaneously advertise a resolved path."""
    doc = ksc.make_document(
        [
            ksc.make_entry(
                kernel_id="k1",
                name="n",
                gpu_pct=1.0,
                source_file="/a/b.py",
                method=ksc.METHOD_REJECTED,
            )
        ],
        generated_by="test",
    )
    assert any("rejected_non_path_sentinel" in p for p in ksc.validate_document(doc))


def test_validate_rejects_non_finite_and_out_of_range_confidence():
    """Artifact confidence must remain a finite probability."""
    for confidence in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1, "NaN"):
        doc = ksc.make_document(
            [
                ksc.make_entry(
                    kernel_id="k1",
                    name="n",
                    gpu_pct=1.0,
                    confidence=confidence,
                )
            ],
            generated_by="test",
        )
        assert any("invalid confidence" in problem for problem in ksc.validate_document(doc))


# --- projection from candidates ---------------------------------------------


def test_projection_classifies_each_resolution_tier():
    """Method is derived, since grep resolves without stamping anything."""
    got = tl.build_source_resolution_entries(
        [
            {
                "kernel_id": "k1",
                "name": "a",
                "gpu_pct": 9.0,
                "source_file": "/x/a.py",
                "source_resolution_method": "trace_python_stack",
            },
            {"kernel_id": "k2", "name": "b", "gpu_pct": 8.0, "source_file": "/x/b.py"},
            {"kernel_id": "k3", "name": "c", "gpu_pct": 7.0, "source_file": ""},
            {
                "kernel_id": "k4",
                "name": "d",
                "gpu_pct": 6.0,
                "source_file": "",
                "source_resolution_method": "rejected_non_path_sentinel",
                "source_file_rejected": "AITER (vendor)",
            },
            {
                "kernel_id": "k5",
                "name": "e",
                "gpu_pct": 5.0,
                "source_file": "/x/e.py",
                "source_resolution_method": "llm_fallback",
            },
        ]
    )
    by_id = {e["kernel_id"]: e for e in got}
    assert by_id["k1"]["method"] == ksc.METHOD_TRACE
    assert by_id["k2"]["method"] == ksc.METHOD_GREP
    assert by_id["k3"]["method"] == ksc.METHOD_UNRESOLVED
    assert by_id["k4"]["method"] == ksc.METHOD_REJECTED
    assert by_id["k4"]["rejected_value"] == "AITER (vendor)"
    assert by_id["k5"]["method"] == ksc.METHOD_LLM_FALLBACK


def test_written_artifact_satisfies_its_own_contract(tmp_path):
    out = tmp_path / ksc.SOURCE_RESOLUTION_FILENAME
    tl.write_source_resolution_artifact(
        [
            {
                "kernel_id": "k1",
                "name": "a",
                "gpu_pct": 5.0,
                "source_file": "/x/a.py",
                "source_resolution_method": "trace_python_stack",
            }
        ],
        out,
        framework="sglang",
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert ksc.validate_document(doc) == []
    assert doc["framework"] == "sglang"


# --- degrade, don't abort, against an older installed contract module -------
#
# tracelens_analysis.py runs as a standalone subprocess and imports the
# *installed* hyperloom, which need not match this checkout (cf.
# runtime/source-mirrors/). A contract module that predates the method-name
# constants this script reads must degrade to a fallback method name rather
# than raise AttributeError and kill the run.


def test_candidate_method_falls_back_without_the_constants(monkeypatch):
    """_candidate_resolution_method degrades to grep/unresolved with no contract."""
    monkeypatch.setattr(tl, "_KSC", None)
    assert tl._candidate_resolution_method({"source_file": "/repo/k.cu"}) == "name_grep"
    assert tl._candidate_resolution_method({}) == "unresolved"


def test_stamped_method_survives_a_missing_known_methods(monkeypatch):
    """A stamped method is echoed back even if KNOWN_METHODS is unavailable."""

    class _OldContract:
        # Newer constant present, but the KNOWN_METHODS set is absent.
        METHOD_ACTIVE_FINDER = "active_finder"

    monkeypatch.setattr(tl, "_KSC", _OldContract())
    item = {"source_resolution_method": "active_finder", "source_file": "/repo/k.cu"}
    # KNOWN_METHODS is missing -> the stamp is not recognized and falls to the
    # path-present grep verdict rather than raising.
    assert tl._candidate_resolution_method(item) == "name_grep"
