###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the kernel source-resolution artifact and its review tier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl

from hyperloom.common import kernel_source_contract as ksc


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


def test_entry_carries_audit_history_when_supplied():
    """Optional review history must survive document reconstruction."""
    entry = ksc.make_entry(
        kernel_id="k001",
        name="k",
        gpu_pct=1.0,
        previous_source_file="/repo/old.py",
        previous_method=ksc.METHOD_SYMBOL_INDEX,
    )
    assert entry["previous_source_file"] == "/repo/old.py"
    assert entry["previous_method"] == ksc.METHOD_SYMBOL_INDEX


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


def test_validate_allows_a_non_patchable_gate_verdict_with_a_source():
    """A vendor GEMM's dispatcher resolves under gate_non_patchable -- valid."""
    doc = ksc.make_document(
        [
            ksc.make_entry(
                kernel_id="k1",
                name="Cijk_Ailk_Bljk",
                gpu_pct=1.0,
                source_file="/a/dispatcher.cu",
                method=ksc.METHOD_GATE_NON_PATCHABLE,
            )
        ],
        generated_by="test",
    )
    assert ksc.validate_document(doc) == []


def test_validate_allows_a_non_patchable_gate_verdict_without_a_source():
    """A bare non-patchable symbol carries no source -- also valid."""
    doc = ksc.make_document(
        [
            ksc.make_entry(
                kernel_id="k1",
                name="Cijk_Ailk_Bljk",
                gpu_pct=1.0,
                method=ksc.METHOD_GATE_NON_PATCHABLE,
            )
        ],
        generated_by="test",
    )
    assert ksc.validate_document(doc) == []


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


def test_projection_echoes_each_tracelens_resolver_method():
    """Each candidate carries a resolve_kernel_source method; an unknown one degrades to unresolved."""
    got = tl.build_source_resolution_entries(
        [
            {
                "kernel_id": "k1",
                "name": "a",
                "gpu_pct": 9.0,
                "source_file": "/x/a.py",
                "source_resolution_method": "triton_ast",
            },
            {
                "kernel_id": "k2",
                "name": "b",
                "gpu_pct": 8.0,
                "source_file": "/x/b.cu",
                "source_resolution_method": "symbol_index",
            },
            {
                "kernel_id": "k3",
                "name": "c",
                "gpu_pct": 7.0,
                "source_file": "",
                "source_resolution_method": "unresolved",
            },
            {
                "kernel_id": "k4",
                "name": "Cijk_Ailk_Bljk",
                "gpu_pct": 6.0,
                "source_file": "/x/dispatcher.cu",
                "source_resolution_method": "gate_non_patchable",
            },
            {
                "kernel_id": "k5",
                "name": "e",
                "gpu_pct": 5.0,
                "source_file": "/x/e.cu",
                # A method the current contract does not know degrades, never raises.
                "source_resolution_method": "some_retired_tier",
            },
        ]
    )
    by_id = {e["kernel_id"]: e for e in got}
    assert by_id["k1"]["method"] == ksc.METHOD_TRITON_AST
    assert by_id["k2"]["method"] == ksc.METHOD_SYMBOL_INDEX
    assert by_id["k3"]["method"] == ksc.METHOD_UNRESOLVED
    assert by_id["k4"]["method"] == ksc.METHOD_GATE_NON_PATCHABLE
    assert by_id["k4"]["source_file"] == "/x/dispatcher.cu"
    assert by_id["k5"]["method"] == ksc.METHOD_UNRESOLVED


def test_written_artifact_satisfies_its_own_contract(tmp_path):
    out = tmp_path / ksc.SOURCE_RESOLUTION_FILENAME
    tl.write_source_resolution_artifact(
        [
            {
                "kernel_id": "k1",
                "name": "a",
                "gpu_pct": 5.0,
                "source_file": "/x/a.py",
                "source_resolution_method": "symbol_index",
            }
        ],
        out,
        framework="sglang",
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert ksc.validate_document(doc) == []
    assert doc["framework"] == "sglang"


# --- degrade, don't abort, against an older installed contract module -------


def test_candidate_method_falls_back_without_the_constants(monkeypatch):
    """_candidate_resolution_method degrades to unresolved when the contract is absent."""
    monkeypatch.setattr(tl, "_KSC", None)
    assert tl._candidate_resolution_method({"source_file": "/repo/k.cu"}) == "unresolved"
    assert tl._candidate_resolution_method({}) == "unresolved"


def test_stamped_method_survives_a_missing_known_methods(monkeypatch):
    """An unrecognized stamp degrades to unresolved rather than raising."""

    class _OldContract:
        # No KNOWN_METHODS set present.
        METHOD_UNRESOLVED = "unresolved"

    monkeypatch.setattr(tl, "_KSC", _OldContract())
    item = {"source_resolution_method": "symbol_index", "source_file": "/repo/k.cu"}
    assert tl._candidate_resolution_method(item) == "unresolved"
