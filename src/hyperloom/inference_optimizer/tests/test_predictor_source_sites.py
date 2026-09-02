# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the predictor's source-location lookup.

The documents under test are built with ``kernel_source_contract.make_entry`` /
``make_document`` rather than hand-written dicts, so a producer-side field
rename shows up here instead of being silently tolerated by a stale fixture.

The gate on ``unresolved`` / ``rejected_non_path_sentinel`` is the point of the
module: those tiers still emit entries, and forwarding their paths would put a
location in the prompt that does not exist.
"""

from __future__ import annotations

import json

from hyperloom.common.kernel_source_contract import (
    METHOD_ACTIVE_FINDER,
    METHOD_CURATED,
    METHOD_REJECTED,
    METHOD_UNRESOLVED,
    SOURCE_RESOLUTION_FILENAME,
    make_document,
    make_entry,
    validate_document,
)
from hyperloom.orchestrator.predictor import source_sites as ss


def _write(tmp_path, entries, *, schema_version=None):
    """Write a resolution document beside a fake analysis.md; return its path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    analysis_md = run_dir / "analysis.md"
    analysis_md.write_text("# Performance Analysis Report\n", encoding="utf-8")
    doc = make_document(entries, generated_by="test", model_name="m", framework="vllm")
    if schema_version is not None:
        doc["schema_version"] = schema_version
    (run_dir / SOURCE_RESOLUTION_FILENAME).write_text(json.dumps(doc), encoding="utf-8")
    return analysis_md


_RESOLVED = dict(
    kernel_id="k001",
    name="torch_gemm",
    gpu_pct=14.1,
    source_file="tuned_gemm.py",
    source_line=395,
    source_function="torch_gemm",
    method=METHOD_ACTIVE_FINDER,
)


def test_documents_under_test_satisfy_the_producer_contract():
    """Guard the fixtures themselves, so a contract change is not absorbed here."""
    doc = make_document([make_entry(**_RESOLVED)], generated_by="test")
    assert validate_document(doc) == []


def test_resolved_entry_is_indexed_by_both_id_and_name(tmp_path):
    sites = ss.load_source_sites(_write(tmp_path, [make_entry(**_RESOLVED)]))
    expected = {
        "source_file": "tuned_gemm.py",
        "source_line": 395,
        "source_function": "torch_gemm",
    }
    assert sites["k001"] == expected
    assert sites["torch_gemm"] == expected


def test_unresolved_and_rejected_tiers_are_dropped(tmp_path):
    """A failed tier still emits an entry; its path must not reach the prompt."""
    entries = [
        make_entry(kernel_id="k900", name="a", gpu_pct=1.0, method=METHOD_UNRESOLVED),
        make_entry(
            kernel_id="k901",
            name="b",
            gpu_pct=1.0,
            source_file="<unknown>",
            method=METHOD_REJECTED,
        ),
        make_entry(**_RESOLVED),
    ]
    sites = ss.load_source_sites(_write(tmp_path, entries))
    assert "k900" not in sites and "a" not in sites
    assert "k901" not in sites and "b" not in sites
    assert sites["torch_gemm"]["source_file"] == "tuned_gemm.py"


def test_entry_without_a_path_is_dropped_even_on_a_good_tier(tmp_path):
    entries = [
        make_entry(kernel_id="k001", name="a", gpu_pct=1.0, method=METHOD_CURATED, source_file=""),
    ]
    assert ss.load_source_sites(_write(tmp_path, entries)) == {}


def test_incompatible_major_version_is_ignored(tmp_path):
    path = _write(tmp_path, [make_entry(**_RESOLVED)], schema_version="2.0.0")
    assert ss.load_source_sites(path) == {}


def test_missing_artifact_and_empty_input_return_empty(tmp_path):
    assert ss.load_source_sites(None) == {}
    assert ss.load_source_sites("") == {}
    assert ss.load_source_sites(tmp_path / "nope" / "analysis.md") == {}


def test_malformed_json_returns_empty(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "analysis.md").write_text("x", encoding="utf-8")
    (run_dir / SOURCE_RESOLUTION_FILENAME).write_text("{not json", encoding="utf-8")
    assert ss.load_source_sites(run_dir / "analysis.md") == {}


def test_id_hit_wins_over_a_colliding_name(tmp_path):
    """An exact id match must not be shadowed by another row's name."""
    entries = [
        make_entry(
            kernel_id="k002",
            name="k001",
            gpu_pct=1.0,
            source_file="wrong.py",
            method=METHOD_CURATED,
        ),
        make_entry(
            kernel_id="k001",
            name="right",
            gpu_pct=1.0,
            source_file="right.py",
            method=METHOD_CURATED,
        ),
    ]
    sites = ss.load_source_sites(_write(tmp_path, entries))
    assert sites["k001"]["source_file"] == "right.py"


def test_attach_sites_matches_by_id_then_name(tmp_path):
    sites = ss.load_source_sites(_write(tmp_path, [make_entry(**_RESOLVED)]))
    rows = ss.attach_sites(
        [
            {"kernel_id": "k001", "name": "renamed_since", "source_file": "stale.py"},
            {"name": "torch_gemm"},
        ],
        sites,
    )
    assert rows[0]["source_file"] == "tuned_gemm.py"
    assert rows[0]["source_line"] == 395
    assert rows[1]["source_function"] == "torch_gemm"


def test_attach_sites_leaves_unmatched_rows_honestly_bare(tmp_path):
    """No frame is a normal outcome, not something to invent."""
    rows = ss.attach_sites([{"kernel_id": "kZZZ", "name": "x", "source_file": "kept.py"}], {})
    assert rows[0]["source_file"] == "kept.py"
    assert rows[0]["source_line"] is None
    assert rows[0]["source_function"] is None


def test_attach_sites_does_not_mutate_shared_state_rows(tmp_path):
    """Rows come off last_trace_analyze and are shared with other readers."""
    sites = ss.load_source_sites(_write(tmp_path, [make_entry(**_RESOLVED)]))
    original = {"kernel_id": "k001", "name": "torch_gemm"}
    ss.attach_sites([original], sites)
    assert original == {"kernel_id": "k001", "name": "torch_gemm"}


def test_artifact_is_found_in_the_run_dir_above_the_report(tmp_path):
    """The deterministic route's real layout, caught by the probe on a live session.

    TraceLens writes analysis.md into <run_dir>/tracelens/ while the resolution
    artifact stays in <run_dir>. Looking only beside the report found nothing,
    silently, on every deterministic-route session.
    """
    run_dir = tmp_path / "20260902T070926Z_tl-4b51a343"
    tracelens = run_dir / "tracelens"
    tracelens.mkdir(parents=True)
    (tracelens / "analysis.md").write_text("# report", encoding="utf-8")
    doc = make_document([make_entry(**_RESOLVED)], generated_by="test")
    (run_dir / SOURCE_RESOLUTION_FILENAME).write_text(json.dumps(doc), encoding="utf-8")

    sites = ss.load_source_sites(tracelens / "analysis.md")
    assert sites["k001"]["source_line"] == 395


def test_a_sibling_artifact_still_wins(tmp_path):
    """The bypass route writes both into one directory; that must keep working."""
    run_dir = tmp_path / "run"
    tracelens = run_dir / "tracelens"
    tracelens.mkdir(parents=True)
    (tracelens / "analysis.md").write_text("# report", encoding="utf-8")
    beside = make_document([make_entry(**{**_RESOLVED, "source_file": "beside.py"})], generated_by="t")
    above = make_document([make_entry(**{**_RESOLVED, "source_file": "above.py"})], generated_by="t")
    (tracelens / SOURCE_RESOLUTION_FILENAME).write_text(json.dumps(beside), encoding="utf-8")
    (run_dir / SOURCE_RESOLUTION_FILENAME).write_text(json.dumps(above), encoding="utf-8")

    assert ss.load_source_sites(tracelens / "analysis.md")["k001"]["source_file"] == "beside.py"
