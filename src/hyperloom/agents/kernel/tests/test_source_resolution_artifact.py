###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the kernel source-resolution artifact and its review tier.

The artifact exists so that "where does this kernel live, and how do we know"
has one versioned answer on disk instead of a scatter of candidate fields. That
only holds if the schema is enforced, so these pin the envelope, the per-entry
keys, and the guard rails on the tier allowed to rewrite entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402
from _llm_source_review import build_review_prompt, review_resolution_document  # noqa: E402

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


def test_entry_carries_audit_history_when_supplied():
    """Optional review history must survive document reconstruction."""
    entry = ksc.make_entry(
        kernel_id="k001",
        name="k",
        gpu_pct=1.0,
        previous_source_file="/repo/old.py",
        previous_method=ksc.METHOD_TRACE,
    )
    assert entry["previous_source_file"] == "/repo/old.py"
    assert entry["previous_method"] == ksc.METHOD_TRACE


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
        doc = _doc_with(
            ksc.make_entry(
                kernel_id="k1",
                name="n",
                gpu_pct=1.0,
                confidence=confidence,
            )
        )
        assert any("invalid confidence" in problem for problem in ksc.validate_document(doc))


# --- projection from candidates ---------------------------------------------


def test_projection_classifies_each_resolution_tier():
    """Method is derived, since grep resolves without stamping anything."""
    got = tl.build_source_resolution_entries(
        [
            {"kernel_id": "k1", "name": "a", "gpu_pct": 9.0, "source_file": "/x/a.py",
             "source_resolution_method": "trace_python_stack"},
            {"kernel_id": "k2", "name": "b", "gpu_pct": 8.0, "source_file": "/x/b.py"},
            {"kernel_id": "k3", "name": "c", "gpu_pct": 7.0, "source_file": ""},
            {"kernel_id": "k4", "name": "d", "gpu_pct": 6.0, "source_file": "",
             "source_resolution_method": "rejected_non_path_sentinel",
             "source_file_rejected": "AITER (vendor)"},
            {"kernel_id": "k5", "name": "e", "gpu_pct": 5.0, "source_file": "/x/e.py",
             "source_resolution_method": "llm_fallback"},
        ]
    )
    by_id = {e["kernel_id"]: e for e in got}
    assert by_id["k1"]["method"] == ksc.METHOD_TRACE
    assert by_id["k2"]["method"] == ksc.METHOD_GREP
    assert by_id["k3"]["method"] == ksc.METHOD_UNRESOLVED
    assert by_id["k4"]["method"] == ksc.METHOD_REJECTED
    assert by_id["k4"]["rejected_value"] == "AITER (vendor)"
    assert by_id["k5"]["method"] == ksc.METHOD_LLM_FALLBACK
    assert by_id["k5"]["method"] != ksc.METHOD_LLM


def test_written_artifact_satisfies_its_own_contract(tmp_path):
    out = tmp_path / ksc.SOURCE_RESOLUTION_FILENAME
    tl.write_source_resolution_artifact(
        [{"kernel_id": "k1", "name": "a", "gpu_pct": 5.0, "source_file": "/x/a.py",
          "source_resolution_method": "trace_python_stack"}],
        out,
        framework="sglang",
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert ksc.validate_document(doc) == []
    assert doc["framework"] == "sglang"


# --- review tier guard rails -------------------------------------------------


def _doc_with(*entries):
    return ksc.make_document(list(entries), generated_by="test")


def _reply(revisions):
    def _complete(_prompt, _model, _timeout):
        return json.dumps({"revisions": revisions})

    return _complete


def test_review_may_unresolve_a_confidently_wrong_entry():
    """The failure the deterministic tiers cannot self-detect.

    aten::fill_ has no defining source; the trace tier attributes it to
    whichever business file called it, and that path passes every mechanical
    check. Only a reviewer looking at the whole entry can reject it.
    """
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="aten::fill_", gpu_pct=9.0,
                       source_file="/repo/moe.py", method=ksc.METHOD_TRACE)
    )
    out, notes = review_resolution_document(
        doc,
        framework_roots=("/repo",),
        complete=_reply([{"kernel_id": "k1", "action": "unresolve", "reason": "builtin"}]),
        model="m",
    )
    entry = out["entries"][0]
    assert entry["source_file"] == ""
    assert entry["method"] == ksc.METHOD_UNRESOLVED
    assert entry["previous_source_file"] == "/repo/moe.py"
    assert notes


def test_review_cannot_invent_a_path():
    """A rewrite must land somewhere verifiable, or the original stands."""
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file="/repo/a.py", method=ksc.METHOD_TRACE)
    )
    out, notes = review_resolution_document(
        doc,
        framework_roots=("/repo",),
        complete=_reply([{"kernel_id": "k1", "action": "rewrite",
                          "source_file": "/tmp/invented.py", "reason": "guess"}]),
        model="m",
    )
    entry = out["entries"][0]
    assert entry["source_file"] == "/repo/a.py"
    assert entry["method"] == ksc.METHOD_TRACE
    assert "previous_source_file" not in entry
    assert any("rejected unverifiable path" in n for n in notes)


def test_review_denies_rewrites_when_the_path_contract_is_unavailable(monkeypatch, tmp_path):
    """An unusable guard denies the rewrite instead of waving it through.

    ``path_is_acceptable`` is the only thing standing between a generated path
    and ``source_file``. When the contract module cannot be imported the tier
    has no way to verify anything, so a rewrite must fail closed -- otherwise
    losing the import silently turns the guard off.
    """
    from _llm_source_review import _apply_revision

    monkeypatch.setattr("_llm_source_review._KSC", None)
    entry = {"kernel_id": "k1", "source_file": "/repo/a.py", "method": "name_grep"}
    note = _apply_revision(
        entry,
        {"kernel_id": "k1", "action": "rewrite", "source_file": "/etc/passwd"},
        (str(tmp_path),),
    )
    assert entry["source_file"] == "/repo/a.py"
    assert entry["method"] == "name_grep"
    assert "previous_source_file" not in entry
    assert "path contract unavailable" in note


def test_review_stores_the_symlink_target_it_validated(tmp_path):
    """Keeping the link would let the location leave the roots after the check."""
    target = tmp_path / "implementation.py"
    target.write_text("def kernel(): pass\n", encoding="utf-8")
    link = tmp_path / "kernel.py"
    link.symlink_to(target)
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file=str(tmp_path / "wrong.py"), method=ksc.METHOD_TRACE)
    )
    out, _ = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply([{"kernel_id": "k1", "action": "rewrite",
                          "source_file": str(link), "reason": "defines it"}]),
        model="m",
    )
    assert out["entries"][0]["source_file"] == str(target)


def test_review_accepts_a_rewrite_to_a_file_that_exists(tmp_path):
    """A rewrite must land on a real file under a known root."""
    real = tmp_path / "right.py"
    real.write_text("def kernel(): pass\n", encoding="utf-8")
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file=str(tmp_path / "wrong.py"), method=ksc.METHOD_TRACE)
    )
    out, _ = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply([{"kernel_id": "k1", "action": "rewrite",
                          "source_file": str(real), "reason": "defines it"}]),
        model="m",
    )
    entry = out["entries"][0]
    assert entry["source_file"] == str(real)
    assert entry["method"] == ksc.METHOD_LLM
    assert entry["previous_source_file"].endswith("wrong.py")
    # Line and function described the old file; carrying them over would lie.
    assert entry["source_line"] is None
    assert entry["source_function"] == ""


def test_line_annotated_rewrite_is_stored_as_an_openable_path(tmp_path):
    """Call-site metadata is split from the path before downstream use."""
    real = tmp_path / "right.py"
    real.write_text("def kernel(): pass\n", encoding="utf-8")
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1",
            name="n",
            gpu_pct=9.0,
            source_file=str(tmp_path / "wrong.py"),
            method=ksc.METHOD_TRACE,
        )
    )
    out, _ = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply(
            [{
                "kernel_id": "k1",
                "action": "rewrite",
                "source_file": f"{real}(247): kernel",
            }]
        ),
        model="m",
    )
    entry = out["entries"][0]
    assert entry["source_file"] == str(real)
    assert entry["source_line"] == 247
    assert entry["source_function"] == "kernel"
    assert Path(entry["source_file"]).is_file()


def test_a_line_suffix_only_difference_is_not_a_rewrite(tmp_path):
    """Adding call-site syntax to the same file must not create fake history."""
    real = tmp_path / "same.py"
    real.write_text("def kernel(): pass\n", encoding="utf-8")
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1",
            name="n",
            gpu_pct=9.0,
            source_file=str(real),
            method=ksc.METHOD_TRACE,
        )
    )
    out, notes = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply(
            [{
                "kernel_id": "k1",
                "action": "rewrite",
                "source_file": f"{real}(1): kernel",
            }]
        ),
        model="m",
    )
    assert out["entries"][0]["source_file"] == str(real)
    assert "previous_source_file" not in out["entries"][0]
    assert notes == []


def test_line_annotated_paths_are_still_verifiable(tmp_path):
    """TraceLens reports call sites as "path.py(247): fn"; that is a real path.

    Measured on a live session, 29 of 36 resolved entries carried this suffix.
    Checking existence without stripping it would reject every one of them.
    """
    real = tmp_path / "moe.py"
    real.write_text("def kernel(): pass\n", encoding="utf-8")
    roots = (str(tmp_path),)
    assert ksc.path_is_acceptable(str(real), roots)
    assert ksc.path_is_acceptable(f"{real}(247)", roots)
    assert ksc.path_is_acceptable(f"{real}(247): kernel", roots)
    assert ksc.strip_line_suffix(f"{real}(247): kernel") == str(real)
    # Stripping must not resurrect a path that is simply absent.
    assert not ksc.path_is_acceptable(f"{tmp_path}/gone.py(1): f", roots)


def test_symlink_cannot_escape_a_framework_root(tmp_path):
    """Root containment is checked on resolved targets, not lexical paths."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def kernel(): pass\n", encoding="utf-8")
    escaping = root / "escaping.py"
    escaping.symlink_to(outside)
    inside = root / "inside.py"
    inside.write_text("def kernel(): pass\n", encoding="utf-8")
    internal = root / "internal.py"
    internal.symlink_to(inside)
    assert not ksc.path_is_acceptable(str(escaping), (str(root),))
    assert ksc.path_is_acceptable(str(internal), (str(root),))
    assert ksc.canonical_source_path(str(escaping), (str(root),)) == ""
    assert ksc.canonical_source_path(str(internal), (str(root),)) == str(inside)


def test_review_preview_does_not_read_an_escaping_current_symlink(monkeypatch, tmp_path):
    """A current entry is previewed only through a validated canonical target."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("review_outside_secret\n", encoding="utf-8")
    link = root / "kernel.py"
    link.symlink_to(outside)
    entry = ksc.make_entry(
        kernel_id="k1",
        name="n",
        gpu_pct=9.0,
        source_file=str(link),
        method=ksc.METHOD_TRACE,
    )
    reads: list[str] = []
    monkeypatch.setattr("_llm_source_review._preview", lambda path: reads.append(path) or "leaked")

    prompt = build_review_prompt(
        [entry],
        with_preview=True,
        framework_roots=(str(root),),
    )

    assert reads == []
    assert "review_outside_secret" not in prompt
    assert "leaked" not in prompt


def test_review_rejects_a_plausible_path_that_does_not_exist(tmp_path):
    """Under-a-root is not enough: the file must be there.

    A model can emit a perfectly plausible path inside the framework tree. If
    root membership alone qualified it, the backend would be handed a fabricated
    file -- the wrong-source failure this pipeline exists to prevent.
    """
    invented = tmp_path / "python" / "sglang" / "srt" / "does_not_exist.py"
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file="", method=ksc.METHOD_UNRESOLVED)
    )
    out, notes = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply([{"kernel_id": "k1", "action": "rewrite",
                          "source_file": str(invented), "reason": "looks right"}]),
        model="m",
    )
    assert out["entries"][0]["source_file"] == ""
    assert any("rejected unverifiable path" in n for n in notes)


def test_review_keeps_entries_below_the_gpu_floor_untouched():
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="tiny", gpu_pct=0.01,
                       source_file="/repo/a.py", method=ksc.METHOD_TRACE)
    )
    out, notes = review_resolution_document(
        doc, framework_roots=("/repo",), complete=_reply([]), model="m"
    )
    assert out["entries"][0]["source_file"] == "/repo/a.py"
    assert any("GPU share" in n for n in notes)


def test_unusable_reply_leaves_the_table_alone():
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file="/repo/a.py", method=ksc.METHOD_TRACE)
    )

    def _garbage(_p, _m, _t):
        return "not json at all"

    out, notes = review_resolution_document(
        doc, framework_roots=("/repo",), complete=_garbage, model="m"
    )
    assert out["entries"][0]["source_file"] == "/repo/a.py"
    assert any("unusable reply" in n for n in notes)


def test_call_failure_leaves_the_table_alone():
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file="/repo/a.py", method=ksc.METHOD_TRACE)
    )

    def _boom(_p, _m, _t):
        raise RuntimeError("gateway 401")

    out, notes = review_resolution_document(
        doc, framework_roots=("/repo",), complete=_boom, model="m"
    )
    assert out["entries"][0]["source_file"] == "/repo/a.py"
    assert any("llm call failed" in n for n in notes)


def test_review_call_error_is_redacted_from_notes_and_log():
    """Provider failures expose only stable exception metadata."""
    class ProviderError(RuntimeError):
        """Represent a provider response carrying sensitive details."""

        code = "gateway_timeout"

    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1",
            name="n",
            gpu_pct=9.0,
            source_file="/repo/a.py",
            method=ksc.METHOD_TRACE,
        )
    )

    def _boom(_prompt, _model, _timeout):
        raise ProviderError(
            "https://gateway.example/v1?token=query-secret "
            "Authorization: Bearer header-secret"
        )

    logs: list[str] = []
    out, notes = review_resolution_document(
        doc,
        framework_roots=("/repo",),
        complete=_boom,
        model="m",
        log=logs.append,
    )
    recorded = "\n".join([*notes, *logs])
    assert out["entries"][0]["source_file"] == "/repo/a.py"
    assert "ProviderError" in recorded
    assert "code=gateway_timeout" in recorded
    assert "gateway.example" not in recorded
    assert "query-secret" not in recorded
    assert "header-secret" not in recorded
    assert "Authorization" not in recorded


def test_review_discards_staged_changes_when_path_validation_raises(monkeypatch, tmp_path):
    """A later validation exception cannot commit an earlier revision."""
    replacement = tmp_path / "replacement.py"
    replacement.write_text("def kernel(): pass\n", encoding="utf-8")
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1",
            name="a",
            gpu_pct=9.0,
            source_file="/repo/a.py",
            method=ksc.METHOD_TRACE,
        ),
        ksc.make_entry(
            kernel_id="k2",
            name="b",
            gpu_pct=8.0,
            source_file="/repo/b.py",
            method=ksc.METHOD_TRACE,
        ),
    )
    original_entries = json.loads(json.dumps(doc["entries"]))

    class PathValidationError(RuntimeError):
        """Represent a failing path guard."""

    original_helper = ksc.canonical_source_path

    def _guard(path, roots):
        """Delegate ordinary paths and fail on the second revision."""
        if path == str(replacement):
            raise PathValidationError("https://secret.example/?token=path-secret")
        return original_helper(path, roots)

    monkeypatch.setattr(ksc, "canonical_source_path", _guard)
    out, notes = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply([
            {"kernel_id": "k1", "action": "unresolve"},
            {"kernel_id": "k2", "action": "rewrite", "source_file": str(replacement)},
        ]),
        model="m",
    )

    assert out["entries"] == original_entries
    assert out["llm_audit"]["review"]["outcome"] == "validation_error"
    assert any("PathValidationError" in note for note in notes)
    assert all("path-secret" not in note for note in notes)


def test_review_parse_exception_leaves_entries_untouched(monkeypatch):
    """A parser exception is advisory and cannot alter the source table."""
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1",
            name="n",
            gpu_pct=9.0,
            source_file="/repo/a.py",
            method=ksc.METHOD_TRACE,
        )
    )
    original_entries = json.loads(json.dumps(doc["entries"]))

    def _raise(_reply):
        """Simulate a parser failure with sensitive response context."""
        raise ValueError("https://secret.example/?token=parse-secret")

    monkeypatch.setattr("_llm_source_review.parse_revisions", _raise)
    out, notes = review_resolution_document(
        doc,
        framework_roots=("/repo",),
        complete=lambda *_args: "{}",
        model="m",
    )

    assert out["entries"] == original_entries
    assert any("ValueError" in note for note in notes)
    assert all("parse-secret" not in note for note in notes)


def test_revision_for_unknown_kernel_is_reported_not_applied():
    """An id nobody asked about rejects the batch rather than being skipped.

    A reply naming a kernel that was never sent is not a reply about the batch
    that was sent, so the entry it does not mention keeps its own resolution
    instead of being silently left to a partial review.
    """
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file="/repo/a.py", method=ksc.METHOD_TRACE)
    )
    out, notes = review_resolution_document(
        doc,
        framework_roots=("/repo",),
        complete=_reply([{"kernel_id": "ghost", "action": "unresolve"}]),
        model="m",
    )
    assert out["entries"][0]["source_file"] == "/repo/a.py"
    assert any("unknown kernel_id" in n for n in notes)


def test_one_unreadable_gpu_pct_does_not_disable_the_whole_review(tmp_path):
    """A single bad row must not cost every other row its review.

    The caller wraps this tier in a blanket handler, so an exception raised
    while ranking would surface as "review skipped" for the entire run. An
    unreadable share ranks as zero, which drops that row below the floor and
    leaves it untouched -- the rest of the table is still reviewed.
    """
    real = tmp_path / "right.py"
    real.write_text("def kernel(): pass\n", encoding="utf-8")
    hot = ksc.make_entry(kernel_id="k1", name="hot", gpu_pct=9.0,
                         source_file=str(tmp_path / "wrong.py"), method=ksc.METHOD_TRACE)
    broken = ksc.make_entry(kernel_id="k2", name="broken", gpu_pct=1.0,
                            source_file="/repo/b.py", method=ksc.METHOD_TRACE)
    broken["gpu_pct"] = "12%"
    out, _ = review_resolution_document(
        _doc_with(hot, broken),
        framework_roots=(str(tmp_path), "/repo"),
        complete=_reply([
            {"kernel_id": "k1", "action": "rewrite", "source_file": str(real)},
        ]),
        model="m",
    )
    assert out["entries"][0]["source_file"] == str(real)
    assert out["entries"][1]["source_file"] == "/repo/b.py"


def test_missing_review_id_rejects_the_entire_batch():
    """A truncated reply is not equivalent to an explicit keep decision."""
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1", name="a", gpu_pct=9.0,
            source_file="/repo/a.py", method=ksc.METHOD_TRACE,
        ),
        ksc.make_entry(
            kernel_id="k2", name="b", gpu_pct=8.0,
            source_file="/repo/b.py", method=ksc.METHOD_TRACE,
        ),
    )
    out, notes = review_resolution_document(
        doc,
        complete=_reply([{"kernel_id": "k1", "action": "unresolve"}]),
        model="m",
    )
    assert [entry["source_file"] for entry in out["entries"]] == [
        "/repo/a.py",
        "/repo/b.py",
    ]
    assert any("missing=" in note for note in notes)
    assert out["llm_audit"]["review"]["outcome"] == "protocol_error"


def test_duplicate_review_id_rejects_the_entire_batch():
    """Repeated revisions cannot overwrite their own audit history."""
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="k1", name="a", gpu_pct=9.0,
            source_file="/repo/a.py", method=ksc.METHOD_TRACE,
        )
    )
    revisions = [
        {"kernel_id": "k1", "action": "unresolve"},
        {"kernel_id": "k1", "action": "unresolve"},
    ]
    out, notes = review_resolution_document(
        doc,
        complete=_reply(revisions),
        model="m",
    )
    assert out["entries"][0]["source_file"] == "/repo/a.py"
    assert any("duplicate=" in note for note in notes)


def test_revision_for_an_unsent_entry_rejects_the_entire_batch():
    """An entry below the review floor cannot be modified by an extra ID."""
    doc = _doc_with(
        ksc.make_entry(
            kernel_id="hot", name="a", gpu_pct=9.0,
            source_file="/repo/a.py", method=ksc.METHOD_TRACE,
        ),
        ksc.make_entry(
            kernel_id="cold", name="b", gpu_pct=0.1,
            source_file="/repo/b.py", method=ksc.METHOD_TRACE,
        ),
    )
    revisions = [
        {"kernel_id": "hot", "action": "keep"},
        {"kernel_id": "cold", "action": "unresolve"},
    ]
    out, notes = review_resolution_document(
        doc,
        complete=_reply(revisions),
        model="m",
    )
    assert out["entries"][1]["source_file"] == "/repo/b.py"
    assert any("extra/unknown kernel_id" in note for note in notes)


# --- revisions must reach the pipeline, not just the artifact ---------------


def test_revision_is_folded_back_onto_the_candidate():
    """A review that only edits the artifact is inert.

    Dispatch reads kernel_candidates.json; the artifact is an audit view. Unless
    the revision is written back, the review tier changes nothing that runs.
    """
    candidates = [
        {"kernel_id": "k1", "name": "foo_kernel", "gpu_pct": 9.0,
         "source_file": "/sgl-workspace/aiter/csrc/wrong.cpp", "source_type": "hip_cpp"}
    ]
    entries = [
        ksc.make_entry(
            kernel_id="k1", name="foo_kernel", gpu_pct=9.0,
            source_file="/sgl-workspace/aiter/ops/right.py", method=ksc.METHOD_LLM,
        )
    ]
    entries[0]["previous_source_file"] = "/sgl-workspace/aiter/csrc/wrong.cpp"
    entries[0]["previous_method"] = ksc.METHOD_TRACE

    assert tl.apply_resolution_entries_to_candidates(entries, candidates) == 1
    got = candidates[0]
    assert got["source_file"] == "/sgl-workspace/aiter/ops/right.py"
    assert got["source_path"] == "/sgl-workspace/aiter/ops/right.py"
    assert got["source_resolution_previous_file"].endswith("wrong.cpp")
    assert got["source_resolution_previous_method"] == ksc.METHOD_TRACE
    # source_type drives reusable_native_kernel, so it must be recomputed.
    assert got["source_type"] == "python"
    assert "reusable_native_kernel" in got
    assert "skip_reason" in got


def test_unreviewed_entries_leave_candidates_untouched():
    """Only entries carrying previous_source_file were revised."""
    candidates = [{"kernel_id": "k1", "name": "n", "gpu_pct": 9.0,
                   "source_file": "/repo/a.py", "source_type": "python"}]
    entries = [ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                              source_file="/repo/a.py", method=ksc.METHOD_TRACE)]
    assert tl.apply_resolution_entries_to_candidates(entries, candidates) == 0
    assert candidates[0]["source_file"] == "/repo/a.py"
    assert "source_resolution_previous_file" not in candidates[0]


def test_unresolve_clears_the_candidate_source():
    """Dropping to unresolved must also clear it downstream, not only here."""
    candidates = [{"kernel_id": "k1", "name": "aten::fill_", "gpu_pct": 9.0,
                   "source_file": "/repo/moe.py", "source_type": "python"}]
    entries = [ksc.make_entry(kernel_id="k1", name="aten::fill_", gpu_pct=9.0,
                              source_file="", method=ksc.METHOD_UNRESOLVED)]
    entries[0]["previous_source_file"] = "/repo/moe.py"
    entries[0]["previous_method"] = ksc.METHOD_TRACE

    assert tl.apply_resolution_entries_to_candidates(entries, candidates) == 1
    got = candidates[0]
    assert got["source_file"] == ""
    assert got["reusable_native_kernel"] is False


def _aiter_candidate(**over):
    """A candidate carrying the curated metadata an op_to_source hit stamps."""
    item = {
        "kernel_id": "k1",
        "name": "fused_moe",
        "gpu_pct": 9.0,
        "source_file": "/repo/aiter/impl.cu",
        "source_type": "hip_cpp",
        "kernel_sources": ["/repo/aiter/impl.cu"],
        "kernel_kind": "aiter_ck",
        "source_framework": "aiter",
        "prebuilt_binary": "/repo/aiter/impl.co",
        "runtime_backend": "aiter",
        "launcher_source_file": "/repo/sglang/launch.py",
        "source_promoted_from_launcher": True,
        "tracelens_launcher_path": "/repo/sglang/launch.py(10): launch",
        "kernel_path": "/repo/sglang/launch.py(10): launch",
        "vendor_dispatch_wrapper": True,
        "runtime_generated_kernel": True,
        "source_resolution_confidence": 0.91,
        "op_to_source_kind": "dispatch",
        "op_to_source_patchable": True,
    }
    item.update(over)
    return item


def _rewrite_to(path):
    entry = ksc.make_entry(
        kernel_id="k1", name="fused_moe", gpu_pct=9.0,
        source_file=path, method=ksc.METHOD_LLM,
    )
    entry["previous_source_file"] = "/repo/aiter/impl.cu"
    entry["previous_method"] = ksc.METHOD_TRACE
    return entry


def test_a_rewrite_clears_metadata_describing_the_old_source():
    """Otherwise the candidate describes two sources and readers disagree.

    forge_submit._resolve_framework consults source_framework before it looks
    at source_file, so a stale value routes a vLLM rewrite as aiter.
    """
    item = _aiter_candidate()
    assert tl.apply_resolution_entries_to_candidates([_rewrite_to("/repo/vllm/new.py")], [item]) == 1
    assert item["source_file"] == "/repo/vllm/new.py"
    for stale in (
        "kernel_sources",
        "kernel_kind",
        "source_framework",
        "prebuilt_binary",
        "runtime_backend",
        "launcher_source_file",
        "source_promoted_from_launcher",
        "tracelens_launcher_path",
        "kernel_path",
        "vendor_dispatch_wrapper",
        "source_resolution_confidence",
        "op_to_source_kind",
        "op_to_source_patchable",
    ):
        assert stale not in item, stale
    assert item["runtime_generated_kernel"] is False


def test_a_stale_aiter_asm_kind_cannot_skip_a_rewritten_kernel():
    """classify_patchability reads kernel_kind; keeping it skips the new source."""
    item = _aiter_candidate(kernel_kind="aiter_asm")
    tl.apply_resolution_entries_to_candidates([_rewrite_to("/repo/vllm/new.py")], [item])
    assert "aiter_asm" not in str(item.get("skip_reason") or "")


def test_a_curated_resolution_is_not_overridable():
    """op_to_source.json names the real compute core; a file head cannot outrank it."""
    item = _aiter_candidate(
        op_to_source_status="resolved",
        source_resolution_method=ksc.METHOD_CURATED,
    )
    entry = _rewrite_to("/repo/vllm/new.py")
    assert tl.apply_resolution_entries_to_candidates([entry], [item]) == 0
    assert item["source_file"] == "/repo/aiter/impl.cu"
    assert item["kernel_kind"] == "aiter_ck"
    # The artifact must not advertise a revision that was not applied.
    assert entry["review_rejected"] == "curated_resolution_not_overridable"
    assert entry["source_file"] == "/repo/aiter/impl.cu"


@pytest.mark.parametrize("status", ["non_rewritable", "no_kernel"])
def test_a_curated_negative_verdict_is_not_overridable(status):
    """A curated terminal miss is authoritative even when it keeps a launcher."""
    item = _aiter_candidate(
        op_to_source_status=status,
        source_resolution_method=ksc.METHOD_CURATED,
    )
    entry = _rewrite_to("/repo/vllm/new.py")

    assert tl.apply_resolution_entries_to_candidates([entry], [item]) == 0
    assert item["source_file"] == "/repo/aiter/impl.cu"
    assert entry["review_rejected"] == "curated_resolution_not_overridable"


def test_audit_history_survives_artifact_rebuild(tmp_path, monkeypatch):
    """Applied revisions remain reversible after candidates are re-projected."""
    old = tmp_path / "old.py"
    new = tmp_path / "new.py"
    old.write_text("def old(): pass\n", encoding="utf-8")
    new.write_text("def new(): pass\n", encoding="utf-8")
    candidates = [{
        "kernel_id": "k1",
        "name": "kernel",
        "gpu_pct": 9.0,
        "source_file": str(old),
        "source_type": "python",
        "source_resolution_method": ksc.METHOD_TRACE,
    }]

    def _review(doc, *, log_path):
        """Inject one reviewed rewrite without making a network call."""
        assert log_path is None
        entry = doc["entries"][0]
        entry["previous_source_file"] = entry["source_file"]
        entry["previous_method"] = entry["method"]
        entry["source_file"] = str(new)
        entry["method"] = ksc.METHOD_LLM

    monkeypatch.setattr(tl, "_review_source_resolution", _review)
    out_path = tmp_path / ksc.SOURCE_RESOLUTION_FILENAME
    assert tl.write_source_resolution_artifact(candidates, out_path) == out_path
    entry = json.loads(out_path.read_text(encoding="utf-8"))["entries"][0]
    assert entry["source_file"] == str(new)
    assert entry["previous_source_file"] == str(old)
    assert entry["previous_method"] == ksc.METHOD_TRACE
    assert candidates[0]["source_resolution_previous_file"] == str(old)
    assert candidates[0]["source_resolution_previous_method"] == ksc.METHOD_TRACE


def test_review_runs_without_any_opt_in(tmp_path):
    """The tier is unconditional; nothing in the environment gates it."""
    real = tmp_path / "right.py"
    real.write_text("def kernel(): pass\n", encoding="utf-8")
    doc = _doc_with(
        ksc.make_entry(kernel_id="k1", name="n", gpu_pct=9.0,
                       source_file=str(tmp_path / "wrong.py"), method=ksc.METHOD_TRACE)
    )
    out, _ = review_resolution_document(
        doc,
        framework_roots=(str(tmp_path),),
        complete=_reply([{"kernel_id": "k1", "action": "rewrite",
                          "source_file": str(real), "reason": "defines it"}]),
        model="m",
    )
    assert out["entries"][0]["source_file"] == str(real)
