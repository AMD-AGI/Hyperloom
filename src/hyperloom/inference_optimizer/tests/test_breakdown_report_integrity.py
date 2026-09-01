# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report integrity: reported counts must mean what the schema says they mean.

These tests pin contracts the breakdown pipeline previously stated in comments
but never enforced:

* ``CapabilityEntry.keeps`` counts kernels adopted at *integrate*. A KEEP that
  only cleared the micro benchmark is not an adoption and must not inflate it.
* ``keeps`` counts distinct kernels, so a kernel re-tried across runs is one
  adoption. ``attempts`` deliberately stays a count of invocation rows -- "how
  many tries did this take" is the question it answers -- so the two fields
  carry different units on purpose.
* Only a ``KEEP`` verdict is an adoption. ``NEEDS_REVIEW`` and a missing
  decision mean the verdict is not in yet, and a lane holding one must not
  read as ``kept``.
* Timeline de-duplication must not fold two different tasks into one row just
  because they share an action and a wall-clock second.
* A section skipped for lack of data still carries evidence ("this never ran");
  that evidence must reach the reader instead of being dropped silently.
* A sweep variant whose ``benchmark_report.json`` is missing, truncated, empty,
  or not a JSON object is ``failed`` / ``skipped``, never silently ``ok``.
  ``abort_reason.json`` without a report is ``failed``. ``ok`` does not require
  selectable throughput.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown import collectors
from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder
from hyperloom.inference_optimizer.breakdown.reporters import cross_section, llm_prompt
from hyperloom.inference_optimizer.breakdown.reporters._renderers import sweep as sw
from hyperloom.inference_optimizer.breakdown.reporters.base import RenderedSection
from hyperloom.inference_optimizer.breakdown.schema import SweepPoint


def _integrate_state(kernel_id: str, decision: str, gain: float | None = None) -> dict[str, Any]:
    """Build a ``state`` carrying one integrate verdict for ``kernel_id``.

    Args:
        kernel_id (str): Kernel the integrate attempt refers to.
        decision (str): Terminal integrate decision, e.g. ``"KEEP"`` / ``"REVERT"``.
        gain (float | None): Best end-to-end gain percentage, if any.

    Returns:
        dict[str, Any]: A minimal ``state.json`` shaped mapping.
    """
    return {
        "kernel_integrate_attempts": {
            "attempt-1": {
                "kernel_id": kernel_id,
                "last_decision": decision,
                "best_gain_pct": gain,
            }
        }
    }


# Capability counting


def test_micro_only_keep_is_not_an_integrate_adoption() -> None:
    """A KEEP with no integrate record is micro-only and must not count as adopted."""
    invs = [{"kernel_id": "k1", "decision": "KEEP", "micro_speedup": 1.4}]

    cap = collectors.collect_capability_summary({}, [], [], forge_invocations=invs)

    assert cap["forge"]["attempts"] == 1
    assert cap["forge"]["keeps"] == 0, "micro-only KEEP must not be reported as an integrate adoption"
    assert cap["forge"]["micro_only_keeps"] == 1
    # Not "kept": nothing was adopted end-to-end, so the executive summary must
    # not advertise this lane as a capability that paid off.
    assert cap["forge"]["status"] == "attempted"


def test_integrate_adoption_is_counted_as_keep() -> None:
    """A KEEP confirmed by an integrate verdict is a real adoption."""
    invs = [{"kernel_id": "k1", "decision": "KEEP"}]

    cap = collectors.collect_capability_summary(_integrate_state("k1", "KEEP", 7.5), [], [], forge_invocations=invs)

    assert cap["forge"]["keeps"] == 1
    assert cap["forge"]["status"] == "kept"
    assert cap["forge"]["e2e_gain_pct"] == 7.5
    assert cap["forge"].get("micro_only_keeps", 0) == 0


def test_same_kernel_retried_across_runs_counts_once() -> None:
    """One kernel stamped KEEP in several runs is one adoption, not several."""
    invs = [
        {"kernel_id": "k1", "decision": "KEEP", "run_id": "run-a"},
        {"kernel_id": "k1", "decision": "KEEP", "run_id": "run-b"},
    ]

    cap = collectors.collect_capability_summary(_integrate_state("k1", "KEEP", 3.0), [], [], forge_invocations=invs)

    assert cap["forge"]["keeps"] == 1, "distinct kernels, not invocation rows"


def test_reverted_kernel_counted_once_across_runs() -> None:
    """A kernel reverted at integrate stays a single revert across retries."""
    invs = [
        {"kernel_id": "k1", "decision": "KEEP", "run_id": "run-a"},
        {"kernel_id": "k1", "decision": "KEEP", "run_id": "run-b"},
    ]

    cap = collectors.collect_capability_summary(_integrate_state("k1", "REVERT"), [], [], forge_invocations=invs)

    assert cap["forge"]["keeps"] == 0
    assert cap["forge"]["reverts"] == 1
    assert cap["forge"]["status"] == "reverted"


def test_micro_only_and_adopted_kernels_are_tallied_separately() -> None:
    """Mixed lane: one adopted kernel, one micro-only, counted in their own buckets."""
    invs = [
        {"kernel_id": "k1", "decision": "KEEP"},
        {"kernel_id": "k2", "decision": "KEEP"},
    ]

    cap = collectors.collect_capability_summary(_integrate_state("k1", "KEEP", 4.0), [], [], forge_invocations=invs)

    assert cap["forge"]["keeps"] == 1
    assert cap["forge"]["micro_only_keeps"] == 1
    assert cap["forge"]["status"] == "kept"


def test_needs_review_is_not_an_adoption() -> None:
    """``NEEDS_REVIEW`` means the verdict is not in, not that it was a win."""
    invs = [{"kernel_id": "k1", "decision": "KEEP"}]

    cap = collectors.collect_capability_summary(_integrate_state("k1", "NEEDS_REVIEW"), [], [], forge_invocations=invs)

    assert cap["forge"]["keeps"] == 0
    assert cap["forge"]["status"] != "kept"
    assert cap["forge"]["pending_integrate"] == 1


def test_missing_integrate_decision_is_not_an_adoption() -> None:
    """An empty decision is an undecided verdict, reachable on the fault path."""
    invs = [{"kernel_id": "k1", "decision": "KEEP"}]

    cap = collectors.collect_capability_summary(_integrate_state("k1", ""), [], [], forge_invocations=invs)

    assert cap["forge"]["keeps"] == 0
    assert cap["forge"]["pending_integrate"] == 1


def test_adopted_patch_is_not_undone_by_a_reverted_sibling() -> None:
    """``kernel_integrate_attempts`` is keyed by kernel|patch|args, so one
    kernel holds several rows. Folding them by kernel id must not let whichever
    row happens to be iterated last decide the outcome.
    """
    state = {
        "kernel_integrate_attempts": {
            # Ordered so the REVERT is visited last: overwriting loses the KEEP.
            "k1|patchA|": {"kernel_id": "k1", "last_decision": "KEEP", "best_gain_pct": 9.0},
            "k1|patchB|": {"kernel_id": "k1", "last_decision": "REVERT", "best_gain_pct": -5.0},
        }
    }

    cap = collectors.collect_capability_summary(
        state, [], [], forge_invocations=[{"kernel_id": "k1", "decision": "KEEP"}]
    )

    assert cap["forge"]["keeps"] == 1, "an adopted patch survives a reverted sibling"
    assert cap["forge"]["status"] == "kept"
    assert cap["forge"]["e2e_gain_pct"] == 9.0


def test_kernel_with_only_reverted_patches_is_not_adopted() -> None:
    """Folding must not manufacture an adoption out of two reverts."""
    state = {
        "kernel_integrate_attempts": {
            "k1|patchA|": {"kernel_id": "k1", "last_decision": "REVERT", "best_gain_pct": -5.0},
            "k1|patchB|": {"kernel_id": "k1", "last_decision": "REVERT", "best_gain_pct": -2.0},
        }
    }

    cap = collectors.collect_capability_summary(
        state, [], [], forge_invocations=[{"kernel_id": "k1", "decision": "KEEP"}]
    )

    assert cap["forge"]["keeps"] == 0
    assert cap["forge"]["status"] == "reverted"


def test_pending_verdict_loses_to_a_decided_one() -> None:
    """A decided sibling outranks an undecided one."""
    state = {
        "kernel_integrate_attempts": {
            "k1|patchA|": {"kernel_id": "k1", "last_decision": "NEEDS_REVIEW", "best_gain_pct": 1.0},
            "k1|patchB|": {"kernel_id": "k1", "last_decision": "KEEP", "best_gain_pct": 4.0},
        }
    }

    cap = collectors.collect_capability_summary(
        state, [], [], forge_invocations=[{"kernel_id": "k1", "decision": "KEEP"}]
    )

    assert cap["forge"]["keeps"] == 1
    assert cap["forge"].get("pending_integrate", 0) == 0


def test_geak_result_marks_capability_attempted_without_native_run_dir() -> None:
    """GEAK e2e does not use the native kernel-agent invocation layout."""
    geak = {
        "engaged": True,
        "status": "ok",
        "kernels_attempted": [{"kernel_id": "k1"}],
        "accepted_kernels": [],
    }

    cap = collectors.collect_capability_summary({}, [], [], geak=geak)

    assert cap["geak"]["attempts"] == 1
    assert cap["geak"]["keeps"] == 0
    assert cap["geak"]["status"] == "attempted"


def test_geak_configured_without_result_remains_not_attempted() -> None:
    """Selecting the backend is not evidence that its route actually ran."""
    geak = {
        "engaged": True,
        "status": "missing",
        "error_class": "no_result",
        "accepted_kernels": [],
        "accepted_heads": [],
    }

    cap = collectors.collect_capability_summary(
        {"kernel_optimizer": "geak"},
        [],
        [],
        geak=geak,
    )

    assert cap["geak"]["attempts"] == 0
    assert cap["geak"]["keeps"] == 0
    assert cap["geak"]["status"] == "not_attempted"


def test_promoted_geak_route_marks_capability_kept() -> None:
    """A promoted GEAK route must not still report ``not_attempted``."""
    state = {
        "optimization_stack": [
            {
                "action": "geak_e2e",
                "variant_name": "geak_e2e",
                "source": "geak_e2e",
            }
        ]
    }
    geak = {
        "engaged": True,
        "status": "ok",
        "accepted_kernels": [{"kernel_id": "k1"}, {"kernel_id": "k2"}],
    }

    cap = collectors.collect_capability_summary(state, [], [], geak=geak)

    assert cap["geak"]["attempts"] == 2
    # ONE promotion is ONE keep. The canonical ledger books a single adoption
    # for the route-level win, and the two counters must not disagree.
    assert cap["geak"]["keeps"] == 1
    assert cap["geak"]["status"] == "kept"


# Fragment identity


def test_item_filename_seq_matches_envelope_seq(tmp_path: Path) -> None:
    """A keyless item spends one sequence number, not two.

    The filename and the envelope must agree, or a ``seq=N`` trace line points
    at a file that does not exist.
    """
    rec = Recorder(tmp_path, producer="coordinator")

    path = rec.record_item("measurements", {"value": 1})

    envelope = json.loads(path.read_text(encoding="utf-8"))
    filename_seq = int(path.stem.rsplit("-", 1)[-1])
    assert filename_seq == envelope["seq"]


def test_sequence_numbers_stay_monotonic_across_item_kinds(tmp_path: Path) -> None:
    """Sharing a number must not break the ordering assembler sorts on."""
    rec = Recorder(tmp_path, producer="coordinator")

    first = rec.record_item("measurements", {"value": 1})
    second = rec.record_item("measurements", {"value": 2}, key="keyed")
    third = rec.record_item("measurements", {"value": 3})

    seqs = [json.loads(p.read_text(encoding="utf-8"))["seq"] for p in (first, second, third)]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "sequence numbers must stay unique"


def test_keys_that_slugify_alike_get_distinct_fragments(tmp_path: Path) -> None:
    """``ck#1`` and ``ck-1`` are different kernels and need different files."""
    rec = Recorder(tmp_path, producer="coordinator")

    first = rec.record_item("measurements", {"value": 1}, key="ck#1")
    second = rec.record_item("measurements", {"value": 2}, key="ck-1")

    assert first != second, "sanitizing must not fold two keys onto one file"
    assert json.loads(first.read_text(encoding="utf-8"))["payload"]["value"] == 1
    assert json.loads(second.read_text(encoding="utf-8"))["payload"]["value"] == 2


def test_same_key_still_rewrites_one_fragment(tmp_path: Path) -> None:
    """Idempotence across retries is the whole point of a stable key."""
    rec = Recorder(tmp_path, producer="coordinator")

    first = rec.record_item("measurements", {"value": 1}, key="k1")
    second = rec.record_item("measurements", {"value": 2}, key="k1")

    assert first == second
    assert len(list(tmp_path.glob("measurements__*.json"))) == 1


def test_legacy_reuse_does_not_resurrect_the_collision(tmp_path: Path) -> None:
    """A legacy filename only belongs to a key that sanitizing left untouched.

    ``a/b`` and ``a:b`` both sanitize to ``a-b``. Reusing a legacy ``a-b.json``
    for either of them puts the digest back where it started: two entities
    merged into one file.
    """
    rec = Recorder(tmp_path, producer="coordinator")
    legacy = tmp_path / "measurements__coordinator__a-b.json"
    legacy.write_text(
        json.dumps({"section": "measurements", "kind": "item", "seq": 1, "payload": {"who": "a/b"}}),
        encoding="utf-8",
    )

    written = rec.record_upsert_item("measurements", {"who": "a:b"}, key="a:b")

    assert written != legacy, "a sanitized key must not claim an ambiguous legacy file"
    assert json.loads(legacy.read_text(encoding="utf-8"))["payload"]["who"] == "a/b", "legacy left intact"
    assert json.loads(written.read_text(encoding="utf-8"))["payload"]["who"] == "a:b"


def test_fragment_written_under_the_old_name_keeps_that_name(tmp_path: Path) -> None:
    """A resumed session must update its fragment, not fork a second one."""
    rec = Recorder(tmp_path, producer="coordinator")
    legacy = tmp_path / "measurements__coordinator__k1.json"
    legacy.write_text(
        json.dumps({"section": "measurements", "kind": "item", "seq": 1, "payload": {"value": 1}}),
        encoding="utf-8",
    )

    written = rec.record_upsert_item("measurements", {"value": 2}, key="k1")

    assert written == legacy
    assert len(list(tmp_path.glob("measurements__*.json"))) == 1, "resume must not duplicate the fragment"
    assert json.loads(written.read_text(encoding="utf-8"))["payload"]["value"] == 2


# Timeline de-duplication


def test_distinct_tasks_in_the_same_second_are_not_folded() -> None:
    """Two tasks sharing an action and a second are two events, not one."""
    state = {
        "explore_attempts": [
            {"ts": "2026-08-24T10:00:00Z", "task_id": "task-1", "status": "succeeded"},
            {"ts": "2026-08-24T10:00:00Z", "task_id": "task-2", "status": "succeeded"},
        ]
    }

    events = collectors.collect_phase_timeline(None, state, [])

    assert len(events) == 2, "task_id must participate in the de-dup key"
    assert {e["task_id"] for e in events} == {"task-1", "task-2"}


def test_exporter_keeps_distinct_tasks_from_recorder_fragments() -> None:
    """The exporter merges recorder fragments and must fold them as the collector does.

    Recorder-only rows never pass through the collector, so an exporter with its
    own weaker identity silently drops events the collector would have kept.
    """
    from hyperloom.inference_optimizer.breakdown.exporter import _merge_phase_timeline

    fragment = [
        {"ts": "2026-08-24T10:00:00Z", "action": "explore", "task_id": "task-1", "status": "succeeded"},
        {"ts": "2026-08-24T10:00:00Z", "action": "explore", "task_id": "task-2", "status": "succeeded"},
    ]

    merged = _merge_phase_timeline(fragment, [])

    assert len(merged) == 2, "exporter must not fold two tasks into one event"
    assert {e.get("task_id") for e in merged} == {"task-1", "task-2"}


def test_collector_and_exporter_agree_on_event_identity() -> None:
    """The two paths must not disagree about what counts as the same event."""
    from hyperloom.inference_optimizer.breakdown.exporter import _merge_phase_timeline

    rows = [
        {"ts": "2026-08-24T10:00:00Z", "task_id": "task-1", "status": "succeeded"},
        {"ts": "2026-08-24T10:00:00Z", "task_id": "task-2", "status": "succeeded"},
        {"ts": "2026-08-24T10:00:00Z", "task_id": "task-1", "status": "succeeded"},
    ]
    via_collector = collectors.collect_phase_timeline(None, {"explore_attempts": rows}, [])
    via_exporter = _merge_phase_timeline([dict(r, action="explore") for r in rows], [])

    assert len(via_collector) == len(via_exporter)


def test_exporter_still_folds_a_fragment_echo_of_a_collector_row() -> None:
    """The fold that exists for a reason must survive the fix."""
    from hyperloom.inference_optimizer.breakdown.exporter import _merge_phase_timeline

    row = {
        "ts": "2026-08-24T10:00:00Z",
        "action": "explore",
        "task_id": "task-1",
        "change": "explore",
        "status": "succeeded",
    }

    merged = _merge_phase_timeline([dict(row)], [dict(row)])

    assert len(merged) == 1


def test_true_duplicate_rows_are_still_folded() -> None:
    """De-dup still collapses rows that are genuinely the same event."""
    state = {
        "explore_attempts": [
            {"ts": "2026-08-24T10:00:00Z", "task_id": "task-1", "status": "succeeded"},
            {"ts": "2026-08-24T10:00:00Z", "task_id": "task-1", "status": "succeeded"},
        ]
    }

    events = collectors.collect_phase_timeline(None, state, [])

    assert len(events) == 1


# Skipped-section evidence


def test_skipped_section_evidence_is_not_dropped() -> None:
    """A skipped section's warning still reaches ``data_quality_flags``."""
    sec = RenderedSection(
        section_id="sweep",
        title="Sweep",
        key_facts=["No sweep run this session."],
        warnings=["sweep never ran this session"],
        skipped=True,
    )

    flags = cross_section._data_quality_flags({}, [sec])

    assert any("sweep" in f for f in flags), "skipped sections must not vanish silently"
    assert any("never ran" in f for f in flags)


def test_skipped_section_without_warnings_still_reports_absence() -> None:
    """Absence itself is reportable even when the renderer logged no warning."""
    sec = RenderedSection(
        section_id="roofline",
        title="Roofline",
        key_facts=["No roofline snapshot recorded."],
        skipped=True,
    )

    flags = cross_section._data_quality_flags({}, [sec])

    assert any("roofline" in f for f in flags)


def test_section_without_producer_is_not_a_data_quality_flag() -> None:
    """A section nothing produces says nothing about this session."""
    sec = RenderedSection(
        section_id="data_provenance",
        title="Data Provenance",
        key_facts=["No provenance recorded."],
        skipped=True,
    )

    flags = cross_section._data_quality_flags({}, [sec])

    assert not any("data_provenance" in f for f in flags)


def test_every_registered_section_reaches_the_report() -> None:
    """A section that renders but is never grouped is invisible work.

    ``geak_invocations`` / ``forge_invocations`` rendered for months without a
    group entry, so the report quoted their adoption counts while showing none
    of the attempts behind them.
    """
    from hyperloom.inference_optimizer.breakdown.reporters import compose
    from hyperloom.inference_optimizer.breakdown.reporters.base import REGISTRY

    grouped = {section_id for _, ids in compose.SECTION_GROUPS for section_id in ids}
    registered = {section_id for section_id, _ in REGISTRY}

    assert not (registered - grouped), "registered renderers missing from SECTION_GROUPS"
    assert not (grouped - registered), "SECTION_GROUPS references sections nothing renders"


def test_sections_without_producer_list_matches_reality(tmp_path: Path) -> None:
    """Recompute the dead-section list; the constant must still match.

    Wiring up a producer, or registering another section nothing fills, has to
    fail here rather than silently drift. The scan reads ``breakdown.get("x")``
    literals, so a renderer that computes its key at runtime would be missed --
    none do today.
    """
    import re

    from hyperloom.inference_optimizer.breakdown import exporter
    from hyperloom.inference_optimizer.breakdown.recorder import recorder as recorder_mod

    available = set(exporter.build(tmp_path).keys())
    available |= set(getattr(recorder_mod, "SECTION_SHAPES", {}) or {})
    available |= set(getattr(recorder_mod, "DERIVED_SECTIONS", ()) or ())

    renderers_dir = Path(cross_section.__file__).resolve().parent / "_renderers"
    without_producer: set[str] = set()
    for path in sorted(renderers_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        keys = set(re.findall(r'breakdown\.get\("([a-z_]+)"', path.read_text(encoding="utf-8")))
        if keys - available:
            without_producer.add(path.stem)

    assert without_producer == set(cross_section._SECTIONS_WITHOUT_PRODUCER)


def test_live_section_warnings_are_unchanged() -> None:
    """Non-skipped sections keep their existing flag format."""
    sec = RenderedSection(
        section_id="roofline",
        title="Roofline",
        warnings=["ceiling unavailable"],
        skipped=False,
    )

    flags = cross_section._data_quality_flags({}, [sec])

    assert "[roofline] ceiling unavailable" in flags


# Report output


def test_data_quality_flags_survive_an_llm_summary() -> None:
    """A model that ignores the flags must not be able to erase them.

    The flags are where a skipped section's evidence ends up, and the LLM
    summary replaces the deterministic one wholesale, so a narrative that never
    mentions them would otherwise delete them from the report.
    """
    from hyperloom.inference_optimizer.breakdown.reporters.compose import render_session_report

    class _SilentLLM:
        """Answers well-formed JSON that never mentions a flag."""

        def complete(self, *, system: str, user: str) -> str:
            """Return a summary with no data-quality content.

            Args:
                system (str): System prompt (ignored).
                user (str): User prompt (ignored).

            Returns:
                str: A valid response envelope.
            """
            return json.dumps({"executive_summary": "Everything went fine.", "section_narratives": {}})

    result = render_session_report({}, llm_client=_SilentLLM())

    assert result.used_llm
    assert "Everything went fine." in result.markdown
    assert "Data quality flags" in result.markdown, "flags must not depend on the model repeating them"


def test_capability_table_shows_unadopted_outcomes() -> None:
    """``keeps`` alone cannot distinguish a failed lane from a pending one."""
    from hyperloom.inference_optimizer.breakdown.reporters._renderers import capability_summary as cap_renderer

    rendered = cap_renderer.render(
        {
            "capability_summary": {
                "forge": {
                    "status": "attempted",
                    "attempts": 3,
                    "keeps": 0,
                    "micro_only_keeps": 2,
                    "pending_integrate": 1,
                    "reverts": 1,
                    "e2e_gain_pct": 4.5,
                }
            }
        }
    )

    assert "micro_only=2" in rendered.markdown_block
    assert "pending_integrate=1" in rendered.markdown_block
    assert "reverts=1" in rendered.markdown_block
    assert "e2e_gain" in rendered.markdown_block


# LLM narrative guard rails


def _parse(exec_summary: str = "ok", **narratives: str) -> dict[str, Any]:
    """Run ``parse_llm_response`` over a well-formed response envelope.

    Args:
        exec_summary (str): Executive summary the model supposedly returned.
        **narratives (str): Section narratives keyed by section id.

    Returns:
        dict[str, Any]: The parsed and sanitized result.
    """
    return llm_prompt.parse_llm_response(
        json.dumps({"executive_summary": exec_summary, "section_narratives": narratives})
    )


def test_ordinary_narrative_survives_untouched() -> None:
    """Guard rails must not disturb prose that respected the brief."""
    prose = "Throughput improved after the aiter variant was adopted. No regressions were seen."

    out = _parse(exec_summary=prose, sweep=prose)

    assert out["executive_summary"] == prose
    assert out["section_narratives"]["sweep"] == prose


def test_prose_containing_an_inline_angle_bracket_survives() -> None:
    """Only a line *opening* a block is a threat; ``<`` mid-sentence is prose."""
    prose = "Tail latency stayed < 5ms while throughput rose."

    out = _parse(sweep=prose)

    assert out["section_narratives"]["sweep"] == prose


def test_unterminated_html_comment_is_rejected() -> None:
    """An unclosed comment comments out every section after this one."""
    out = _parse(sweep="Looks fine.\n<!-- unterminated")

    assert out["section_narratives"]["sweep"] == ""


def test_any_block_opener_rejects_the_whole_narrative() -> None:
    """Repairing the prose would leave text the model never wrote.

    Enumerating safe HTML was the wrong shape for this: CommonMark opens a
    block many ways and missing one fails silently, so the rule matches the
    act of opening a block instead.
    """
    openers = [
        "## Injected Heading\nThe sweep found a better concurrency.",
        "Here is the config:\n```yaml\nkey: value\n```",
        "Summary line\n===",
        "<div>raw</div>\nreal prose",
        "<!DOCTYPE html>\nprose",
        "<?php echo 1; ?>\nprose",
        "<![CDATA[ raw ]]>\nprose",
        "<p>paragraph</p>",
        "<pre>fixed</pre>",
        "<blockquote>quoted</blockquote>",
    ]

    for source in openers:
        out = _parse(sweep=source)
        assert out["section_narratives"]["sweep"] == "", f"block opener slipped through: {source!r}"


def test_overlong_narrative_is_dropped_whole() -> None:
    """Half a truncated sentence reads worse than the deterministic fallback."""
    out = _parse(sweep="x" * (llm_prompt._MAX_NARRATIVE_CHARS + 1))

    assert out["section_narratives"]["sweep"] == ""


def test_overlong_executive_summary_is_dropped_whole() -> None:
    """The summary has its own, larger ceiling."""
    out = _parse(exec_summary="x" * (llm_prompt._MAX_EXEC_SUMMARY_CHARS + 1))

    assert out["executive_summary"] == ""


def test_multi_paragraph_prose_is_still_accepted() -> None:
    """Rejecting block openers must not reject ordinary paragraph breaks."""
    prose = "The sweep raised concurrency.\n\nNo accuracy regression was observed."

    out = _parse(sweep=prose)

    assert out["section_narratives"]["sweep"] == prose


# Sweep variant status


_SWEEP_POINT_KEYS = frozenset(SweepPoint.__annotations__)
_SWEEP_STATUSES = frozenset({"ok", "failed", "skipped"})


def _sweep_variant_dir(session_dir: Path, name: str) -> Path:
    """Create an empty ``runs/sweep/<task>/<name>`` variant directory.

    Args:
        session_dir (Path): Session root used as the sweep parent.
        name (str): Variant directory name (``variant_*_conc*_isl*_osl*``).

    Returns:
        Path: The created variant directory.
    """
    path = session_dir / "runs" / "sweep" / "task-1" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_report_text(variant_dir: Path, text: str) -> None:
    """Write ``benchmark_report.json`` under ``variant_dir``.

    Args:
        variant_dir (Path): Sweep variant directory.
        text (str): File contents, well-formed or deliberately corrupt.
    """
    (variant_dir / "benchmark_report.json").write_text(text, encoding="utf-8")


def _write_abort_reason(
    variant_dir: Path,
    name: str,
    *,
    error_class: str,
    error: str,
) -> None:
    """Write a grid-runner shaped ``abort_reason.json``.

    Args:
        variant_dir (Path): Sweep variant directory.
        name (str): Variant name stored on the marker.
        error_class (str): Short failure class.
        error (str): Failure detail.
    """
    (variant_dir / "abort_reason.json").write_text(
        json.dumps(
            {
                "variant": name,
                "error_class": error_class,
                "error": error,
                "extra_args": "",
                "aborted_at_utc": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def _ok_benchmark_report() -> dict[str, Any]:
    """Return a readable successful benchmark report with metrics.

    Returns:
        dict[str, Any]: A report the collector treats as ``status=ok``.
    """
    return {
        "success": True,
        "output_throughput": 800.0,
        "mean_ttft_ms": 50.0,
        "mean_tpot_ms": 10.0,
        "mean_e2el_ms": 1000.0,
    }


def _install_sweep_case(session_dir: Path, kind: str) -> str:
    """Create one sweep variant matching ``kind`` and return its directory name.

    Args:
        session_dir (Path): Session root.
        kind (str): Case id from the status-matrix parametrize table.

    Returns:
        str: The variant directory name.
    """
    name = "variant_0_conc32_isl128_osl128"
    variant_dir = _sweep_variant_dir(session_dir, name)
    if kind == "ok_full":
        _write_report_text(variant_dir, json.dumps(_ok_benchmark_report()))
    elif kind == "empty_object":
        _write_report_text(variant_dir, "{}")
    elif kind == "success_true_no_metrics":
        _write_report_text(variant_dir, json.dumps({"success": True}))
    elif kind == "zero_tput":
        _write_report_text(variant_dir, json.dumps({"success": True, "output_throughput": 0.0}))
    elif kind == "success_false_error":
        _write_report_text(variant_dir, json.dumps({"success": False, "error": "OOM killed"}))
    elif kind == "success_false_errors_list":
        _write_report_text(
            variant_dir,
            json.dumps({"success": False, "errors": ["CUDA OOM", "server died"]}),
        )
    elif kind == "success_false_no_reason":
        _write_report_text(variant_dir, json.dumps({"success": False}))
    elif kind == "truncated":
        _write_report_text(variant_dir, '{"success": true, "output_throughput":')
    elif kind == "empty_file":
        _write_report_text(variant_dir, "")
    elif kind == "top_level_list":
        _write_report_text(variant_dir, json.dumps([{"output_throughput": 1.0}]))
    elif kind == "missing":
        pass
    elif kind == "abort_reason_only":
        _write_abort_reason(
            variant_dir, name, error_class="no_benchmark_workspace", error="no benchmark_* workspace produced"
        )
    elif kind == "unreadable_with_abort":
        _write_report_text(variant_dir, '{"success": true, "output_throughput":')
        _write_abort_reason(
            variant_dir,
            name,
            error_class="no_benchmark_workspace",
            error="no benchmark_* workspace produced",
        )
    elif kind == "long_errors":
        _write_report_text(
            variant_dir,
            json.dumps({"success": False, "errors": ["E" * 800, "tail"]}),
        )
    elif kind == "secret_error":
        _write_report_text(
            variant_dir,
            json.dumps({"success": False, "error": "auth failed Bearer sensitive-token"}),
        )
    else:
        raise ValueError(kind)
    return name


@pytest.mark.parametrize(
    ("kind", "expected_status", "error_contains", "expect_warning", "tput"),
    [
        ("ok_full", "ok", None, False, 800.0),
        ("empty_object", "ok", None, False, None),
        ("success_true_no_metrics", "ok", None, False, None),
        ("success_false_error", "failed", ("OOM killed",), False, None),
        ("success_false_errors_list", "failed", ("CUDA OOM", "server died"), False, None),
        ("success_false_no_reason", "failed", ("success=false",), False, None),
        ("truncated", "failed", ("unreadable",), True, None),
        ("empty_file", "failed", ("unreadable",), True, None),
        ("top_level_list", "failed", ("unreadable",), True, None),
        ("missing", "skipped", None, False, None),
        ("abort_reason_only", "failed", ("no_benchmark_workspace",), False, None),
        ("unreadable_with_abort", "failed", ("no_benchmark_workspace",), True, None),
        ("long_errors", "failed", ("EEE",), False, None),
        ("secret_error", "failed", ("[REDACTED]",), False, None),
    ],
)
def test_sweep_variant_status_matrix_matches_report_readability(
    tmp_path: Path,
    kind: str,
    expected_status: str,
    error_contains: tuple[str, ...] | None,
    expect_warning: bool,
    tput: float | None,
) -> None:
    """Each on-disk report shape maps to the documented status / error / warning."""
    name = _install_sweep_case(tmp_path, kind)
    warnings: list[str] = []
    rows = collectors.collect_sweep(tmp_path, {}, warnings)["all_variants"]
    assert len(rows) == 1
    row = rows[0]
    assert row["variant_name"] == name
    assert row["status"] == expected_status
    if expected_status == "failed":
        assert row["error"]
        for fragment in error_contains or ():
            assert fragment in row["error"], row["error"]
    else:
        assert row["error"] is None
    assert row["output_throughput_tok_s"] == tput
    has_parse_warning = any("failed to parse" in w for w in warnings)
    assert has_parse_warning is expect_warning, warnings
    if expected_status == "skipped":
        assert row["benchmark_report_path"] is None
    if kind == "long_errors":
        assert len(row["error"]) <= 500
        assert "tail" not in row["error"]
    if kind == "secret_error":
        assert "sensitive-token" not in row["error"]


def test_sweep_variant_rows_share_a_stable_key_set(tmp_path: Path) -> None:
    """ok / failed / skipped rows expose the same SweepPoint keys.

    Table-shaped consumers (DataFrame conversion, column-wise parsers) depend
    on every row having the same key set, including ``error`` as null on
    non-failed rows.
    """
    ok_dir = _sweep_variant_dir(tmp_path, "variant_0_conc32_isl128_osl128")
    _write_report_text(ok_dir, json.dumps(_ok_benchmark_report()))
    failed_dir = _sweep_variant_dir(tmp_path, "variant_1_conc64_isl128_osl128")
    _write_report_text(failed_dir, '{"not json')
    _sweep_variant_dir(tmp_path, "variant_2_conc128_isl128_osl128")

    rows = collectors.collect_sweep(tmp_path, {}, [])["all_variants"]
    assert {row["status"] for row in rows} == _SWEEP_STATUSES
    for row in rows:
        assert set(row) == _SWEEP_POINT_KEYS
        assert row["status"] in _SWEEP_STATUSES
        assert "error" in row
        if row["status"] == "failed":
            assert row["error"]


def test_corrupt_sweep_report_is_not_counted_as_success_by_the_renderer(tmp_path: Path) -> None:
    """Collector output fed to the renderer: failed and skipped are counted apart."""
    ok_dir = _sweep_variant_dir(tmp_path, "variant_0_conc32_isl128_osl128")
    _write_report_text(ok_dir, json.dumps(_ok_benchmark_report()))
    bad_dir = _sweep_variant_dir(tmp_path, "variant_1_conc64_isl128_osl128")
    _write_report_text(bad_dir, '{"success": true, "output_throughput":')
    _sweep_variant_dir(tmp_path, "variant_2_conc128_isl128_osl128")

    section = collectors.collect_sweep(tmp_path, {}, [])
    out = sw.render({"sweep": section})

    summary = next(f for f in out.key_facts if f.startswith("Sweep grid="))
    assert "success=1" in summary, out.key_facts
    assert "failed=1" in summary, out.key_facts
    assert "skipped=1" in summary, out.key_facts
    assert any("Best point" in f and "tput=800.0" in f for f in out.key_facts), out.key_facts
    assert out.decisions[0].kind == "kept"
    assert "800.00" in out.markdown_block
    assert "50.00" in out.markdown_block
    assert "unreadable" in out.markdown_block
    assert "| conc | isl | osl | status | tput | ttft | e2el | error |" in out.markdown_block
    by_status = {v["status"] for v in section["all_variants"]}
    assert by_status == {"ok", "failed", "skipped"}


def test_empty_ok_rows_do_not_invent_a_best_point() -> None:
    """ok-without-metrics must not become kept / Best point tput=None."""
    variants = [
        {"conc": 32, "isl": 128, "osl": 128, "status": "ok", "output_throughput_tok_s": None, "ttft_mean_ms": None},
        {"conc": 64, "isl": 128, "osl": 128, "status": "ok", "output_throughput_tok_s": None, "ttft_mean_ms": None},
    ]
    out = sw.render({"sweep": {"grid_size": 2, "all_variants": variants}})
    summary = next(f for f in out.key_facts if f.startswith("Sweep grid="))
    assert "success=2" in summary
    assert not any("Best point" in f for f in out.key_facts), out.key_facts
    assert out.decisions[0].kind == "attempted"


def test_success_true_zero_throughput_is_still_ok(tmp_path: Path) -> None:
    """A landed success=true report is ok even when throughput is 0.

    Does not assert the extracted metric: ``_benchmark_report_metrics`` currently
    collapses 0 through an ``or`` chain, which is a known defect (see the
    xfail sibling), not the status contract under test here.
    """
    _install_sweep_case(tmp_path, "zero_tput")
    row = collectors.collect_sweep(tmp_path, {}, [])["all_variants"][0]
    assert row["status"] == "ok"
    assert row["error"] is None


@pytest.mark.xfail(
    strict=True,
    reason="known defect: _benchmark_report_metrics uses `or`, so throughput 0 is indistinguishable from missing",
)
def test_zero_throughput_is_preserved_not_collapsed_to_none(tmp_path: Path) -> None:
    """Measured 0 tok/s should survive as 0.0, not become null."""
    _install_sweep_case(tmp_path, "zero_tput")
    row = collectors.collect_sweep(tmp_path, {}, [])["all_variants"][0]
    assert row["output_throughput_tok_s"] == 0.0
