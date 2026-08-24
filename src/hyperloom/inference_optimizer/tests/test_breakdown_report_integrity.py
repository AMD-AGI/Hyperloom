# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report integrity: reported counts must mean what the schema says they mean.

These tests pin contracts the breakdown pipeline previously stated in comments
but never enforced:

* ``CapabilityEntry.keeps`` counts kernels adopted at *integrate*. A KEEP that
  only cleared the micro benchmark is not an adoption and must not inflate it.
* ``keeps`` / ``attempts`` count distinct kernels, not invocation rows, so a
  kernel re-tried across runs is still one kernel.
* Timeline de-duplication must not fold two different tasks into one row just
  because they share an action and a wall-clock second.
* A section skipped for lack of data still carries evidence ("this never ran");
  that evidence must reach the reader instead of being dropped silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown import collectors
from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder
from hyperloom.inference_optimizer.breakdown.reporters import cross_section, llm_prompt
from hyperloom.inference_optimizer.breakdown.reporters.base import RenderedSection


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


def test_headings_are_stripped_from_narrative() -> None:
    """A heading would re-parent every deterministic block that follows."""
    out = _parse(sweep="## Injected Heading\nThe sweep found a better concurrency.")

    assert "##" not in out["section_narratives"]["sweep"]
    assert "better concurrency" in out["section_narratives"]["sweep"]


def test_code_fence_is_stripped_from_narrative() -> None:
    """An unbalanced fence swallows the rest of the document."""
    out = _parse(sweep="Here is the config:\n```yaml\nkey: value\n```\nThat is all.")

    assert "```" not in out["section_narratives"]["sweep"]


def test_thematic_break_and_block_html_are_stripped() -> None:
    """Setext rules promote the previous line; block HTML escapes the slot."""
    out = _parse(sweep="Summary line\n===\n<div>raw</div>\nreal prose here")

    cleaned = out["section_narratives"]["sweep"]
    assert "===" not in cleaned
    assert "<div>" not in cleaned
    assert "real prose here" in cleaned


def test_overlong_narrative_is_dropped_whole() -> None:
    """Half a truncated sentence reads worse than the deterministic fallback."""
    out = _parse(sweep="x" * (llm_prompt._MAX_NARRATIVE_CHARS + 1))

    assert out["section_narratives"]["sweep"] == ""


def test_overlong_executive_summary_is_dropped_whole() -> None:
    """The summary has its own, larger ceiling."""
    out = _parse(exec_summary="x" * (llm_prompt._MAX_EXEC_SUMMARY_CHARS + 1))

    assert out["executive_summary"] == ""


def test_mostly_structural_output_is_dropped_whole() -> None:
    """A model that returned a document, not a paragraph, contributes nothing."""
    out = _parse(sweep="# One\n## Two\n### Three\n```\n```\nstray")

    assert out["section_narratives"]["sweep"] == ""
