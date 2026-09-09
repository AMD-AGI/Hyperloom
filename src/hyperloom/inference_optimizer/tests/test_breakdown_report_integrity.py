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


def test_no_renderer_reads_a_key_nothing_produces(tmp_path: Path) -> None:
    """Every renderer's key must be one the exporter or recorder can fill.

    Renderers used to be allowed to read keys nothing ever wrote: they were
    skipped on every report and a suppression list kept them from being
    flagged. Registering another such section has to fail here rather than
    ship a section that can only ever be empty. The scan reads
    ``breakdown.get("x")`` literals, so a renderer that computes its key at
    runtime would be missed -- none do today.
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

    assert without_producer == set()


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
            "timeline": [
                {
                    "type": "kernel",
                    "ext": {
                        "outcome": {
                            "by_source": {
                                "kernel_rewrite": {
                                    "attempted": 3,
                                    "keeps": 0,
                                    "micro_only_keeps": 2,
                                    "needs_review": 1,
                                    "reverts": 1,
                                    "e2e_gain_pct": 4.5,
                                }
                            }
                        }
                    },
                }
            ]
        }
    )

    assert "micro_only=2" in rendered.markdown_block
    assert "pending_review=1" in rendered.markdown_block
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


_SWEEP_STATUSES = frozenset({"ok", "failed", "skipped"})


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
