# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the per-run experience ledger (loop/experience.py).

Covers signature extraction, objective constraint distillation/dedup/cap,
prompt rendering, and best-effort disk flush. Filesystem via tmp_path."""

from __future__ import annotations

from kernelforge.loop.experience import (
    ExperienceLedger,
    _extract_signature,
)


# ── _extract_signature ────────────────────────────────────────────────────────


def test_extract_signature_prefers_marker_line():
    text = "some prelude\nRuntimeError: invalid cast!\ntrailing noise"
    assert _extract_signature(text) == "RuntimeError: invalid cast!"


def test_extract_signature_falls_back_to_first_nonempty():
    text = "\n   \nplain first line\nsecond"
    assert _extract_signature(text) == "plain first line"


def test_extract_signature_empty():
    assert _extract_signature("") == ""
    assert _extract_signature("\n  \n") == ""


def test_extract_signature_truncates_to_180():
    long = "error " + "x" * 500
    assert len(_extract_signature(long)) == 180


# ── constraint distillation ───────────────────────────────────────────────────


def test_distill_fastmath_bool(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(1, outcome="build-fail", error_text="got #arith.fastmath<True> attribute")
    assert any("FastMathFlags" in c for c in led.constraints)


def test_distill_invalid_cast(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(1, outcome="build-fail", error_text="Invalid cast! backend")
    assert any("copy-atom width" in c for c in led.constraints)


def test_distill_deduplicates(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(1, outcome="fail", error_text="invalid cast")
    led.record_iteration(2, outcome="fail", error_text="invalid cast again")
    cast_constraints = [c for c in led.constraints if "copy-atom width" in c]
    assert len(cast_constraints) == 1


def test_constraint_cap_drops_oldest(tmp_path):
    led = ExperienceLedger(str(tmp_path), max_constraints=3)
    for i in range(6):
        led.memory.add(f"rule-{i}")
    assert led.constraints == ["rule-3", "rule-4", "rule-5"]


def test_add_constraint_ignores_blank(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.memory.add("")
    led.memory.add("   ")
    assert led.constraints == []


# ── recording + rendering ──────────────────────────────────────────────────────


def test_record_populates_entry_fields(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(
        3,
        outcome="REVERT_PERF",
        diff_summary="  a.py | 2 +-  ",
        error_text="prelude\nAssertionError: not faster\n",
    )
    e = led.entries[0]
    assert e.iteration == 3
    assert e.outcome == "REVERT_PERF"
    assert e.diff_summary == "a.py | 2 +-"
    assert e.error_sig == "AssertionError: not faster"


def test_render_for_prompt_includes_constraints_and_recent(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(1, outcome="build-fail", error_text="invalid cast", diff_summary="k.py | 1 +")
    rendered = led.render_for_prompt()
    assert "## Observed toolchain constraints" in rendered
    assert "## Recent iterations" in rendered
    assert "iter 1: build-fail" in rendered
    assert "error: " in rendered


def test_render_for_prompt_constraints_only(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(1, outcome="build-fail", error_text="invalid cast")
    rendered = led.render_for_prompt(include_recent=False)
    assert "## Observed toolchain constraints" in rendered
    assert "## Recent iterations" not in rendered


def test_render_for_prompt_keeps_only_recent_k(tmp_path):
    led = ExperienceLedger(str(tmp_path), keep_recent=2)
    for i in range(1, 5):
        led.record_iteration(i, outcome=f"iter{i}")
    rendered = led.render_for_prompt()
    assert "iter 3: iter3" in rendered
    assert "iter 4: iter4" in rendered
    assert "iter 1: iter1" not in rendered


def test_render_empty_ledger_is_blank(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    assert led.render_for_prompt() == ""


def test_diff_summary_capped_to_eight_lines(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    diff = "\n".join(f"line{i}" for i in range(20))
    led.record_iteration(1, outcome="KEEP", diff_summary=diff)
    rendered = led.render_for_prompt()
    assert "    line7" in rendered
    assert "    line8" not in rendered


# ── flush ──────────────────────────────────────────────────────────────────────


def test_flush_writes_file(tmp_path):
    led = ExperienceLedger(str(tmp_path))
    led.record_iteration(1, outcome="KEEP")
    assert led.path.exists()
    content = led.path.read_text()
    assert content.startswith("# Forge experience ledger")
    assert "iter 1: KEEP" in content


# ── the shared distillation core ──────────────────────────────────────────────


def test_both_ledgers_keep_their_own_truncation_and_cap():
    """One mechanism, two calibrations: the wording and limits stay per-ledger."""
    from kernelforge.fusion.loop import FusionExperienceLedger
    from kernelforge.fusion.loop import _extract_signature as fusion_signature

    long_line = "error: " + "x" * 400

    assert len(_extract_signature(long_line)) == 180
    assert len(fusion_signature(long_line)) == 200
    assert ExperienceLedger("/tmp").memory.max_constraints == 15
    assert FusionExperienceLedger().memory.max_constraints == 12
