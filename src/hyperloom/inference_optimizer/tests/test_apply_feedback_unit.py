# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the structured apply-failure feedback helpers.

Covers ApplyFeedback (de)serialisation and mandate rendering, the patch
source-context extractor (modification + deletion targets, prefix stripping,
missing-file fallbacks), the generic source_context_for_file primitive, and
the build_apply_feedback factory.
"""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors import _apply_feedback as af
from hyperloom.orchestrator.actions.executors._apply_feedback import (
    ApplyFeedback,
    build_apply_feedback,
    read_patch_source_context,
    source_context_for_file,
)


# ---------------------------------------------------------------------------
# ApplyFeedback dataclass
# ---------------------------------------------------------------------------

def test_from_dict_defaults_for_missing_keys():
    fb = ApplyFeedback.from_dict({})
    assert fb.patch == ""
    assert fb.channel == "nogit"
    assert fb.tried_levels == []
    assert fb.stderr == ""
    assert fb.rejected_hunks == ""
    assert fb.source_context == ""


def test_to_dict_from_dict_roundtrip():
    fb = ApplyFeedback(
        patch="/x/p.patch",
        channel="git",
        tried_levels=[1, 0],
        stderr="err",
        rejected_hunks="rej",
        source_context="ctx",
    )
    restored = ApplyFeedback.from_dict(fb.to_dict())
    assert restored == fb


def test_format_for_mandate_minimal_only_header_and_channel():
    fb = ApplyFeedback(patch="/a/b/only.patch", channel="nogit")
    block = fb.format_for_mandate()
    assert "only.patch" in block
    assert "Channel: nogit" in block
    assert "stderr" not in block
    assert "Rejected hunks" not in block
    assert "Source context" not in block


def test_format_for_mandate_all_sections():
    fb = ApplyFeedback(
        patch="/a/full.patch",
        channel="git",
        tried_levels=[1],
        stderr="boom",
        rejected_hunks="@@ -1 +1 @@",
        source_context="  1| code",
    )
    block = fb.format_for_mandate()
    assert "Tried strip levels: [1]" in block
    assert "boom" in block
    assert "Rejected hunks" in block
    assert "Source context" in block


# ---------------------------------------------------------------------------
# read_patch_source_context
# ---------------------------------------------------------------------------

def test_read_context_modification_with_ab_prefix(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n", encoding="utf-8")
    patch_text = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -10,1 +10,1 @@\n"
        "-line10\n"
        "+line10_patched\n"
    )
    ctx = read_patch_source_context(patch_text, tmp_path, radius=6)
    assert "mod.py" in ctx
    assert "line10" in ctx


def test_read_context_deletion_uses_minus_side(tmp_path):
    target = tmp_path / "gone.py"
    target.write_text("a\nb\nc\n", encoding="utf-8")
    patch_text = (
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,3 +0,0 @@\n"
        "-a\n-b\n-c\n"
    )
    ctx = read_patch_source_context(patch_text, tmp_path, radius=6)
    assert "gone.py" in ctx


def test_read_context_no_target_returns_empty(tmp_path):
    ctx = read_patch_source_context("just some text\n", tmp_path)
    assert ctx == ""


def test_read_context_empty_file_returns_empty(tmp_path):
    target = tmp_path / "empty.py"
    target.write_text("", encoding="utf-8")
    patch_text = "--- a/empty.py\n+++ b/empty.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert read_patch_source_context(patch_text, tmp_path) == ""


def test_read_context_absolute_target(tmp_path):
    target = tmp_path / "abs.py"
    target.write_text("x\ny\nz\n", encoding="utf-8")
    patch_text = (
        f"--- {target}\n"
        f"+++ {target}\n"
        "@@ -1 +1 @@\n"
        "-x\n+x2\n"
    )
    ctx = read_patch_source_context(patch_text, tmp_path)
    assert str(target) in ctx


# ---------------------------------------------------------------------------
# source_context_for_file
# ---------------------------------------------------------------------------

def test_source_context_for_file_empty_path_returns_empty():
    assert source_context_for_file("   ") == ""


def test_source_context_for_file_missing_returns_empty(tmp_path):
    assert source_context_for_file(str(tmp_path / "nope.py")) == ""


def test_source_context_for_file_absolute_with_symbol(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def a():\n    pass\n\ndef target():\n    return 1\n", encoding="utf-8")
    ctx = source_context_for_file(str(f), symbol="def target", window=4)
    assert "target" in ctx
    assert "code.py" in ctx


def test_source_context_for_file_relative_with_search_roots(tmp_path):
    f = tmp_path / "rel.py"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    ctx = source_context_for_file("rel.py", search_roots=[tmp_path], window=2)
    assert "rel.py" in ctx


def test_source_context_for_file_empty_file(tmp_path):
    f = tmp_path / "blank.py"
    f.write_text("", encoding="utf-8")
    assert source_context_for_file(str(f)) == ""


def test_source_context_for_file_symbol_not_found_centres_top(tmp_path):
    f = tmp_path / "nosym.py"
    f.write_text("l1\nl2\nl3\n", encoding="utf-8")
    ctx = source_context_for_file(str(f), symbol="ZZZ", window=2)
    assert "nosym.py" in ctx


# ---------------------------------------------------------------------------
# build_apply_feedback
# ---------------------------------------------------------------------------

def test_build_apply_feedback_without_root():
    fb = build_apply_feedback("/tmp/p.patch", channel="git", tried_levels=[1], stderr="e")
    assert isinstance(fb, ApplyFeedback)
    assert fb.channel == "git"
    assert fb.tried_levels == [1]
    assert fb.stderr == "e"
    assert fb.source_context == ""


def test_build_apply_feedback_with_root_adds_context(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("a\nb\nc\nd\n", encoding="utf-8")
    patch_file = tmp_path / "p.patch"
    patch_file.write_text(
        "--- a/mod.py\n+++ b/mod.py\n@@ -2 +2 @@\n-b\n+b2\n",
        encoding="utf-8",
    )
    fb = build_apply_feedback(patch_file, channel="nogit", framework_root=tmp_path)
    assert fb.source_context
    assert "mod.py" in fb.source_context


def test_build_apply_feedback_with_root_unreadable_patch(tmp_path):
    # Missing patch file → context stays empty.
    fb = build_apply_feedback(tmp_path / "missing.patch", channel="git", framework_root=tmp_path)
    assert fb.source_context == ""


# ---------------------------------------------------------------------------
# Exception-guard branches (helpers must swallow and return "")
# ---------------------------------------------------------------------------

def test_read_patch_source_context_swallows_exceptions(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("parse blew up")

    monkeypatch.setattr(af, "_read_source_context_impl", _boom)
    assert read_patch_source_context("--- a/x\n+++ b/x\n", tmp_path) == ""


def test_source_context_for_file_swallows_exceptions(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("resolve blew up")

    monkeypatch.setattr(af, "_source_context_for_file_impl", _boom)
    assert source_context_for_file("/tmp/whatever.py") == ""
