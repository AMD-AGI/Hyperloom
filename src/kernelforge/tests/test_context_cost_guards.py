# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guards on the two things that make a Forge session's context expensive.

Measured over 316 end-to-end Forge runs, ~72% of the LLM bill is proportional
to how large each agent session's context grows (cache write + cache read),
and the median session writes ~133k tokens of it. These pin the two places
this repo can act on that: what it inlines into every prompt, and whether a
missing output filter stays visible.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge import rtk
from kernelforge.knowledge.local_index import _strip_frontmatter, build_forge_knowledge


def test_frontmatter_is_dropped_from_a_knowledge_map() -> None:
    """title/kind/scope/updated describe the file to its maintainer, not the agent."""
    text = "---\ntitle: x\nkind: index\n---\n\n# Real map\n\nbody"
    assert _strip_frontmatter(text) == "# Real map\n\nbody"


def test_an_unclosed_fence_is_not_treated_as_frontmatter() -> None:
    """Swallowing a whole map to save 200 tokens is not a saving."""
    text = "---\ntitle: x\n\n# Real map\n\nbody"
    assert _strip_frontmatter(text) == text


def test_a_map_without_frontmatter_is_untouched() -> None:
    text = "# Real map\n\nbody"
    assert _strip_frontmatter(text) == text


def test_rendered_knowledge_carries_no_yaml_metadata(tmp_path: Path) -> None:
    """The strip has to survive the whole render path, not just the helper."""
    pillar = tmp_path / "hardware"
    pillar.mkdir()
    (pillar / "INDEX.md").write_text(
        "---\ntitle: secret-metadata-marker\nupdated: 2026-01-01\n---\n\n# Hardware map\n\nrouting",
        encoding="utf-8",
    )
    (tmp_path / "common_methodology").mkdir()
    (tmp_path / "common_methodology" / "INDEX.md").write_text("# Methodology map\n", encoding="utf-8")

    rendered = build_forge_knowledge(tmp_path)
    assert "secret-metadata-marker" not in rendered
    assert "# Hardware map" in rendered
    assert "routing" in rendered


def test_a_missing_output_filter_is_reportable_not_silent(monkeypatch) -> None:
    """rtk degrading silently is by design; degrading invisibly is not.

    Verified absent in the CI environment that produced those 316 runs: no
    rtk binary anywhere on the shared filesystem and zero occurrences in the
    end-to-end run logs, so every ninja and git dump reached the agent's
    context in full.
    """
    monkeypatch.setattr(rtk, "_RTK_PATH", None)
    warning = rtk.unavailable_warning()
    assert "rtk is not on PATH" in warning
    # The warning names the in-tree installer rather than the upstream URL:
    # a reader who follows the URL lands on `cargo install`, while
    # `kernelforge install-rtk` is the path this repository supports and tests.
    assert "kernelforge install-rtk" in warning

    monkeypatch.setattr(rtk, "_RTK_PATH", "/usr/bin/rtk")
    assert rtk.unavailable_warning() == ""
