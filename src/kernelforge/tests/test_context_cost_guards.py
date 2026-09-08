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

import pytest

from kernelforge import rtk
from kernelforge.knowledge import local_index
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


def _lang_section(block: str, name: str) -> str:
    """The rendered text of one ``languages/<name>/`` section."""
    marker = f"## languages/{name}/"
    start = block.index(marker)
    nxt = block.find("\n## ", start + 1)
    return block[start : nxt if nxt != -1 else len(block)]


def _kb(tmp_path: Path) -> Path:
    """A knowledge root with two language folders, each with a real map."""
    for pillar in ("hardware", "common_methodology"):
        (tmp_path / pillar).mkdir()
        (tmp_path / pillar / "INDEX.md").write_text(f"# {pillar} map\n", encoding="utf-8")
    langs = tmp_path / "languages"
    for name, body in (("triton", "TRITON-BODY-MARKER"), ("gluon", "GLUON-BODY-MARKER")):
        folder = langs / name
        folder.mkdir(parents=True)
        (folder / "INDEX.md").write_text(
            f"---\ntitle: meta\n---\n\n# {name.title()} on AMD — knowledge map\n\n{body}\n",
            encoding="utf-8",
        )
    return tmp_path


def test_the_primary_language_map_is_inlined_whole(tmp_path) -> None:
    """The language the task is actually written in still arrives in full."""
    root = _kb(tmp_path)
    block = build_forge_knowledge(root, language=("triton", "gluon"))
    assert "TRITON-BODY-MARKER" in _lang_section(block, "triton")


def test_the_carried_language_map_is_a_pointer_not_a_body(tmp_path) -> None:
    """Triton carries Gluon so the move is known -- knowing it costs a path, not a map.

    The carried map used to be inlined whole, which put roughly 2.7k tokens of
    a language the task is not written in ahead of every turn of every session.
    """
    root = _kb(tmp_path)
    block = build_forge_knowledge(root, language=("triton", "gluon"))
    section = _lang_section(block, "gluon")
    assert "GLUON-BODY-MARKER" not in section
    # The affordance survives: what the folder is, and the exact path to read.
    assert "Gluon on AMD" in section
    assert str(root / "languages" / "gluon" / "INDEX.md") in section


def test_deferral_follows_the_reading_order_not_the_folder_name(tmp_path) -> None:
    """A Gluon task inlines Gluon and defers Triton -- the pairing is symmetric."""
    root = _kb(tmp_path)
    block = build_forge_knowledge(root, language=("gluon", "triton"))
    assert "GLUON-BODY-MARKER" in _lang_section(block, "gluon")
    assert "TRITON-BODY-MARKER" not in _lang_section(block, "triton")


def test_a_lone_language_is_never_deferred(tmp_path) -> None:
    """With one language there is nothing being carried; deferring it would lose the map."""
    root = _kb(tmp_path)
    block = build_forge_knowledge(root, language="triton")
    assert "TRITON-BODY-MARKER" in _lang_section(block, "triton")


def test_a_carried_folder_without_a_map_still_lists_its_files(tmp_path) -> None:
    """A flat listing is already short; deferring it would point at a file that is not there."""
    root = _kb(tmp_path)
    bare = root / "languages" / "bare"
    bare.mkdir()
    (bare / "card.md").write_text("# A card\n", encoding="utf-8")
    section = _lang_section(build_forge_knowledge(root, language=("triton", "bare")), "bare")
    assert "card.md" in section
    assert "Map not inlined" not in section


def test_deferring_the_carried_map_actually_shrinks_the_block() -> None:
    """The point of the change is size, so measure it on the maps that ship.

    A synthetic two-line map would make the pointer look like a regression --
    the saving is real only because the real Gluon map is thousands of tokens.
    """
    root = Path(local_index.__file__).resolve().parent.parent / "data" / "local_knowledge"
    if not (root / "languages" / "gluon" / "INDEX.md").is_file():
        pytest.skip("shipped knowledge tree not present in this checkout")
    paired = build_forge_knowledge(root, language=("triton", "gluon"))
    solo = build_forge_knowledge(root, language="triton")
    # What inlining the carried map would have cost: the paired block before
    # this change was exactly the solo block plus the whole Gluon map.
    carried = len(local_index._render_level(root, "languages/gluon"))
    saved = (len(solo) + carried) - len(paired)
    # Worth doing at all: the deferral has to buy back most of the carried map.
    assert saved > 0.8 * carried, f"only saved {saved} of {carried} chars"
