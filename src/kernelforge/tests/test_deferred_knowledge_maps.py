# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ablation that defers every knowledge map to a pointer.

The maps are the bulk of a kernel-backend system prompt and that prompt is the
cached prefix, so they are re-read on every turn of every session. Deferring
them is a behaviour trade, not a free win, which is why it is off by default --
these tests pin the default and the shape of the alternative rather than argue
for either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelforge.knowledge.local_index import build_forge_knowledge


@pytest.fixture
def kb(tmp_path: Path) -> Path:
    """A knowledge tree with a mandatory pillar and two language folders."""
    for rel, title in (
        ("hardware", "The chip"),
        ("common_methodology", "How to optimize"),
        ("languages/triton", "Triton on AMD"),
        ("languages/gluon", "Gluon on AMD"),
    ):
        folder = tmp_path / rel
        folder.mkdir(parents=True)
        (folder / "INDEX.md").write_text(
            f"# {title} — knowledge map\n\n## Start here\n\n- card.md — a card\n" + ("filler line\n" * 200),
            encoding="utf-8",
        )
        (folder / "card.md").write_text("# A card\n", encoding="utf-8")
    return tmp_path


def test_the_maps_are_inlined_unless_the_ablation_asks_otherwise(kb: Path) -> None:
    text = build_forge_knowledge(kb, language="triton")
    assert "filler line" in text
    assert "Map not inlined" not in text


def test_deferring_replaces_every_map_with_a_pointer_to_it(kb: Path) -> None:
    text = build_forge_knowledge(kb, language="triton", defer_all=True)
    assert "filler line" not in text
    for rel in ("hardware", "common_methodology", "languages/triton"):
        assert str(kb / rel / "INDEX.md") in text, rel


def test_a_deferred_pillar_still_names_itself(kb: Path) -> None:
    """A pointer an agent cannot tell apart from another pointer is not navigation."""
    text = build_forge_knowledge(kb, language="triton", defer_all=True)
    assert "The chip" in text
    assert "How to optimize" in text
    assert "Triton on AMD" in text


def test_deferring_is_what_makes_the_block_small(kb: Path) -> None:
    inlined = build_forge_knowledge(kb, language="triton")
    deferred = build_forge_knowledge(kb, language="triton", defer_all=True)
    assert len(deferred) < len(inlined) / 4


def test_a_carried_language_and_a_deferred_pillar_read_differently(kb: Path) -> None:
    """Gluon is a move the campaign may make; hardware is the map itself."""
    carried = build_forge_knowledge(kb, language=("triton", "gluon"))
    assert "crosses into `languages/gluon`" in carried

    everything = build_forge_knowledge(kb, language=("triton", "gluon"), defer_all=True)
    assert "crosses into" not in everything
    assert everything.count("before opening any card under it") == 4


def test_the_preamble_survives_the_ablation(kb: Path) -> None:
    """Absolute-path guidance is the reason a pointer is followable at all."""
    text = build_forge_knowledge(kb, language="triton", defer_all=True)
    assert "ABSOLUTE path" in text
    assert f"Knowledge root (KB): {kb}" in text


def test_a_pillar_with_no_map_is_left_inlined(tmp_path: Path) -> None:
    """A flat listing is already short; deferring it would cost a Read for nothing."""
    folder = tmp_path / "hardware"
    folder.mkdir(parents=True)
    (folder / "note.md").write_text("# A note\n\nSome prose.\n", encoding="utf-8")
    text = build_forge_knowledge(tmp_path, defer_all=True)
    assert "note.md" in text
    assert "Map not inlined" not in text


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_the_knob_reads_the_environment(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from kernelforge.config import Config

    monkeypatch.setenv("KERNELFORGE_DEFER_KNOWLEDGE_MAPS", value)
    assert Config().defer_knowledge_maps is True


def test_the_knob_is_on_when_the_environment_says_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernelforge.config import Config

    monkeypatch.delenv("KERNELFORGE_DEFER_KNOWLEDGE_MAPS", raising=False)
    assert Config().defer_knowledge_maps is True


@pytest.mark.parametrize("value", ["0", "false", "NO"])
def test_the_environment_can_still_turn_the_pointers_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from kernelforge.config import Config

    monkeypatch.setenv("KERNELFORGE_DEFER_KNOWLEDGE_MAPS", value)
    assert Config().defer_knowledge_maps is False


def test_an_explicit_choice_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same trap ``include_mori_kb`` documents: False must not read as unset."""
    from kernelforge.config import Config

    monkeypatch.setenv("KERNELFORGE_DEFER_KNOWLEDGE_MAPS", "1")
    assert Config(defer_knowledge_maps=False).defer_knowledge_maps is False


def test_the_kernel_backend_prompt_carries_the_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernelforge.config import Config
    from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt

    monkeypatch.delenv("KERNELFORGE_DEFER_KNOWLEDGE_MAPS", raising=False)
    inlined = build_single_kernel_backend_prompt(Config(defer_knowledge_maps=False), "triton")
    deferred = build_single_kernel_backend_prompt(Config(), "triton")
    # triton carries gluon, so the default already defers exactly that one map.
    assert inlined.count("Map not inlined") == 1
    assert "crosses into `languages/gluon`" in inlined
    assert deferred.count("Map not inlined") == 4
    assert "crosses into" not in deferred
    assert len(deferred) < len(inlined)
