# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which tasks are served the sgl-amd-mono framework knowledge, and that what it routes to ships."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from kernelforge.config import Config
from kernelforge.config.resources import resource_path
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.kernel_backends.constants import includes_sgl_amd_mono
from kernelforge.knowledge.local_wiki.index import build_forge_knowledge
from kernelforge.workflows.forge_loop.roles.analysis_evidence import AnalysisEvidenceMixin

_SECTION = "## framework/sgl_amd_mono/"
_MONO_SOURCE = "/work/sgl-amd-mono/sgl_amd_mono/qwen3/ffn.py"
_KB = resource_path("local_knowledge")
_FOLDER = _KB / "framework" / "sgl_amd_mono"
_LOCAL_REF = re.compile(r"`((?:overall|models|skills)/[\w./-]+\.md)`")
_PILLAR_REF = re.compile(r"`((?:hardware|common_methodology|framework|languages)/[\w./-]+\.md)`")


@pytest.mark.parametrize(
    ("backend", "sources", "expected"),
    [
        ("mono", None, True),
        ("flydsl", [_MONO_SOURCE], True),
        ("flydsl", ["/work/kernels/sgl_amd_mono_notes/ffn.py"], False),
        ("triton", ["/work/vllm/attention/ops/prefill.py"], False),
    ],
)
def test_the_framework_knowledge_follows_the_backend_or_the_package(backend, sources, expected):
    """A monokernel campaign, or any campaign on the package's sources; a directory that merely contains the name is
    neither."""
    assert includes_sgl_amd_mono(backend, sources) is expected


def test_the_framework_level_sits_between_the_pillars_and_the_languages():
    block = build_forge_knowledge(_KB, language=("flydsl",), frameworks=("sgl_amd_mono",))

    assert block.index("## common_methodology/") < block.index(_SECTION) < block.index("## languages/flydsl/")
    assert _SECTION not in build_forge_knowledge(_KB, language=("flydsl",))


def test_framework_levels_keep_their_reading_order_whatever_order_they_are_named_in():
    block = build_forge_knowledge(_KB, frameworks=("sgl_amd_mono", "aiter"))

    assert block.index("## framework/aiter/") < block.index(_SECTION)


def test_a_framework_that_does_not_exist_is_refused_rather_than_skipped():
    """A misspelt name would otherwise serve the task one level fewer, silently."""
    with pytest.raises(ValueError):
        build_forge_knowledge(_KB, frameworks=("sgl-amd-mono",))


def test_a_monokernel_prompt_carries_the_framework_map():
    prompt = build_single_kernel_backend_prompt(Config(gpu_target="gfx942", defer_knowledge_maps=False), "mono")

    assert _SECTION in prompt
    assert "overall/monokernel_interface.md" in prompt


def test_a_campaign_on_the_package_is_served_the_map_whatever_its_backend():
    config = Config(gpu_target="gfx942", defer_knowledge_maps=False)
    on_package = build_single_kernel_backend_prompt(config, "flydsl", source_paths=[_MONO_SOURCE])
    elsewhere = build_single_kernel_backend_prompt(config, "flydsl", source_paths=["/work/kernels/gemm.py"])

    assert _SECTION in on_package
    assert _SECTION not in elsewhere


class _Orchestrator(AnalysisEvidenceMixin):
    """The slice of a campaign the orchestration knowledge index reads."""

    def __init__(self, backend: str, sources: list[str]) -> None:
        self.config = SimpleNamespace(local_knowledge_dir=str(_KB), defer_knowledge_maps=False)
        self.ic = SimpleNamespace(kernel_backend=backend, backend=backend)
        self._sources = sources

    def _target_source_files(self) -> list[str]:
        return self._sources


@pytest.mark.parametrize(
    ("backend", "sources", "expected"),
    [("mono", ["/work/driver.py"], True), ("flydsl", [_MONO_SOURCE], True), ("flydsl", ["/work/gemm.py"], False)],
)
def test_specialists_and_analysis_read_the_same_framework_level_as_the_implementer(backend, sources, expected):
    """The orchestration context is a second assembly of the knowledge block; it must not drop a level."""
    index = _Orchestrator(backend, sources)._orchestration_knowledge_index()

    assert (_SECTION in index) is expected


def test_every_card_on_disk_is_registered_in_the_map():
    """An unregistered card is one no agent is routed to."""
    index = (_FOLDER / "INDEX.md").read_text(encoding="utf-8")
    cards = sorted(path for path in _FOLDER.rglob("*.md") if path.name != "INDEX.md")

    assert cards
    for card in cards:
        assert card.name in index, f"{card.relative_to(_FOLDER)} is not registered in INDEX.md"


@pytest.mark.parametrize(
    "card",
    sorted([*_FOLDER.rglob("*.md"), _KB / "common_methodology" / "optimization" / "lever_monokernel.md"]),
    ids=lambda path: path.relative_to(_KB).as_posix(),
)
def test_every_card_a_card_routes_to_exists(card):
    """A route to a missing card sends the agent to a Read that fails."""
    text = card.read_text(encoding="utf-8")

    for ref in _LOCAL_REF.findall(text):
        assert (_FOLDER / ref).is_file(), f"{card.name} routes to missing framework/sgl_amd_mono/{ref}"
    for ref in _PILLAR_REF.findall(text):
        assert (_KB / ref).is_file(), f"{card.name} routes to missing {ref}"
