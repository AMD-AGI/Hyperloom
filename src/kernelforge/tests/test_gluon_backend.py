# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gluon kernel backend registration, the triton<->gluon knowledge pairing, and detection.

Gluon is Triton's low-level dialect, not a separate toolchain: same frontend,
same JIT, same lowering, same cache. Two consequences are load-bearing enough to
lock down here.

First, the two backends carry each other's knowledge layer. A Triton campaign
has to know that dropping to Gluon is an available move rather than a different
project, and a Gluon kernel still needs the shared compile-pipeline and
ISA-verification cards that only exist under ``languages/triton/``. The pairing
is also what lets ``languages/gluon/`` stay thin instead of restating the
substrate -- so if it silently breaks, the Gluon tree becomes wrong rather than
merely smaller.

Second, detection order. A Gluon file necessarily imports triton and routinely
keeps a ``@triton.jit`` sibling as its fallback, in a directory named after
triton -- aiter's paged-MQA-logits ships exactly that shape. Matching Triton
first would send every such kernel to the wrong kernel_backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.kernel_backends.constants import (
    KERNEL_BACKENDS,
    resolve_language_dir,
    resolve_language_dirs,
)
from kernelforge.loop.campaign_config import infer_kernel_backend


_GPU = "gfx950"


@pytest.fixture()
def config() -> Config:
    return Config(gpu_target=_GPU)


# ─── registration ───


def test_gluon_is_a_registered_backend():
    assert "gluon" in KERNEL_BACKENDS


def test_gluon_renders_a_forge_loop_prompt(config):
    assert build_single_kernel_backend_prompt(config, "gluon")


# ─── the triton <-> gluon knowledge pairing ───


class TestLanguagePairing:
    """Both backends must reach both language folders, in their own order."""

    def test_gluon_leads_with_gluon_and_keeps_triton(self, config):
        root = Path(config.local_knowledge_dir)
        assert resolve_language_dirs("gluon", root) == ("gluon", "triton")

    def test_triton_leads_with_triton_and_gains_gluon(self, config):
        root = Path(config.local_knowledge_dir)
        assert resolve_language_dirs("triton", root) == ("triton", "gluon")

    @pytest.mark.parametrize("kernel_backend", ["gluon", "triton"])
    def test_both_prompts_carry_both_layers(self, config, kernel_backend):
        prompt = build_single_kernel_backend_prompt(config, kernel_backend)
        assert "languages/gluon" in prompt, f"{kernel_backend}: no Gluon knowledge layer"
        assert "languages/triton" in prompt, f"{kernel_backend}: no Triton knowledge layer"

    def test_an_unpaired_backend_is_unaffected(self, config):
        """The pairing is opt-in per backend, not a change to the default."""
        root = Path(config.local_knowledge_dir)
        assert resolve_language_dirs("flydsl", root) == ("flydsl",)
        assert resolve_language_dirs("hipblaslt", root) == ()

    def test_missing_folder_degrades_instead_of_emitting_a_dead_section(self, tmp_path):
        """A checkout without one folder loses that layer, not the whole pairing."""
        (tmp_path / "languages" / "triton").mkdir(parents=True)
        assert resolve_language_dirs("triton", tmp_path) == ("triton",)
        assert resolve_language_dirs("gluon", tmp_path) == ("triton",)

    def test_primary_accessor_still_returns_one_name(self, config):
        """``resolve_language_dir`` keeps its old contract for callers wanting one."""
        root = Path(config.local_knowledge_dir)
        assert resolve_language_dir("gluon", root) == "gluon"
        assert resolve_language_dir("triton", root) == "triton"


class TestKnowledgeBuilderAcceptsASequence:
    """``build_forge_knowledge`` had to widen for the pairing to be expressible."""

    @staticmethod
    def _sections(block: str) -> list[str]:
        return [line for line in block.splitlines() if line.startswith("## languages/")]

    def test_a_bare_string_still_works(self, config):
        from kernelforge.knowledge import build_forge_knowledge

        block = build_forge_knowledge(config.local_knowledge_dir, language="gluon")
        assert self._sections(block) == [
            "## languages/gluon/  —  base: %s" % (Path(config.local_knowledge_dir) / "languages" / "gluon")
        ]

    def test_a_sequence_renders_in_order(self, config):
        from kernelforge.knowledge import build_forge_knowledge

        block = build_forge_knowledge(config.local_knowledge_dir, language=("gluon", "triton"))
        assert [s.split("/")[1] for s in self._sections(block)] == ["gluon", "triton"]

    def test_duplicates_collapse(self, config):
        """A folder must never be rendered twice into one prompt."""
        from kernelforge.knowledge import build_forge_knowledge

        block = build_forge_knowledge(config.local_knowledge_dir, language=("triton", "gluon", "triton"))
        assert [s.split("/")[1] for s in self._sections(block)] == ["triton", "gluon"]

    def test_none_and_empty_add_no_layer(self, config):
        from kernelforge.knowledge import build_forge_knowledge

        for empty in (None, (), ("",)):
            block = build_forge_knowledge(config.local_knowledge_dir, language=empty)
            assert self._sections(block) == [], f"{empty!r} produced a language layer"


# ─── the knowledge tree itself ───


class TestKnowledgeTree:
    """The cards the prompts route to must exist and be reachable.

    ``build_forge_knowledge`` loads ``INDEX.md`` whole and leaves the rest on
    disk, so a card the index names but that is not there is a dangling pointer
    the agent only discovers mid-session.
    """

    @pytest.fixture()
    def gluon_root(self, config) -> Path:
        return Path(config.local_knowledge_dir) / "languages" / "gluon"

    def test_index_exists(self, gluon_root):
        assert (gluon_root / "INDEX.md").is_file()

    @pytest.mark.parametrize(
        "card",
        [
            "API_docs/programming_model.md",
            "API_docs/layouts.md",
            "API_docs/amd_targets.md",
            "skills/optimize/gluon_levers/overview.md",
            "skills/optimize/gluon_levers/forge_integration.md",
        ],
    )
    def test_card_exists(self, gluon_root, card):
        assert (gluon_root / card).is_file(), f"missing Gluon card: {card}"

    def test_index_is_loaded_into_the_prompt(self, config):
        """The map is inlined; the cards stay on disk and are Read on demand."""
        prompt = build_single_kernel_backend_prompt(config, "gluon")
        assert "Gluon on AMD — knowledge map" in prompt


# ─── detection ───

_GLUON_KERNEL = """\
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    idx = gl.arange(0, BLOCK, layout=layout)
    gl.store(y_ptr + idx, gl.load(x_ptr + idx, mask=idx < n), mask=idx < n)
"""

# The shape aiter ships: one file, one public entry, a Gluon path and a
# @triton.jit fallback selected at dispatch.
_MIXED_KERNEL = (
    _GLUON_KERNEL
    + """

@triton.jit
def add_kernel_triton(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    tl.store(y_ptr + idx, tl.load(x_ptr + idx, mask=idx < n), mask=idx < n)

def add(x, y, n):
    return add_kernel if _use_gluon() else add_kernel_triton
"""
)

_TRITON_KERNEL = """\
import triton
import triton.language as tl

# NOTE: a Gluon rewrite of this kernel was considered and rejected -- the
# autotune search has not converged yet, so the cheaper axes are not exhausted.
@triton.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    tl.store(y_ptr + idx, tl.load(x_ptr + idx, mask=idx < n), mask=idx < n)
"""


class TestInferKernelBackend:
    """Gluon must be recognized ahead of Triton, and on evidence not vocabulary."""

    @pytest.fixture(autouse=True)
    def _no_env_override(self, monkeypatch):
        monkeypatch.delenv("FORGE_KERNEL_BACKEND", raising=False)

    def test_a_gluon_kernel_infers_the_gluon_kernel_backend(self, tmp_path):
        path = tmp_path / "kernel.py"
        path.write_text(_GLUON_KERNEL)
        assert infer_kernel_backend([path]) == "gluon"

    def test_a_mixed_file_infers_gluon_not_triton(self, tmp_path):
        """The lower-level language leads; the Triton layer is carried anyway."""
        path = tmp_path / "kernel.py"
        path.write_text(_MIXED_KERNEL)
        assert infer_kernel_backend([path]) == "gluon"

    def test_a_gluon_kernel_under_a_triton_directory_still_infers_gluon(self, tmp_path):
        """The directory name is not the language.

        aiter keeps Gluon kernels under ``ops/triton/``, and this is the shape
        that would fool a path heuristic. Detection reads the source instead.
        """
        path = tmp_path / "ops" / "triton" / "attention" / "k.py"
        path.parent.mkdir(parents=True)
        path.write_text(_GLUON_KERNEL)
        assert infer_kernel_backend([path]) == "gluon"

    def test_merely_mentioning_gluon_does_not_infer_gluon(self, tmp_path):
        """Detection keys on an import or a decorator, never on the word."""
        path = tmp_path / "kernel.py"
        path.write_text(_TRITON_KERNEL)
        assert infer_kernel_backend([path]) == "triton"

    def test_the_aiter_framework_arm_still_outranks_the_language(self, tmp_path):
        """Pre-existing precedence, locked here because Gluon makes it visible.

        ``infer_kernel_backend`` picks the FRAMEWORK kernel backend for anything under aiter,
        whatever language the kernel is written in -- that is how Triton and HIP
        kernels in aiter have always been routed, and Gluon does not change it.

        The consequence is worth knowing: ``aiter`` has no language layer
        (``resolve_language_dirs("aiter", ...) == ()``), so an aiter-hosted
        Gluon kernel gets the framework cards and no Gluon authoring cards. Pass
        ``--kernel-backend gluon`` explicitly for such a campaign, or accept that
        the language knowledge is absent. Changing the precedence would re-route
        every existing aiter campaign, so it is deliberately left alone.
        """
        path = tmp_path / "aiter" / "ops" / "triton" / "attention" / "k.py"
        path.parent.mkdir(parents=True)
        path.write_text(_GLUON_KERNEL)
        assert infer_kernel_backend([path]) == "aiter"


# ─── the escalation hint ───


class TestTritonEscalationHint:
    """A Triton campaign must be told the drop to Gluon is a move it can make.

    The prompt used to answer a codegen ceiling with "suggest CK or FlyDSL",
    which reads as "stop and recommend a different project". Converged autotune
    plus low MFMA utilization is a scheduling limit, and the response is one
    level down in the same toolchain.
    """

    @pytest.fixture()
    def triton_prompt(self, config) -> str:
        return build_single_kernel_backend_prompt(config, "triton")

    def test_names_the_escalation(self, triton_prompt):
        assert "Escalating to Gluon" in triton_prompt

    def test_states_the_trigger(self, triton_prompt):
        """Converged search + idle matrix core, explicitly not 'hardware limit'."""
        assert "Autotune converged" in triton_prompt
        # Matched on the collapsed text: the prompt is hard-wrapped, so any
        # phrase long enough to be meaningful spans a newline in the source.
        collapsed = " ".join(triton_prompt.split())
        assert "matrix core far from peak is NOT" in collapsed

    def test_routes_to_the_forge_shape_card_before_the_edit(self, triton_prompt):
        assert "forge_integration.md" in triton_prompt

    def test_does_not_present_it_as_someone_elses_job(self, triton_prompt):
        assert "You may do this yourself" in triton_prompt


class TestAiterKernelBackendRoutesToAuthoringKnowledge:
    """aiter has no language layer, so it must at least name the route.

    ``resolve_language_dirs("aiter", ...)`` is empty by design -- aiter kernels
    are written in six different languages and inlining all six maps would swamp
    the prompt. But the prompt used to list only ``framework/aiter/``,
    ``hardware/`` and ``common_methodology/``, so a campaign that decided it
    needed to author a kernel had no route from the prompt to any authoring
    folder at all. That is not Gluon-specific; Gluon only made it visible,
    because aiter is where production Gluon lives (``ops/triton/`` holds Gluon
    kernels behind a ``@triton.jit`` fallback).
    """

    @pytest.fixture()
    def aiter_prompt(self, config) -> str:
        return build_single_kernel_backend_prompt(config, "aiter")

    def test_has_no_language_layer(self, config):
        """The premise: this is why the pointer has to be in the prompt text."""
        root = Path(config.local_knowledge_dir)
        assert resolve_language_dirs("aiter", root) == ()

    def test_names_the_authoring_route(self, aiter_prompt):
        assert "languages/<lang>/" in aiter_prompt

    @pytest.mark.parametrize("lang", ["triton", "gluon", "hip", "ck", "flydsl"])
    def test_every_authoring_language_is_reachable(self, aiter_prompt, lang):
        assert f"languages/{lang}/" in aiter_prompt

    def test_says_the_layer_is_not_inlined(self, aiter_prompt):
        """Otherwise the agent waits for a map that never arrives."""
        assert "NOT inlined" in aiter_prompt

    def test_warns_that_the_path_is_not_the_language(self, aiter_prompt):
        collapsed = " ".join(aiter_prompt.split())
        assert "aiter keeps Gluon kernels under `ops/triton/`" in collapsed


class TestGluonPromptDiscipline:
    """What the Gluon prompt must carry, beyond the shared kernel backend contract."""

    @pytest.fixture()
    def gluon_prompt(self, config) -> str:
        return build_single_kernel_backend_prompt(config, "gluon")

    def test_probes_the_toolchain_before_writing(self, gluon_prompt):
        """Gluon is triton.experimental and has shipped release-to-release breakage."""
        assert "triton.experimental" in gluon_prompt
        assert "PROBE" in gluon_prompt

    def test_states_the_same_file_dispatch_shape(self, gluon_prompt):
        """A new file is not committed by a KEEP unless the campaign allowlisted it."""
        assert "SAME TRACKED FILE" in gluon_prompt
        assert "--commit-new-path" in gluon_prompt

    def test_warns_that_env_vars_are_part_of_the_measurement(self, gluon_prompt):
        assert "TRITON_ENABLE_LLIR_SCHED" in gluon_prompt

    def test_does_not_hardcode_tuning_numbers(self, gluon_prompt):
        """Tile sizes and TFLOPS belong in the cards, which are versioned and dated."""
        for memorized in ("1489", "256x256x64", "5255"):
            assert memorized not in gluon_prompt, (
                f"gluon prompt hardcodes {memorized!r}; it belongs in a knowledge card"
            )
