"""Structural contracts for the two fellow prompt assembly paths.

Locks XML tag boundaries (<knowledge>, <skill>, <workspace>, <coordination>),
per-fellow Coordination Rules fingerprints, and full rendered-prompt sha256
snapshots so that any refactor that silently changes prompt output is caught.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

import kernelforge.fellows as _fellows_pkg
from kernelforge.config import Config
from kernelforge.fellows.base import build_single_fellow_prompt
from kernelforge.fellows.constants import FELLOW_BACKENDS


_GPU = "gfx950"
_KB_SENTINEL = "KB_SENTINEL_VALUE"

# Directory of the fellows package as actually loaded, so the intellikit prompt's
# embedded PATCH_CO path does not make hashes depend on the checkout location.
_FELLOWS_ABS = os.path.abspath(os.path.dirname(_fellows_pkg.__file__))


def _norm(text: str) -> str:
    return text.replace(_FELLOWS_ABS, "<FELLOWS_ROOT>")


def _sha256(text: str) -> str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


@pytest.fixture()
def forge_loop_prompts(monkeypatch):
    """build_single_fellow_prompt for every backend, with mocked knowledge."""
    monkeypatch.setattr(
        "kernelforge.knowledge.build_forge_knowledge",
        lambda *a, **k: _KB_SENTINEL,
    )
    config = Config(gpu_target=_GPU)
    return {
        backend: build_single_fellow_prompt(config, f"{backend}-fellow")
        for backend in FELLOW_BACKENDS
    }


# ---------- snapshot hashes -------------------------------------------------

# The fixture mocks ``build_forge_knowledge`` to a constant sentinel, so these
# hashes cover the prompt TEMPLATE only. A change to which knowledge folders a
# backend is served (``resolve_language_dirs``) leaves every hash here alone --
# which makes an unexpected diff in this table a precise signal that prompt text
# moved, not that knowledge assembly did.
_SHA256_FORGE_LOOP: dict[str, str] = {
    "aiter":      "3f0b7ebcdaa71aeed07863b6ebe4b5e6616c28589ab59938f30577c1fab49a53",
    "ck":         "7e6794a4dd0b17a087896083248bc7040c4c22a08985ea5bbdcca514cd731570",
    "flydsl":     "356c429f55c796de99790f653a8cd16b549f0aca8dd9365081e6affeca109f2e",
    "fusion":     "f461db9611095eb9a8c423bde73c9012985eae9d2cbc6afa93a508d09cc291b3",
    "gluon":      "d8a4f4cfc9aa4ce32958d44e51f637f040e13d7c35165160f2d78fdf9efcdf59",
    "hip":        "5217b3f51175ce13024d97d907a7bb63c786e09f14f69b78a6b515615984e32b",
    "hipblaslt":  "4a6be121500463f42bb82d78656e7d03de80917fb2c15dd3da9226605b9524c9",
    "intellikit": "26d3e33deaebe7a741059427a8e676e85eec2a52c3c1b5fa7136d4780bcc93a7",
    "triton":     "bca0355881da2717dcddde7f1c0ad3daf5e5f5978f54735e79f6b5a7502c9a9b",
}


class TestRenderedPromptSnapshots:
    """Full sha256 of every rendered prompt — catches any output change."""

    def test_forge_loop_backends(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            got = _sha256(prompt)
            assert got == _SHA256_FORGE_LOOP[backend], (
                f"{backend}: forge-loop prompt changed (got {got!r}, "
                f"expected {_SHA256_FORGE_LOOP[backend]!r})"
            )


class TestForgeLoopPath:
    """Forge-loop prompts carry knowledge but never skill/workspace/coordination."""

    def test_knowledge_block_present(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert f"<knowledge>\n{_KB_SENTINEL}\n</knowledge>" in prompt, (
                f"{backend}: forge-loop prompt missing <knowledge> block"
            )

    def test_no_skill_tag(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert "<skill>" not in prompt, f"{backend}: forge-loop prompt has <skill>"

    def test_no_workspace_tag(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert "<workspace>" not in prompt, f"{backend}: forge-loop prompt has <workspace>"

    def test_no_coordination_tag(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert "<coordination>" not in prompt, (
                f"{backend}: forge-loop prompt has <coordination>"
            )

    def test_gpu_target_present(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert _GPU in prompt, f"{backend}: forge-loop prompt missing GPU target"


# ---------- shared edit-surface / sweep contract ----------------------------

_SWEEP_CARD = "cheap_sweeps.md"
_EDIT_SURFACE_CARD = "edit_surface_reach.md"
_LOOP_FORM_CARD = "loop_form_and_pipelining.md"


class TestEditSurfaceAndSweepContract:
    """The sweep contract is shared, always resident, and no longer self-erasing.

    Two campaigns lost their largest available win on a fellow whose prompt never
    mentioned sweeps at all, because the contract lived only in the Triton
    prompt. It now lives in a ``common_methodology/`` card that every fellow
    receives, with an always-resident pointer in each prompt (the knowledge tree
    is Read-on-demand, so a card nobody opens teaches nothing).
    """

    def test_every_fellow_points_at_the_sweep_card(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert _SWEEP_CARD in prompt, (
                f"{backend}: prompt does not name the shared sweep card"
            )
            assert "FORGE_SWEEP_" in prompt, (
                f"{backend}: prompt does not carry the sweep-knob contract"
            )
            assert "sweep_const" in prompt, (
                f"{backend}: prompt does not carry the sweep echo contract"
            )

    def test_every_fellow_points_at_the_edit_surface_card(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert _EDIT_SURFACE_CARD in prompt, (
                f"{backend}: prompt does not name the edit-surface card"
            )
            assert "editable_sources" in prompt, (
                f"{backend}: prompt never names the editable source list"
            )
            assert "os.environ" in prompt, (
                f"{backend}: prompt does not state the os.environ converse"
            )
            # The declared list is a floor: `agent.py` tells repository tasks
            # that any tracked non-protected implementation file is editable,
            # so a prompt presenting the list as the boundary contradicts the
            # rest of its own assembly -- in the direction that lost the
            # campaigns.
            assert "FLOOR, not a ceiling" in prompt, (
                f"{backend}: prompt presents the editable list as a ceiling"
            )

    def test_every_fellow_carries_the_boolean_parse_warning(self, forge_loop_prompts):
        """A knob that cannot be turned off is the sweep bug the echo cannot catch.

        The echo prints the string the host sent, not the value the source made
        of it, so `bool("0")` returning True makes the OFF point time the ON
        kernel and still come back confirmed.
        """
        for backend, prompt in forge_loop_prompts.items():
            assert 'bool("0")' in prompt, (
                f"{backend}: prompt does not warn that a bool-cast swept string "
                "is always True"
            )

    def test_no_fellow_tells_the_implementer_to_collapse_the_knobs(
        self, forge_loop_prompts
    ):
        """A knob deleted mid-campaign is an axis no later session re-opens."""
        for backend, prompt in forge_loop_prompts.items():
            lowered = prompt.lower()
            assert "collapse the knobs back" not in lowered, (
                f"{backend}: prompt still tells the implementer to delete its "
                "own sweep knobs"
            )
            assert "dead weight in the delivered kernel" not in lowered, (
                f"{backend}: prompt still calls a shipped sweep knob dead weight"
            )
            assert "keep the knobs" in lowered, (
                f"{backend}: prompt does not tell the implementer to keep the "
                "sweep knobs through the search"
            )

    def test_sweep_contract_is_not_owned_by_one_fellow(self):
        """No fellow prompt module may re-privatize the shared contract."""
        fellows_root = Path(_FELLOWS_ABS)
        owners = [
            path.relative_to(fellows_root).as_posix()
            for path in sorted(fellows_root.rglob("prompts.py"))
            if "FORGE_SWEEP_" in path.read_text(encoding="utf-8")
        ]
        assert owners == [], (
            "the sweep contract belongs in local_knowledge/common_methodology/, "
            f"not in a single fellow prompt: {owners}"
        )


class TestSharedCardsAreReachable:
    """The new cards must be reachable through the real knowledge index.

    ``build_forge_knowledge`` loads ``common_methodology/INDEX.md`` whole, so a
    card that is not registered there is invisible to every fellow no matter what
    the prompt says.
    """

    @pytest.fixture()
    def knowledge_block(self) -> str:
        config = Config(gpu_target=_GPU)
        return build_single_fellow_prompt(config, "flydsl-fellow")

    def test_cards_exist_on_disk(self):
        root = Path(Config(gpu_target=_GPU).local_knowledge_dir)
        for card in (_SWEEP_CARD, _EDIT_SURFACE_CARD, _LOOP_FORM_CARD):
            assert (root / "common_methodology" / "optimization" / card).is_file(), (
                f"missing shared card: {card}"
            )

    def test_assembled_knowledge_references_both_cards(self, knowledge_block):
        for card in (_SWEEP_CARD, _EDIT_SURFACE_CARD, _LOOP_FORM_CARD):
            assert f"optimization/{card}" in knowledge_block, (
                f"{card} is not reachable from the assembled knowledge block"
            )

    def test_loop_form_card_reaches_a_triton_kernel_context(self):
        """The loop-form rule must land in a Triton kernel's context specifically.

        It is registered twice on purpose: once in ``common_methodology/INDEX.md``
        (every fellow) and once in ``languages/triton/INDEX.md``, because the
        recognition signature -- a ``while`` bounded by a ``tl.load`` -- is Triton
        syntax and a Triton author routes through the language map, not the
        methodology one.
        """
        prompt = build_single_fellow_prompt(
            Config(gpu_target=_GPU),
            "triton-fellow",
            task_type="image_kernel",
            source_paths=["vllm/attention/ops/triton_sparse_attn_prefill.py"],
        )
        assert f"optimization/{_LOOP_FORM_CARD}" in prompt
        assert prompt.count(_LOOP_FORM_CARD) >= 2, (
            "expected the card in both the common_methodology and languages/triton maps"
        )


class TestDocumentedSweepHelper:
    """The helper the card shows must round-trip a boolean knob.

    Every fellow now receives this card, so whatever it shows is what eight
    backends will paste into a kernel. ``type(default)(value)`` is ``bool(value)``
    for a boolean default, and ``bool("0")`` and ``bool("false")`` are both True:
    the OFF point then benchmarks the ON configuration, while the echo -- which
    reports the string the host sent, never the value the source computed --
    marks the point confirmed. The sweep closes a live axis it never varied,
    through the one contract that exists to prevent exactly that.
    """

    @pytest.fixture()
    def sweep_const(self):
        card = (
            Path(Config(gpu_target=_GPU).local_knowledge_dir)
            / "common_methodology"
            / "optimization"
            / _SWEEP_CARD
        )
        blocks = re.findall(
            r"```python\n(.*?)```", card.read_text(encoding="utf-8"), re.DOTALL
        )
        assert len(blocks) == 1, (
            f"{_SWEEP_CARD}: expected exactly one python block to lock, "
            f"found {len(blocks)}"
        )
        namespace: dict = {"os": os}
        exec(compile(blocks[0], str(card), "exec"), namespace)  # noqa: S102
        assert "_sweep_const" in namespace, (
            f"{_SWEEP_CARD}: the documented block no longer defines _sweep_const"
        )
        return namespace["_sweep_const"]

    def test_unset_knob_keeps_the_default(self, sweep_const, monkeypatch):
        monkeypatch.delenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", raising=False)
        assert sweep_const("USE_FUSED_EPILOGUE", True) is True

    @pytest.mark.parametrize("token", ["0", "false", "False", "no", "off", " 0 "])
    def test_a_boolean_knob_can_be_turned_off(self, sweep_const, monkeypatch, token):
        monkeypatch.setenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", token)
        assert sweep_const("USE_FUSED_EPILOGUE", True) is False, (
            f"{token!r} left the flag on: the OFF point would time the ON kernel"
        )

    @pytest.mark.parametrize("token", ["1", "true", "TRUE", "yes", "on"])
    def test_a_boolean_knob_can_be_turned_on(self, sweep_const, monkeypatch, token):
        monkeypatch.setenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", token)
        assert sweep_const("USE_FUSED_EPILOGUE", False) is True

    def test_an_unreadable_boolean_is_refused_not_guessed(
        self, sweep_const, monkeypatch
    ):
        """A typo must fail the point, not silently time the default again."""
        monkeypatch.setenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", "maybe")
        with pytest.raises(ValueError, match="USE_FUSED_EPILOGUE"):
            sweep_const("USE_FUSED_EPILOGUE", True)

    @pytest.mark.parametrize(
        ("raw", "default", "expected"),
        [("64", 32, 64), ("1.5", 1.0, 1.5), ("nhwc", "nchw", "nhwc")],
    )
    def test_non_boolean_defaults_still_convert(
        self, sweep_const, monkeypatch, raw, default, expected
    ):
        monkeypatch.setenv("FORGE_SWEEP_BLOCK_H", raw)
        assert sweep_const("BLOCK_H", default) == expected

    def test_every_read_echoes(self, sweep_const, monkeypatch, capsys):
        monkeypatch.setenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", "0")
        sweep_const("USE_FUSED_EPILOGUE", True)
        assert "sweep_const: USE_FUSED_EPILOGUE 0" in capsys.readouterr().out
