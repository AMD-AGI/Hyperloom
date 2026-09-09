"""Structural contracts for the two kernel backend prompt assembly paths."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

import kernelforge.kernel_backends as _kernel_backends_pkg
from kernelforge.config import Config
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.kernel_backends.constants import KERNEL_BACKENDS


_GPU = "gfx950"
_KB_SENTINEL = "KB_SENTINEL_VALUE"

# Directory of the kernel backends package as actually loaded, so a prompt that embeds a tool path does not make
# hashes depend on the checkout location.
_KERNEL_BACKENDS_ABS = os.path.abspath(os.path.dirname(_kernel_backends_pkg.__file__))


def _norm(text: str) -> str:
    return text.replace(_KERNEL_BACKENDS_ABS, "<KERNEL_BACKENDS_ROOT>")


def _sha256(text: str) -> str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


@pytest.fixture()
def forge_loop_prompts(monkeypatch):
    """build_single_kernel_backend_prompt for every backend, with mocked knowledge."""
    monkeypatch.setattr(
        "kernelforge.knowledge.build_forge_knowledge",
        lambda *a, **k: _KB_SENTINEL,
    )
    config = Config(gpu_target=_GPU)
    return {backend: build_single_kernel_backend_prompt(config, backend) for backend in KERNEL_BACKENDS}


# ---------- snapshot hashes -------------------------------------------------

# The fixture mocks ``build_forge_knowledge`` to a constant sentinel, so these
# hashes cover the prompt TEMPLATE only. A change to which knowledge folders a
# backend is served (``resolve_language_dirs``) leaves every hash here alone --
# which makes an unexpected diff in this table a precise signal that prompt text
# moved, not that knowledge assembly did.
# Every hash below has been re-snapshotted three times: once in the KernelForge
# -> Hyperloom merge (which rewrote one runnable command in a shared knowledge
# card -- the old package name in ``python3 -m <pkg>.mcp_server.tools.bench``),
# once for the backend-vocabulary rename, which reaches the prompt TEXT because
# each backend introduces itself by name ("You are the CK kernel backend --"),
# and once for the local_knowledge card renames (cheap_sweeps.md ->
# lever_cheap_sweeps.md and friends). That last one moved every hash even
# though only ck/hip/triton prompts.py changed, because the two cards every
# backend is pointed at live in the shared prompt_utils.py preamble.
# The intellikit backend's removal moved only aiter's hash: its prompt listed
# `languages/asm/` in the language-folder routing, and that folder went with the
# backend (diffed: one line changed, nothing else).
# The token-efficiency pass moved aiter/ck/hip/triton and nothing else. Two
# edits, both diffed line by line against the previous rendering: the four
# prompts stopped naming `build`/`test`/`bench`/`pmc`/`registers` as tools --
# this loop has Bash and the driver, those tools do not exist, and the framing
# paragraph in orchestrator/agent.py used to spend a sentence per session
# translating the names back into shell -- and triton's Gluon-escalation section
# deferred its mechanics to the two cards it already routes to, keeping the
# trigger, the ownership claim and the route resident (see
# test_gluon_backend.py::TestTritonEscalationHint, which pins exactly those).
# Each time the rendered prompts were diffed line by line against their previous
# rendering; for the card renames every changed line was a card name and nothing
# else moved. See test_rename_completeness.py for the tree-wide check.
_SHA256_FORGE_LOOP: dict[str, str] = {
    "aiter": "322db8617f4b69ce31a3b582cdda4ed09c4037a411f8161b81afb07874d23385",
    "ck": "8c8bd5b1b15e4f21bf70e729c3831de55a8efb7f29e868f99d5e9f73ed0e908e",
    "flydsl": "59115fbf5dd6c4cd22dc0c547d7a95c9992b64b6ac3f8f5f8a88853f03055937",
    "fusion": "d158dc07a0d00e0b36c5bc6d5e20d2f207285517829f5b96131b582ee4df3d3d",
    "gluon": "f127190e0da7240c7b05a6951d7f046cc88c7ce145383daf483d69ad8f4123cd",
    "hip": "7399928977cf188ae30f49fc0087386285d0fd5c131cd70b3a9064095f03fba7",
    "hipblaslt": "1ccbabae411cb958862fe9bf3cfbe5b1b9406467af18fa689e3bba1ccd2d646b",
    "triton": "67345584efeba90afc87959e75583167c11bb9b9ac9a0275dbddbed2f945037a",
}


class TestRenderedPromptSnapshots:
    """Full sha256 of every rendered prompt — catches any output change."""

    def test_forge_loop_backends(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            got = _sha256(prompt)
            assert got == _SHA256_FORGE_LOOP[backend], (
                f"{backend}: forge-loop prompt changed (got {got!r}, expected {_SHA256_FORGE_LOOP[backend]!r})"
            )


class TestNoInventedTools:
    """No backend prompt may name a tool the forge loop does not hand a session.

    The implementer gets Bash and the measurement driver. It has never had
    `build`, `test`, `bench`, `pmc` or `registers` tools, but four of these
    prompts instructed it to use them, and ``make_agent_fn`` compensated with a
    sentence -- carried in the cached prefix of every session, of every campaign
    -- explaining that those five names meant shell commands. A prompt that
    names the mechanism needs no such correction, so the correction is gone and
    this is what keeps it gone. `Read`, `Edit`, `Grep`, `Glob`, `Bash` and `Task`
    are real, hence the allowlist rather than a ban on the word "tool".
    """

    _INVENTED = ("`build` tool", "`test` tool", "`bench` tool", "`pmc` tool", "`registers` tool")

    def test_no_backend_prompt_names_an_invented_tool(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            collapsed = " ".join(prompt.split())
            for name in self._INVENTED:
                assert name not in collapsed, f"{backend}: prompt names the nonexistent {name}"


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
            assert "<coordination>" not in prompt, f"{backend}: forge-loop prompt has <coordination>"

    def test_gpu_target_present(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert _GPU in prompt, f"{backend}: forge-loop prompt missing GPU target"


# ---------- shared edit-surface / sweep contract ----------------------------

_SWEEP_CARD = "lever_cheap_sweeps.md"
_EDIT_SURFACE_CARD = "lever_edit_surface.md"
_LOOP_FORM_CARD = "lever_loop_form.md"


class TestEditSurfaceAndSweepContract:
    """The sweep contract is shared, always resident, and no longer self-erasing."""

    def test_every_kernel_backend_points_at_the_sweep_card(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert _SWEEP_CARD in prompt, f"{backend}: prompt does not name the shared sweep card"
            assert "FORGE_SWEEP_" in prompt, f"{backend}: prompt does not carry the sweep-knob contract"
            assert "sweep_const" in prompt, f"{backend}: prompt does not carry the sweep echo contract"

    def test_every_kernel_backend_points_at_the_edit_surface_card(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert _EDIT_SURFACE_CARD in prompt, f"{backend}: prompt does not name the edit-surface card"
            assert "editable_sources" in prompt, f"{backend}: prompt never names the editable source list"
            assert "os.environ" in prompt, f"{backend}: prompt does not state the os.environ converse"
            # The declared list is a floor: `agent.py` tells repository tasks that any tracked non-protected
            # implementation file is editable, so a prompt presenting the list as the boundary contradicts the rest of
            # its own assembly -- in the direction that lost the campaigns.
            assert "FLOOR, not a ceiling" in prompt, f"{backend}: prompt presents the editable list as a ceiling"

    def test_every_kernel_backend_carries_the_boolean_parse_warning(self, forge_loop_prompts):
        """A knob that cannot be turned off is the sweep bug the echo cannot catch."""
        for backend, prompt in forge_loop_prompts.items():
            assert 'bool("0")' in prompt, (
                f"{backend}: prompt does not warn that a bool-cast swept string is always True"
            )

    def test_no_kernel_backend_tells_the_implementer_to_collapse_the_knobs(self, forge_loop_prompts):
        """A knob deleted mid-campaign is an axis no later session re-opens."""
        for backend, prompt in forge_loop_prompts.items():
            lowered = prompt.lower()
            assert "collapse the knobs back" not in lowered, (
                f"{backend}: prompt still tells the implementer to delete its own sweep knobs"
            )
            assert "dead weight in the delivered kernel" not in lowered, (
                f"{backend}: prompt still calls a shipped sweep knob dead weight"
            )
            assert "keep the knobs" in lowered, (
                f"{backend}: prompt does not tell the implementer to keep the sweep knobs through the search"
            )

    def test_sweep_contract_is_not_owned_by_one_kernel_backend(self):
        """No kernel backend prompt module may re-privatize the shared contract."""
        kernel_backends_root = Path(_KERNEL_BACKENDS_ABS)
        owners = [
            path.relative_to(kernel_backends_root).as_posix()
            for path in sorted(kernel_backends_root.rglob("prompts.py"))
            if "FORGE_SWEEP_" in path.read_text(encoding="utf-8")
        ]
        assert owners == [], (
            "the sweep contract belongs in local_knowledge/common_methodology/, "
            f"not in a single kernel_backend prompt: {owners}"
        )


class TestSharedCardsAreReachable:
    """The new cards must be reachable through the real knowledge index."""

    @pytest.fixture()
    def knowledge_block(self) -> str:
        # Maps inlined on purpose: what is under test here is the *content* of
        # the INDEX maps -- whether they register the cards at all. Deferral
        # (the default) does not touch those files, it only replaces the inlined
        # copy with a pointer to them, so the registration is asserted against
        # the inlined form and the pointer path is covered separately below.
        config = Config(gpu_target=_GPU, defer_knowledge_maps=False)
        return build_single_kernel_backend_prompt(config, "flydsl")

    def test_cards_exist_on_disk(self):
        root = Path(Config(gpu_target=_GPU).local_knowledge_dir)
        for card in (_SWEEP_CARD, _EDIT_SURFACE_CARD, _LOOP_FORM_CARD):
            assert (root / "common_methodology" / "optimization" / card).is_file(), f"missing shared card: {card}"

    def test_assembled_knowledge_references_both_cards(self, knowledge_block):
        for card in (_SWEEP_CARD, _EDIT_SURFACE_CARD, _LOOP_FORM_CARD):
            assert f"optimization/{card}" in knowledge_block, (
                f"{card} is not reachable from the assembled knowledge block"
            )

    def test_loop_form_card_reaches_a_triton_kernel_context(self):
        """The loop-form rule must land in a Triton kernel's context specifically."""
        prompt = build_single_kernel_backend_prompt(
            Config(gpu_target=_GPU, defer_knowledge_maps=False),
            "triton",
            task_type="image_kernel",
            source_paths=["vllm/attention/ops/triton_sparse_attn_prefill.py"],
        )
        assert f"optimization/{_LOOP_FORM_CARD}" in prompt
        assert prompt.count(_LOOP_FORM_CARD) >= 2, (
            "expected the card in both the common_methodology and languages/triton maps"
        )

    def test_the_default_prompt_still_reaches_the_cards_through_the_pointers(self):
        """Under the default the maps are pointers, so reachability is a chain.

        Nothing is inlined, so the guarantee the two tests above make about the
        assembled prompt has to be re-made one link further out: the prompt must
        name the INDEX of each pillar that registers a card, and that INDEX --
        the file an agent is told to ``Read`` -- must actually register it.
        """
        root = Path(Config(gpu_target=_GPU).local_knowledge_dir)
        prompt = build_single_kernel_backend_prompt(
            Config(gpu_target=_GPU),
            "triton",
            task_type="image_kernel",
            source_paths=["vllm/attention/ops/triton_sparse_attn_prefill.py"],
        )
        assert _LOOP_FORM_CARD not in prompt, "the default must not inline the maps"

        for pillar, cards in (
            ("common_methodology", (_SWEEP_CARD, _EDIT_SURFACE_CARD, _LOOP_FORM_CARD)),
            ("languages/triton", (_LOOP_FORM_CARD,)),
        ):
            index = root / pillar / "INDEX.md"
            assert str(index) in prompt, f"the prompt never points at {pillar}/INDEX.md"
            registered = index.read_text()
            for card in cards:
                assert card in registered, f"{card} is not registered in {pillar}/INDEX.md"


class TestDocumentedSweepHelper:
    """The helper the card shows must round-trip a boolean knob."""

    @pytest.fixture()
    def sweep_const(self):
        card = Path(Config(gpu_target=_GPU).local_knowledge_dir) / "common_methodology" / "optimization" / _SWEEP_CARD
        blocks = re.findall(r"```python\n(.*?)```", card.read_text(encoding="utf-8"), re.DOTALL)
        assert len(blocks) == 1, f"{_SWEEP_CARD}: expected exactly one python block to lock, found {len(blocks)}"
        namespace: dict = {"os": os}
        exec(compile(blocks[0], str(card), "exec"), namespace)  # noqa: S102
        assert "_sweep_const" in namespace, f"{_SWEEP_CARD}: the documented block no longer defines _sweep_const"
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

    def test_an_unreadable_boolean_is_refused_not_guessed(self, sweep_const, monkeypatch):
        """A typo must fail the point, not silently time the default again."""
        monkeypatch.setenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", "maybe")
        with pytest.raises(ValueError, match="USE_FUSED_EPILOGUE"):
            sweep_const("USE_FUSED_EPILOGUE", True)

    @pytest.mark.parametrize(
        ("raw", "default", "expected"),
        [("64", 32, 64), ("1.5", 1.0, 1.5), ("nhwc", "nchw", "nhwc")],
    )
    def test_non_boolean_defaults_still_convert(self, sweep_const, monkeypatch, raw, default, expected):
        monkeypatch.setenv("FORGE_SWEEP_BLOCK_H", raw)
        assert sweep_const("BLOCK_H", default) == expected

    def test_every_read_echoes(self, sweep_const, monkeypatch, capsys):
        monkeypatch.setenv("FORGE_SWEEP_USE_FUSED_EPILOGUE", "0")
        sweep_const("USE_FUSED_EPILOGUE", True)
        assert "sweep_const: USE_FUSED_EPILOGUE 0" in capsys.readouterr().out
