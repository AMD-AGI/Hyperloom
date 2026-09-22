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
from kernelforge.loop.scoring import canonical_gate_prompt, runs_task_suite_acceptance


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
# The acceptance-gate correction moved all nine, the only re-snapshot so far
# that had to: the loop stopped running the task's declared suite for every
# backend but assembly, and each prompt still described the removed step.
# Diffed line by line. The eight non-assembly prompts swapped the nine-line
# task-config paragraph for the seven-line driver one; assembly kept its
# paragraph less the warm-start clause, which no path performs any more; the
# five agent-facing sentences that repeated the claim outside the shared
# paragraph were corrected (aiter/ck/hip/triton step 4, flydsl step 4, ck's
# iron rule, fusion's stop condition, gluon's smaller-shape trap); "GATE" as a
# name for the performance target became "performance target" in ck/hip/
# hipblaslt, since the word now has a defined meaning beside it; and hipblaslt
# gained the paragraph it had never rendered despite three stop conditions
# resting on it.
# Sending the sweep to a copy moved the eight source backends and left assembly
# byte-identical, which is the expected shape: the two edits are both in the
# shared prompt_utils.py preamble, and assembly renders neither card. Diffed
# line by line -- twenty lines in each of the eight, and the changed content is
# one identical block in all of them (same md5 over the diff bodies). The
# edit-surface bullet gained the converse it had always implied but never said
# (reaching someone else's environment constant is in scope, authoring a new
# read in the deliverable is not), and the sweep bullet moved the knobs into a
# copy under forge_experiments/ and replaced "KEEP the knobs ... strip at
# submission" with what the submitted kernel carries instead.
_SHA256_FORGE_LOOP: dict[str, str] = {
    "aiter": "4998333601ec64f50f489f991ba6a8a3bd8884bf6489ad05b45dca5fb1e0c9d0",
    "assembly": "67ce0c680f6b603d7c656feb1f1cc1f5eaf1bf4f6afc5d9f368b0361dd4b1492",
    "ck": "ced4d49611f49aaf9cc83dbfd6503a96d134cb635e9c2f293b2aa4eac41e8432",
    "flydsl": "f0e3277302f47ab8588c366a457008932094f17ad59b3421eed4a719ad21e193",
    "fusion": "f8dff507ff548abd7887bd5482e58fef11be1ee0907680e112f442e46e3bb3d5",
    "gluon": "6c14947fa8acc317a280b9fbb4dea22a9e21ad145a669db1242d6e690b35e125",
    "hip": "5b3732328ff0e5c3f341236677ff409dc5a10851bb874ab4566378fa2f8217ba",
    "hipblaslt": "0025272647704539649c243fd756d4433f0e0f73c77b4b5341985f69f780c0c6",
    "triton": "14d0f939891735764ed7d425134b527adbdcd3a6c52f82be8331f465b9ace8e7",
}


class TestRenderedPromptSnapshots:
    """Full sha256 of every rendered prompt — catches any output change."""

    def test_forge_loop_backends(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            got = _sha256(prompt)
            assert got == _SHA256_FORGE_LOOP[backend], (
                f"{backend}: forge-loop prompt changed (got {got!r}, expected {_SHA256_FORGE_LOOP[backend]!r})"
            )


class TestAcceptanceGateMatchesTheLoop:
    """Every prompt states the gate the loop will actually apply to that backend.

    The description and the loop's Step 7 were two copies of one fact, and they
    drifted: the step became assembly's alone while all eight other prompts kept
    telling their agent that forge runs the task's `compile_command` and
    `correctness_command` on every candidate. An agent optimizes against the
    criterion it is given, so that is not a stale sentence -- it points the
    implementer at a verdict no longer formed and at a diagnostic never
    produced. Both sides now read ``runs_task_suite_acceptance``.
    """

    def test_each_prompt_carries_the_gate_its_backend_is_judged_by(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            assert canonical_gate_prompt(backend) in prompt, (
                f"{backend}: prompt does not state the acceptance gate the loop applies to it"
            )

    def test_only_a_task_suite_backend_claims_forge_runs_the_task_config(self, forge_loop_prompts):
        for backend, prompt in forge_loop_prompts.items():
            claims = "`correctness_command`" in prompt
            assert claims is runs_task_suite_acceptance(backend), (
                f"{backend}: prompt {'claims' if claims else 'omits'} the task-config gate while the loop "
                f"{'runs' if runs_task_suite_acceptance(backend) else 'does not run'} it"
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

    @pytest.fixture()
    def source_loop_prompts(self, forge_loop_prompts):
        # Assembly edits one .s behind a fixed launcher, without Python sweep knobs.
        return {name: prompt for name, prompt in forge_loop_prompts.items() if name != "assembly"}

    def test_assembly_has_a_fixed_edit_surface(self, forge_loop_prompts):
        prompt = forge_loop_prompts["assembly"]
        assert "Only the task's declared .s file is editable" in prompt
        assert "Python launcher, binding manifest, driver, ABI and specialization are frozen" in prompt
        assert "There is no LLM PORT phase and no handwritten replacement seed" in prompt
        assert "FLOOR, not a ceiling" not in prompt
        assert "FORGE_SWEEP_" not in prompt

    def test_every_kernel_backend_points_at_the_sweep_card(self, source_loop_prompts):
        for backend, prompt in source_loop_prompts.items():
            assert _SWEEP_CARD in prompt, f"{backend}: prompt does not name the shared sweep card"
            assert "FORGE_SWEEP_" in prompt, f"{backend}: prompt does not carry the sweep-knob contract"
            assert "sweep_const" in prompt, f"{backend}: prompt does not carry the sweep echo contract"

    def test_every_kernel_backend_points_at_the_edit_surface_card(self, source_loop_prompts):
        for backend, prompt in source_loop_prompts.items():
            assert _EDIT_SURFACE_CARD in prompt, f"{backend}: prompt does not name the edit-surface card"
            assert "editable_sources" in prompt, f"{backend}: prompt never names the editable source list"
            assert "os.environ" in prompt, f"{backend}: prompt does not state the os.environ converse"
            # The declared list is a floor: `agent.py` tells repository tasks that any tracked non-protected
            # implementation file is editable, so a prompt presenting the list as the boundary contradicts the rest of
            # its own assembly -- in the direction that lost the campaigns.
            assert "FLOOR, not a ceiling" in prompt, f"{backend}: prompt presents the editable list as a ceiling"

    def test_every_kernel_backend_carries_the_boolean_parse_warning(self, source_loop_prompts):
        """A knob that cannot be turned off is the sweep bug the echo cannot catch."""
        for backend, prompt in source_loop_prompts.items():
            assert 'bool("0")' in prompt, (
                f"{backend}: prompt does not warn that a bool-cast swept string is always True"
            )

    def test_every_kernel_backend_keeps_sweep_plumbing_out_of_what_it_submits(self, source_loop_prompts):
        """Sweep knobs are scaffolding for the copy; the kernel that ships carries the literal they picked.

        The prompt used to say the opposite -- keep the knobs in the source for the whole search, strip them at
        submission if anyone asks. Nobody asked, so every campaign delivered its scaffolding: sixty environment reads
        and three hundred echo lines in one shipped kernel, indistinguishable to a reader from live configuration.
        Sending the sweep to a copy costs the search nothing and is the only version of the rule that holds without a
        cleanup step somebody has to remember.
        """
        for backend, prompt in source_loop_prompts.items():
            lowered = prompt.lower()
            assert "keep the knobs" not in lowered, (
                f"{backend}: prompt still tells the implementer to ship its own sweep knobs"
            )
            assert "a copy of the kernel" in lowered, (
                f"{backend}: prompt does not send the sweep to a copy"
            )
            assert "winning literal" in lowered, (
                f"{backend}: prompt does not say the submitted kernel carries the literal, not the read"
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
