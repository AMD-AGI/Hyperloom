# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, PatchRef, Recipe, apply
import pytest

from hyperloom_multibranch.specialist.agent_import import KeptChange, KeptChangeError, import_delta, read_kept_change
from hyperloom_multibranch.specialist.agent_manifest import ConfigSource

SGLANG = "https://github.com/sgl-project/sglang"
PIN = "1" * 40
OWN = PatchRef("patch/sglang/" + "e" * 64 + ".patch", "e" * 64)
ROOT = Recipe(
    "sha256:" + "d" * 64,
    launch=LaunchConfig("--tp 1 --mem-fraction-static 0.8", {"SGLANG_A": "1"}),
    components=(ComponentPatches("sglang", SGLANG, PIN, (OWN,)),),
)


class Store:
    """Stores each patch under its sha256, as the experiment repository does."""

    def __init__(self) -> None:
        self.texts: dict[str, str] = {}

    async def __call__(self, component: str, text: str) -> PatchRef:
        sha256 = hashlib.sha256(text.encode()).hexdigest()
        self.texts[sha256] = text
        return PatchRef(f"patch/{component}/{sha256}.patch", sha256)


def _kept(seq: int, patch: str, args: str = ROOT.launch.server_args, envs: dict[str, str] | None = None) -> KeptChange:
    return KeptChange(seq, patch, args, dict(ROOT.launch.envs) if envs is None else envs, summary=f"keep {seq}")


def _patches(recipe: Recipe) -> tuple[PatchRef, ...]:
    [component] = recipe.components
    return component.patches


async def test_the_first_keep_adds_its_patch_on_top_of_the_root_s_own() -> None:
    store = Store()

    delta = await import_delta(_kept(1, "diff A\n"), root=ROOT, head=ROOT, component="sglang", store=store)

    assert delta is not None
    made = apply(ROOT, delta)
    assert [store.texts.get(ref.sha256) for ref in _patches(made)] == [None, "diff A\n"]
    assert _patches(made)[0] == OWN and made.launch == ROOT.launch


async def test_a_later_keep_replaces_the_import_the_head_committed_with_the_new_cumulative_patch() -> None:
    store = Store()
    first = await import_delta(_kept(1, "diff A\n"), root=ROOT, head=ROOT, component="sglang", store=store)
    assert first is not None
    head = apply(ROOT, first)

    delta = await import_delta(_kept(2, "diff A+B\n"), root=ROOT, head=head, component="sglang", store=store)

    assert delta is not None
    made = apply(head, delta)
    assert [store.texts.get(ref.sha256) for ref in _patches(made)] == [None, "diff A+B\n"]


async def test_a_rejected_keep_rides_in_the_next_import_against_the_head_that_never_took_it() -> None:
    store = Store()

    delta = await import_delta(_kept(2, "diff A+B\n"), root=ROOT, head=ROOT, component="sglang", store=store)

    assert delta is not None
    assert [store.texts.get(ref.sha256) for ref in _patches(apply(ROOT, delta))] == [None, "diff A+B\n"]


async def test_the_agent_s_server_arguments_and_environment_replace_the_head_s() -> None:
    kept = _kept(1, "", args="--tp 1 --kv-cache-dtype fp8_e4m3", envs={"SGLANG_B": "2"})

    delta = await import_delta(kept, root=ROOT, head=ROOT, component="sglang", store=Store())

    assert delta is not None
    made = apply(ROOT, delta)
    assert made.launch == LaunchConfig("--tp 1 --kv-cache-dtype fp8_e4m3", {"SGLANG_B": "2"})
    assert _patches(made) == (OWN,)


async def test_a_keep_that_changes_nothing_on_the_head_imports_nothing() -> None:
    store = Store()
    first = await import_delta(_kept(1, "diff A\n"), root=ROOT, head=ROOT, component="sglang", store=store)
    assert first is not None

    again = await import_delta(
        _kept(2, "diff A\n"), root=ROOT, head=apply(ROOT, first), component="sglang", store=store
    )

    assert again is None
    assert await import_delta(_kept(1, ""), root=ROOT, head=ROOT, component="sglang", store=store) is None


def test_a_keep_reads_the_agent_s_config_through_its_config_source() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")
    state = {"server_flags": "--tp 1 --enable-torch-compile", "server_env": {"SGLANG_A": "0"}}

    kept = read_kept_change(3, "diff\n", state, source, "fused rmsnorm")

    assert kept == KeptChange(3, "diff\n", "--tp 1 --enable-torch-compile", {"SGLANG_A": "0"}, "fused rmsnorm")


def test_a_keep_whose_config_the_source_cannot_read_is_refused_naming_the_file() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")

    with pytest.raises(KeptChangeError, match="agent/config.json"):
        read_kept_change(1, "diff\n", {"server_flags": "--tp 1"}, source, "")
