# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
from pathlib import Path

from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    apply,
)
import pytest

from hyperloom_multibranch.specialist.agent_import import (
    KeepStores,
    KeptChange,
    KeptChangeError,
    added_at,
    import_delta,
    read_kept_change,
    with_tables,
)
from hyperloom_multibranch.specialist.agent_manifest import ConfigSource
from hyperloom_multibranch.specialist.worktree import Upstream

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


async def _no_table(env: str, table: str) -> KernelArtifact:
    raise AssertionError(f"no table was expected, got one for {env}")


def _stores(store: "Store", table=_no_table) -> KeepStores:
    return KeepStores(store, table)


def _kept(seq: int, patch: str, args: str = ROOT.launch.server_args, envs: dict[str, str] | None = None) -> KeptChange:
    return KeptChange(seq, patch, args, dict(ROOT.launch.envs) if envs is None else envs, summary=f"keep {seq}")


def _patches(recipe: Recipe) -> tuple[PatchRef, ...]:
    [component] = recipe.components
    return component.patches


async def test_the_first_keep_adds_its_patch_on_top_of_the_root_s_own() -> None:
    store = Store()

    delta = await import_delta(_kept(1, "diff A\n"), root=ROOT, head=ROOT, component="sglang", stores=_stores(store))

    assert delta is not None
    made = apply(ROOT, delta)
    assert [store.texts.get(ref.sha256) for ref in _patches(made)] == [None, "diff A\n"]
    assert _patches(made)[0] == OWN and made.launch == ROOT.launch


async def test_a_later_keep_replaces_the_import_the_head_committed_with_the_new_cumulative_patch() -> None:
    store = Store()
    first = await import_delta(_kept(1, "diff A\n"), root=ROOT, head=ROOT, component="sglang", stores=_stores(store))
    assert first is not None
    head = apply(ROOT, first)

    delta = await import_delta(_kept(2, "diff A+B\n"), root=ROOT, head=head, component="sglang", stores=_stores(store))

    assert delta is not None
    made = apply(head, delta)
    assert [store.texts.get(ref.sha256) for ref in _patches(made)] == [None, "diff A+B\n"]


async def test_a_rejected_keep_rides_in_the_next_import_against_the_head_that_never_took_it() -> None:
    store = Store()

    delta = await import_delta(_kept(2, "diff A+B\n"), root=ROOT, head=ROOT, component="sglang", stores=_stores(store))

    assert delta is not None
    assert [store.texts.get(ref.sha256) for ref in _patches(apply(ROOT, delta))] == [None, "diff A+B\n"]


async def test_the_agent_s_server_arguments_and_environment_replace_the_head_s() -> None:
    kept = _kept(1, "", args="--tp 1 --kv-cache-dtype fp8_e4m3", envs={"SGLANG_B": "2"})

    delta = await import_delta(kept, root=ROOT, head=ROOT, component="sglang", stores=_stores(Store()))

    assert delta is not None
    made = apply(ROOT, delta)
    assert made.launch == LaunchConfig("--tp 1 --kv-cache-dtype fp8_e4m3", {"SGLANG_B": "2"})
    assert _patches(made) == (OWN,)


async def test_a_keep_that_changes_nothing_on_the_head_imports_nothing() -> None:
    store = Store()
    first = await import_delta(_kept(1, "diff A\n"), root=ROOT, head=ROOT, component="sglang", stores=_stores(store))
    assert first is not None

    again = await import_delta(
        _kept(2, "diff A\n"), root=ROOT, head=apply(ROOT, first), component="sglang", stores=_stores(store)
    )

    assert again is None
    assert await import_delta(_kept(1, ""), root=ROOT, head=ROOT, component="sglang", stores=_stores(store)) is None


def test_a_keep_reads_the_agent_s_config_through_its_config_source() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")
    state = {"server_flags": "--tp 1 --enable-torch-compile", "server_env": {"SGLANG_A": "0"}}

    kept = read_kept_change(3, "diff\n", state, source, "fused rmsnorm")

    assert kept == KeptChange(3, "diff\n", "--tp 1 --enable-torch-compile", {"SGLANG_A": "0"}, "fused rmsnorm")


def test_a_keep_whose_config_the_source_cannot_read_is_refused_naming_the_file() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")

    with pytest.raises(KeptChangeError, match="agent/config.json"):
        read_kept_change(1, "diff\n", {"server_flags": "--tp 1"}, source, "")


async def test_a_keep_of_a_component_new_on_the_head_names_the_upstream_its_package_was_built_from() -> None:
    bare = Recipe("sha256:" + "d" * 64, launch=ROOT.launch)
    upstream = Upstream(SGLANG, "2" * 40)

    imported = await import_delta(
        _kept(1, "diff A\n"), root=bare, head=bare, component="sglang", stores=_stores(Store())
    )
    assert imported is not None
    delta = added_at(imported, bare, upstream)

    [change] = delta.components
    assert (change.upstream_repo, change.upstream_ref) == (SGLANG, "2" * 40)
    [made] = apply(bare, delta).components
    assert (made.upstream_repo, made.upstream_ref) == (SGLANG, "2" * 40)


TABLE_ENV = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"


def _artifact(text: str, target: str = "gemm_a8w8") -> KernelArtifact:
    sha256 = hashlib.sha256(text.encode()).hexdigest()
    load = {"load": {"via": "env_file", "env": TABLE_ENV}}
    return KernelArtifact(KernelKind.GEMM, target, f"kernel/gemm/{target}/{sha256}.csv", sha256, load)


def test_a_keep_carries_the_text_of_each_table_file_of_the_agent_s_own_as_it_was_at_the_keep(tmp_path: Path) -> None:
    owned = tmp_path / "worktree"
    (owned / "configs").mkdir(parents=True)
    table = owned / "configs" / "tuned.csv"
    table.write_text("M,N,K\n64,5120,5120\n", encoding="utf-8")
    (tmp_path / "elsewhere.csv").write_text("M,N,K\n", encoding="utf-8")
    envs = {
        TABLE_ENV: str(table),
        "AITER_CONFIG_GEMM_BF16": str(tmp_path / "elsewhere.csv"),
        "AITER_CONFIG_FMOE": "configs/tuned.csv",
        "SGLANG_TABLE": str(table),
    }

    kept = with_tables(_kept(1, "", envs=envs), owned)
    table.write_text("rewritten after the keep\n", encoding="utf-8")

    assert kept.tables == {str(table): "M,N,K\n64,5120,5120\n"}


async def _table(env: str, table: str) -> KernelArtifact:
    assert env == TABLE_ENV
    return _artifact(table)


def _with_table(seq: int, path: str, table: str) -> KeptChange:
    kept = _kept(seq, "", envs={**ROOT.launch.envs, TABLE_ENV: path})
    return KeptChange(kept.seq, "", kept.server_args, kept.envs, kept.summary, {path: table})


async def test_a_table_the_agent_keeps_in_its_workspace_is_imported_as_a_kernel_artifact_not_a_path() -> None:
    path = "/branches/b01/worktree/tuned.csv"
    head = Recipe(ROOT.base_image_digest, LaunchConfig(ROOT.launch.server_args, {**ROOT.launch.envs, TABLE_ENV: path}))

    delta = await import_delta(
        _with_table(1, path, "rows\n"), root=ROOT, head=head, component="sglang", stores=_stores(Store(), _table)
    )

    assert delta is not None
    made = apply(head, delta)
    assert made.kernels == (_artifact("rows\n"),)
    assert TABLE_ENV not in made.launch.envs


async def test_a_later_table_replaces_the_imported_one_and_a_keep_without_it_drops_it() -> None:
    path, stores = "/w/tuned.csv", _stores(Store(), _table)
    imported = await import_delta(
        _with_table(1, path, "rows A\n"), root=ROOT, head=ROOT, component="sglang", stores=stores
    )
    assert imported is not None
    head = apply(ROOT, imported)

    replaced = await import_delta(
        _with_table(2, path, "rows A+B\n"), root=ROOT, head=head, component="sglang", stores=stores
    )
    dropped = await import_delta(_kept(3, ""), root=ROOT, head=head, component="sglang", stores=stores)
    same = await import_delta(_with_table(4, path, "rows A\n"), root=ROOT, head=head, component="sglang", stores=stores)

    assert replaced is not None and apply(head, replaced).kernels == (_artifact("rows A+B\n"),)
    assert dropped is not None and apply(head, dropped).kernels == ()
    assert same is None
