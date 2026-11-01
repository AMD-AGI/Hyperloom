# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
from dataclasses import dataclass
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
from hyperloom.measure.compile import COMPONENT_LAYOUTS, LocalBlobStore, PackageBase, apply_patch_stack
import pytest

from hyperloom_multibranch.specialist.agent_import import (
    ComponentStart,
    KeepStores,
    KeptChange,
    KeptChangeError,
    added_at,
    import_delta,
    read_kept_change,
    read_snapshot,
    with_tables,
)
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.specialist import agent_import
from hyperloom_multibranch.specialist.agent_import import SourceIncrement
from hyperloom_multibranch.specialist.agent_manifest import ConfigSource
from hyperloom_multibranch.specialist.worktree import BranchWorktree, SourceBase, Upstream

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


async def _no_table(env: str, table: str, rows_only: bool) -> KernelArtifact:
    raise AssertionError(f"no table was expected, got one for {env}")


async def _from_root(imported: tuple[PatchRef, ...], cumulative: str) -> str:
    """The increment over a head that imported nothing yet: the keep's whole patch."""
    assert imported == (), f"the head imported nothing, got {imported}"
    return cumulative


def _stores(store: "Store", table=_no_table, increment: SourceIncrement = _from_root) -> KeepStores:
    return KeepStores(store, table, increment)


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


async def test_the_agent_s_server_arguments_and_environment_replace_the_head_s() -> None:
    kept = _kept(1, "", args="--tp 1 --kv-cache-dtype fp8_e4m3", envs={"SGLANG_B": "2"})

    delta = await import_delta(kept, root=ROOT, head=ROOT, component="sglang", stores=_stores(Store()))

    assert delta is not None
    made = apply(ROOT, delta)
    assert made.launch == LaunchConfig("--tp 1 --kv-cache-dtype fp8_e4m3", {"SGLANG_B": "2"})
    assert _patches(made) == (OWN,)


async def test_a_keep_with_no_source_change_and_the_head_s_launch_imports_nothing() -> None:
    assert await import_delta(_kept(1, ""), root=ROOT, head=ROOT, component="sglang", stores=_stores(Store())) is None


def test_a_keep_reads_the_agent_s_config_through_its_config_source() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")
    state = {"server_flags": "--tp 1 --enable-torch-compile", "server_env": {"SGLANG_A": "0"}}

    kept = read_kept_change(3, "diff\n", state, source, "fused rmsnorm", ROOT.launch)

    assert kept == KeptChange(3, "diff\n", "--tp 1 --enable-torch-compile", {"SGLANG_A": "0"}, "fused rmsnorm")


def test_a_keep_leaves_out_the_workload_variables_the_measurement_sets_and_names_them() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")
    state = {"server_flags": "--tp 1", "server_env": {"SGLANG_A": "0", "ISL": "1024", "osl": "1024"}}

    kept = read_kept_change(1, "", state, source, "", ROOT.launch)

    assert kept.envs == {"SGLANG_A": "0"}
    assert kept.dropped == {"ISL": "1024", "osl": "1024"}


def test_a_keep_whose_config_the_source_cannot_read_is_refused_naming_the_file() -> None:
    source = ConfigSource("agent/config.json", "server_flags", "server_env")

    with pytest.raises(KeptChangeError, match="agent/config.json"):
        read_kept_change(1, "diff\n", {"server_flags": "--tp 1"}, source, "", ROOT.launch)


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

    kept = with_tables(_kept(1, "", envs=envs), (owned,))
    table.write_text("rewritten after the keep\n", encoding="utf-8")

    assert kept.tables == {str(table): "M,N,K\n64,5120,5120\n"}


async def _table(env: str, table: str, rows_only: bool) -> KernelArtifact:
    assert env == TABLE_ENV and rows_only
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


def test_a_keep_reads_what_the_agent_accepted_on_top_of_its_start_from_a_result_file() -> None:
    """An agent that reports only the flags and NAME=VALUE environment it accepted runs with them over its start."""

    source = ConfigSource("result.json", "accepted_config.flags", "accepted_config.env", "checkout", adds=True)
    state = {"accepted_config": {"flags": "--mem-fraction-static 0.9 --page-size 16", "env": f"{TABLE_ENV}=/t.csv"}}

    kept = read_kept_change(1, "", state, source, "", ROOT.launch)

    assert kept.server_args == "--tp 1 --mem-fraction-static 0.9 --page-size 16"
    assert kept.envs == {"SGLANG_A": "1", TABLE_ENV: "/t.csv"}


def test_an_environment_string_that_is_not_name_value_words_is_refused_naming_the_file() -> None:
    source = ConfigSource("result.json", "flags", "env", "checkout", adds=True)

    with pytest.raises(KeptChangeError, match="result.json"):
        read_kept_change(1, "", {"flags": "", "env": "--not-an-assignment"}, source, "", ROOT.launch)


def _listing(*paths: object) -> str:
    image = "/sgl-workspace/aiter/aiter/configs"
    defaults = [f"{image}/a8w8_blockscale_bpreshuffle_tuned_gemm.csv", f"{image}/model_configs/dsv3.csv"]
    return ":".join([*defaults, *map(str, paths)])


def test_a_variable_listing_the_image_s_tables_before_the_agent_s_own_carries_only_the_agent_s_rows(
    tmp_path: Path,
) -> None:
    own, other = tmp_path / "worktree" / "a.csv", tmp_path / "agent" / "b.csv"
    for path, rows in ((own, "1,5120,5120\n64,5120,5120\n"), (other, "64,5120,5120\n8192,5120,5120\n")):
        path.parent.mkdir(parents=True)
        path.write_text(f"M,N,K\n{rows}", encoding="utf-8")
    envs = {TABLE_ENV: _listing(own, other), "AITER_CONFIG_GEMM_BF16": _listing(tmp_path / "elsewhere.csv")}

    kept = with_tables(_kept(1, "", envs=envs), (own.parent, other.parent))

    assert kept.tables == {envs[TABLE_ENV]: "M,N,K\n1,5120,5120\n64,5120,5120\n8192,5120,5120\n"}


def test_tables_of_one_variable_under_other_headers_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("M,N,K\n1,1,1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("M,N,K,us\n1,1,1,2.0\n", encoding="utf-8")
    envs = {TABLE_ENV: _listing(tmp_path / "a.csv", tmp_path / "b.csv")}

    with pytest.raises(KeptChangeError, match="headers"):
        with_tables(_kept(1, "", envs=envs), (tmp_path,))


def test_a_variable_listing_only_the_image_s_tables_stays_a_launch_value(tmp_path: Path) -> None:
    kept = with_tables(_kept(1, "", envs={TABLE_ENV: _listing()}), (tmp_path,))

    assert kept.tables == {}


ROOT_COMMIT = "a" * 40


def _arbor_keep(outbox: Path, value: str, copied: dict[str, bytes], digests: dict[str, str] | None = None) -> None:
    """Snapshot 1 of ``outbox`` as Arbor writes one whose ``TABLE_ENV`` is the pathsep list ``value``: each part in
    ``copied`` copied under ``tables/<VAR>/<k>-<basename>`` in list order, the other parts in its path only."""

    directory = outbox / "0001"
    (directory / "tables" / TABLE_ENV).mkdir(parents=True)
    files = []
    for k, part in enumerate(value.split(":")):
        if part not in copied:
            continue
        file = f"tables/{TABLE_ENV}/{k}-{Path(part).name}"
        (directory / file).write_bytes(copied[part])
        digest = (digests or {}).get(part, hashlib.sha256(copied[part]).hexdigest())
        files.append({"path": part, "file": file, "sha256": digest, "bytes": len(copied[part])})
    keep = {"number": 1, "previous": None, "root": ROOT_COMMIT, "parent": ROOT_COMMIT, "commit": "b" * 40,
            "change": "own rows", "tables": {TABLE_ENV: {"path": value, "files": files}}}  # fmt: skip
    (directory / "keep.json").write_text(json.dumps(keep), encoding="utf-8")
    (directory / "change.patch").write_text("", encoding="utf-8")
    config = {"server_flags": ROOT.launch.server_args, "server_env": {TABLE_ENV: value}}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _read(outbox: Path, copies: tuple[str, ...]) -> KeptChange:
    return read_snapshot(outbox, 1, ROOT_COMMIT, "".join, ComponentStart(ROOT.launch, {}, copies))


def _start_files(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    """MB's private copies of the image's default tables, the start's rows and the agent's own rows, on disk."""

    agent = tmp_path / "agent"
    defaults = [agent / "start-tables" / TABLE_ENV / "a.csv", agent / "start-tables" / TABLE_ENV / "model" / "b.csv"]
    start, own = agent / "start-tables" / "files" / "rows.csv", agent / "workspace" / "own.csv"
    texts = ("M,N,K\n1,5120,5120\n", "M,N,K\n2,5120,5120\n", "M,N,K\n64,5120,5120\n", "M,N,K\n256,5120,5120\n")
    for path, text in zip((*defaults, start, own), texts, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return defaults, start, own


def test_a_snapshot_reads_every_file_a_multi_part_aiter_value_lists_as_one_table_over_the_image_s(
    tmp_path: Path,
) -> None:
    defaults, start, own = _start_files(tmp_path)
    value = ":".join([*map(str, defaults), _listing(start, own)])
    _arbor_keep(tmp_path / "outbox", value, {str(path): path.read_bytes() for path in (*defaults, start, own)})

    kept = _read(tmp_path / "outbox", tuple(map(str, defaults)))

    on_disk = with_tables(_kept(1, "", envs={TABLE_ENV: _listing(start, own)}), (tmp_path,))
    assert kept.envs[TABLE_ENV] == value and kept.uncopied == ()
    assert (
        kept.tables == {value: on_disk.tables[_listing(start, own)]} == {value: "M,N,K\n64,5120,5120\n256,5120,5120\n"}
    )


def test_the_tables_of_a_vllm_image_s_aiter_are_the_image_s_not_the_agent_s_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ruby, 2026-10-31: in the vLLM image aiter lives in the Python's site-packages, not the SGLang workspace, and its
    # default tables were read as the agent's own rows.
    configs = tmp_path / "site-packages" / "aiter" / "configs"
    monkeypatch.setattr(agent_import, "image_aiter_configs", lambda: configs)
    defaults = [configs / "bf16_tuned_gemm.csv", configs / "model_configs" / "llama70B_bf16_tuned_gemm.csv"]
    own = tmp_path / "worktree" / "arbor" / "aiter_tables" / "qwen3_bf16_tuned_gemm.csv"
    texts = ("M,N,K\n1,4096,4096\n", "M,N,K\n2,4096,4096\n", "M,N,K\n8,6144,4096\n")
    for path, text in zip((*defaults, own), texts, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    value = ":".join(map(str, (*defaults, own)))
    _arbor_keep(tmp_path / "outbox", value, {str(path): path.read_bytes() for path in (*defaults, own)})

    kept = _read(tmp_path / "outbox", ())

    assert kept.envs[TABLE_ENV] == value and kept.tables == {value: "M,N,K\n8,6144,4096\n"}


def test_a_multi_part_value_whose_copy_does_not_match_its_sha256_or_is_missing_is_refused(tmp_path: Path) -> None:
    defaults, start, own = _start_files(tmp_path)
    value = ":".join(map(str, (*defaults, start, own)))
    copied = {str(path): path.read_bytes() for path in (*defaults, start, own)}
    _arbor_keep(tmp_path / "bad", value, copied, digests={str(own): "0" * 64})
    _arbor_keep(tmp_path / "missing", value, copied)
    (tmp_path / "missing" / "0001" / "tables" / TABLE_ENV / "3-own.csv").unlink()
    _arbor_keep(tmp_path / "unlisted", value, {str(path): copied[str(path)] for path in (*defaults, own)})

    with pytest.raises(KeptChangeError, match="sha256"):
        _read(tmp_path / "bad", tuple(map(str, defaults)))
    with pytest.raises(KeptChangeError, match="cannot be read"):
        _read(tmp_path / "missing", tuple(map(str, defaults)))
    with pytest.raises(KeptChangeError, match=str(start)):
        _read(tmp_path / "unlisted", tuple(map(str, defaults)))


#: The installed ``config.py`` the source tests edit: lines far enough apart that git writes a hunk for each.
LINES = [f"line_{number} = {number}\n" for number in range(1, 31)]
BARE = Recipe("sha256:" + "d" * 64, launch=ROOT.launch, components=(ComponentPatches("vllm", SGLANG, PIN),))


@dataclass
class Source:
    """An agent's workspace at the root of a session whose experiment repository builds on the same installed file."""

    workspace: BranchWorktree
    repo: GitExperimentRepo
    root: str
    root_commit: str
    installed: PackageBase
    blobs: LocalBlobStore

    def keep(self, **lines: str) -> str:
        """The workspace's cumulative patch once each named line holds its new text."""
        text = [f"{name} = {lines[name]}\n" if name in lines else line for line, name in _named(LINES)]
        (self.workspace.path / "vllm" / "config.py").write_text("".join(text))
        return self.workspace.diff_from(self.root_commit, "vllm")

    def stores(self, head: str) -> KeepStores:
        async def increment(imported: tuple[PatchRef, ...], cumulative: str) -> str:
            texts = [await self.repo.patch_text(head, ref.path) for ref in imported]
            return self.workspace.diff_between(self.root_commit, texts, cumulative, "vllm")

        return KeepStores(self.repo.store_patch, _no_table, increment)

    async def imported(self, kept: KeptChange, branch: str, head: str) -> str:
        """``kept`` imported against ``head`` and committed onto ``branch``, as the gate commits a gain."""
        recipe = await self.repo.read_recipe(head)
        delta = await import_delta(kept, root=BARE, head=recipe, component="vllm", stores=self.stores(head))
        assert delta is not None
        return await self.repo.commit(branch, head, apply(recipe, delta), f"keep {kept.seq}")

    def config(self, recipe: Recipe) -> str:
        stack = recipe.component("vllm")
        assert stack is not None
        [entry] = apply_patch_stack(stack, COMPONENT_LAYOUTS["vllm"], self.installed, self.blobs).entries
        return self.blobs.get(entry.blob).decode()


def _named(lines: list[str]) -> list[tuple[str, str]]:
    return [(line, line.split(" = ")[0]) for line in lines]


async def _source(tmp_path: Path) -> Source:
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_text("".join(LINES))
    installed = PackageBase("vllm", package, None, None)
    base = SourceBase(tmp_path / "source", locator=lambda _layout: installed)
    base.build(["vllm"])
    workspace = base.worktree("b01", tmp_path / "workspace")
    blobs = LocalBlobStore(tmp_path / "blobs")
    repo = GitExperimentRepo(tmp_path / "experiments.git", blobs, lambda _layout: installed)
    root = await repo.init_root(BARE)
    await repo.create_branch("b01", root)
    return Source(workspace, repo, root, workspace.commit(), installed, blobs)


def _expected(**lines: str) -> str:
    return "".join(f"{name} = {lines[name]}\n" if name in lines else line for line, name in _named(LINES))


async def test_a_later_keep_stacks_only_its_own_change_on_the_imports_the_head_committed(tmp_path: Path) -> None:
    source = await _source(tmp_path)
    first = await source.imported(_kept(1, source.keep(line_2="20")), "b01", source.root)

    second = _kept(2, source.keep(line_2="20", line_25="250"))
    head = await source.repo.read_recipe(first)
    delta = await import_delta(second, root=BARE, head=head, component="vllm", stores=source.stores(first))

    assert delta is not None
    [change] = delta.components
    assert change.remove_patches == ()
    [added] = change.add_patches
    text = source.blobs.get(added.sha256).decode()
    assert "+line_25 = 250" in text and "line_2 =" not in text
    made = apply(head, delta)
    assert made.component("vllm").patches[:-1] == head.component("vllm").patches
    assert source.config(made) == _expected(line_2="20", line_25="250")


async def test_a_rejected_keep_s_change_rides_in_the_next_import_against_the_head_that_never_took_it(
    tmp_path: Path,
) -> None:
    source = await _source(tmp_path)
    source.keep(line_2="20")
    second = _kept(2, source.keep(line_2="20", line_25="250"))

    delta = await import_delta(second, root=BARE, head=BARE, component="vllm", stores=source.stores(source.root))

    assert delta is not None
    assert source.config(apply(BARE, delta)) == _expected(line_2="20", line_25="250")


async def test_a_keep_the_head_already_is_imports_nothing(tmp_path: Path) -> None:
    source = await _source(tmp_path)
    first = await source.imported(_kept(1, source.keep(line_2="20")), "b01", source.root)

    again = _kept(2, source.keep(line_2="20"))
    head = await source.repo.read_recipe(first)

    assert await import_delta(again, root=BARE, head=head, component="vllm", stores=source.stores(first)) is None


async def test_a_keep_carries_an_earlier_unmerged_keep_that_clashes_with_integration_as_its_one_conflict(
    tmp_path: Path,
) -> None:
    """Run 4aef3318's b01: integration holds b04's patch over the lines of b01's first keep, so that keep never merged.
    A merge carries b01's lineage from the merge base, so the first keep's patch is the conflict and the second keep's
    hunk lands beside b04's."""

    source = await _source(tmp_path)
    b04_text = source.keep(line_2="2000", line_10="1000")
    await source.repo.create_branch("b04", source.root)
    b04 = await source.imported(_kept(1, b04_text), "b04", source.root)
    integration = await source.repo.commit_integration(
        source.root, (await source.repo.merge(b04, source.root)).delta, (b04,)
    )
    first = await source.imported(_kept(1, source.keep(line_2="20")), "b01", source.root)
    second = await source.imported(_kept(2, source.keep(line_2="20", line_25="250")), "b01", first)

    merge = await source.repo.merge(second, integration)

    first_patch = (await source.repo.read_recipe(first)).components[0].patches[0].path
    assert [conflict.path for conflict in merge.patch_conflicts] == [first_patch]
    assert merge.carried == (first, second) and merge.replaced_patches == ()
    merged = apply(await source.repo.read_recipe(integration), merge.delta)
    assert source.config(merged) == _expected(line_2="2000", line_10="1000", line_25="250")
