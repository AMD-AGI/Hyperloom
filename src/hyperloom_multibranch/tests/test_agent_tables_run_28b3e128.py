# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuned GEMM tables of run 28b3e128 (Qwen3-14B-FP8) as the session held them, in ``data/agent_tables``:

- ``b01_listed.csv``: the rows b01's agent listed in its launch value of the variable, after the image's default tables.
- ``integration_ee3ab820.csv``: the integration head's rows-only table (blob ``ee3ab820``).
- ``b04_d67f257c.csv``: b04's earlier table (blob ``d67f257c``), whose rows the tuner did not measure have a time of 0.
- ``b02_accepted.csv``: the rows of the table b02's agent reported in ``accepted_config.env`` for the shapes integration
  also has.

b01 and integration set six shapes to other rows; b01's row is faster only at M=8192 N=5120 K=5120.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import (
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore, kernel_attachments, kernel_load_env, kernel_rows_only
from hyperloom_multibranch.repo.merge import Stored, carry, stack_kernel_slot
from hyperloom_multibranch.specialist.agent_import import (
    KeepStores,
    KeptChange,
    import_delta,
    read_kept_change,
    with_tables,
)
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import ENV, install

DATA = Path(__file__).parent / "data" / "agent_tables"
TARGET = "qwen3-14b-qkv-o"
IMAGE_TABLES = "/sgl-workspace/aiter/aiter/configs"
START = LaunchConfig("--fp8-gemm-backend aiter --kv-cache-dtype fp8_e4m3")
BASE = Recipe("rocm/sglang@sha256:" + "c" * 64, START)
#: The image's default tables b01's launch value listed before its own, as its recipe 70b23a1 has them.
DEFAULTS = (
    f"{IMAGE_TABLES}/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
    f"{IMAGE_TABLES}/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv",
    f"{IMAGE_TABLES}/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)


def _data(name: str) -> bytes:
    return (DATA / name).read_bytes()


def _rows(table: bytes) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(row["M"], row["N"], row["K"]): row for row in csv.DictReader(io.StringIO(table.decode()))}


class Tables:
    """The session's blob store, as both the toolbox and the experiment repository reach it."""

    def __init__(self, root: Path) -> None:
        self.blobs = LocalBlobStore(root)

    def keep(self, data: bytes) -> str:
        self.blobs.put(data)
        return hashlib.sha256(data).hexdigest()

    def artifact(self, data: bytes) -> KernelArtifact:
        self.keep(data)
        return mint_kernel_artifact(KernelKind.GEMM, TARGET, ENV, data, rows_only=True)

    async def put(self, env: str, table: str, rows_only: bool) -> KernelArtifact:
        assert env == ENV and rows_only
        return self.artifact(table.encode())

    def stored(self, configs: Path) -> Stored:
        def no_patch(_component: str, _text: str) -> PatchRef:
            raise AssertionError("no patch was expected")

        return Stored(lambda _stack, _paths: {}, lambda a: self.blobs.get(a.sha256), no_patch, self.keep, configs)


async def _no_patch(_component: str, _text: str) -> PatchRef:
    raise AssertionError("no patch was expected")


async def _no_source(imported: tuple[PatchRef, ...], cumulative: str) -> str:
    return cumulative


async def _imported(kept: KeptChange, tables: Tables) -> Recipe:
    stores = KeepStores(_no_patch, tables.put, _no_source)
    delta = await import_delta(kept, root=BASE, head=BASE, component="sglang", stores=stores)
    assert delta is not None
    return apply(BASE, delta)


@pytest.fixture
def configs(tmp_path) -> Path:
    return install(tmp_path / "image")


async def test_a_table_listed_in_a_launch_value_merges_with_integration_s_into_one_artifact(
    tmp_path: Path, configs: Path
) -> None:
    """Every merge of b01's commits was refused at test build: the recipe set the variable and integration's artifact
    set it again."""

    table = tmp_path / "b01" / "worktree" / "listed.csv"
    table.parent.mkdir(parents=True)
    table.write_bytes(_data("b01_listed.csv"))
    tables = Tables(tmp_path / "blobs")
    kept = KeptChange(1, "", START.server_args, {ENV: ":".join([*DEFAULTS, str(table)])})

    candidate = await _imported(with_tables(kept, (table.parent,)), tables)
    head = apply(BASE, RecipeDelta(add_kernels=(tables.artifact(_data("integration_ee3ab820.csv")),)))
    merge = carry("i1", BASE, candidate, head, tables.stored(configs))
    merged = apply(head, merge.delta)

    assert ENV not in candidate.launch.envs and [kernel_load_env(a) for a in candidate.kernels] == [ENV]
    (united,) = merged.kernels
    assert kernel_rows_only(united) and ENV not in merged.launch.envs
    _, envs = kernel_attachments(merged.kernels, merged.launch.envs, tables.blobs)
    assert list(envs) == [ENV]
    shapes = set(_rows(_data("b01_listed.csv"))) | set(_rows(_data("integration_ee3ab820.csv")))
    assert set(_rows(tables.blobs.get(united.sha256))) == shapes


async def test_a_shape_both_tables_set_takes_the_row_the_tuner_timed_faster(tmp_path: Path, configs: Path) -> None:
    tables = Tables(tmp_path / "blobs")
    listed, integration = _data("b01_listed.csv"), _data("integration_ee3ab820.csv")
    theirs, ours = tables.artifact(listed), tables.artifact(integration)

    merge = carry("i1", BASE, _with(theirs), _with(ours), tables.stored(configs))

    (united,) = merge.united_tables
    rows = _rows(tables.blobs.get(united.table.sha256))
    shared = set(_rows(listed)) & set(_rows(integration))
    faster = {
        shape: min(_rows(listed)[shape], _rows(integration)[shape], key=lambda row: float(row["us"]))
        for shape in shared
    }
    assert len(rows) == len(_rows(listed)) + len(_rows(integration)) - 6
    assert {shape: rows[shape] for shape in shared} == faster
    assert rows[("8192", "5120", "5120")]["us"] == "223.8886"
    assert {shape for shape in shared if rows[shape] == _rows(listed)[shape]} == {("8192", "5120", "5120")}
    assert {(s["M"], s["N"], s["K"]) for s in united.kept_head_shapes} == shared - {("8192", "5120", "5120")}


def test_a_row_the_tuner_did_not_time_does_not_win_a_shape_by_its_zero_time(tmp_path: Path, configs: Path) -> None:
    tables = Tables(tmp_path / "blobs")
    untimed, integration = _data("b04_d67f257c.csv"), _data("integration_ee3ab820.csv")

    kept = stack_kernel_slot(tables.artifact(untimed), tables.artifact(integration), tables.stored(configs))

    assert kept is not None
    rows, head_rows = _rows(tables.blobs.get(kept.sha256)), _rows(integration)
    zero = {
        shape for shape, row in _rows(untimed).items() if head_rows.get(shape, row) != row and float(row["us"]) == 0
    }
    assert len(zero) == 10 and all(rows[shape] == head_rows[shape] for shape in zero)


async def test_a_table_an_agent_reports_in_its_result_file_is_imported_as_an_artifact(
    tmp_path: Path, configs: Path
) -> None:
    """b02's keep changed nothing MB could read: its table was only in ``accepted_config.env`` of its result file."""

    checkout = tmp_path / "b02" / "session" / "agent" / "src"
    table = checkout / "mb-run" / "geak" / "eval" / "tuning" / "deploy" / "files" / "accepted.csv"
    table.parent.mkdir(parents=True)
    table.write_bytes(_data("b02_accepted.csv"))
    result = {
        "accepted_config": {"flags": "", "env": f"{ENV}={table}"},
        "accepted_kernels": [{"short_name": "gemm_a8w8_blockscale_bpreshuffle", "artifact": str(table)}],
    }
    manifest = AgentManifest.read(
        {
            "source": "/geak",
            "commit": "0881e29473e5554f535ece584bfd15e2c09db906",
            "config_file": "mb-run/handoff.json",
            "config": {},
            "launch": "python3 interface/run_e2e.py mb-run/handoff.json mb-run/result.json",
            "keep": {"path": "mb-run/geak/eval/workflow_return.json", "field": "accepted_kernels", "where": {}},
            "config_source": {
                "path": "mb-run/geak/eval/workflow_return.json",
                "root": "checkout",
                "server_args": "accepted_config.flags",
                "envs": "accepted_config.env",
                "adds": True,
            },
        }
    )
    tables = Tables(tmp_path / "blobs")
    assert manifest.keep is not None and manifest.config_source is not None
    assert manifest.keep.count(json.loads(json.dumps(result))) == 1

    kept = read_kept_change(1, "", result, manifest.config_source, "", START)
    candidate = await _imported(with_tables(kept, (tmp_path / "b02" / "worktree", checkout.parent)), tables)
    head = apply(BASE, RecipeDelta(add_kernels=(tables.artifact(_data("integration_ee3ab820.csv")),)))
    merged = apply(head, carry("i1", BASE, candidate, head, tables.stored(configs)).delta)

    (artifact,) = candidate.kernels
    assert kernel_rows_only(artifact) and tables.blobs.get(artifact.sha256) == _data("b02_accepted.csv")
    assert candidate.launch == START
    _, envs = kernel_attachments(merged.kernels, merged.launch.envs, tables.blobs)
    assert list(envs) == [ENV]


def _with(artifact: KernelArtifact) -> Recipe:
    return apply(BASE, RecipeDelta(add_kernels=(artifact,)))
