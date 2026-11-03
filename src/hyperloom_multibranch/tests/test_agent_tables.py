# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuned GEMM tables an agent keeps, carried into a recipe and merged with integration's. The tables are synthetic, in
aiter's column layout: ``HEAD`` is integration's, ``LISTED`` sets two of its shapes to other rows, one faster and one
slower, and ``UNTIMED`` sets one of them to a row the tuner wrote without timing it."""

import csv
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
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import ENV, HEADER, install

TARGET = "gemm-a8w8"
IMAGE_TABLES = "/sgl-workspace/aiter/aiter/configs"
START = LaunchConfig("--fp8-gemm-backend aiter --kv-cache-dtype fp8_e4m3")
BASE = Recipe("rocm/sglang@sha256:" + "c" * 64, START)
#: The image's default tables an agent's launch value lists before its own.
DEFAULTS = (
    f"{IMAGE_TABLES}/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
    f"{IMAGE_TABLES}/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv",
)
FASTER, SLOWER, NEW = ("1", "1024", "1024"), ("16", "1024", "1024"), ("32", "1024", "1024")


def _table(*rows: tuple[tuple[str, str, str], float, str]) -> bytes:
    lines = [HEADER]
    for (m, n, k), us, kernel in rows:
        measured = "1.00,100.00" if us else "0.0,0.0"
        lines.append(f"gfx950,256,{m},{n},{k},ck,1,0,{us:.4f},{kernel},{measured},0.0")
    return "".join(f"{line}\n" for line in lines).encode()


HEAD = _table((FASTER, 10.0, "kernel_a"), (SLOWER, 10.0, "kernel_a"), (("64", "1024", "1024"), 12.0, "kernel_b"))
LISTED = _table((FASTER, 8.0, "kernel_c"), (SLOWER, 14.0, "kernel_c"), (NEW, 15.0, "kernel_b"))
UNTIMED = _table((FASTER, 0.0, "kernel_c"), (NEW, 0.0, "kernel_c"))


def _rows(table: bytes) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(row["M"], row["N"], row["K"]): row for row in csv.DictReader(io.StringIO(table.decode()))}


class Tables:
    """The session's blob store, as both the toolbox and the experiment repository reach it."""

    def __init__(self, root: Path) -> None:
        self.blobs = LocalBlobStore(root)

    def keep(self, data: bytes) -> str:
        return self.blobs.put(data)

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


def _with(artifact: KernelArtifact) -> Recipe:
    return apply(BASE, RecipeDelta(add_kernels=(artifact,)))


@pytest.fixture
def configs(tmp_path) -> Path:
    return install(tmp_path / "image")


async def test_a_table_listed_in_a_launch_value_merges_with_integration_s_into_one_artifact(
    tmp_path: Path, configs: Path
) -> None:
    """A recipe that set the variable beside integration's artifact for it would set it twice and fail to build."""

    table = tmp_path / "b01" / "worktree" / "listed.csv"
    table.parent.mkdir(parents=True)
    table.write_bytes(LISTED)
    tables = Tables(tmp_path / "blobs")
    kept = KeptChange(1, "", START.server_args, {ENV: ":".join([*DEFAULTS, str(table)])})

    candidate = await _imported(with_tables(kept, (table.parent,)), tables)
    head = _with(tables.artifact(HEAD))
    merged = apply(head, carry("i1", BASE, candidate, head, tables.stored(configs)).delta)

    assert ENV not in candidate.launch.envs and [kernel_load_env(a) for a in candidate.kernels] == [ENV]
    (united,) = merged.kernels
    assert kernel_rows_only(united) and ENV not in merged.launch.envs
    _, envs = kernel_attachments(merged.kernels, merged.launch.envs, tables.blobs)
    assert list(envs) == [ENV]
    assert set(_rows(tables.blobs.get(united.sha256))) == set(_rows(LISTED)) | set(_rows(HEAD))


async def test_a_shape_both_tables_set_takes_the_row_the_tuner_timed_faster(tmp_path: Path, configs: Path) -> None:
    tables = Tables(tmp_path / "blobs")

    merge = carry("i1", BASE, _with(tables.artifact(LISTED)), _with(tables.artifact(HEAD)), tables.stored(configs))

    (united,) = merge.united_tables
    rows = _rows(tables.blobs.get(united.table.sha256))
    assert len(rows) == 4
    assert rows[FASTER] == _rows(LISTED)[FASTER] and rows[SLOWER] == _rows(HEAD)[SLOWER]
    assert [(s["M"], s["N"], s["K"]) for s in united.kept_head_shapes] == [SLOWER]


def test_a_row_the_tuner_did_not_time_does_not_win_a_shape_by_its_zero_time(tmp_path: Path, configs: Path) -> None:
    tables = Tables(tmp_path / "blobs")

    kept = stack_kernel_slot(tables.artifact(UNTIMED), tables.artifact(HEAD), tables.stored(configs))

    assert kept is not None
    rows = _rows(tables.blobs.get(kept.sha256))
    assert rows[FASTER] == _rows(HEAD)[FASTER] and rows[NEW] == _rows(UNTIMED)[NEW]


async def test_a_table_an_agent_reports_in_its_result_file_is_imported_as_an_artifact(
    tmp_path: Path, configs: Path
) -> None:
    """A keep whose table is named only in ``accepted_config.env`` of the agent's result file changes no launch."""

    checkout = tmp_path / "b02" / "session" / "agent" / "src"
    table = checkout / "mb-run" / "eval" / "accepted.csv"
    table.parent.mkdir(parents=True)
    table.write_bytes(LISTED)
    result = {
        "accepted_config": {"flags": "", "env": f"{ENV}={table}"},
        "accepted_kernels": [{"short_name": "gemm_a8w8_blockscale_bpreshuffle", "artifact": str(table)}],
    }
    manifest = AgentManifest.read(
        {
            "source": "/agent",
            "commit": "0" * 40,
            "config_file": "mb-run/handoff.json",
            "config": {},
            "launch": "python3 interface/run_e2e.py mb-run/handoff.json mb-run/result.json",
            "keep": {"path": "mb-run/eval/workflow_return.json", "field": "accepted_kernels", "where": {}},
            "config_source": {
                "path": "mb-run/eval/workflow_return.json",
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
    head = _with(tables.artifact(HEAD))
    merged = apply(head, carry("i1", BASE, candidate, head, tables.stored(configs)).delta)

    (artifact,) = candidate.kernels
    assert kernel_rows_only(artifact) and tables.blobs.get(artifact.sha256) == LISTED
    assert candidate.launch == START
    _, envs = kernel_attachments(merged.kernels, merged.launch.envs, tables.blobs)
    assert list(envs) == [ENV]
