# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuned table a specialist puts is the table the test's server reads, on a real GPU (W85G).

``test_kernel_artifact.py`` proved ``put_kernel_artifact`` against the file-protocol stand-in runner. Here the specialist
is the real MCP service, the toolbox runs the real runner on two cards, and the server is SGLang with aiter serving
Qwen3-8B at TP1. Specialist ``s2`` puts a bf16 GEMM table (the image's own, plus a ``torch`` row for each linear layer of
the model at the decode batch sizes) and carries the artifact in the delta of an ``explore`` test; specialist ``s1`` runs
the same test without it. What must hold: the refusals of the tool (a second target for the variable is
``kernel_variable_taken``, an empty table, a target that leaves its directory, a variable and a kind the measurement
cannot load); the table's bytes in the blob store; the artifact test's ``framework_base.json`` naming the table's hash
as the table of the variable while the default test names none; the server's log naming the placed table in its
tuned-config lookups; both tests' merge directories private; the shared tables and ``/tmp/aiter_configs`` as they were.
What was measured is printed (``PROBE``).

Needs what ``test_toolbox_kinds_gpu`` needs, with ``HL_W13_CARDS`` naming two cards, and the image's aiter tables at
``/sgl-workspace``; anywhere else the module is skipped.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import TABLE_EXERCISED, TestResult, Validity
from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, RecipeDelta
from hyperloom.measure.framework.base import DEFAULT_CONFIGS, FRAMEWORK_BASE_FILE, load_snapshot, tree_files
from hyperloom.measure.framework.merge import PRIVATE
from hyperloom.measure.points import default_canonical
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.toolbox.tests.integration_inputs import IMAGE, baseline_recipe
from hyperloom_multibranch.toolbox.tests.integration_rig import Specialist, Stack, Variant, build_stack
from hyperloom_multibranch.toolbox.tests.test_framework_isolation_gpu import _landed
from hyperloom_multibranch.toolbox.tests.test_framework_isolation_parallel_gpu import (
    AITER_MERGE_DIR,
    SHARED,
    USE_AITER,
    VARIABLE,
    _installed_aiter_names_the_variable,
)
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import (  # noqa: F401  (workload_environment is a fixture)
    CARDS,
    INFERENCEX,
    MODEL,
    _probe,
    _run_root,
    pytestmark,
    workload_environment,
)

pytestmark = [
    pytestmark,
    pytest.mark.skipif(not SHARED.is_file(), reason=f"needs the image's aiter table {SHARED}"),
    pytest.mark.skipif(not _installed_aiter_names_the_variable(), reason=f"the installed aiter has no {VARIABLE}_FILE"),
]

TARGET = "gemm_bf16"
#: Makes aiter log the lookup of every shape it dispatches, hits included; the line names the table it looked in.
LOG_LOOKUPS = "AITER_LOG_TUNED_CONFIG"
ENVS = {USE_AITER: "1", LOG_LOOKUPS: "1"}
#: The decode batch sizes of a small test: the rows put are for these.
DECODE_BATCHES = (1, 2, 4, 8, 12, 16, 24, 32)
TEST_WAIT_SEC = 3300
LOOKUP = re.compile(r"(?:tuned config|tuned on cu_num = \d+) in (\S+?)[, ]")
TORCH_ROW = {"libtype": "torch", "solidx": "0", "splitK": "0", "us": "0.0", "kernelName": "torch", "err_ratio": "0.0"}
SHAPE_ROW = {"bias": "False", "dtype": "torch.bfloat16", "outdtype": "torch.bfloat16", "scaleAB": "False"}
SHAPE_ROW |= {"bpreshuffle": "False", "tflops": "0.0", "bw": "0.0"}


def _linear_shapes(model: Path) -> list[tuple[int, int]]:
    """The ``(N, K)`` of the four linear layers of a dense Qwen3 layer at TP1, from the model's own config."""
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    hidden, inner, heads, kv = (
        config[k] for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")
    )
    head_dim = config.get("head_dim", hidden // heads)
    return [((heads + 2 * kv) * head_dim, hidden), (hidden, heads * head_dim), (2 * inner, hidden), (hidden, inner)]


def tuned_table() -> str:
    """The image's own bf16 table in its format, plus a ``torch`` row for each linear layer at the decode batch sizes."""
    with SHARED.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    template = dict(zip(header, next(row for row in body if row[0] == "gfx950"), strict=True))
    shape = [header.index(column) for column in ("M", "N", "K")]
    taken = {tuple(row[i] for i in shape) for row in body}
    added = []
    for n, k in _linear_shapes(Path(MODEL)):
        for m in DECODE_BATCHES:
            row = {**template, **SHAPE_ROW, **TORCH_ROW, "M": str(m), "N": str(n), "K": str(k)}
            if (str(m), str(n), str(k)) not in taken:
                added.append([row[column] for column in header])
    lines = [header, *body, *added]
    return "".join(",".join(line) + "\n" for line in lines)


def _lookups(landed: Path) -> list[str]:
    """The tuned-config lookups the test's server logged, one line each."""
    logs = "\n".join(p.read_text(errors="replace") for p in sorted(landed.rglob("server*.log")))
    return [line.strip() for line in logs.splitlines() if LOOKUP.search(line)]


def _put_rows_hit(lookups: list[str]) -> list[str]:
    """The lookups that found a row: the image's table has none of the ``torch`` libtype, the rows put are all of it."""
    return [line for line in lookups if "is tuned on cu_num" in line and "libtype is torch" in line]


def _named_tables(lookups: list[str]) -> set[str]:
    return {match.group(1) for line in lookups if (match := LOOKUP.search(line))}


async def _refusals(stack: Stack, table: str) -> dict[str, str]:
    """What the service answers a specialist whose table cannot be loaded; none of it may reach the blob store."""
    s3 = Specialist(stack, "s3")
    before = sorted(stack.blobs.root.iterdir())
    base = {"kind": "gemm", "target": TARGET, "env": VARIABLE, "content": table}
    cases = {
        "kernel_variable_taken": {"target": "gemm_bf16_other", "content": table + "# not the same table\n"},
        "empty-table": {"content": ""},
        "target-escapes": {"target": "../x"},
        "env-name": {"env": "not a name"},
        "kind-without-a-load": {"kind": "fusion"},
    }
    replies = {
        name: await s3.refused("put_kernel_artifact", **{**base, **overrides}) for name, overrides in cases.items()
    }
    assert sorted(stack.blobs.root.iterdir()) == before, "a refused table is not stored"
    return replies


def _assert_refused(replies: dict[str, str]) -> None:
    assert "kernel_variable_taken" in replies["kernel_variable_taken"] and TARGET in replies["kernel_variable_taken"]
    assert "content" in replies["empty-table"]
    assert "kernel_artifact_invalid" in replies["target-escapes"]
    assert "kernel_artifact_invalid" in replies["env-name"]
    assert "kind" in replies["kind-without-a-load"]


def _variant(root: Path) -> Variant:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=INFERENCEX, capture_output=True, text=True, check=True
    ).stdout.strip()
    spec = spec_context(
        root,
        image_ref="local",
        image_digest=IMAGE,
        gpu_type="MI355X",
        inferencex_commit=commit,
        benchmark_backend="bypass",
    )
    canonical = default_canonical(
        root / "canonical", primary_concurrency=4, config_path=default_baseline_config(), model_path=MODEL
    )
    return Variant(
        tuple(CARDS),
        spec,
        canonical,
        IMAGE,
        runner_envs={"INFERENCEX_REPO": INFERENCEX, "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC": "900"},
        timeout_sec=2400,
        poll_interval_sec=2.0,
    )


async def _run(stack: Stack, specialist: Specialist, delta: RecipeDelta) -> TestResult:
    submitted = await specialist.call(
        "submit_test", kind="test", purpose="explore", parent_commit=stack.baseline_commit, delta=delta.to_document()
    )
    return await asyncio.wait_for(stack.toolbox.outcome(json.loads(submitted)["request_id"]), TEST_WAIT_SEC)


def _facts(stack: Stack, result: TestResult) -> dict[str, Any]:
    landed = _landed(stack.root / "session", result.request_id)
    return {"base": json.loads((landed / FRAMEWORK_BASE_FILE).read_text()), "lookups": _lookups(landed)}


async def test_a_put_table_is_the_table_the_server_of_the_test_reads() -> None:
    assert len(CARDS) == 2, f"HL_W13_CARDS must name two cards, it is {CARDS}"
    root = _run_root("w85g")
    tables_before, merged_before = tree_files(Path(DEFAULT_CONFIGS)), tree_files(AITER_MERGE_DIR)
    table = tuned_table()
    sha = hashlib.sha256(table.encode()).hexdigest()
    async with build_stack(root, _variant(root), baseline_recipe()) as stack:
        s1, s2 = Specialist(stack, "s1"), Specialist(stack, "s2")
        artifact = json.loads(
            await s2.call("put_kernel_artifact", kind="gemm", target=TARGET, env=VARIABLE, content=table)
        )
        refusals = await _refusals(stack, table)
        stored = stack.blobs.get(sha)
        config = ConfigDelta(envs=ENVS)
        with_table, without = await asyncio.gather(
            _run(stack, s2, RecipeDelta(config=config, add_kernels=(KernelArtifact.from_document(artifact),))),
            _run(stack, s1, RecipeDelta(config=config)),
        )
        facts = {"table": _facts(stack, with_table), "default": _facts(stack, without)}
        snapshot = load_snapshot(root / "session" / "framework")
    merged_name = f"{snapshot.variables[VARIABLE].tuned}.csv"
    _probe(
        "kernel-artifact",
        artifact=artifact,
        rows=len(table.splitlines()) - 1,
        refusals=refusals,
        validity={"table": with_table.validity.value, "default": without.validity.value},
        output_throughput={
            name: result.rounds[-1].metrics.get("output_throughput")
            for name, result in (("table", with_table), ("default", without))
        },
        point_tables={name: fact["base"]["point_tables"] for name, fact in facts.items()},
        merge_files={name: fact["base"]["merge_files"] for name, fact in facts.items()},
        merge={
            name: {k: fact["base"][k] for k in ("merge_isolation", "merge_dir", "merge_patched_processes")}
            for name, fact in facts.items()
        },
        lookup_tables={name: sorted(_named_tables(fact["lookups"])) for name, fact in facts.items()},
        lookup_lines={name: [line[:300] for line in fact["lookups"][:3]] for name, fact in facts.items()},
        torch_rows_hit={name: len(_put_rows_hit(fact["lookups"])) for name, fact in facts.items()},
        table_hits={
            name: [hit.to_document() for hit in result.table_hits]
            for name, result in (("table", with_table), ("default", without))
        },
        shared_tables_unchanged=tree_files(Path(DEFAULT_CONFIGS)) == tables_before,
        aiter_merge_dir_unchanged=tree_files(AITER_MERGE_DIR) == merged_before,
    )

    _assert_refused(refusals)
    assert artifact["ref"] == f"kernel/gemm/{TARGET}/{sha}.csv" and artifact["sha256"] == sha, artifact
    assert stored == table.encode(), "the bytes are in the session's blob store"
    for result in (with_table, without):
        assert result.validity is Validity.SUCCEEDED, (result.failure_class, result.failure_reason)
        assert result.rounds[-1].metrics.get("output_throughput", 0) > 0
    _assert_server_reads_the_table(facts, with_table, sha, merged_name)
    _assert_table_hits(with_table, without, sha, facts["table"]["lookups"])
    _assert_isolated(facts, {"table": with_table, "default": without})
    assert tree_files(Path(DEFAULT_CONFIGS)) == tables_before, "the shared tables are as they were"
    assert tree_files(AITER_MERGE_DIR) == merged_before, "nothing reached /tmp/aiter_configs"


def _assert_server_reads_the_table(facts: dict[str, Any], result: TestResult, sha: str, merged_name: str) -> None:
    placed = f"k-{sha[:16]}"
    table, default = facts["table"], facts["default"]
    tables = table["base"]["point_tables"]
    assert list(tables) == [VARIABLE] and [t["sha256"] for t in tables[VARIABLE]] == [sha], tables
    assert Path(tables[VARIABLE][0]["source"]).name == placed
    assert f"/sandboxes/{result.sandbox.sandbox_id}/" in tables[VARIABLE][0]["source"], "placed in its own sandbox"
    assert default["base"]["point_tables"] == {}, "the default test names no table of its own"
    assert merged_name in default["base"]["merge_files"], "the default test reads aiter's merged view"
    assert merged_name not in table["base"]["merge_files"], "a variable that lists one table is not merged"
    assert table["lookups"], "the server logged no tuned-config lookups"
    assert _put_rows_hit(table["lookups"]), "no shape of the model hit one of the torch rows that were put"
    named = _named_tables(table["lookups"])
    assert named and all(Path(name).name.endswith(placed) for name in named), named
    assert not any(placed in name for name in _named_tables(default["lookups"])), "the default test never saw it"


def _assert_table_hits(with_table: TestResult, without: TestResult, sha: str, lookups: list[str]) -> None:
    """The result's count of the table's hits is the count of the log's hit lines that name the table, done by hand."""
    (hits,) = with_table.table_hits
    named = [line for line in lookups if f"k-{sha[:16]}" in line and "is tuned on cu_num" in line]
    assert (hits.env, hits.target, hits.sha256, hits.status) == (VARIABLE, TARGET, sha, TABLE_EXERCISED), hits
    assert hits.hit_lines == len(named) and 1 <= hits.rows_hit <= hits.rows, (hits, len(named))
    assert with_table.runtime_findings and "exercised" in with_table.runtime_findings[0]
    assert without.table_hits == () and without.runtime_findings == (), "a test that put no table has no finding"


def _assert_isolated(facts: dict[str, Any], results: dict[str, TestResult]) -> None:
    for name, fact in facts.items():
        base = fact["base"]
        assert base["isolation"] == "private_copies" and base["matches_session"] is True
        assert base["framework_base_changed"] is False and base["restored"] == {"start": [], "end": []}
        assert base["shared_sha256_start"] == base["shared_sha256_end"] == base["session_sha256"]
        assert base["merge_isolation"] == PRIVATE and base["merge_chain_failure"] is None, base["merge_reason"]
        assert f"/sandboxes/{results[name].sandbox.sandbox_id}/" in base["merge_dir"], "below its own sandbox"
    assert facts["table"]["base"]["merge_dir"] != facts["default"]["base"]["merge_dir"]


#: A table in the shape of a hand-written one: put accepts it (the tool checks the target, the variable and the size, not
#: the columns), and aiter, which indexes the rows by ``gfx`` and ``cu_num``, cannot read it.
UNREADABLE_TABLE = "M,N,K,libtype,solidx\n4096,4096,4096,torch,0\n"


async def test_a_table_aiter_cannot_read_fails_its_test_and_the_log_says_why() -> None:
    root = _run_root("w85g-unreadable")
    async with build_stack(root, _variant(root), baseline_recipe()) as stack:
        s1 = Specialist(stack, "s1")
        put = await s1.call("put_kernel_artifact", kind="gemm", target=TARGET, env=VARIABLE, content=UNREADABLE_TABLE)
        delta = RecipeDelta(config=ConfigDelta(envs=ENVS), add_kernels=(KernelArtifact.from_document(json.loads(put)),))
        result = await _run(stack, s1, delta)
        logs = sorted((root / "session" / "tests" / result.request_id).rglob("server*.log"))
    errors = [
        line.strip()[:300]
        for log in logs
        for line in log.read_text(errors="replace").splitlines()
        if re.search(r"KeyError|Traceback|Error", line)
    ]
    _probe(
        "unreadable-table",
        validity=result.validity.value,
        failure_class=result.failure_class,
        failure_reason=result.failure_reason,
        server_logs=len(logs),
        error_lines=errors[-8:],
    )
    assert result.validity is Validity.CANDIDATE_FAILED, result.failure_reason
    assert (result.failure_reason or "").startswith("python_exception: KeyError"), result.failure_reason
    assert "gfx" in result.failure_reason, "the cause names the column the table lacks"
