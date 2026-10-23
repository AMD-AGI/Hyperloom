# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A few tuned rows put with ``rows_only`` are read over the image's default tables, on a real GPU (W115).

The specialist is the real MCP service, the toolbox runs the real runner on one card, and the server is SGLang with aiter
serving Qwen3-8B at TP1. The rows are four, all ``torch`` at a decode step of four requests: gate_up, which the image's
own Qwen3-8B table tunes with ``triton``; qkv and o, which other models' tables in the image tune; and down, which no
table does. Without the override aiter would rewrite the table files and raise ``Found N duplicate shape entries``
before the server started. What must hold: the server boots and the test succeeds; ``framework_base.json`` says which
rows were taken out of which of the image's tables, as many as the tables hold; the server's log shows the rows hit, the
overridden shapes read as ``torch``; no duplicate-entry error in any log; the shared tables and ``/tmp/aiter_configs``
as they were. What was measured is printed (``PROBE``).

Needs what ``test_kernel_artifact_gpu`` needs, with ``HL_W13_CARDS`` naming one card.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, RecipeDelta
from hyperloom.contracts.measurement import Validity
from hyperloom.measure.framework_base import DEFAULT_CONFIGS, tree_files
from hyperloom.measure.framework_merge import PRIVATE
from hyperloom_multibranch.toolbox.tests.integration_inputs import baseline_recipe
from hyperloom_multibranch.toolbox.tests.integration_rig import Specialist, build_stack
from hyperloom_multibranch.toolbox.tests.test_framework_isolation_parallel_gpu import AITER_MERGE_DIR, SHARED, VARIABLE
from hyperloom_multibranch.toolbox.tests.test_kernel_artifact_gpu import (  # noqa: F401  (the fixtures and skips)
    ENVS,
    SHAPE_ROW,
    TARGET,
    TORCH_ROW,
    _facts,
    _linear_shapes,
    _named_tables,
    _put_rows_hit,
    _run,
    _variant,
    pytestmark,
    workload_environment,
)
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import CARDS, MODEL, _probe, _run_root

MODEL_TABLE = Path(DEFAULT_CONFIGS) / "model_configs" / "qwen3_8b_bf16_tuned_gemm.csv"
#: What a row of a bf16 table is told apart by; the test's own reading of the image's tables, not the toolbox's.
KEY = ("gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle")
DUPLICATES = re.compile(r"Found \d+ duplicate shape entries")
DECODE_M = "4"


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        header, *rows = csv.reader(handle)
    return header, rows


def _image_tables() -> list[Path]:
    """What aiter reads for the variable when it is unset: the default table and the per-model ones."""
    models = sorted(p for p in MODEL_TABLE.parent.glob("*bf16_tuned_gemm*.csv") if "untuned" not in p.name)
    return [SHARED, *models]


def rows_to_put() -> list[dict[str, str]]:
    """Four ``torch`` rows at M=4, one for each linear layer of the model, in the image's own header."""
    header, table = _read(SHARED)
    template = dict(zip(header, next(row for row in table if row[0] == "gfx950"), strict=True))
    return [
        {**template, **SHAPE_ROW, **TORCH_ROW, "M": DECODE_M, "N": str(n), "K": str(k)}
        for n, k in _linear_shapes(Path(MODEL))
    ]


def image_rows_taken(rows: list[dict[str, str]]) -> dict[str, int]:
    """How many rows of each of the image's tables have the key of one of ``rows``: what the test must have taken out."""
    keys = {tuple(row[column] for column in KEY) for row in rows}
    taken = {}
    for path in _image_tables():
        header, body = _read(path)
        indices = [header.index(column) for column in KEY]
        count = sum(tuple(line[i] for i in indices) in keys for line in body)
        if count:
            taken[path.name] = count
    return taken


def _tuned_as(lookups: list[str], row: dict[str, str]) -> list[str]:
    """The libtype of each lookup the server logged for the shape of ``row``."""
    wanted = f"shape is M:{row['M']}, N:{row['N']}, K:{row['K']} "
    return [m.group(1) for line in lookups if wanted in line and (m := re.search(r"libtype is (\w+)", line))]


async def test_rows_put_with_rows_only_are_read_over_the_image_tables_and_override_the_shape_they_share() -> None:
    assert len(CARDS) == 1, f"HL_W13_CARDS must name one card, it is {CARDS}"
    root = _run_root("w115")
    tables_before, merged_before = tree_files(Path(DEFAULT_CONFIGS)), tree_files(AITER_MERGE_DIR)
    rows = rows_to_put()
    header = list(rows[0])
    content = "".join(",".join(line) + "\n" for line in [header, *([row[c] for c in header] for row in rows)])
    expected = image_rows_taken(rows)
    async with build_stack(root, _variant(root), baseline_recipe()) as stack:
        s1 = Specialist(stack, "s1")
        artifact = json.loads(
            await s1.call(
                "put_kernel_artifact", kind="gemm", target=TARGET, env=VARIABLE, content=content, rows_only=True
            )
        )
        delta = RecipeDelta(config=ConfigDelta(envs=ENVS), add_kernels=(KernelArtifact.from_document(artifact),))
        result = await _run(stack, s1, delta)
        facts = _facts(stack, result)
        logs = "\n".join(
            p.read_text(errors="replace") for p in sorted((root / "session" / "tests").rglob("server*.log"))
        )
    base = facts["base"]
    _probe(
        "rows-only",
        artifact=artifact,
        rows_put=len(rows),
        expected_taken=expected,
        bytes_put=len(content),
        default_bytes=SHARED.stat().st_size,
        validity=result.validity.value,
        failure=[result.failure_class, result.failure_reason],
        output_throughput=result.rounds[-1].metrics.get("output_throughput") if result.rounds else None,
        rows_only=base["rows_only"],
        point_tables=base["point_tables"],
        merge={k: base[k] for k in ("merge_isolation", "merge_dir", "merge_patched_processes", "merge_files")},
        shapes_read_as={row["N"] + "x" + row["K"]: _tuned_as(facts["lookups"], row) for row in rows},
        rows_hit=len(_put_rows_hit(facts["lookups"])),
        lookup_tables=sorted(_named_tables(facts["lookups"])),
        duplicate_errors=len(DUPLICATES.findall(logs)),
        shared_tables_unchanged=tree_files(Path(DEFAULT_CONFIGS)) == tables_before,
        aiter_merge_dir_unchanged=tree_files(AITER_MERGE_DIR) == merged_before,
    )

    assert artifact["metadata"]["load"] == {"via": "env_file", "env": VARIABLE, "rows_only": True}
    assert stack.blobs.get(artifact["sha256"]) == content.encode() and len(content) < SHARED.stat().st_size / 10
    assert result.validity is Validity.SUCCEEDED, (result.failure_class, result.failure_reason)
    assert result.rounds[-1].metrics.get("output_throughput", 0) > 0
    assert not DUPLICATES.search(logs), "aiter met a shape in two of the tables it was given"
    laid = base["rows_only"][VARIABLE]
    assert expected and MODEL_TABLE.name in expected, "the image no longer tunes the shapes this test overrides"
    assert laid["rows"] == 4 and laid["dropped"] == expected and laid["overridden"] == sum(expected.values()), laid
    sources = {Path(t["source"]).name for t in base["point_tables"][VARIABLE]}
    assert sources == {f"k-{artifact['sha256'][:16]}", *expected}, "the rows file and the tables that lost rows"
    for row in rows:
        assert set(_tuned_as(facts["lookups"], row)) == {"torch"}, f"{row['N']}x{row['K']}: not read from the rows put"
    assert _put_rows_hit(facts["lookups"]), "no shape of the model hit a row that was put"
    assert Path(base["merge_dir"]) in {Path(name).parent for name in _named_tables(facts["lookups"])}
    assert base["merge_isolation"] == PRIVATE and base["isolation"] == "private_copies"
    assert base["framework_base_changed"] is False and base["matches_session"] is True
    assert tree_files(Path(DEFAULT_CONFIGS)) == tables_before, "the shared tables are as they were"
    assert tree_files(AITER_MERGE_DIR) == merged_before, "nothing reached /tmp/aiter_configs"
