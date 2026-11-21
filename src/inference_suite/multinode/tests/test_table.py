# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PD default table composes each row as InferenceX 3d55815's scripts print it in DRY_RUN."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_suite.frameworks import Framework
from inference_suite.multinode.table import (
    ATOM_TABLE,
    SGLANG_TABLE,
    RoleShape,
    TableError,
    TableQuery,
    compose,
    table_text,
)

_GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "pd_table_golden.json").read_text(encoding="utf-8"))


def _query(case: dict) -> TableQuery:
    return TableQuery(
        model_name=case["model"],
        prefill=RoleShape(*case["prefill_shape"]),
        decode=RoleShape(*case["decode_shape"]),
        max_concurrency=case["max_concurrency"],
    )


@pytest.mark.parametrize("case", _GOLDEN, ids=[f"{c['framework']}-{c['model']}-{c['shape']}" for c in _GOLDEN])
def test_row_composes_as_the_script_prints_it(case: dict) -> None:
    hit = compose(Framework(case["framework"]), _query(case))
    for role, launch in (("prefill", hit.prefill), ("decode", hit.decode)):
        assert list(launch.args) == case[role]["args"], role
        for key, value in case[role]["env"].items():
            assert launch.env.get(key) == value, f"{role} env {key}"


def test_every_row_has_a_golden() -> None:
    import yaml

    sglang = {key.removesuffix("-DI") for key in yaml.safe_load(table_text(SGLANG_TABLE))}
    atom = set(yaml.safe_load(table_text(ATOM_TABLE)))
    covered = {(c["framework"], c["model"]) for c in _GOLDEN}
    assert {("sglang", key) for key in sglang} | {("atom", key) for key in atom} <= covered


def test_qwen35_matches_the_hyperloom_baseline() -> None:
    """The per-role args Hyperloom's ``baseline_args.py`` composed for Qwen3.5-397B-A17B-FP8 at TP 8."""
    hit = compose(Framework.SGLANG, TableQuery("Qwen3.5-397B-A17B-FP8", RoleShape(8), RoleShape(8)))
    graph = [str(n) for n in range(1, 129)]
    base = [
        *("--decode-log-interval", "1000", "--log-level", "warning", "--watchdog-timeout", "3600"),
        *("--load-balance-method", "round_robin", "--kv-cache-dtype", "fp8_e4m3", "--attention-backend", "aiter"),
        *("--disaggregation-transfer-backend", "mori", "--moe-dense-tp-size", "1"),
    ]
    assert list(hit.prefill.args) == [
        *("--tp-size", "8", *base, "--mem-fraction-static", "0.8", "--max-running-requests", "128"),
        *("--chunked-prefill-size", "262144", "--cuda-graph-bs", *graph, "--disable-radix-cache"),
    ]
    assert list(hit.decode.args) == [
        *("--tp-size", "8", *base, "--mem-fraction-static", "0.85", "--max-running-requests", "128"),
        *("--cuda-graph-bs", *graph, "--prefill-round-robin-balance"),
    ]


def test_unknown_model_is_refused() -> None:
    with pytest.raises(TableError, match="no row"):
        compose(Framework.SGLANG, TableQuery("NoSuchModel", RoleShape(8), RoleShape(8)))


def test_agentic_keeps_the_prefill_radix_cache() -> None:
    hit = compose(Framework.SGLANG, TableQuery("DeepSeek-R1", RoleShape(8), RoleShape(8), agentic=True))
    assert "--disable-radix-cache" not in hit.prefill.args
