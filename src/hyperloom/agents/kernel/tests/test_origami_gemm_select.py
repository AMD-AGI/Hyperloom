#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the deterministic Origami GEMM fallback selector."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "origami_gemm_select.py"
_SPEC = importlib.util.spec_from_file_location("origami_gemm_select_tool", _MODULE_PATH)
assert _SPEC and _SPEC.loader
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def test_shape_records_normalizes_and_deduplicates(tmp_path):
    source = tmp_path / "shapes.json"
    source.write_text(
        json.dumps(
            {
                "shapes": [
                    {"M": 16, "N": 32, "K": 128},
                    {"m": "16", "n": "32", "k": "128"},
                    {"M": 0, "N": 32, "K": 128},
                    {"not": "a shape"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert selector._shape_records(source) == [{"M": 16, "N": 32, "K": 128}]


def test_classify_dispatch_distinguishes_csv_selected_default():
    default_name = "a8w8_blockscale_default"

    assert selector.classify_dispatch(None, default_kernel_name=default_name) == (
        "fallback",
        "",
    )
    assert selector.classify_dispatch(
        {"kernelName": ""},
        default_kernel_name=default_name,
    ) == ("fallback", "")
    assert selector.classify_dispatch(
        {"kernelName": default_name},
        default_kernel_name=default_name,
    ) == ("csv_default_template", default_name)
    assert selector.classify_dispatch(
        {"kernelName": "a8w8_blockscale_other"},
        default_kernel_name=default_name,
    ) == ("csv", "a8w8_blockscale_other")
    assert selector.classify_dispatch(
        {"kernelName": float("nan")},
        default_kernel_name=default_name,
    )[0] == "invalid_csv"


def test_rank_shape_retries_cache_policy():
    class Config:
        def __init__(self, kernel_id):
            self.kernel_id = kernel_id
            self.cache_hints_a = 0
            self.cache_hints_b = 0

    class Origami:
        @staticmethod
        def compute_total_latency(_problem, _hardware, cfg):
            if cfg.cache_hints_b != 4:
                return float("inf")
            return {1: 20.0, 2: 10.0}[cfg.kernel_id]

    configs = [(1, Config(1)), (2, Config(2))]
    ranked, mode = selector.rank_shape(Origami(), object(), object(), configs)

    assert mode == "nt_b"
    assert ranked == [(2, 10.0), (1, 20.0)]


def test_benchmark_gate_requires_strict_measured_win():
    assert selector.benchmark_is_faster(9.0, 10.0) is True
    assert selector.benchmark_is_faster(10.0, 10.0) is False
    assert selector.benchmark_is_faster(10.1, 10.0) is False
    assert selector.benchmark_is_faster(9.95, 10.0, min_speedup=1.01) is False


def test_select_writes_only_true_fallback_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    tuned = tmp_path / "active.csv"
    tuned.write_text("gfx,cu_num,M,N,K,kernelName\n", encoding="utf-8")
    output = tmp_path / "out"

    default_kernel = types.SimpleNamespace(name="default-kernel")
    selected_kernel = types.SimpleNamespace(name="origami-kernel")
    monkeypatch.setattr(
        selector,
        "_load_kernel_table",
        lambda _root: {selector.DEFAULT_KERNEL_ID: default_kernel, 3: selected_kernel},
    )
    monkeypatch.setattr(selector, "_resolve_aiter_root", lambda _value="": tmp_path)
    monkeypatch.setattr(selector, "_build_configs", lambda *_args: [(3, object())])
    monkeypatch.setattr(selector, "_make_problem", lambda *_args: object())
    monkeypatch.setattr(
        selector,
        "rank_shape",
        lambda *_args: ([(3, 12.5)], "base"),
    )
    monkeypatch.setattr(
        selector,
        "_benchmark_shape",
        lambda *_args: {
            "benchmark_status": "ok",
            "benchmark_selected_us": 8.0,
            "benchmark_default_us": 10.0,
            "benchmark_speedup": 1.25,
            "benchmark_error_ratio": 0.0,
            "benchmark_use_origami": True,
        },
    )

    def resolve(m, _n, _k, _path):
        if m == 16:
            return None
        return {"kernelName": "default-kernel", "kernelId": selector.DEFAULT_KERNEL_ID}

    monkeypatch.setattr(
        selector,
        "_aiter_runtime",
        lambda: (resolve, "gfx950", 304, str(tuned)),
    )
    fake_origami = types.SimpleNamespace(get_hardware_for_device=lambda _idx: object())
    monkeypatch.setitem(sys.modules, "origami", fake_origami)

    result = selector.select(
        {
            "shapes": [
                {"M": 16, "N": 4096, "K": 8192},
                {"M": 32, "N": 4096, "K": 8192},
            ],
            "tuned_csv": str(tuned),
            "output_dir": str(output),
        }
    )

    assert result["status"] == "ok"
    assert result["selected_shapes"] == 1
    with Path(result["candidate_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["M"] == "16"
    assert rows[0]["kernelId"] == "3"
    assert rows[0]["kernelName"] == "origami-kernel"
    assert float(rows[0]["us"]) == 8.0
    assert result["env_value"] == result["merged_csv"]
    with Path(result["merged_csv"]).open(newline="", encoding="utf-8") as handle:
        merged_rows = list(csv.DictReader(handle))
    assert len(merged_rows) == 1
    assert merged_rows[0]["kernelName"] == "origami-kernel"

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert [row["dispatch_source"] for row in report["rows"]] == [
        "fallback",
        "csv_default_template",
    ]


def test_main_fails_closed_when_origami_is_unavailable(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "shapes": [{"M": 16, "N": 32, "K": 128}],
                "tuned_csv": str(tmp_path / "missing.csv"),
                "output_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )

    assert selector.main(["--input-json", str(input_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "skipped"
    assert result["candidate"] is False
    assert result["reason"] == "selector_unavailable"


def test_main_disabled_does_not_read_input_or_write_artifacts(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.delenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", raising=False)
    output = tmp_path / "out"

    assert selector.main(["--input-json", str(tmp_path / "missing.json")]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "skipped",
        "candidate": False,
        "reason": "disabled",
    }
    assert not output.exists()
