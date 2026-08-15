# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the infersim benchmark backend + projection bridge.

No GPU and no Infera install: the projection call is monkeypatched, so these
verify backend selection, argv construction, benchmark-spec parsing, model
preset resolution, the metrics->report mapping, and that a simulated run flows
through Hyperloom's measurement extractor unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb
from hyperloom.orchestrator.actions.executors import infersim_bridge as ib
from hyperloom.orchestrator.actions.executors import infersim_runner
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement,
    is_valid_measurement,
)


def test_infersim_backend_selected(monkeypatch):
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "infersim")
    assert bb.resolve_backend_name() == "infersim"
    backend = bb.resolve_backend()
    assert backend.name == "infersim"
    cmd = backend.build_command(
        python_exe="PY",
        config_path=Path("/cfg.yaml"),
        output_dir=Path("/out"),
    )
    assert cmd == [
        "PY",
        "-m",
        "hyperloom.orchestrator.actions.executors.infersim_runner",
        "benchmark",
        "--benchmark-config",
        "/cfg.yaml",
        "--output-dir",
        "/out",
        "--run-mode",
        "local",
    ]


def test_infersim_backend_lifecycle_ineligible():
    backend = bb.InfersimBackend()
    verdict = backend.lifecycle_eligibility({"framework": "sglang"})
    assert verdict is not None
    assert verdict["eligible"] is False


def test_infersim_interpreter_prefers_env(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_INFERSIM_PYTHON", "/opt/infera/bin/python")
    assert bb.InfersimBackend().resolve_interpreter() == "/opt/infera/bin/python"


def test_spec_from_benchmark_parses_envs(monkeypatch):
    monkeypatch.delenv(ib.ENV_EP, raising=False)
    monkeypatch.delenv(ib.ENV_PP, raising=False)
    bench = {
        "framework": "vllm",
        "model": "/models/Qwen-Qwen3-14B",
        "precision": "fp8",
        "envs": {"TP": 4, "CONC": 128, "ISL": 2048, "OSL": 256},
    }
    spec = ib.spec_from_benchmark(bench)
    assert spec.framework == "vllm"
    assert spec.tp == 4
    assert spec.conc == 128
    assert spec.isl == 2048
    assert spec.osl == 256
    assert spec.weight_dtype == "fp8"


def test_spec_parses_ep_from_server_args(monkeypatch):
    monkeypatch.delenv(ib.ENV_EP, raising=False)
    bench = {
        "framework": "sglang",
        "model": "/models/gpt-oss-120b",
        "envs": {"TP": 8, "EXTRA_SGLANG_ARGS": "--ep-size 8 --foo 1"},
    }
    spec = ib.spec_from_benchmark(bench)
    assert spec.ep == 8


def test_resolve_preset_heuristics_and_override(monkeypatch):
    monkeypatch.delenv(ib.ENV_MODEL, raising=False)
    assert ib.resolve_preset("/models/gpt-oss-120b") == "gpt_oss_120B"
    assert ib.resolve_preset("/data/Qwen-Qwen3-14B") == "qwen3_14B"
    assert ib.resolve_preset("/some/unknown-model") is None
    monkeypatch.setenv(ib.ENV_MODEL, "custom_preset")
    assert ib.resolve_preset("/models/gpt-oss-120b") == "custom_preset"


def _fake_metrics() -> ib.ProjMetrics:
    return ib.ProjMetrics(
        output_throughput=9000.0,
        request_throughput=8.78,
        total_token_throughput=18000.0,
        ttft_ms=25.0,
        tpot_ms=6.5,
        itl_ms=6.5,
        e2el_ms=6650.0,
        decode_tps_per_gpu=9000.0,
        memory_per_gpu_gb=70.0,
        max_concurrency=64,
        calibrated=False,
        replica_gpus=1,
    )


def test_raw_result_from_metrics_shape():
    spec = ib.ServingSpec(framework="sglang", model_path="/m", tp=1, conc=64, isl=1024, osl=1024)
    raw = ib.raw_result_from_metrics(spec, _fake_metrics())
    assert raw["output_throughput"] == 9000.0
    assert raw["mean_ttft_ms"] == 25.0
    assert raw["mean_tpot_ms"] == 6.5
    assert raw["mean_e2el_ms"] == 6650.0
    assert raw["total_output_tokens"] == 64 * 1024
    assert raw["infersim_decode_tps_per_gpu"] == 9000.0


def _write_bench(tmp_path: Path) -> Path:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/models/gpt-oss-120b",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 64, "ISL": 1024, "OSL": 1024},
        }
    }
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_runner_end_to_end_with_mocked_projection(tmp_path, monkeypatch):
    """The runner writes a Magpie-compatible report from projected metrics."""
    monkeypatch.setattr(ib, "project", lambda spec: _fake_metrics())

    cfg_path = _write_bench(tmp_path)
    rc = infersim_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0

    workspaces = list((tmp_path / "out").glob("benchmark_sglang_*"))
    assert len(workspaces) == 1
    ws = workspaces[0]
    report = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["bypass_analysis"]["backend"] == "infersim"

    m = extract_benchmark_measurement(report, workspace=ws)
    assert is_valid_measurement(m) is True
    assert m["output_throughput"] == 9000.0
    assert m["ttft_mean_ms"] == 25.0


def test_runner_projection_failure_emits_failed_report(tmp_path, monkeypatch):
    def boom(spec):
        raise ib.InfersimBridgeError("no preset resolvable")

    monkeypatch.setattr(ib, "project", boom)
    cfg_path = _write_bench(tmp_path)
    rc = infersim_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 1

    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    report = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["success"] is False
    assert any("no preset resolvable" in e for e in report["errors"])


def test_runner_cli_rejects_non_local(tmp_path):
    rc = infersim_runner.main(
        [
            "benchmark",
            "--benchmark-config",
            str(tmp_path / "c.yaml"),
            "--output-dir",
            str(tmp_path / "o"),
            "--run-mode",
            "docker",
        ]
    )
    assert rc == 2


def test_runner_server_phase_is_noop_success(tmp_path):
    rc = infersim_runner.main(
        [
            "benchmark",
            "--benchmark-config",
            str(tmp_path / "c.yaml"),
            "--output-dir",
            str(tmp_path / "o"),
            "--phase",
            "server",
        ]
    )
    assert rc == 0


def test_resolve_workload_prefers_explicit_env(tmp_path, monkeypatch):
    wl = tmp_path / "custom_workload.yaml"
    wl.write_text("work_group: t\n", encoding="utf-8")
    monkeypatch.setenv(ib.ENV_WORKLOAD, str(wl))
    spec = ib.ServingSpec(framework="sglang", model_path="/m")
    workload, extra_env = ib._resolve_workload_and_env(spec)
    assert workload == str(wl.resolve())
    assert "INFERSIM_MODEL" not in extra_env


def test_resolve_workload_uses_template_for_preset(monkeypatch):
    monkeypatch.delenv(ib.ENV_WORKLOAD, raising=False)
    monkeypatch.setenv(ib.ENV_MODEL, "gpt_oss_120B")
    spec = ib.ServingSpec(framework="sglang", model_path="/models/gpt-oss-120b")
    workload, extra_env = ib._resolve_workload_and_env(spec)
    assert Path(workload).name == "infersim_workload.yaml"
    assert extra_env["INFERSIM_MODEL"] == "gpt_oss_120B"


def _write_anchor(path: Path, *, model: str, real_weights: bool, decode_ms: float,
                  quant=None, kv="bf16", aiter=True) -> None:
    """Minimal benchmark artifact in the shape benchmark_vllm.py emits."""
    path.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "measured": {"model": {"prefill_ms": 10.0, "decode_ms": decode_ms}},
                "sweep": [{"batch": 16, "prefill_ms": 10.0, "decode_ms": decode_ms}],
                "meta": {
                    "model": model,
                    "batch": 16,
                    "input_len": 1024,
                    "tp": 1,
                    "quantization": quant,
                    "kv_cache_dtype": kv,
                    "use_aiter": aiter,
                    "real_weights": real_weights,
                    "load_format": "auto" if real_weights else "dummy",
                },
            }
        ),
        encoding="utf-8",
    )


def test_anchor_is_real_weights_detection(tmp_path):
    real, dummy = tmp_path / "r.json", tmp_path / "d.json"
    _write_anchor(real, model="m", real_weights=True, decode_ms=9.0)
    _write_anchor(dummy, model="m", real_weights=False, decode_ms=5.0)
    assert ib._anchor_is_real_weights(str(real)) is True
    assert ib._anchor_is_real_weights(str(dummy)) is False
    assert ib._anchor_is_real_weights(str(tmp_path / "missing.json")) is False


def test_recipe_from_spec_extracts_attention_backend():
    spec = ib.ServingSpec(
        framework="sglang",
        model_path="/models/x",
        extra_server_args="--attention-backend aiter --max-num-seqs 64",
    )
    recipe = ib.recipe_from_spec(spec)
    assert recipe["attention_backend"] == "aiter"
    assert recipe["weight_dtype"] == "bf16"


def test_select_anchor_prefers_explicit_env(tmp_path, monkeypatch):
    a = tmp_path / "explicit.json"
    _write_anchor(a, model="m", real_weights=True, decode_ms=9.0)
    monkeypatch.setenv(ib.ENV_ANCHOR, str(a))
    choice = ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m"))
    assert choice is not None
    assert choice.path == str(a)
    assert choice.regime_distance == 0


def test_select_anchor_none_without_store(monkeypatch):
    monkeypatch.delenv(ib.ENV_ANCHOR, raising=False)
    monkeypatch.delenv(ib.ENV_ANCHOR_STORE, raising=False)
    assert ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m")) is None


def _write_curve(path: Path, points: list[tuple[int, float]]) -> None:
    """Artifact carrying an explicit decode-vs-batch curve."""
    path.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "sweep": [
                    {"batch": b, "prefill_ms": 10.0, "decode_ms": d} for b, d in points
                ],
                "meta": {"model": "m", "tp": 1, "input_len": 1024},
            }
        )
    )


@pytest.mark.parametrize(
    "points, sane",
    [
        ([(1, 4.0), (8, 6.4), (32, 12.0)], True),      # ordinary rising curve
        ([(16, 12.0)], True),                          # single point: narrow, valid
        ([(8, 6.0), (16, 5.7)], True),                 # -5%: run-to-run noise
        ([(16, 16.3), (64, 1.4)], False),              # differencing degenerated
        ([(4, 11.6), (32, 9.5)], False),               # decode faster at 8x batch
        ([(8, 0.0)], False),                           # non-positive timing
        ([], False),                                   # nothing measured
    ],
)
def test_anchor_curve_sanity_gate(tmp_path, points, sane):
    p = tmp_path / "curve.json"
    _write_curve(p, points)
    assert ib.anchor_curve_is_sane(str(p)) is sane


def test_anchor_curve_sanity_gate_missing_file(tmp_path):
    assert ib.anchor_curve_is_sane(str(tmp_path / "nope.json")) is False


@pytest.mark.parametrize(
    "args, expected",
    [
        ("", (None, 0)),
        ("--attention-backend triton", (None, 0)),
        ('--speculative-config \'{"method": "deepseek_mtp", '
         '"num_speculative_tokens": 3}\'', ("deepseek_mtp", 3)),
        ("--speculative-algorithm NEXTN --speculative-num-steps 3", ("NEXTN", 3)),
        ("--speculative-algorithm EAGLE3", ("EAGLE3", 1)),
        ("--method mtp --num-speculative-tokens 3", ("mtp", 3)),
        ("--method fp8", (None, 0)),
    ],
)
def test_parse_speculative_across_frameworks(args, expected):
    assert ib.parse_speculative(args) == expected


def test_recipe_marks_speculative_candidates_apart():
    """A speculating candidate must not share a regime with a plain one.

    Speculation changes how many tokens a step emits, so reusing a
    non-speculative anchor for it silently under-predicts throughput.
    """
    plain = ib.recipe_from_spec(ib.ServingSpec(framework="vllm", model_path="m"))
    mtp = ib.recipe_from_spec(
        ib.ServingSpec(
            framework="atom",
            model_path="m",
            extra_server_args="--method mtp --num-speculative-tokens 3",
        )
    )
    assert plain["speculative"] == "off"
    assert mtp["speculative"] == "spec:3"
    assert plain["speculative"] != mtp["speculative"]


def test_select_anchor_rejects_insane_anchor(tmp_path, monkeypatch):
    """A corrupt curve is worse than no anchor: fall back to pure analysis.

    Applies even to an operator-pinned anchor, which is the path most likely to
    point at a hand-picked artifact nobody re-validated.
    """
    bad = tmp_path / "bad.json"
    _write_curve(bad, [(16, 16.3), (64, 1.4)])
    monkeypatch.setenv(ib.ENV_ANCHOR, str(bad))
    monkeypatch.delenv(ib.ENV_ANCHOR_STORE, raising=False)
    assert ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m")) is None

    good = tmp_path / "good.json"
    _write_curve(good, [(16, 12.0), (64, 20.0)])
    monkeypatch.setenv(ib.ENV_ANCHOR, str(good))
    choice = ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m"))
    assert choice is not None and choice.path == str(good)
