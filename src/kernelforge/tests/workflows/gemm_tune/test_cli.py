# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI contract: GPU resolution, provenance, and unconditional live tuning."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from kernelforge.workflows.gemm_tune import pipeline as cli_mod
from kernelforge.cli.gemm_tune_command import gemm_tune
from kernelforge.workflows.gemm_tune.dense_shapes import MGrid
from kernelforge.workflows.gemm_tune.tuners.base import TuneResult


def _model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model


def _stub_preflight(monkeypatch, *, soft=None, hard=None):
    monkeypatch.setattr(
        "kernelforge.workflows.gemm_tune.aiter_preflight.collect",
        lambda: {"soft": soft or [], "hard": hard or [], "aligned": False},
    )


def test_help_carries_no_knowledge_base_options():
    """Tuning has no knowledge base, so no option may imply one exists."""
    result = CliRunner().invoke(gemm_tune, ["run", "--help"])

    assert result.exit_code == 0
    for option in ("--kb-read", "--kb-accept-candidate", "--kb-strict-lib"):
        assert option not in result.output


def test_every_runnable_tuner_is_executed(tmp_path, monkeypatch):
    """Nothing may stand between a runnable tuner and a real tuning run."""
    _stub_preflight(monkeypatch, soft=["AITER_COMMIT unset"])
    model = _model_dir(tmp_path)
    output = tmp_path / "output"
    executed: list[str] = []

    class _Tuner:
        def __init__(self, name):
            self.name = name

        def execute(self):
            executed.append(self.name)
            return TuneResult(tuner_name=self.name, status="no_improvement")

    monkeypatch.setattr(cli_mod, "_create_tuner", lambda name, ctx: _Tuner(name))
    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(model),
            "--framework",
            "vllm",
            "--precision",
            "fp8",
            "--gpu-type",
            "mi300x",
            "--tuner",
            "a8w8",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert executed == ["a8w8"]
    report = json.loads((output / "result.json").read_text())
    assert [t["tuner"] for t in report["tuners_run"]] == ["a8w8"]
    assert all("kb_cache" not in t for t in report["tuners_run"])


def test_hard_aiter_mismatch_aborts_before_tuning(tmp_path, monkeypatch):
    _stub_preflight(monkeypatch, hard=["MISALIGNED: serving and tuner roots differ"])
    model = _model_dir(tmp_path)
    executed = []

    class _Tuner:
        def execute(self):
            executed.append(True)
            return TuneResult(tuner_name="a8w8", status="no_improvement")

    monkeypatch.setattr(cli_mod, "_create_tuner", lambda _name, _ctx: _Tuner())
    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(model),
            "--framework",
            "vllm",
            "--precision",
            "fp8",
            "--gpu-type",
            "mi300x",
            "--tuner",
            "a8w8",
            "--skip-gpu-check",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code != 0
    assert "MISALIGNED" in result.output
    assert executed == []


def test_aiter_preflight_defect_propagates(tmp_path, monkeypatch):
    def _boom():
        raise RuntimeError("preflight defect")

    monkeypatch.setattr("kernelforge.workflows.gemm_tune.aiter_preflight.collect", _boom)

    with pytest.raises(RuntimeError, match="^preflight defect$"):
        cli_mod._aiter_preflight_problems(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_demand_report_loader_defect_propagates(monkeypatch):
    def _boom(_path):
        raise RuntimeError("loader defect")

    monkeypatch.setattr("kernelforge.workflows.gemm_tune.evidence.load_demand", _boom)

    with pytest.raises(RuntimeError, match="^loader defect$"):
        cli_mod._load_demand_report("demand.json")


def test_aiter_preflight_returns_hard_problems(tmp_path, monkeypatch):
    _stub_preflight(monkeypatch, soft=["note"], hard=["MISALIGNED"])

    assert cli_mod._aiter_preflight_problems(tmp_path) == ["MISALIGNED"]


def test_cli_resolves_auto_once_and_records_effective_gpu(tmp_path, monkeypatch):
    from kernelforge.workflows.gemm_tune import router

    _stub_preflight(monkeypatch)
    model = _model_dir(tmp_path)
    output = tmp_path / "output"
    calls = []

    def detect():
        calls.append(True)
        return "gfx942"

    monkeypatch.setattr(router, "detect_arch", detect)
    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(model),
            "--framework",
            "vllm",
            "--precision",
            "bf16",
            "--gpu-type",
            "auto",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [True]
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "result.json").read_text())
    assert plan["gpu_type"] == "mi300x"
    assert report["gpu_type"] == "mi300x"
    assert '"gpu_type": "auto"' not in json.dumps([plan, report]).lower()


def test_cli_auto_detection_failure_aborts_before_model_or_tuning(tmp_path, monkeypatch):
    from kernelforge.workflows.gemm_tune import model_analyzer, router

    monkeypatch.setattr(router, "detect_arch", lambda: "")
    monkeypatch.setattr(
        model_analyzer,
        "analyze_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model analysis must not start")),
    )
    output = tmp_path / "output"

    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(tmp_path / "model"),
            "--framework",
            "vllm",
            "--precision",
            "bf16",
            "--gpu-type",
            "auto",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "--gpu-type" in result.output
    assert not (output / "plan.json").exists()


def test_plan_routes_the_same_inputs_as_run_and_run_builds_the_tuner_context(tmp_path, monkeypatch):
    from kernelforge.workflows.gemm_tune import router as router_mod

    _stub_preflight(monkeypatch)
    model = _model_dir(tmp_path)
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "[aiter] shape is M:512, N:1536, K:7168, "
        "not found tuned config in /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv\n",
        encoding="utf-8",
    )
    sources = {
        "untuned_csv": tmp_path / "untuned.csv",
        "moe_untuned_csv": tmp_path / "moe.csv",
        "shapes_json": tmp_path / "shapes.json",
        "shapes_manifest": tmp_path / "manifest.json",
        "tunableop_input": tmp_path / "tunableop.txt",
    }
    for path in sources.values():
        path.write_text("{}\n", encoding="utf-8")

    selection_calls = []
    contexts = []

    def select_tuners(_profile, target, evidence):
        selection_calls.append(vars(target) | vars(evidence))
        return [router_mod.TunerSpec("a8w8_blockscale", skip_reason="not supported in contract test")]

    def execute_tuners(_specs, ctx, _deadline):
        contexts.append(ctx)
        return [], []

    monkeypatch.setattr(router_mod, "select_tuners", select_tuners)
    monkeypatch.setattr(cli_mod, "_execute_tuners", execute_tuners)

    common = [
        "--model-path",
        str(model),
        "--framework",
        "vllm",
        "--precision",
        "bf16",
        "--gpu-type",
        "mi300x",
        "--kernel-signature-log",
        str(server_log),
        "--untuned-csv",
        str(sources["untuned_csv"]),
        "--tunableop-input",
        str(sources["tunableop_input"]),
    ]
    output = tmp_path / "output"
    runner = CliRunner()
    run_result = runner.invoke(
        gemm_tune,
        [
            "run",
            *common,
            "--shapes-json",
            str(sources["shapes_json"]),
            "--shapes-manifest",
            str(sources["shapes_manifest"]),
            "--moe-untuned-csv",
            str(sources["moe_untuned_csv"]),
            "--iters",
            "7",
            "--warmup",
            "3",
            "--min-improvement-pct",
            "1.5",
            "--m-grid",
            "exact",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )
    plan_result = runner.invoke(gemm_tune, ["plan", *common])

    assert run_result.exit_code == 0, run_result.output
    assert plan_result.exit_code == 0, plan_result.output
    assert len(selection_calls) == 2
    run_route, plan_route = selection_calls
    assert run_route == plan_route

    assert plan_route["kernel_signature_log"] == str(server_log)
    assert plan_route["has_untuned_csv"] is True
    assert plan_route["has_shapes_json"] is True
    assert plan_route["has_tunableop_input"] is True
    assert plan_route["demand_report"] is not None
    assert [demand["table"] for demand in plan_route["demand_report"]["demands"]] == ["a8w8_blockscale_tuned_gemm.csv"]

    (ctx,) = contexts
    assert (ctx.iters, ctx.warmup, ctx.min_improvement_pct) == (7, 3, 1.5)
    assert ctx.m_grid is MGrid.EXACT
    assert {
        "untuned_csv": ctx.untuned_csv,
        "moe_untuned_csv": ctx.moe_untuned_csv,
        "shapes_json": ctx.shapes_json,
        "shapes_manifest": ctx.shapes_manifest,
        "demand_json": ctx.demand_json,
        "tunableop_input": ctx.tunableop_input,
        "kernel_signature_log": ctx.kernel_signature_log,
    } == {
        **sources,
        "demand_json": output / "demand.json",
        "kernel_signature_log": server_log,
    }


def _run_with_ceiling(tmp_path, monkeypatch, *, routed: list, extra: list[str]) -> list[str]:
    """Invoke ``gemm-tune run`` over a fixed routed set and report what executed.

    A ``routed`` entry is a tuner name, or a ``(name, skip_reason)`` pair for one the router refused.
    """
    _stub_preflight(monkeypatch)
    model = _model_dir(tmp_path)
    executed: list[str] = []

    class _Tuner:
        def __init__(self, name):
            self.name = name

        def execute(self):
            executed.append(self.name)
            return TuneResult(tuner_name=self.name, status="no_improvement")

    from kernelforge.workflows.gemm_tune import router as router_mod
    from kernelforge.workflows.gemm_tune.router import TunerSpec

    def _spec(index, entry):
        name, skip_reason = entry if isinstance(entry, tuple) else (entry, None)
        return TunerSpec(
            name,
            skip_reason=skip_reason,
            priority=10 * (index + 1),
            estimated_minutes=0 if skip_reason else 20,
        )

    monkeypatch.setattr(cli_mod, "_create_tuner", lambda name, ctx: _Tuner(name))
    # The CLI imports select_tuners inside the command body, so the patch has to land on the router module it reads
    # from.
    monkeypatch.setattr(
        router_mod,
        "select_tuners",
        lambda *_a, **_k: [_spec(index, entry) for index, entry in enumerate(routed)],
    )
    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(model),
            "--framework",
            "vllm",
            "--precision",
            "fp8",
            "--gpu-type",
            "mi300x",
            "--skip-gpu-check",
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return executed


def test_the_tuner_ceiling_keeps_the_highest_priority_ones(tmp_path, monkeypatch):
    """A share paying for two of three tuners must not start the third."""
    executed = _run_with_ceiling(
        tmp_path,
        monkeypatch,
        routed=["fmoe_ck", "a8w8", "dense_bf16"],
        extra=["--max-tuners", "2"],
    )

    assert executed == ["fmoe_ck", "a8w8"]


def test_no_tuner_ceiling_runs_every_routed_tuner(tmp_path, monkeypatch):
    """Zero means no ceiling was derived, never "run nothing"."""
    executed = _run_with_ceiling(
        tmp_path,
        monkeypatch,
        routed=["fmoe_ck", "a8w8", "dense_bf16"],
        extra=["--max-tuners", "0"],
    )

    assert executed == ["fmoe_ck", "a8w8", "dense_bf16"]


def test_a_ceiling_wider_than_the_routed_set_drops_nothing(tmp_path, monkeypatch):
    executed = _run_with_ceiling(
        tmp_path,
        monkeypatch,
        routed=["fmoe_ck", "a8w8"],
        extra=["--max-tuners", "5"],
    )

    assert executed == ["fmoe_ck", "a8w8"]


def test_a_router_skipped_tuner_does_not_spend_a_ceiling_slot(tmp_path, monkeypatch):
    """A refused tuner books no time, so the share still pays for as many tuners as can actually run."""
    executed = _run_with_ceiling(
        tmp_path,
        monkeypatch,
        routed=[
            ("vllm_moe_triton", "MoE Triton sweep measures bf16 only; awq is not tuned"),
            "a8w8",
            "dense_bf16",
        ],
        extra=["--max-tuners", "2"],
    )

    assert executed == ["a8w8", "dense_bf16"]
