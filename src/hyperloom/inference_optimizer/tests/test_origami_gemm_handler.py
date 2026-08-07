# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator coverage for the Origami GEMM fallback pre-tuner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState


def test_pre_tuning_env_is_inserted_before_existing_env_args(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    seed = tmp_path / "origami.csv"
    seed.write_text("M,N,K\n", encoding="utf-8")

    cmd = krh._with_gemm_pre_tuning_env(
        ["env", "E2E_METRIC=output", "python3", "tune.py"],
        {
            "_origami_pre_tuning_envs": {
                "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(seed)
            }
        },
    )

    assert cmd[:3] == [
        "env",
        f"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE={seed}",
        "E2E_METRIC=output",
    ]


@pytest.mark.asyncio
async def test_origami_fallback_shapes_pre_tuning_seed(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    caplog.set_level("INFO", logger=krh.__name__)
    state = SharedState(
        precision="fp8",
        framework="vllm",
        model_path="/models/block-fp8",
        gpu_type="mi355x",
    )
    state.save(tmp_path)

    kernel_root = tmp_path / "kernel-agent"
    tool = kernel_root / "tools" / "origami_gemm_select.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# placeholder\n", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_root))
    monkeypatch.setattr(krh, "_resolve_aiter_root_for_forge", lambda: "/aiter")

    captured: dict[str, object] = {}

    async def fake_run(cmd: list[str], *, timeout_sec: int):
        captured["cmd"] = cmd
        captured["timeout_sec"] = timeout_sec
        input_path = Path(cmd[cmd.index("--input-json") + 1])
        data = json.loads(input_path.read_text(encoding="utf-8"))
        candidate = Path(data["output_dir"]) / "origami_a8w8_blockscale.csv"
        candidate.write_text(
            "gfx,cu_num,M,N,K,libtype,kernelId,splitK,kernelName\n"
            "gfx950,304,16,4096,8192,ck,3,0,origami-kernel\n",
            encoding="utf-8",
        )
        return (
            0,
            json.dumps(
                {
                    "status": "ok",
                    "candidate": True,
                    "observed_shapes": 1,
                    "fallback_shapes": 1,
                    "benchmarked_shapes": 1,
                    "selected_shapes": 1,
                    "report_path": str(Path(data["output_dir"]) / "report.json"),
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                    "env_value": str(candidate),
                    "rows": [
                        {
                            "status": "selected",
                            "M": 16,
                            "N": 4096,
                            "K": 8192,
                            "kernelId": 3,
                            "benchmark_selected_us": 8.0,
                            "benchmark_default_us": 10.0,
                            "benchmark_speedup": 1.25,
                        }
                    ],
                }
            ),
            "",
        )

    monkeypatch.setattr(krh, "_run_subprocess", fake_run)

    result = await krh._run_origami_gemm_fallback(
        {
            "task_id": "origami-test",
            "precision": "fp8",
            "quant_type": "blockscale",
            "shapes_json": [{"M": 16, "N": 4096, "K": 8192}],
        },
        session_dir=tmp_path,
    )

    assert result["selector"] == "origami"
    assert result["pre_tuning_seed"] is True
    assert result["selected_shapes"] == 1
    assert "requires_e2e_validation" not in result
    assert "--input-json" in captured["cmd"]
    assert "ORIGAMI_GEMM_START" in caplog.text
    assert "ORIGAMI_GEMM_SUMMARY" in caplog.text
    assert "ORIGAMI_GEMM_WIN" in caplog.text


@pytest.mark.asyncio
async def test_run_gemm_tuning_seeds_then_calls_geak(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    caplog.set_level("INFO", logger=krh.__name__)
    seed = tmp_path / "origami-merged.csv"
    seed.write_text("M,N,K\n16,4096,8192\n", encoding="utf-8")
    candidate = {
        "status": "ok",
        "candidate": True,
        "selector": "origami",
        "selected_shapes": 2,
        "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        "env_value": str(seed),
    }
    called: dict[str, object] = {}

    async def fake_origami(_payload, *, session_dir):
        assert session_dir == tmp_path
        return dict(candidate)

    async def fake_geak(payload, *, session_dir):
        called["payload"] = payload
        assert session_dir == tmp_path
        return {"status": "complete", "backend": "geak", "decision": "KEEP"}

    monkeypatch.setattr(krh, "_run_origami_gemm_fallback", fake_origami)
    monkeypatch.setattr(krh, "_run_geak_gemm_tuning", fake_geak)

    result = await krh.run_gemm_tuning_handler(
        {"task_id": "origami-seeds-geak"},
        session_dir=tmp_path,
    )

    assert result["backend"] == "geak"
    assert called["payload"]["_origami_pre_tuning_envs"] == {
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(seed)
    }
    assert result["origami_fallback"]["candidate"] is True
    assert result["origami_fallback"]["applied_before_backend"] is True
    assert result["origami_fallback"]["authoritative_backend"] == "geak"
    assert "ORIGAMI_GEMM_INJECT" in caplog.text
    assert "ORIGAMI_GEMM_BACKEND backend=geak" in caplog.text


@pytest.mark.asyncio
async def test_run_gemm_tuning_seeds_then_calls_forge(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    caplog.set_level("INFO", logger=krh.__name__)
    seed = tmp_path / "origami-merged.csv"
    seed.write_text("M,N,K\n16,4096,8192\n", encoding="utf-8")

    async def fake_origami(_payload, *, session_dir):
        assert session_dir == tmp_path
        return {
            "status": "ok",
            "candidate": True,
            "selected_shapes": 1,
            "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            "env_value": str(seed),
        }

    async def fake_forge(payload, *, session_dir):
        assert session_dir == tmp_path
        assert payload["_origami_pre_tuning_envs"] == {
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(seed)
        }
        return {"status": "complete", "backend": "forge", "decision": "KEEP"}

    monkeypatch.setattr(krh, "_run_origami_gemm_fallback", fake_origami)
    monkeypatch.setattr(krh, "_run_forge_gemm_tuning", fake_forge)

    result = await krh.run_gemm_tuning_handler(
        {"task_id": "origami-seeds-forge"},
        session_dir=tmp_path,
    )

    assert result["backend"] == "forge"
    assert "ORIGAMI_GEMM_INJECT" in caplog.text
    assert "ORIGAMI_GEMM_BACKEND backend=forge" in caplog.text


@pytest.mark.asyncio
async def test_run_gemm_tuning_continues_when_origami_errors(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", "1")
    caplog.set_level("INFO", logger=krh.__name__)

    async def broken_origami(_payload, *, session_dir):
        raise RuntimeError("origami unavailable")

    async def fake_geak(_payload, *, session_dir):
        return {"status": "skipped", "backend": "geak", "decision": "REVERT"}

    monkeypatch.setattr(krh, "_run_origami_gemm_fallback", broken_origami)
    monkeypatch.setattr(krh, "_run_geak_gemm_tuning", fake_geak)

    result = await krh.run_gemm_tuning_handler(
        {"task_id": "origami-fail-soft"},
        session_dir=tmp_path,
    )

    assert result["backend"] == "geak"
    assert result["origami_fallback"]["reason"] == "selector_error"
    assert "ORIGAMI_GEMM_ERROR reason=selector_exception" in caplog.text
    assert "ORIGAMI_GEMM_BACKEND backend=geak" in caplog.text


def test_pre_tuning_env_is_blocked_when_origami_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", raising=False)
    seed = tmp_path / "origami.csv"
    seed.write_text("M,N,K\n", encoding="utf-8")
    original = ["env", "E2E_METRIC=output", "python3", "tune.py"]

    cmd = krh._with_gemm_pre_tuning_env(
        original,
        {
            "_origami_pre_tuning_envs": {
                "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(seed)
            }
        },
    )

    assert cmd is original


@pytest.mark.asyncio
async def test_origami_helper_disabled_is_zero_touch(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", raising=False)
    workspace = tmp_path / "runs" / "gemm_tuning" / "disabled" / "origami"

    result = await krh._run_origami_gemm_fallback(
        {"task_id": "disabled"},
        session_dir=tmp_path,
    )

    assert result == {
        "status": "skipped",
        "reason": "disabled",
        "candidate": False,
    }
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_handler_disabled_skips_all_origami_paths(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("HYPERLOOM_ORIGAMI_GEMM_FALLBACK", raising=False)
    caplog.set_level("INFO", logger=krh.__name__)
    payload = {"task_id": "disabled-handler"}
    called: dict[str, object] = {}

    async def forbidden_origami(*_args, **_kwargs):
        raise AssertionError("disabled handler must not call Origami")

    async def fake_geak(received, *, session_dir):
        called["payload"] = received
        assert session_dir == tmp_path
        return {"status": "complete", "backend": "geak", "decision": "KEEP"}

    monkeypatch.setattr(krh, "_run_origami_gemm_fallback", forbidden_origami)
    monkeypatch.setattr(krh, "_run_geak_gemm_tuning", fake_geak)

    result = await krh.run_gemm_tuning_handler(payload, session_dir=tmp_path)

    assert called["payload"] is payload
    assert "_origami_pre_tuning_envs" not in payload
    assert "origami_fallback" not in result
    assert "ORIGAMI_GEMM_" not in caplog.text
