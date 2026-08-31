# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the optional GPU-specialist rebench helper.

``run_grid`` / Magpie are mocked, so these exercise port resolution,
leased-card reporting, env-pair parsing, result shapes, and the CLI ``main``.
"""

from __future__ import annotations

import json
import pytest

from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.specialists import rebench as sr


def _variant_result(**overrides) -> VariantResult:
    """A real ``VariantResult``, which is what ``run_grid`` returns.

    Built from the dataclass rather than a namespace so a field the production
    code reads under the wrong name fails here instead of silently yielding
    ``None`` in the emitted result.
    """
    fields = {
        "name": "rebench",
        "extra_server_args": "",
        "extra_envs": {},
        "status": "succeeded",
        "output_throughput": 1234.5,
        "ttft_mean_ms": 10.0,
        "tpot_mean_ms": 2.0,
        "error": "",
        "nonfatal_warnings": ["w1"],
    }
    fields.update(overrides)
    return VariantResult(**fields)


def test_resolve_port_auto_and_explicit() -> None:
    assert 0 < sr._resolve_port(None) <= 65535
    assert 0 < sr._resolve_port(0) <= 65535
    assert sr._resolve_port(12345) == 12345
    assert sr._resolve_port(8888) == 8888


def test_current_leased_cards_precedence(monkeypatch) -> None:
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    assert sr._current_leased_cards() == ""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    assert sr._current_leased_cards() == "4,5"
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "6,7")
    assert sr._current_leased_cards() == "6,7"


def test_parse_env_pairs() -> None:
    assert sr._parse_env_pairs(None) == {}
    assert sr._parse_env_pairs(["A=1", "B=x=y", "malformed", "=nope", "  C =3"]) == {"A": "1", "B": "x=y", "C": "3"}


@pytest.mark.asyncio
async def test_run_specialist_rebench_success(tmp_path, monkeypatch) -> None:
    fake = _variant_result(workspace=str(tmp_path / "ws"))
    seen: dict = {}

    async def _fake_run_grid(**kwargs):
        seen.update(kwargs)
        return [fake]

    monkeypatch.setattr(sr, "default_baseline_config", lambda: str(tmp_path / "base.yaml"))
    monkeypatch.setattr(sr, "materialize_config_with_envs", lambda *a, **k: tmp_path / "mat.yaml")
    monkeypatch.setattr(sr, "run_grid", _fake_run_grid)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    res = await sr.run_specialist_rebench(
        config_path=None,
        output_dir=str(tmp_path / "out"),
        base_extra_args="--kv-cache-dtype fp8_e4m3",
        extra_envs={"E": "1"},
        port=12321,
    )
    assert res["ok"] is True
    assert res["status"] == "succeeded"
    assert res["output_throughput"] == 1234.5
    assert res["port"] == 12321
    assert res["gpu_ids"] == "4,5,6,7"
    assert "w1" in res["warnings"]
    assert seen["base_extra_args"] == "--kv-cache-dtype fp8_e4m3"
    assert seen["grid"][0].extra_server_args == ""
    assert seen["preclean_before_run"] is False


@pytest.mark.asyncio
async def test_run_specialist_rebench_failed_status(tmp_path, monkeypatch) -> None:
    fake = _variant_result(
        status="failed",
        output_throughput=None,
        ttft_mean_ms=None,
        tpot_mean_ms=None,
        workspace="",
        error="boom",
        nonfatal_warnings=[],
    )

    async def _fake_run_grid(**kwargs):
        return [fake]

    monkeypatch.setattr(sr, "materialize_config_with_envs", lambda *a, **k: tmp_path / "mat.yaml")
    monkeypatch.setattr(sr, "run_grid", _fake_run_grid)

    res = await sr.run_specialist_rebench(
        config_path=str(tmp_path / "c.yaml"),
        output_dir=str(tmp_path / "o"),
    )
    assert res["ok"] is False
    assert any("rebench_failed" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_run_specialist_rebench_no_result(tmp_path, monkeypatch) -> None:
    async def _fake_run_grid(**kwargs):
        return []

    monkeypatch.setattr(sr, "materialize_config_with_envs", lambda *a, **k: tmp_path / "m.yaml")
    monkeypatch.setattr(sr, "run_grid", _fake_run_grid)

    res = await sr.run_specialist_rebench(
        config_path=str(tmp_path / "c.yaml"),
        output_dir=str(tmp_path / "o"),
    )
    assert res["ok"] is False
    assert "no result" in res["error"]


@pytest.mark.asyncio
async def test_run_specialist_rebench_exception_surfaced(tmp_path, monkeypatch) -> None:
    async def _boom(**kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(sr, "materialize_config_with_envs", lambda *a, **k: tmp_path / "m.yaml")
    monkeypatch.setattr(sr, "run_grid", _boom)

    res = await sr.run_specialist_rebench(
        config_path=str(tmp_path / "c.yaml"),
        output_dir=str(tmp_path / "o"),
    )
    assert res["ok"] is False
    assert "kaboom" in res["error"]


def test_main_success_prints_json(tmp_path, monkeypatch, capsys) -> None:
    async def _fake(**kwargs):
        return {"ok": True, "output_throughput": 5.0}

    monkeypatch.setattr(sr, "run_specialist_rebench", _fake)
    rc = sr.main(["--output", str(tmp_path / "o")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_main_failure_return_code(tmp_path, monkeypatch, capsys) -> None:
    async def _fake(**kwargs):
        return {"ok": False, "error": "x"}

    monkeypatch.setattr(sr, "run_specialist_rebench", _fake)
    rc = sr.main(["--output", str(tmp_path / "o"), "--env", "A=1", "--extra-args=--foo bar"])
    assert rc == 1
