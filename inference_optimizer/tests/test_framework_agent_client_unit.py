# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``framework_agent_client``: fa binary resolution, the sync
subprocess wrapper (success / not-found / timeout), the async phase runner
error branches, and the ``phase_discover`` request shaping."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import framework_agent_client as fac


# -- repo_url_for_framework -----------------------------------------------
def test_repo_url_for_framework_known_and_unknown() -> None:
    assert fac.repo_url_for_framework("sglang").endswith("sglang.git")
    assert fac.repo_url_for_framework("nope") == ""


# -- _resolve_fa_binary ----------------------------------------------------
def test_resolve_fa_binary_explicit_env(tmp_path, monkeypatch) -> None:
    fa = tmp_path / "fa"
    fa.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FA_BIN", str(fa))
    assert fac._resolve_fa_binary() == str(fa)


def test_resolve_fa_binary_via_path(monkeypatch) -> None:
    monkeypatch.delenv("FA_BIN", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    monkeypatch.setattr(fac.shutil, "which", lambda _n: "/usr/bin/fa")
    assert fac._resolve_fa_binary() == "/usr/bin/fa"


def test_resolve_fa_binary_via_root(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FA_BIN", raising=False)
    monkeypatch.setattr(fac.shutil, "which", lambda _n: None)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "fa").write_text("#!/bin/sh\n")
    monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path))
    assert fac._resolve_fa_binary() == str(scripts / "fa")


def test_resolve_fa_binary_none(monkeypatch) -> None:
    monkeypatch.delenv("FA_BIN", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    monkeypatch.setattr(fac.shutil, "which", lambda _n: None)
    assert fac._resolve_fa_binary() is None


# -- _run_fa_subcommand_sync ----------------------------------------------
def test_run_fa_subcommand_sync_ok(monkeypatch) -> None:
    def _run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, '{"ok": 1}', "")

    monkeypatch.setattr(fac.subprocess, "run", _run)
    rc, out, err = fac._run_fa_subcommand_sync("fa", "phase-discover", Path("/x"), 5.0)
    assert (rc, out, err) == (0, '{"ok": 1}', "")


def test_run_fa_subcommand_sync_not_found(monkeypatch) -> None:
    def _run(cmd, **kw):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(fac.subprocess, "run", _run)
    rc, _out, err = fac._run_fa_subcommand_sync("fa", "phase-discover", Path("/x"), 5.0)
    assert rc == 127
    assert "not found" in err


def test_run_fa_subcommand_sync_timeout(monkeypatch) -> None:
    def _run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd="fa", timeout=5.0)

    monkeypatch.setattr(fac.subprocess, "run", _run)
    rc, _out, err = fac._run_fa_subcommand_sync("fa", "phase-discover", Path("/x"), 5.0)
    assert rc == 124
    assert "timed out" in err


# -- _invoke_fa_phase ------------------------------------------------------
@pytest.mark.asyncio
async def test_invoke_fa_phase_no_binary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_binary", lambda: None)
    with pytest.raises(RuntimeError, match="fa binary not found"):
        await fac._invoke_fa_phase(
            subcommand="phase-discover", request={}, session_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_invoke_fa_phase_nonzero_rc(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_binary", lambda: "fa")
    monkeypatch.setattr(
        fac, "_run_fa_subcommand_sync",
        lambda *a, **k: (3, "", "boom"),
    )
    with pytest.raises(RuntimeError, match="exited rc=3"):
        await fac._invoke_fa_phase(
            subcommand="phase-discover", request={"a": 1}, session_dir=tmp_path,
        )
    # temp request file is cleaned up
    assert not list((tmp_path / ".fa-tmp").glob("phase-*.json"))


@pytest.mark.asyncio
async def test_invoke_fa_phase_invalid_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_binary", lambda: "fa")
    monkeypatch.setattr(
        fac, "_run_fa_subcommand_sync",
        lambda *a, **k: (0, "not json", ""),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        await fac._invoke_fa_phase(
            subcommand="phase-discover", request={}, session_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_invoke_fa_phase_happy_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_binary", lambda: "fa")
    monkeypatch.setattr(
        fac, "_run_fa_subcommand_sync",
        lambda *a, **k: (0, '{"candidates": []}', ""),
    )
    out = await fac._invoke_fa_phase(
        subcommand="phase-discover", request={}, session_dir=tmp_path,
    )
    assert out == {"candidates": []}


# -- phase_discover --------------------------------------------------------
@pytest.mark.asyncio
async def test_phase_discover_shapes_request_and_dedups_keywords(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["subcommand"] = subcommand
        captured["request"] = request
        return {"batch_id": "b1", "candidates": []}

    monkeypatch.setattr(fac, "_invoke_fa_phase", _fake_invoke)
    out = await fac.phase_discover(
        model="m", framework="SGLang", gpu_type="mi300x",
        gaps=[{"area": "moe"}], session_dir=tmp_path,
        keywords=["Fused", "fused", " MoE ", ""], max_candidates=3, batch_id="b1",
    )
    assert out["batch_id"] == "b1"
    req = captured["request"]
    assert req["framework"] == "sglang"  # normalized
    assert req["repo_url"].endswith("sglang.git")  # resolved from framework
    assert req["max_search_candidates"] == 3
    # keywords lowercased, trimmed, de-duplicated preserving order
    assert req["keywords"] == ["fused", "moe"]
