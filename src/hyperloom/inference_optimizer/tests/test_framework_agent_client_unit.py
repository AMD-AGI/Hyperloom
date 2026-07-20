# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``framework_agent_client``: fa binary resolution, the sync
subprocess wrapper (success / not-found / timeout), the async phase runner
error branches, and the ``phase_discover`` request shaping."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.orchestrator.framework import client as fac

# Reference the resolver's own constant so the two never drift apart.
_FA_MODULE = fac._FA_MODULE


# -- repo_url_for_framework -----------------------------------------------
def test_repo_url_for_framework_known_and_unknown() -> None:
    assert fac.repo_url_for_framework("sglang").endswith("sglang.git")
    assert fac.repo_url_for_framework("nope") == ""


# -- _resolve_fa_command ---------------------------------------------------
def test_resolve_fa_command_is_module_invocation(monkeypatch) -> None:
    """``fa`` runs as ``[python, -m, <module>]``, independent of $PATH."""
    monkeypatch.setattr(fac.sys, "executable", "/fake/python3")
    assert fac._resolve_fa_command() == ["/fake/python3", "-m", _FA_MODULE]


# -- _run_fa_subcommand_sync ----------------------------------------------
def test_run_fa_subcommand_sync_ok(monkeypatch) -> None:
    def _run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, '{"ok": 1}', "")

    monkeypatch.setattr(fac.subprocess, "run", _run)
    rc, out, err = fac._run_fa_subcommand_sync(["fa"], "phase-discover", Path("/x"), 5.0)
    assert (rc, out, err) == (0, '{"ok": 1}', "")


def test_run_fa_subcommand_sync_builds_full_command(monkeypatch) -> None:
    """The cmd prefix is preserved verbatim and the subcommand + IO flags are
    appended, so a multi-token module fallback prefix runs correctly."""
    captured: dict = {}

    def _run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "{}", "")

    monkeypatch.setattr(fac.subprocess, "run", _run)
    prefix = [sys.executable, "-m", _FA_MODULE]
    fac._run_fa_subcommand_sync(prefix, "phase-discover", Path("/req.json"), 5.0)
    assert captured["cmd"] == [*prefix, "phase-discover", "--request", "/req.json", "--out", "-"]


def test_run_fa_subcommand_sync_not_found(monkeypatch) -> None:
    def _run(cmd, **kw):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(fac.subprocess, "run", _run)
    rc, _out, err = fac._run_fa_subcommand_sync(["fa"], "phase-discover", Path("/x"), 5.0)
    assert rc == 127
    assert "not found" in err


def test_run_fa_subcommand_sync_timeout(monkeypatch) -> None:
    def _run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd="fa", timeout=5.0)

    monkeypatch.setattr(fac.subprocess, "run", _run)
    rc, _out, err = fac._run_fa_subcommand_sync(["fa"], "phase-discover", Path("/x"), 5.0)
    assert rc == 124
    assert "timed out" in err


def test_module_fallback_entry_starts_in_real_subprocess() -> None:
    """Smoke: ``python -m <module> schema`` launches and emits valid JSON."""
    # Inherit the parent env so the child keeps PYTHONPATH.
    proc = subprocess.run(
        [sys.executable, "-m", _FA_MODULE, "schema"],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "schema" in payload.get("subcommands_available", [])


# -- _invoke_fa_phase ------------------------------------------------------
@pytest.mark.asyncio
async def test_invoke_fa_phase_uses_resolved_command(tmp_path, monkeypatch) -> None:
    """The resolved command prefix is threaded verbatim into the sync runner."""
    captured: dict = {}

    def _fake_run(cmd_prefix, subcommand, request_path, timeout_sec):
        captured["cmd_prefix"] = cmd_prefix
        return (0, '{"candidates": []}', "")

    monkeypatch.setattr(fac, "_resolve_fa_command", lambda: [sys.executable, "-m", _FA_MODULE])
    monkeypatch.setattr(fac, "_run_fa_subcommand_sync", _fake_run)
    out = await fac._invoke_fa_phase(
        subcommand="phase-discover",
        request={},
        session_dir=tmp_path,
    )
    assert out == {"candidates": []}
    assert captured["cmd_prefix"] == [sys.executable, "-m", _FA_MODULE]


@pytest.mark.asyncio
async def test_invoke_fa_phase_nonzero_rc(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_command", lambda: ["fa"])
    monkeypatch.setattr(
        fac,
        "_run_fa_subcommand_sync",
        lambda *a, **k: (3, "", "boom"),
    )
    with pytest.raises(RuntimeError, match="exited rc=3"):
        await fac._invoke_fa_phase(
            subcommand="phase-discover",
            request={"a": 1},
            session_dir=tmp_path,
        )
    # temp request file is cleaned up
    assert not list((tmp_path / ".fa-tmp").glob("phase-*.json"))


@pytest.mark.asyncio
async def test_invoke_fa_phase_invalid_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_command", lambda: ["fa"])
    monkeypatch.setattr(
        fac,
        "_run_fa_subcommand_sync",
        lambda *a, **k: (0, "not json", ""),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        await fac._invoke_fa_phase(
            subcommand="phase-discover",
            request={},
            session_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_invoke_fa_phase_happy_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fac, "_resolve_fa_command", lambda: ["fa"])
    monkeypatch.setattr(
        fac,
        "_run_fa_subcommand_sync",
        lambda *a, **k: (0, '{"candidates": []}', ""),
    )
    out = await fac._invoke_fa_phase(
        subcommand="phase-discover",
        request={},
        session_dir=tmp_path,
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
        model="m",
        framework="SGLang",
        gpu_type="mi300x",
        gaps=[{"area": "moe"}],
        session_dir=tmp_path,
        keywords=["Fused", "fused", " MoE ", ""],
        max_candidates=3,
        batch_id="b1",
    )
    assert out["batch_id"] == "b1"
    req = captured["request"]
    assert req["framework"] == "sglang"  # normalized
    assert req["repo_url"].endswith("sglang.git")  # resolved from framework
    assert req["max_search_candidates"] == 3
    assert req["keywords"] == ["fused", "moe"]


@pytest.mark.asyncio
async def test_phase_discover_plumbs_exclusion_memory(tmp_path, monkeypatch) -> None:
    """excluded_candidate_ids + failed_candidate_context reach the request
    (deduped / truncated) so fa can hard-filter already-seen candidates."""
    captured: dict = {}

    async def _fake_invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["request"] = request
        return {"batch_id": "b1", "candidates": []}

    monkeypatch.setattr(fac, "_invoke_fa_phase", _fake_invoke)
    failed = [{"ref": f"PR:{i}", "status": "reverted", "why": "x"} for i in range(15)]
    await fac.phase_discover(
        model="m",
        framework="sglang",
        gpu_type="mi300x",
        gaps=[{"area": "moe"}],
        session_dir=tmp_path,
        excluded_candidate_ids=["PR:1", "PR:1", " PR:2 ", ""],
        failed_candidate_context=failed,
    )
    req = captured["request"]
    assert req["excluded_candidate_ids"] == ["PR:1", "PR:2"]
    assert len(req["failed_candidate_context"]) == 10
    assert req["failed_candidate_context"][-1]["ref"] == "PR:14"


@pytest.mark.asyncio
async def test_phase_discover_omits_exclusion_keys_when_empty(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["request"] = request
        return {"batch_id": "b1", "candidates": []}

    monkeypatch.setattr(fac, "_invoke_fa_phase", _fake_invoke)
    await fac.phase_discover(
        model="m",
        framework="sglang",
        gpu_type="mi300x",
        gaps=[{"area": "moe"}],
        session_dir=tmp_path,
    )
    req = captured["request"]
    assert "excluded_candidate_ids" not in req
    assert "failed_candidate_context" not in req


# -- phase_audit -----------------------------------------------------------
@pytest.mark.asyncio
async def test_phase_audit_same_framework_omits_target_framework(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["request"] = request
        return {"recommended_next_step": "skip"}

    monkeypatch.setattr(fac, "_invoke_fa_phase", _fake_invoke)
    await fac.phase_audit(
        candidate={"repo": "sgl-project/sglang"},
        framework="sglang",
        framework_source_roots=["/src/sglang"],
        target_framework="sglang",  # same as framework
        session_dir=tmp_path,
    )
    req = captured["request"]
    assert "target_framework" not in req
    assert "target_framework_source_roots" not in req


@pytest.mark.asyncio
async def test_phase_audit_cross_framework_sets_target_and_roots(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["request"] = request
        return {"recommended_next_step": "author_via_specialist", "layer": "cross_framework"}

    monkeypatch.setattr(fac, "_invoke_fa_phase", _fake_invoke)
    await fac.phase_audit(
        candidate={"repo": "sgl-project/sglang"},
        framework="sglang",
        framework_source_roots=["/src/sglang"],
        target_framework="vllm",
        target_framework_source_roots=["/src/vllm"],
        session_dir=tmp_path,
    )
    req = captured["request"]
    assert req["target_framework"] == "vllm"
    assert req["target_framework_source_roots"] == ["/src/vllm"]


@pytest.mark.asyncio
async def test_phase_audit_cross_framework_without_explicit_roots_omits_key(tmp_path, monkeypatch) -> None:
    """No target_framework_source_roots passed -> key absent (fa falls back to
    framework_source_roots)."""
    captured: dict = {}

    async def _fake_invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["request"] = request
        return {}

    monkeypatch.setattr(fac, "_invoke_fa_phase", _fake_invoke)
    await fac.phase_audit(
        candidate={"repo": "sgl-project/sglang"},
        framework="sglang",
        framework_source_roots=["/src/sglang"],
        target_framework="vllm",
        session_dir=tmp_path,
    )
    req = captured["request"]
    assert req["target_framework"] == "vllm"
    assert "target_framework_source_roots" not in req
