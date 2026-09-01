#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the forge-fusion kernel-agent wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.common.codex_session import (
    CODEX_EXTERNAL_SANDBOX_ENV,
    CODEX_SANDBOX_MODE_ENV,
)


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "forge_fusion.py"
_SPEC = importlib.util.spec_from_file_location("forge_fusion_tool", _MODULE_PATH)
assert _SPEC and _SPEC.loader
forge_fusion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(forge_fusion)


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Restore ``os.environ`` after every test.

    The Claude branch of ``_inject_author_gateway_env`` (exercised directly and
    via ``main``) mutates ``os.environ`` in place by design; the Codex branch is
    a no-op. ``monkeypatch`` does not revert keys the function writes directly,
    so without this snapshot the Claude auth aliases and stability variables
    pollute later auth/endpoint tests in a full-suite run.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _payload(output_dir: Path) -> dict:
    return {
        "trace_path": "/tmp/decode.trace.json.gz",
        "model_path": "/models/zaya",
        "framework": "sglang",
        "output_dir": str(output_dir),
        "discover_mode": "llm",
        "agent_backend": "claude",
        "llm_model": "claude-opus-4-6",
        "agent_sandbox_mode": "workspace-write",
        "max_turns": 7,
        "gpu": "0",
        "timeout": 9,
    }


def _sentinel_payload(text: str) -> dict:
    start = text.index(forge_fusion.RESULT_BEGIN) + len(forge_fusion.RESULT_BEGIN)
    end = text.index(forge_fusion.RESULT_END)
    return json.loads(text[start:end].strip())


def test_build_cmd_maps_core_options(tmp_path):
    cmd = forge_fusion._build_cmd(_payload(tmp_path))

    assert cmd[:3] == [forge_fusion.sys.executable, "-m", "kernelforge.cli"]
    assert cmd[3] == "forge-fuse"
    assert cmd[cmd.index("--trace") + 1] == "/tmp/decode.trace.json.gz"
    assert cmd[cmd.index("--model-path") + 1] == "/models/zaya"
    assert cmd[cmd.index("--framework") + 1] == "sglang"
    assert cmd[cmd.index("--output-dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--agent-backend") + 1] == "claude"
    # The model flag is spelled the way forge-loop spells it; forge-fuse rejects
    # the old --llm-model outright rather than ignoring it.
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-6"
    assert "--llm-model" not in cmd
    assert cmd[cmd.index("--agent-sandbox-mode") + 1] == "workspace-write"
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    assert "--fuse-all-confirmed" in cmd
    assert "--tp" not in cmd
    assert "--block-size" not in cmd


def test_build_cmd_forwards_session_serve_args(tmp_path):
    payload = _payload(tmp_path)
    payload.update({"tp": 8, "block_size": 128, "max_model_len": 13312})
    cmd = forge_fusion._build_cmd(payload)
    assert cmd[cmd.index("--tp") + 1] == "8"
    assert cmd[cmd.index("--block-size") + 1] == "128"
    assert cmd[cmd.index("--max-model-len") + 1] == "13312"


def test_inject_author_gateway_env_adds_stability_defaults(monkeypatch):
    for name in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "IS_SANDBOX",
        "API_TIMEOUT_MS",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "DISABLE_AUTOUPDATER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/api/v1/llm-proxy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-token")
    # IS_SANDBOX is only set when running as root (SWSPLAT-42390): simulate root
    # so the sandbox default is exercised.
    monkeypatch.setattr(forge_fusion.os, "geteuid", lambda: 0, raising=False)

    forge_fusion._inject_author_gateway_env("claude")

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example/api/v1/llm-proxy"
    assert os.environ["ANTHROPIC_API_KEY"] == "anthropic-token"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "anthropic-token"
    assert os.environ["IS_SANDBOX"] == "1"
    assert os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert os.environ["DISABLE_AUTOUPDATER"] == "1"
    assert "API_TIMEOUT_MS" not in os.environ


def test_inject_author_gateway_env_skips_sandbox_when_non_root(monkeypatch):
    # SWSPLAT-42390: as a non-root user, IS_SANDBOX must NOT be set (we do not
    # defeat claude's bypassPermissions guard for sessions that never needed it).
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "IS_SANDBOX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/api/v1/llm-proxy/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "safe-token")
    monkeypatch.setattr(forge_fusion.os, "geteuid", lambda: 1000, raising=False)

    forge_fusion._inject_author_gateway_env("claude")

    assert "IS_SANDBOX" not in os.environ


def test_inject_author_gateway_env_leaves_codex_environment_untouched(monkeypatch):
    """Codex must not inherit Claude auth aliases, sandboxing, or stability knobs."""
    for name in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "IS_SANDBOX",
        "API_TIMEOUT_MS",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "DISABLE_AUTOUPDATER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/Unified/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-token")
    monkeypatch.setattr(forge_fusion.os, "geteuid", lambda: 0, raising=False)
    before = dict(os.environ)

    forge_fusion._inject_author_gateway_env("codex")

    assert dict(os.environ) == before


def test_build_cmd_rejects_invalid_agent_backend(tmp_path):
    payload = _payload(tmp_path)
    payload["agent_backend"] = "anthropic"

    with pytest.raises(ValueError, match="agent_backend"):
        forge_fusion._build_cmd(payload)


def test_build_cmd_forwards_read_only_agent_sandbox(tmp_path):
    payload = _payload(tmp_path)
    payload["agent_backend"] = "codex"
    payload["agent_sandbox_mode"] = "read-only"

    cmd = forge_fusion._build_cmd(payload)

    assert cmd[cmd.index("--agent-sandbox-mode") + 1] == "read-only"


def test_build_cmd_forwards_confirmed_bypass_agent_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv(CODEX_SANDBOX_MODE_ENV, "bypass")
    monkeypatch.setenv(CODEX_EXTERNAL_SANDBOX_ENV, "1")
    payload = _payload(tmp_path)
    payload["agent_backend"] = "codex"
    payload["agent_sandbox_mode"] = "bypass"

    cmd = forge_fusion._build_cmd(payload)

    assert cmd[cmd.index("--agent-sandbox-mode") + 1] == "bypass"


def test_build_cmd_rejects_unconfirmed_bypass_agent_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv(CODEX_SANDBOX_MODE_ENV, "bypass")
    monkeypatch.delenv(CODEX_EXTERNAL_SANDBOX_ENV, raising=False)
    payload = _payload(tmp_path)
    payload["agent_backend"] = "codex"
    payload["agent_sandbox_mode"] = "bypass"

    with pytest.raises(RuntimeError, match=CODEX_EXTERNAL_SANDBOX_ENV):
        forge_fusion._build_cmd(payload)


def test_build_cmd_rejects_invalid_agent_sandbox_mode(tmp_path):
    payload = _payload(tmp_path)
    payload["agent_sandbox_mode"] = "unconfined"

    with pytest.raises(RuntimeError, match="unknown Codex sandbox mode"):
        forge_fusion._build_cmd(payload)


def test_main_passes_timeout_to_tree_runner(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "fusion_loop": {
            "kept": False,
            "best": {},
        },
        "validation": {},
        "artifacts": {},
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")
    captured: dict[str, object] = {}

    class Proc:
        returncode = 0
        stdout = "OUT\n"
        stderr = "ERR\n"

    def fake_run(cmd, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return Proc()

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", fake_run)

    rc = forge_fusion.main(["--input-json", str(input_json)])

    out = capsys.readouterr()
    assert rc == 0
    assert out.out.startswith("OUT\n")
    assert out.err == "ERR\n"
    assert captured["timeout"] == 9
    result = _sentinel_payload(out.out)
    assert result["decision"] == "REVERT"
    assert result["kept"] is False


def test_timeout_sec_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "not-an-int")

    assert forge_fusion._timeout_sec({}) == forge_fusion.DEFAULT_TIMEOUT_SEC


def test_main_timeout_emits_revert_result(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")

    def fake_run(cmd, timeout):
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output="PARTIAL OUT\n",
            stderr=b"PARTIAL ERR\n",
        )

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", fake_run)

    rc = forge_fusion.main(["--input-json", str(input_json)])

    out = capsys.readouterr()
    assert rc == 124
    assert out.out.startswith("PARTIAL OUT\n")
    assert out.err == "PARTIAL ERR\n"
    result = _sentinel_payload(out.out)
    assert result["status"] == "failed"
    assert result["error_class"] == "subprocess_timeout"
    assert result["decision"] == "REVERT"
    assert result["kept"] is False
    assert result["requires_e2e_validation"] is False
    assert json.loads((output_dir / "result.json").read_text(encoding="utf-8")) == result


def test_main_timeout_salvages_micro_keep_and_patch(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    patch = output_dir / "fusion.patch"
    input_json = tmp_path / "input.json"
    payload = _payload(output_dir)
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    def fake_run(cmd, timeout):
        (output_dir / "kernel_keep_checkpoint.json").write_text(
            json.dumps(
                {
                    "kept": True,
                    "kernel_speedup": 2.69,
                    "env_flag": "QWEN_FUSED",
                    "source_file": "/fw/model.py",
                    "repo_root": "/fw",
                }
            ),
            encoding="utf-8",
        )
        patch.write_text("diff --git a/model.py b/model.py\n", encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd, timeout, output="", stderr="")

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", fake_run)

    rc = forge_fusion.main(["--input-json", str(input_json)])

    out = capsys.readouterr()
    assert rc == 124
    result = _sentinel_payload(out.out)
    assert result["kept"] is True
    assert result["decision"] == "KEEP"
    assert result["requires_e2e_validation"] is True
    assert result["salvaged"] is True
    assert result["patch"] == str(patch)
    assert result["kernel_speedup"] == 2.69


def test_main_timeout_does_not_salvage_stale_previous_run(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "kernel_keep_checkpoint.json").write_text(
        json.dumps(
            {
                "kept": True,
                "kernel_speedup": 9.99,
                "env_flag": "STALE_FUSED",
                "source_file": "/fw/stale.py",
                "repo_root": "/fw",
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "fusion.patch").write_text("diff --git a/stale.py b/stale.py\n", encoding="utf-8")
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")

    def fake_run(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout, output="", stderr="")

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", fake_run)

    rc = forge_fusion.main(["--input-json", str(input_json)])

    result = _sentinel_payload(capsys.readouterr().out)
    assert rc == 124
    assert result["kept"] is False
    assert result["decision"] == "REVERT"


def test_run_with_tree_timeout_captures_output():
    cp = forge_fusion._run_with_tree_timeout(
        [
            forge_fusion.sys.executable,
            "-c",
            "import sys; print('hi'); sys.stderr.write('err')",
        ],
        timeout_sec=30,
    )

    assert cp.returncode == 0
    assert "hi" in (cp.stdout or "")
    assert "err" in (cp.stderr or "")


def test_run_with_tree_timeout_reaps_on_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        forge_fusion._run_with_tree_timeout(
            [forge_fusion.sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_sec=1,
        )


def _patch_file(output_dir) -> str:
    """A real patch file, which is what KernelForge's manifest actually names.

    ``artifacts.patch`` is a path, not the diff text, and integrate reads it off
    disk -- so a fixture holding the text would not exercise what is checked.
    """
    path = Path(output_dir) / "fusion.patch"
    path.write_text("diff --git a/foo.py b/foo.py\n", encoding="utf-8")
    return str(path)


def test_normalize_manifest_kept_writes_keep_result(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "fusion_loop": {
            "kept": True,
            "best": {"kernel_speedup": 1.12},
            "best_env_flag": "VLLM_FUSE=1",
            "best_pattern": "decode_fuse",
        },
        "validation": {"kept": True, "kernel_speedup": 1.12},
        "artifacts": {
            "patch": _patch_file(output_dir),
            "changes": [{"path": "foo.py"}],
            # KernelForge sets repo_root exactly when it sets a patch.
            "repo_root": "/repo/root",
        },
        "fusion": {"source_file": str(output_dir / "foo.py")},
        "verdict": "keep",
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "ok"
    assert result["decision"] == "KEEP"
    assert result["kept"] is True
    assert result["requires_e2e_validation"] is True
    assert result["env_flags"] == {"VLLM_FUSE=1": "1"}
    assert result["kernel_repo"] == "/repo/root"
    assert result["artifact_files"] == ["foo.py"]


def test_normalize_manifest_refuses_a_keep_integrate_cannot_apply(tmp_path):
    """Integrate needs a patch and a target file, and returns without them.

    Reported as ok this is lost twice: nothing adopts it, and the status also
    satisfies the KERNEL-entry idempotency gate, so it is never retried either.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "fusion_loop": {
            "kept": True,
            "best": {"kernel_speedup": 1.12},
            "best_env_flag": "VLLM_FUSE=1",
        },
        "artifacts": {"patch": None, "changes": []},
        "fusion": {"source_file": str(output_dir / "foo.py")},
        "verdict": "keep",
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "failed"
    assert result["decision"] == "REVERT"
    assert result["kept"] is False
    assert result["error_class"] == "fusion_artifact_missing"
    # Anything the gate accepts would stop the session from trying again.
    assert result["status"] not in ("ok", "complete", "kept")


@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ("patch_file", "the patch file it named"),
        ("source_file", "a target file"),
        ("repo_root", "a patch root"),
    ],
)
def test_normalize_manifest_checks_each_artifact_it_hands_to_integrate(tmp_path, drop, expected):
    """Verified rather than assumed: the producer is another repository."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    patch = _patch_file(output_dir)
    if drop == "patch_file":
        Path(patch).unlink()
    manifest = {
        "fusion_loop": {"kept": True, "best": {"kernel_speedup": 1.12}},
        "artifacts": {
            "patch": patch,
            "changes": [],
            "repo_root": "" if drop == "repo_root" else "/venv/site-packages",
        },
        "fusion": {"source_file": "" if drop == "source_file" else "/fw/foo.py"},
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["kept"] is False
    assert result["error_class"] == "fusion_artifact_missing"
    assert expected in result["error"]


def _compile_pass_manifest(output_dir, *, kept: bool) -> dict:
    """A claimed framework compile pass: no authoring loop, no validation block."""
    return {
        "schema_version": 2,
        "verdict": "candidate",
        "fusion_loop": None,
        "validation": None,
        "compile_pass": {
            "flag": "VLLM_FUSE_RMSNORM",
            "config_file": "vllm/config.py",
            "baseline_tok_s": 1000.0,
            "enabled_tok_s": 1090.0 if kept else 1005.0,
            "speedup": 1.09 if kept else 1.005,
            "pass_activated": True,
            "validated": kept,
            "kept": kept,
        },
        "artifacts": {
            "patch": _patch_file(output_dir),
            "changes": [{"path": "vllm/config.py"}],
            "repo_root": "/venv/site-packages",
        }
        if kept
        else None,
        "fusion": {"source_file": str(output_dir / "config.py")},
    }


def test_normalize_manifest_keeps_a_claimed_compile_pass(tmp_path):
    """A compile-pass claim reports no fusion_loop, and used to be read as a miss.

    The claim is the cheapest win available -- the framework already shipped the
    kernel, just switched off -- and its patch was being discarded.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps(_compile_pass_manifest(output_dir, kept=True)), encoding="utf-8"
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["kept"] is True
    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["micro_decision"] == "candidate"
    assert result["patch"] == str(output_dir / "fusion.patch")
    assert result["kernel_repo"] == "/venv/site-packages"
    assert result["requires_e2e_validation"] is True
    # The edit lives in the framework source, so there is no runtime flag to set.
    assert result["env_flags"] == {}
    assert result["baseline_env_flags"] == {}
    # The number is a serving ratio; say so rather than let it pass for a
    # microbenchmark one.
    assert result["kernel_speedup"] == 1.09
    assert result["serving_speedup"] == 1.09
    assert result["compile_pass_flag"] == "VLLM_FUSE_RMSNORM"


def test_normalize_manifest_reverts_a_compile_pass_that_did_not_pay(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps(_compile_pass_manifest(output_dir, kept=False)), encoding="utf-8"
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["kept"] is False
    assert result["decision"] == "REVERT"
    assert result["status"] == "complete"
    assert result["micro_decision"] == "no_improvement"
    assert result["patch"] is None
    assert result["requires_e2e_validation"] is False


def test_normalize_manifest_reports_an_llm_outage_as_infrastructure(tmp_path):
    """`llm_unavailable` means the model was never reached, so it is not a verdict.

    The generic no-KEEP shape would call it ``complete``/``no_improvement``, which
    records an outage as an optimization result AND satisfies the KERNEL-entry
    idempotency gate -- one gateway blip would then skip fusion for the whole
    remaining session.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "schema_version": 2,
        "verdict": "llm_unavailable",
        "diagnosis": {"is_candidate": True},
        "fusion": None,
        "fusion_candidates": [],
        "fusion_loop": None,
        "validation": None,
        "artifacts": None,
        "error": {
            "stage": "discovery",
            "class": "llm_unavailable",
            "kind": "api_error",
            "attempts": 4,
            "message": "Error code: 400 - Bad Request",
        },
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=3)

    # Shaped like the timeout result: infrastructure failed, nothing was judged.
    assert result["status"] == "failed"
    assert result["micro_decision"] == "failed"
    assert result["decision"] == "REVERT"
    assert result["kept"] is False
    assert result["requires_e2e_validation"] is False
    assert result["error_class"] == "llm_unavailable"
    assert result["verdict"] == "llm_unavailable"
    assert "api_error" in result["error"]
    assert "4 attempt(s)" in result["error"]
    assert "Error code: 400" in result["error"]


def test_an_llm_outage_leaves_fusion_retryable_at_the_next_kernel_entry(tmp_path):
    """The load-bearing consequence: `status` decides whether fusion runs again.

    ``_fusion_required_before_kernel_opt`` skips fusion once ``last_fusion.status``
    is one of ok/complete/kept, so an outage must NOT report one of those.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps({"schema_version": 2, "verdict": "llm_unavailable", "error": {}}),
        encoding="utf-8",
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=3)

    assert result["status"] not in ("ok", "complete", "kept")


def test_an_llm_outage_verdict_never_discards_a_validated_fusion(tmp_path):
    """A KEEP outranks the outage verdict, however the manifest ends up shaped.

    forge-fusion only overrides the verdict when discovery raised -- and then it has
    no recipes, so no loop and no validation -- but that invariant lives in another
    repository and nothing here can enforce it. Being wrong would throw away a
    measured patch, so the guard is local.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "schema_version": 2,
        "verdict": "llm_unavailable",
        "fusion_loop": {"kept": True, "best": {"kernel_speedup": 1.2}},
        "validation": {"kept": True, "kernel_speedup": 1.2},
        "artifacts": {
            "patch": _patch_file(output_dir),
            "changes": [{"path": "foo.py"}],
            "repo_root": "/venv/site-packages",
        },
        "fusion": {"source_file": str(output_dir / "foo.py")},
        "error": {"kind": "api_error", "attempts": 2, "message": "flaky"},
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=3)

    assert result["kept"] is True
    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["requires_e2e_validation"] is True
    assert result["patch"] == str(output_dir / "fusion.patch")
    assert "error_class" not in result


def test_an_llm_outage_verdict_is_matched_tolerantly(tmp_path):
    """Matching must not fail open: a stray space would fall back to the
    no_improvement mapping, i.e. straight back into the bug this prevents."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps({"schema_version": 2, "verdict": " LLM_Unavailable "}),
        encoding="utf-8",
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=3)

    assert result["error_class"] == "llm_unavailable"
    assert result["status"] == "failed"


def _aborted_manifest(reason, **loop_extra):
    """A manifest for a run that located a recipe, then died before attempting it.

    This is exactly the shape observed on the fleet: discovery succeeded, so
    ``verdict`` is ``candidate`` and ``fusion.env_flag`` names a real flag, while
    ``fusion_loop`` reports zero attempts and no promoted flag.
    """
    loop = {"termination_reason": reason, "attempts": 0, "best": None, "best_env_flag": None}
    loop.update(loop_extra)
    return {
        "schema_version": 2,
        "verdict": "candidate",
        "diagnosis": {"is_candidate": True},
        "fusion": {
            "env_flag": "DEEPSEEK_V4_FUSED_ATTN_REDUCE_INV_ROPE",
            "source_file": "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py",
        },
        "fusion_loop": loop,
        "validation": None,
        "artifacts": None,
    }


def test_normalize_manifest_reports_a_harness_author_abort_as_infrastructure(tmp_path):
    """``harness_author_failed`` means the loop never ran, so it is not a verdict.

    The generic no-KEEP shape would call it ``complete``/``no_improvement``, which
    records an abort as an optimization result AND satisfies the KERNEL-entry
    idempotency gate -- one failed authoring turn would then skip fusion for the
    whole remaining session, even though the recipe had already been located.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps(_aborted_manifest("harness_author_failed")), encoding="utf-8"
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "failed"
    assert result["micro_decision"] == "failed"
    assert result["decision"] == "REVERT"
    assert result["kept"] is False
    assert result["requires_e2e_validation"] is False
    assert result["error_class"] == "harness_author_failed"
    assert "harness_author_failed" in result["error"]
    # The located recipe is carried for the retry, but not as a confirmed flag:
    # nothing measured it, and ``env_flags`` means "flags this run confirmed".
    assert result["located_env_flag"] == "DEEPSEEK_V4_FUSED_ATTN_REDUCE_INV_ROPE"
    assert "DEEPSEEK_V4_FUSED_ATTN_REDUCE_INV_ROPE" in result["error"]
    assert result["env_flags"] == {}
    assert result["baseline_env_flags"] == {}


def test_an_abort_leaves_fusion_retryable_at_the_next_kernel_entry(tmp_path):
    """The load-bearing consequence: ``status`` decides whether fusion runs again.

    ``_fusion_required_before_kernel_opt`` skips fusion once ``last_fusion.status``
    is one of ok/complete/kept, so an abort must NOT report one of those.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps(_aborted_manifest("harness_author_failed")), encoding="utf-8"
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] not in ("ok", "complete", "kept")


def test_a_missing_git_workspace_abort_takes_the_same_path(tmp_path):
    """The handling keys on the termination reason, not on one known failure.

    ``no_git_workspace`` aborts the loop just as early and reaches the same
    normalization, so fixing only the reason that happened to be observed would
    leave an identical defect one code path away.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # attempts absent rather than 0: that abort path builds its LoopResult
    # without ever setting the counter.
    manifest = _aborted_manifest("no_git_workspace")
    del manifest["fusion_loop"]["attempts"]
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "failed"
    assert result["error_class"] == "no_git_workspace"
    assert result["status"] not in ("ok", "complete", "kept")


def test_an_abort_never_discards_a_validated_fusion(tmp_path):
    """A KEEP outranks the abort reason, however the manifest ends up shaped.

    A loop that kept a fusion by definition attempted one, so the two should never
    co-occur -- but that invariant lives in another repository, and being wrong
    would throw away a measured patch, so the guard is local.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = _aborted_manifest(
        "harness_author_failed",
        kept=True,
        attempts=3,
        best={"kernel_speedup": 1.4},
        best_env_flag="DEEPSEEK_V4_FUSED_ATTN_REDUCE_INV_ROPE",
    )
    manifest["artifacts"] = {
        "patch": _patch_file(output_dir),
        "changes": [{"path": "foo.py"}],
        "repo_root": "/venv/site-packages",
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["kept"] is True
    assert result["decision"] == "KEEP"
    assert result["status"] == "ok"
    assert result["requires_e2e_validation"] is True
    assert result["env_flags"] == {"DEEPSEEK_V4_FUSED_ATTN_REDUCE_INV_ROPE": "1"}
    assert "error_class" not in result


def test_a_loop_that_ran_still_reports_no_improvement(tmp_path):
    """Regression guard: only a loop that never attempted is an abort.

    A loop that ran and found nothing worth keeping is a real result and must keep
    reporting ``complete``/``no_improvement`` with its promoted flags, or the fix
    would turn every honest no-improvement into a retry.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = _aborted_manifest(
        "exhausted", attempts=1, best_env_flag="QWEN3_FUSED_QK_NORM_ROPE_KVCACHE"
    )
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "complete"
    assert result["micro_decision"] == "no_improvement"
    assert result["env_flags"] == {"QWEN3_FUSED_QK_NORM_ROPE_KVCACHE": "1"}
    assert "error_class" not in result
    assert "located_env_flag" not in result


def test_an_abort_reason_is_matched_tolerantly(tmp_path):
    """Matching must not fail open: a stray space would fall back to the
    no_improvement mapping, i.e. straight back into the bug this prevents."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps(_aborted_manifest("  harness_author_failed  ")), encoding="utf-8"
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["error_class"] == "harness_author_failed"
    assert result["status"] == "failed"


def test_main_relays_the_outage_sentinel_despite_a_non_zero_exit(tmp_path, monkeypatch, capsys):
    """forge-fusion exits 3 for an unreachable LLM, which is the first non-zero exit
    that still carries a valid manifest.

    The wrapper mirrors the child's exit code, and the handler prefers the sentinel
    over ``rc``; this pins that contract so a later "just trust rc" simplification
    cannot silently degrade the outage into a generic handler failure.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")

    class Proc:
        returncode = 3  # kernelforge.fusion.command.EXIT_LLM_UNAVAILABLE
        stdout = ""
        stderr = ""

    def fake_run(_cmd, _timeout):
        (output_dir / "fusion_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "verdict": "llm_unavailable",
                    "error": {"kind": "api_error", "attempts": 5, "message": "gateway 400 x5"},
                }
            ),
            encoding="utf-8",
        )
        return Proc()

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", fake_run)

    rc = forge_fusion.main(["--input-json", str(input_json)])

    assert rc == 3, "the child's exit code is mirrored, not swallowed"
    result = _sentinel_payload(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["error_class"] == "llm_unavailable"
    assert result["kept"] is False
    # The on-disk fallback has to agree with the sentinel.
    assert json.loads((output_dir / "result.json").read_text(encoding="utf-8")) == result


def test_normalize_manifest_still_reports_a_real_no_opportunity(tmp_path):
    """A run that DID reach the model and found nothing is unchanged: it is a real
    conclusion, and re-running it in the same session would buy nothing."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps({"schema_version": 2, "verdict": "no_opportunity", "error": None}),
        encoding="utf-8",
    )

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "complete"
    assert result["micro_decision"] == "no_improvement"
    assert result["verdict"] == "no_opportunity"
    assert "error_class" not in result


def test_normalize_manifest_prefers_artifacts_repo_root(tmp_path, monkeypatch):
    """kernel_repo must come from the root forge-fusion exported against (authoritative
    for a non-git pip framework), NOT a git toplevel that would break patch apply."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "fusion_loop": {"kept": True, "best": {"kernel_speedup": 1.2}},
        "validation": {"kept": True},
        "artifacts": {
            "patch": _patch_file(output_dir),
            "changes": [{"path": "vllm/x.py"}],
            "repo_root": "/venv/site-packages",
        },
        "fusion": {"source_file": str(output_dir / "x.py")},
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)
    assert result["kernel_repo"] == "/venv/site-packages"


def test_normalize_manifest_missing_file_reports_error(tmp_path):
    output_dir = tmp_path / "missing"
    output_dir.mkdir()

    result = forge_fusion._normalize_manifest(str(output_dir), rc=1)

    assert result["decision"] == "REVERT"
    assert "no fusion_manifest.json" in result["error"]


def test_normalize_manifest_parse_error_reports_error(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text("{not-json", encoding="utf-8")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["decision"] == "REVERT"
    assert "parse error" in result["error"]


def test_main_kept_manifest_emits_keep_result(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = {
        "fusion_loop": {"kept": True, "best": {"kernel_speedup": 1.05}},
        "validation": {},
        "artifacts": {
            "patch": _patch_file(output_dir),
            "changes": [{"path": "foo.py"}],
            "repo_root": "/venv/site-packages",
        },
        "fusion": {"source_file": str(output_dir / "foo.py")},
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(_cmd, _timeout):
        # The run writes its own manifest; main() clears any stale one first.
        _patch_file(output_dir)
        (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return Proc()

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", fake_run)

    rc = forge_fusion.main(["--input-json", str(input_json)])

    out = capsys.readouterr()
    assert rc == 0
    result = _sentinel_payload(out.out)
    assert result["decision"] == "KEEP"
    assert result["kept"] is True
    assert result["requires_e2e_validation"] is True


def test_main_does_not_report_a_previous_runs_manifest(tmp_path, monkeypatch, capsys):
    """The output dir is keyed on the task, so the file outlives the run."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fusion_manifest.json").write_text(
        json.dumps(
            {
                "fusion_loop": {"kept": True, "best": {"kernel_speedup": 1.4}},
                "artifacts": {"patch": _patch_file(output_dir), "changes": []},
                "fusion": {"source_file": "/fw/foo.py"},
            }
        ),
        encoding="utf-8",
    )
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")

    class Proc:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", lambda _cmd, _timeout: Proc())

    forge_fusion.main(["--input-json", str(input_json)])

    result = _sentinel_payload(capsys.readouterr().out)
    assert result["kept"] is False
    assert result["decision"] == "REVERT"
    assert "no fusion_manifest.json" in result["error"]


def test_main_invalid_json_returns_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")

    rc = forge_fusion.main(["--input-json", str(bad)])

    assert rc == 2
    assert "failed" in capsys.readouterr().out


def test_main_missing_required_field_returns_2(tmp_path, capsys):
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps({"model_path": "/m", "framework": "sglang", "output_dir": "/o"}),
        encoding="utf-8",
    )

    rc = forge_fusion.main(["--input-json", str(input_json)])

    assert rc == 2


def test_build_cmd_optional_and_disabled_flags(tmp_path):
    payload = _payload(tmp_path)
    payload.update(
        {
            "decode_batch": 8,
            "ab_isl": 64,
            "ab_osl": 128,
            "framework_root": "/fw",
            "verbose": True,
            "fuse_all_confirmed": False,
        }
    )

    cmd = forge_fusion._build_cmd(payload)

    assert "--decode-batch" in cmd
    assert "--ab-isl" in cmd
    assert "--ab-osl" in cmd
    assert "--framework-root" in cmd
    assert "--verbose" in cmd
    assert "--fuse-all-confirmed" not in cmd


def test_build_cmd_never_disables_authoring_or_validation(tmp_path):
    """Both stages are the point of the run; the orchestrator never opts out."""
    payload = _payload(tmp_path)
    payload.update({"author": False, "validate": False})

    cmd = forge_fusion._build_cmd(payload)

    assert "--no-author" not in cmd
    assert "--no-validate" not in cmd


def test_timeout_sec_prefers_timeout_sec_key():
    assert forge_fusion._timeout_sec({"timeout_sec": 42}) == 42


def test_timeout_sec_accepts_float_strings(monkeypatch):
    monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "42.9")

    assert forge_fusion._timeout_sec({}) == 42


def test_timeout_sec_infinite_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "inf")

    assert forge_fusion._timeout_sec({}) == forge_fusion.DEFAULT_TIMEOUT_SEC


def test_terminate_process_tree_uses_process_group_on_posix(monkeypatch):
    killed: list[tuple[int, int]] = []

    class FakeProc:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(forge_fusion.os, "name", "posix")
    monkeypatch.setattr(
        forge_fusion.os,
        "getpgid",
        lambda pid: 9999 if pid == 1234 else 1111,
        raising=False,
    )
    monkeypatch.setattr(
        forge_fusion.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
        raising=False,
    )
    monkeypatch.setattr(forge_fusion.signal, "SIGTERM", 15, raising=False)

    forge_fusion._terminate_process_tree(FakeProc())

    assert killed == [(9999, 15)]


def test_terminate_process_tree_escalates_to_sigkill_on_posix(monkeypatch):
    killed: list[tuple[int, int]] = []

    class FakeProc:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise forge_fusion.subprocess.TimeoutExpired("cmd", 5)

    monkeypatch.setattr(forge_fusion.os, "name", "posix")
    monkeypatch.setattr(
        forge_fusion.os,
        "getpgid",
        lambda pid: 9999 if pid == 1234 else 1111,
        raising=False,
    )
    monkeypatch.setattr(
        forge_fusion.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
        raising=False,
    )
    monkeypatch.setattr(forge_fusion.signal, "SIGTERM", 15, raising=False)
    monkeypatch.setattr(forge_fusion.signal, "SIGKILL", 9, raising=False)

    forge_fusion._terminate_process_tree(FakeProc())

    assert killed == [(9999, 15), (9999, 9)]


def test_terminate_process_tree_noop_when_already_exited():
    class FakeProc:
        def poll(self):
            return 0

    forge_fusion._terminate_process_tree(FakeProc())


def test_terminate_process_tree_falls_back_when_same_pgid(monkeypatch):
    terminated: list[bool] = []

    class FakeProc:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            terminated.append(True)

        def kill(self):
            pass

    monkeypatch.setattr(forge_fusion.os, "name", "posix")
    monkeypatch.setattr(forge_fusion.os, "getpgid", lambda _pid: 42, raising=False)

    forge_fusion._terminate_process_tree(FakeProc())

    assert terminated == [True]


def test_emit_swallows_write_errors(tmp_path, monkeypatch, capsys):
    def _raise_oserror(*_args, **_kwargs):
        raise OSError("denied")

    monkeypatch.setattr(forge_fusion.Path, "write_text", _raise_oserror)

    forge_fusion._emit({"status": "ok"}, str(tmp_path))

    assert forge_fusion.RESULT_BEGIN in capsys.readouterr().out


def test_new_session_kwargs_empty_on_windows(monkeypatch):
    monkeypatch.setattr(forge_fusion.os, "name", "nt")
    assert forge_fusion._new_session_kwargs() == {}


def test_load_input_json_empty_path_returns_empty_dict():
    assert forge_fusion._load_input_json("") == {}


def test_emit_without_output_dir_only_prints(capsys):
    forge_fusion._emit({"status": "ok"}, "")
    assert forge_fusion.RESULT_BEGIN in capsys.readouterr().out


def test_terminate_process_tree_handles_getpgid_oserror(monkeypatch):
    terminated: list[bool] = []

    class FakeProc:
        pid = 1

        def poll(self):
            return None

        def terminate(self):
            terminated.append(True)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(forge_fusion.os, "name", "posix")
    monkeypatch.setattr(
        forge_fusion.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(OSError("nope")),
        raising=False,
    )

    forge_fusion._terminate_process_tree(FakeProc())
    assert terminated == [True]


def test_as_text_decodes_bytes():
    assert forge_fusion._as_text(b"abc") == "abc"


def test_relay_streams_writes_stdout_and_stderr(capsys):
    forge_fusion._relay_streams("hello", "err")
    captured = capsys.readouterr()
    assert captured.out == "hello"
    assert captured.err == "err"


def test_terminate_process_tree_windows_uses_terminate(monkeypatch):
    terminated: list[bool] = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            terminated.append(True)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(forge_fusion.os, "name", "nt")
    forge_fusion._terminate_process_tree(FakeProc())
    assert terminated == [True]


def test_run_with_tree_timeout_clears_output_when_reap_times_out(monkeypatch):
    class FakeProc:
        def communicate(self, timeout=None):
            if timeout == 1.0:
                raise forge_fusion.subprocess.TimeoutExpired("cmd", 1)
            raise forge_fusion.subprocess.TimeoutExpired("cmd", 30)

    monkeypatch.setattr(forge_fusion.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(forge_fusion, "_terminate_process_tree", lambda _proc: None)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        forge_fusion._run_with_tree_timeout(["echo"], timeout_sec=30)

    assert excinfo.value.output == ""
