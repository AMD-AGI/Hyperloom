#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the forge-fusion kernel-agent wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "forge_fusion.py"
_SPEC = importlib.util.spec_from_file_location("forge_fusion_tool", _MODULE_PATH)
assert _SPEC and _SPEC.loader
forge_fusion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(forge_fusion)


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Restore ``os.environ`` after every test.

    ``_inject_author_gateway_env`` (exercised directly and via ``main``) mutates
    ``os.environ`` in place by design. ``monkeypatch`` does not revert keys the
    function writes directly, so without this snapshot the leaked ``ANTHROPIC_*``
    / stability vars pollute later auth/endpoint tests in a full-suite run.
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
        "llm_model": "claude-opus-4-6",
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

    assert cmd[:4] == [forge_fusion.sys.executable, "-m", "forge_fusion.cli", "run"]
    assert cmd[cmd.index("--trace") + 1] == "/tmp/decode.trace.json.gz"
    assert cmd[cmd.index("--model-path") + 1] == "/models/zaya"
    assert cmd[cmd.index("--framework") + 1] == "sglang"
    assert cmd[cmd.index("--output-dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    assert "--fuse-all-confirmed" in cmd


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
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/api/v1/llm-proxy/v1")
    monkeypatch.setenv("SAFE_API_KEY", "safe-token")

    forge_fusion._inject_author_gateway_env()

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example/api/v1/llm-proxy"
    assert os.environ["ANTHROPIC_API_KEY"] == "safe-token"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "safe-token"
    assert os.environ["IS_SANDBOX"] == "1"
    assert os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert os.environ["DISABLE_AUTOUPDATER"] == "1"
    assert "API_TIMEOUT_MS" not in os.environ


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


def test_normalize_manifest_kept_writes_keep_result(tmp_path, monkeypatch):
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
            "patch": "diff --git a/foo.py",
            "changes": [{"path": "foo.py"}],
        },
        "fusion": {"source_file": str(output_dir / "foo.py")},
        "verdict": "keep",
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(forge_fusion, "_git_toplevel", lambda _path: "/repo/root")

    result = forge_fusion._normalize_manifest(str(output_dir), rc=0)

    assert result["status"] == "ok"
    assert result["decision"] == "KEEP"
    assert result["kept"] is True
    assert result["requires_e2e_validation"] is True
    assert result["env_flags"] == {"VLLM_FUSE=1": "1"}
    assert result["kernel_repo"] == "/repo/root"
    assert result["artifact_files"] == ["foo.py"]


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
        "artifacts": {},
    }
    (output_dir / "fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(_payload(output_dir)), encoding="utf-8")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(forge_fusion, "_run_with_tree_timeout", lambda _cmd, _timeout: Proc())

    rc = forge_fusion.main(["--input-json", str(input_json)])

    out = capsys.readouterr()
    assert rc == 0
    result = _sentinel_payload(out.out)
    assert result["decision"] == "KEEP"
    assert result["kept"] is True
    assert result["requires_e2e_validation"] is True
