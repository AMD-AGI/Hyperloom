# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLI contract for advisory profiler hints handed to llm discovery."""

from __future__ import annotations

import json

from kernelforge.workflows.forge_fusion import discovery_stage

from . import test_anchored_cli as anchored_cli

_HINT = {
    "kind": "add_norm",
    "operation": "Unfused Add + RMSNorm in `DecoderLayer`",
    "projected_e2e_pct": 2.4,
    "members": [{"kernel_names": ["rms_norm_kernel"], "kernel_launcher_path": "toylm.py(12): forward"}],
}


def _hints_file(tmp_path, payload):
    path = tmp_path / "fusion_hints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_llm_discovery_prompt_carries_the_hints(tmp_path, monkeypatch):
    hz = anchored_cli.Harness(tmp_path)
    captured = anchored_cli.TestDiscovery()._stub_agent(monkeypatch, hz, "[]")

    res = hz.run("--discover", "llm", "--fusion-hints", _hints_file(tmp_path, [_HINT]), "--no-author", "--no-validate")

    assert "prompt" in captured, res.output
    assert "## External profiler hints (advisory)" in captured["prompt"]
    assert "Unfused Add + RMSNorm in `DecoderLayer`" in captured["prompt"]


def test_hints_never_reach_a_non_llm_discovery(tmp_path):
    hz = anchored_cli.Harness(tmp_path)

    res = hz.run("--dry-run", "--fusion-hints", _hints_file(tmp_path, [_HINT]))

    assert res.exit_code != 0
    assert "--fusion-hints requires --discover llm" in res.output


def test_a_malformed_hints_file_is_a_usage_error(tmp_path, monkeypatch):
    hz = anchored_cli.Harness(tmp_path)
    monkeypatch.setattr(discovery_stage, "discover_recipes", lambda *a, **k: [])

    res = hz.run("--dry-run", "--discover", "llm", "--fusion-hints", _hints_file(tmp_path, {"kind": "add_norm"}))

    assert res.exit_code != 0
    assert "must hold a JSON list of objects" in res.output
