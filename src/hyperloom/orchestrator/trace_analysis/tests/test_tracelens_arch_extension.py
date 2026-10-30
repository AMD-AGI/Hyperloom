"""The arch JSON served through ``TL_EXTENSION`` reaches TraceLens in this process and in child processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import tracelens_analysis as tla

#: A platform TraceLens (cf0221e3) does not bundle, so only the served extension can resolve it.
_PLATFORM = "MI355X"
_PROBE = (
    f"from TraceLens.Agent.Analysis.utils.arch_utils import load_arch;print(load_arch({_PLATFORM!r})['probe_peak'])"
)


def test_child_process_resolves_served_arch_and_env_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arch_utils = pytest.importorskip("TraceLens.Agent.Analysis.utils.arch_utils")
    monkeypatch.delenv("TL_EXTENSION", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/nonexistent-sentinel")
    assert _PLATFORM not in arch_utils.list_platforms()
    arch_json = tmp_path / "spec.json"
    arch_json.write_text(json.dumps({"probe_peak": 4242}), encoding="utf-8")

    with tla._tracelens_knows_arch(arch_json, _PLATFORM, tmp_path / "out"):
        assert arch_utils.load_arch(_PLATFORM)["probe_peak"] == 4242
        child = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False)

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "4242"
    assert os.environ["PYTHONPATH"] == "/nonexistent-sentinel"
    assert "TL_EXTENSION" not in os.environ
    assert _PLATFORM not in arch_utils.list_platforms()
