# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An unchanged user script benchmarked at its optimization-module boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from hyperloom.inference_optimizer.script_workload import main as script_main

torch = pytest.importorskip("torch")

SCRIPT = textwrap.dedent(
    """
    from toy_opt import apply
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512)).eval()
    model = apply(model)
    with torch.no_grad():
        for _ in range(3):
            print(float(model(torch.randn(256, 512)).norm()))
    """
)

CANDIDATES = {
    "too_lossy": """
        def apply(model):
            return lambda x: model(x) * 1.1
    """,
    "clock": """
        import time
        _real = time.perf_counter
        time.perf_counter = lambda: _real() / 2
        def apply(model):
            return model
    """,
    "reused_buffer": """
        import torch
        def apply(model):
            buffer = torch.empty(256, 512)
            return lambda x: buffer.copy_(model(x))
    """,
    "known_inputs_only": """
        import torch
        _known = [c["args"][0] for c in torch.load({calls!r}, weights_only=False)]
        def apply(model):
            def run(x):
                if any(torch.equal(x, k) for k in _known):
                    return model(x)
                return torch.zeros(x.shape[0], 512)
            return run
    """,
}


def _prepare(user: Path, workspace: Path, script: str) -> int:
    (user / "toy.py").write_text(script)
    (user / "toy_opt.py").write_text("def apply(model):\n    return model\n")
    argv = ["prepare", "--workspace", str(workspace), "--script", str(user / "toy.py"), "--cwd", str(user)]
    argv += ["--python", sys.executable, "--rel-l2", "1e-5", "--held-out", "2", "--timed-passes", "8"]
    argv += ["--warmup-s", "0"]
    return script_main(argv)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("campaign") / "ws"
    assert _prepare(tmp_path_factory.mktemp("user"), workspace, SCRIPT) == 0
    return workspace


def test_prepare_refuses_an_import_after_the_compiler_loaded(tmp_path):
    late = SCRIPT.replace(
        "from toy_opt import apply\nimport torch\n", "import torch._dynamo\nfrom toy_opt import apply\n"
    )
    with pytest.raises(SystemExit, match=r"toy\.py:3, after .*torch\._dynamo.*above `import torch`.*toy\.py:2"):
        _prepare(tmp_path, tmp_path / "ws", late)
    assert not (tmp_path / "ws" / "optimization").exists()
    assert not (tmp_path / "ws" / "benchmark").exists()


def test_prepare_warns_on_an_import_after_torch(tmp_path, capsys):
    after_torch = SCRIPT.replace(
        "from toy_opt import apply\nimport torch\n", "import torch\nfrom toy_opt import apply\n"
    )
    assert _prepare(tmp_path, tmp_path / "ws", after_torch) == 0
    assert "after torch (torch is first imported at" in capsys.readouterr().err


def _measure(workspace: Path, tmp_path: Path, candidate: str | None = None) -> dict:
    module = workspace / "optimization" / "toy_opt.py"
    original = module.read_text()
    if candidate is not None:
        module.write_text(candidate)
    try:
        env = {**os.environ, "RESULT_DIR": str(tmp_path)}
        subprocess.run(["bash", str(workspace / "benchmark" / "custom_gfx11.sh")], env=env, check=True)
    finally:
        module.write_text(original)
    return json.loads((tmp_path / "inferencex_result.json").read_text())


def _failed(result: dict) -> set[str]:
    return {case["case"] for case in result["quality_gate"]["cases"] if not case["passed"]}


def test_prepare_lays_out_the_scope_and_records_once(workspace):
    scope = json.loads((workspace / "benchmark" / "optimization_scope.json").read_text())
    assert scope == {"schema_version": 2, "file": str(workspace / "optimization" / "toy_opt.py"), "function": "apply"}
    store = workspace / "benchmark" / "reference"
    assert sorted(p.name for p in store.iterdir()) == ["import_order.json", "inputs", "references"]
    order = json.loads((store / "import_order.json").read_text())
    assert (order["torch_imported"], order["gpu_initialized"], order["compiler_imported"]) == (False, False, [])
    limits = json.loads((workspace / "benchmark" / "reference" / "references" / "limits.json").read_text())
    assert limits == [1e-5] * 3
    with pytest.raises(SystemExit, match="already prepared"):
        script_main(["prepare", "--workspace", str(workspace), "--script", "toy.py", "--rel-l2", "1"])


def test_identity_passes(workspace, tmp_path):
    result = _measure(workspace, tmp_path)
    assert result["quality_gate"]["passed"], result["quality_gate"]
    assert result["output_throughput"] > 0
    assert result["bench_summary"]["calls_per_invocation"] == 3


@pytest.mark.parametrize(
    ("name", "caught_by"),
    [
        ("too_lossy", {"recorded[0]", "held_out[0]"}),
        ("clock", {"timing_consistent"}),
        ("reused_buffer", {"held_output"}),
        ("known_inputs_only", {"held_out[0]"}),
    ],
)
def test_gate_rejects(workspace, tmp_path, name, caught_by):
    calls = str(workspace / "benchmark" / "reference" / "inputs" / "calls.pt")
    result = _measure(workspace, tmp_path, textwrap.dedent(CANDIDATES[name].format(calls=calls)))
    assert not result["quality_gate"]["passed"]
    assert caught_by <= _failed(result)


def test_measure_refuses_an_edited_script(workspace, tmp_path):
    config = json.loads((workspace / "benchmark" / "script_workload.json").read_text())
    script = Path(config["script"])
    original = script.read_text()
    script.write_text(original + "\n# edited\n")
    try:
        with pytest.raises(subprocess.CalledProcessError):
            _measure(workspace, tmp_path)
    finally:
        script.write_text(original)
