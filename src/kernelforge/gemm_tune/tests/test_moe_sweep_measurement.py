# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MoE sweep script may only report a speedup over a baseline it timed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from kernelforge.gemm_tune.tuners.vllm_moe_triton import _generate_sweep_script

_PROFILE = SimpleNamespace(
    num_experts=8,
    effective_moe_intermediate=1024,
    hidden_size=2048,
    num_experts_per_tok=2,
)

# The sweep runs on a real GPU against a real vLLM; both are stubbed here so the script's own accept/reject logic is
# what the test exercises.
_FAKE_TORCH = """
class _T:
    def sum(self, *a, **k):
        return self

    def __truediv__(self, other):
        return self


bfloat16 = "bfloat16"
float32 = "float32"


def randn(*a, **k):
    return _T()


def softmax(t, **k):
    return t


def topk(t, k, **kw):
    return _T(), _T()


class cuda:
    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def empty_cache():
        pass
"""

_FAKE_FUSED_MOE = """
import os


def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, override_config=None):
    if override_config is None and os.environ.get("FAIL_BASELINE"):
        raise RuntimeError("no kernel for this shape")
    return hidden_states
"""


def _stub_runtime(root: Path) -> Path:
    package = root / "vllm" / "model_executor" / "layers" / "fused_moe"
    package.mkdir(parents=True)
    (root / "torch.py").write_text(_FAKE_TORCH, encoding="utf-8")
    for part in ("vllm", "vllm/model_executor", "vllm/model_executor/layers"):
        (root / part / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text(_FAKE_FUSED_MOE, encoding="utf-8")
    return root


def _sweep(tmp_path: Path, *, fail_baseline: bool) -> tuple[int, dict, dict]:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    script = _generate_sweep_script(
        work_dir,
        _PROFILE,
        batch_sizes=[16],
        iters=1,
        warmup=1,
        gpu_id="0",
        configs=[
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 2,
            }
        ],
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(_stub_runtime(tmp_path / "stubs")),
    }
    if fail_baseline:
        env["FAIL_BASELINE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    kept = json.loads((work_dir / "sweep_results.json").read_text(encoding="utf-8"))
    return proc.returncode, summary, kept


def test_a_config_is_kept_when_the_baseline_was_timed(tmp_path):
    _rc, summary, _kept = _sweep(tmp_path, fail_baseline=False)

    assert summary["status"] == "ok"
    assert [d["M"] for d in summary["shape_details"]] == [16]
    assert summary["shape_details"][0]["baseline_us"] > 0


def test_a_shape_whose_baseline_never_ran_is_not_reported_as_a_win(tmp_path):
    # The default config failing used to stand in as an infinite baseline time, so every candidate came out
    # infinitely faster and was written to sweep_results.json as a micro KEEP.
    rc, summary, kept = _sweep(tmp_path, fail_baseline=True)

    assert kept == {}
    assert summary["shape_details"] == []
    # No shape was timed, so the sweep reports no speedup rather than a 1.00x that reads as a measurement.
    assert summary["best_speedup"] is None
    assert any("did not benchmark" in e for e in summary["errors"])
    assert rc == 1
