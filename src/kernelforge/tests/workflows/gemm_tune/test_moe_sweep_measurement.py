# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MoE sweep script may only report a speedup over a baseline it timed, and writes a file vLLM will load."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from kernelforge.workflows.gemm_tune.tuners.vllm_moe_triton import SweepGrid, _generate_sweep_script

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

# The kwarg API: each call names its config.
_FAKE_FUSED_MOE_KWARG = """
import os
import time


def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, override_config=None):
    if override_config is None:
        if os.environ.get("FAIL_BASELINE"):
            raise RuntimeError("no kernel for this shape")
        time.sleep(0.002)
    elif override_config["BLOCK_SIZE_K"] == int(os.environ.get("SLOW_BLOCK_K", "0")):
        time.sleep(0.005)
    return hidden_states
"""

# The context-manager API, with vLLM's own override_config: it restores the previous config only on a clean exit.
_FAKE_FUSED_MOE_CONTEXT = """
from contextlib import contextmanager

_config = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids):
    if _config is not None and _config["BLOCK_SIZE_K"] == 256:
        raise RuntimeError("OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536")
    return hidden_states
"""

_FAKE_FUSED_MOE_MODULE = """
def get_config_file_name(E, N, dtype, block_shape=None):
    selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name=AMD_Radeon_Pro_W7900D{selector}.json"


def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
    return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "default_for": M, "N": w2_shape[2]}
"""

_CANDIDATE = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _stub_runtime(root: Path, fused_moe_package: str) -> Path:
    package = root / "vllm" / "model_executor" / "layers" / "fused_moe"
    package.mkdir(parents=True)
    (root / "torch.py").write_text(_FAKE_TORCH, encoding="utf-8")
    for part in ("vllm", "vllm/model_executor", "vllm/model_executor/layers"):
        (root / part / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text(fused_moe_package, encoding="utf-8")
    (package / "fused_moe.py").write_text(_FAKE_FUSED_MOE_MODULE, encoding="utf-8")
    return root


def _sweep(
    tmp_path: Path,
    *,
    batch_sizes: list[int],
    configs: list[dict],
    fused_moe_package: str = _FAKE_FUSED_MOE_KWARG,
    tp: int = 1,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, dict, dict, dict]:
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)
    script = _generate_sweep_script(
        work_dir,
        _PROFILE,
        gpu_id="0",
        grid=SweepGrid(batch_sizes=batch_sizes, iters=2, warmup=1, configs=configs),
        tp=tp,
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(_stub_runtime(tmp_path / "stubs", fused_moe_package)),
        **(env_extra or {}),
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    kept = json.loads((work_dir / "sweep_results.json").read_text(encoding="utf-8"))
    table = json.loads((work_dir / "vllm_config_table.json").read_text(encoding="utf-8"))
    return proc.returncode, summary, kept, table


def test_a_config_is_kept_when_the_baseline_was_timed(tmp_path):
    _rc, summary, kept, _table = _sweep(tmp_path, batch_sizes=[16], configs=[_CANDIDATE])

    assert summary["status"] == "ok"
    assert [d["M"] for d in summary["shape_details"]] == [16]
    assert summary["shape_details"][0]["baseline_us"] > 0
    assert kept == {"16": _CANDIDATE}


def test_a_shape_whose_baseline_never_ran_is_not_reported_as_a_win(tmp_path):
    # The default config failing used to stand in as an infinite baseline time, so every candidate came out
    # infinitely faster and was written to sweep_results.json as a micro KEEP.
    rc, summary, kept, table = _sweep(
        tmp_path, batch_sizes=[16], configs=[_CANDIDATE], env_extra={"FAIL_BASELINE": "1"}
    )

    assert kept == {}
    assert table == {}
    assert summary["shape_details"] == []
    # No shape was timed, so the sweep reports no speedup rather than a 1.00x that reads as a measurement.
    assert summary["best_speedup"] is None
    assert any("did not benchmark" in e for e in summary["errors"])
    assert rc == 1


def test_a_candidate_the_device_cannot_run_does_not_become_the_next_baseline(tmp_path):
    # vLLM's override_config leaves a config installed when the call inside it raises; a candidate needing more LDS
    # than a gfx1100 workgroup has then ran as every later "default", and each following shape's baseline failed.
    too_big = {**_CANDIDATE, "BLOCK_SIZE_K": 256}

    _rc, summary, _kept, table = _sweep(
        tmp_path,
        batch_sizes=[4, 8, 16],
        configs=[too_big, _CANDIDATE],
        fused_moe_package=_FAKE_FUSED_MOE_CONTEXT,
    )

    assert not any("did not benchmark" in e for e in summary.get("errors", []))
    assert [d["M"] for d in summary["shape_details"]] == [4, 8, 16]
    assert sorted(table, key=int) == ["4", "8", "16"]


def test_every_timed_batch_size_is_in_the_file_with_its_winner_or_its_default(tmp_path):
    # vLLM serves each batch size from the nearest key, so a file holding only M=4's winner ran it at M=512 as well.
    fast, slow = _CANDIDATE, {**_CANDIDATE, "BLOCK_SIZE_K": 32}

    _rc, _summary, kept, table = _sweep(
        tmp_path, batch_sizes=[4, 512], configs=[slow], env_extra={"SLOW_BLOCK_K": "32"}
    )
    assert kept == {}
    assert table["4"]["default_for"] == 4 and table["512"]["default_for"] == 512

    _rc, _summary, kept, table = _sweep(tmp_path / "again", batch_sizes=[4, 512], configs=[fast])
    assert kept == {"4": fast, "512": fast}
    assert table == {"4": fast, "512": fast}


def test_the_sweep_times_and_names_the_per_rank_shard_of_the_expert_dimension(tmp_path):
    _rc, summary, _kept, table = _sweep(tmp_path, batch_sizes=[16], configs=[], tp=4)

    per_rank = _PROFILE.effective_moe_intermediate // 4
    assert table["16"]["N"] == per_rank
    assert summary["config_file_name"] == f"E=8,N={per_rank},device_name=AMD_Radeon_Pro_W7900D.json"
