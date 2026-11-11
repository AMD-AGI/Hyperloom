# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The matmul_ogs startup hook applies only a measured matching flag row."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel.ogs_delivery import verify_ogs_apply
from kernelforge.workflows.gemm_tune.tuners import ogs_runtime
from kernelforge.workflows.gemm_tune.tuners.vllm_mxfp4_matmul_ogs import _write_hook


@dataclass
class _Flags:
    block_m: int = 32
    block_n: int = 64
    block_k: int = 128
    num_warps: int = 8
    num_stages: int = 2
    split_k: int = 1
    xcd_swizzle: int = 8
    is_persistent: bool = False


_TUNED = {
    "block_m": 16,
    "block_n": 64,
    "block_k": 256,
    "num_warps": 4,
    "num_stages": 1,
    "split_k": 1,
    "xcd_swizzle": 0,
}


def _table(source: Path) -> dict:
    return {
        "schema_version": 1,
        "gpu_target": "gfx1201",
        "gpu_name": "AMD Radeon AI PRO R9700",
        "opt_flags_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "experts": 32,
        "top_k": 4,
        "rows": [{"n": 128, "k": 256, "min_m": 1, "max_m": 4, "flags": _TUNED}],
    }


def _call(lhs: str = "bf16") -> tuple:
    routing = SimpleNamespace(expt_hist=object(), n_expts_tot=32, n_expts_act=4, expected_tokens_per_expt=4)
    return (
        "bf16",
        lhs,
        "fp4",
        SimpleNamespace(weight_scale=object()),
        1,
        4,
        128,
        256,
        routing,
        False,
        False,
        1,
        False,
        False,
        None,
    )


def test_matching_bf16_mxfp4_row_applies_and_emits_one_receipt(tmp_path, capsys):
    source = tmp_path / "opt_flags.py"
    source.write_text("# fake source")
    module = ModuleType("opt_flags")
    module.__file__ = str(source)
    module.torch = SimpleNamespace(
        bfloat16="bf16", cuda=SimpleNamespace(get_device_name=lambda: "AMD Radeon AI PRO R9700")
    )
    module.FP4 = "fp4"
    module.get_rdna_version = lambda: 4
    module._opt_flags_constraints = {}
    module.make_opt_flags = lambda *_args: _Flags()

    ogs_runtime._install_module(module, _table(source), "a" * 64)

    assert module.make_opt_flags(*_call()).block_m == 16
    assert module.make_opt_flags(*_call()).xcd_swizzle == 0
    assert module.make_opt_flags(*_call("fp16")).block_m == 32
    assert capsys.readouterr().out.count("HL_MATMUL_OGS_FLAGS_APPLIED") == 1


def test_source_drift_refuses_the_tuned_hook(tmp_path):
    source = tmp_path / "opt_flags.py"
    source.write_text("# changed source")
    module = ModuleType("opt_flags")
    module.__file__ = str(source)
    module.make_opt_flags = lambda *_args: _Flags()
    table = _table(source)
    table["opt_flags_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="differs from the tuned source"):
        ogs_runtime._install_module(module, table, "a" * 64)


def test_matching_shapes_on_another_board_keep_stock_flags(tmp_path):
    source = tmp_path / "opt_flags.py"
    source.write_text("# fake source")
    module = ModuleType("opt_flags")
    module.__file__ = str(source)
    module.torch = SimpleNamespace(
        bfloat16="bf16", cuda=SimpleNamespace(get_device_name=lambda: "AMD Radeon AI PRO R9600D")
    )
    module.FP4 = "fp4"
    module.get_rdna_version = lambda: 4
    module._opt_flags_constraints = {}
    module.make_opt_flags = lambda *_args: _Flags()

    ogs_runtime._install_module(module, _table(source), "a" * 64)

    assert module.make_opt_flags(*_call()).block_m == 32


def test_table_refuses_unsupported_or_overlapping_flags(tmp_path):
    source = tmp_path / "opt_flags.py"
    source.write_text("# source")
    table = _table(source)
    path = tmp_path / "flags.json"
    table["rows"][0]["flags"]["split_k"] = 2
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="unsupported values"):
        ogs_runtime._read_table(path)
    table["rows"][0]["flags"]["split_k"] = 1
    table["rows"].append(dict(table["rows"][0]))
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="overlapping"):
        ogs_runtime._read_table(path)


def test_serving_receipt_requires_the_current_table_and_latest_server(tmp_path):
    table = tmp_path / "flags.json"
    table.write_text("{}")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    old_log = tmp_path / "old.log"
    old_log.write_text(f"HL_MATMUL_OGS_FLAGS_APPLIED sha256={digest} n=128 k=256 m=1-4\n")
    latest = tmp_path / "latest.log"
    latest.write_text("stock path\n")

    assert verify_ogs_apply([old_log, latest], str(table))["blocks_keep"] is True
    latest.write_text(old_log.read_text())
    assert verify_ogs_apply([old_log, latest], str(table))["verdict"] == "applied"


def test_sitecustomize_installs_the_hook_before_delayed_opt_flags_import(tmp_path):
    package = tmp_path / "fake" / "triton_kernels" / "matmul_ogs_details"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    source = package / "opt_flags.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "from types import SimpleNamespace\n"
        "torch = SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(get_device_name=lambda: 'AMD Radeon AI PRO R9700'))\n"
        "FP4 = 'fp4'\n"
        "_opt_flags_constraints = {}\n"
        "def get_rdna_version(): return 4\n"
        "@dataclass\n"
        "class Flags:\n"
        "    block_m: int = 32\n"
        "    block_n: int = 64\n"
        "    block_k: int = 128\n"
        "    num_warps: int = 8\n"
        "    num_stages: int = 2\n"
        "    split_k: int = 1\n"
        "    xcd_swizzle: int = 8\n"
        "    is_persistent: bool = False\n"
        "def make_opt_flags(*args): return Flags()\n"
    )
    table = tmp_path / "flags.json"
    table.write_text(json.dumps(_table(source)))
    hook = tmp_path / "hook"
    _write_hook(hook)
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "sitecustomize.py").write_text("import os; os.environ['PRIOR_HOOK'] = 'ran'\n")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(hook), str(prior), str(tmp_path / "fake"))),
        ogs_runtime.FLAGS_ENV: str(table),
    }
    script = (
        "from types import SimpleNamespace\n"
        "from triton_kernels.matmul_ogs_details import opt_flags\n"
        "routing = SimpleNamespace(expt_hist=1, n_expts_tot=32, n_expts_act=4, expected_tokens_per_expt=4)\n"
        "args = ('bf16', 'bf16', 'fp4', SimpleNamespace(weight_scale=1), 1, 4, 128, 256, "
        "routing, False, False, 1, False, False, None)\n"
        "print('BLOCK', opt_flags.make_opt_flags(*args).block_m)\n"
        "import os; print('PRIOR', os.environ.get('PRIOR_HOOK'))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "HL_MATMUL_OGS_FLAGS_APPLIED" in result.stdout
    assert "BLOCK 16" in result.stdout
    assert "PRIOR ran" in result.stdout
