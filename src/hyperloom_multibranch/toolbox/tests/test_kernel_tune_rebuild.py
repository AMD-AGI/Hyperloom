# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernel_tune`` warns when its rows make a prebuilt aiter module stale, so the specialist knows a rebuild comes.

The aiter install is the one ``test_table_modules`` describes (aiter's own module entries): rows for the blockscale
bpreshuffle table make ``module_gemm_a8w8_blockscale_bpreshuffle`` and ``module_deepgemm_opus`` stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hyperloom.measure.framework.tests.test_table_modules import BPRESHUFFLE, CONFIG, CORE, TABLES
from hyperloom_multibranch.toolbox.tools.kernel_tune import _output
from hyperloom_multibranch.toolbox.tools.tune_rebuild import rebuild_warning
from hyperloom_multibranch.toolbox.tools.tune_rows import TunedRows
from hyperloom_multibranch.toolbox.tools.tune_tables import TableStatus, TuneTable

_UTF8 = "utf-8"
_ROW = {"M": 9, "N": 9, "K": 9, "kernelName": "bp_new", "verified": True}


def _configs(root: Path, *, module_config: bool = True) -> Path:
    """An aiter install's ``aiter/configs``, with its module config unless told otherwise."""
    package = root / "aiter"
    for relative, text in {"jit/core.py": CORE, **TABLES}.items():
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_text(text, encoding=_UTF8)
    if module_config:
        (package / "jit" / "optCompilerConfig.json").write_text(json.dumps(CONFIG), encoding=_UTF8)
    return package / "configs"


def _table(tmp_path: Path, *, verified: bool = True) -> TuneTable:
    rows_file = tmp_path / "rows_only.csv"
    rows_file.write_text("M,N,K,kernelName\n9,9,9,bp_new\n", encoding=_UTF8)
    rows = TunedRows("M,N,K,kernelName\n9,9,9,bp_new\n", (_ROW,) if verified else (), (), "glm")
    return TuneTable(BPRESHUFFLE, "a8w8_blockscale_bpreshuffle", TableStatus.TUNED, "", rows, rows_file)


def test_rows_for_a_generated_module_warn_of_its_rebuild(tmp_path: Path) -> None:
    warning = rebuild_warning([_table(tmp_path)], _configs(tmp_path))

    assert warning is not None
    assert "module_deepgemm_opus, module_gemm_a8w8_blockscale_bpreshuffle" in warning
    assert "rebuilds those modules at its server's first prefill" in warning


def test_no_verified_rows_or_no_module_config_claims_nothing(tmp_path: Path) -> None:
    assert rebuild_warning([_table(tmp_path, verified=False)], _configs(tmp_path / "a")) is None
    assert rebuild_warning([_table(tmp_path)], _configs(tmp_path / "b", module_config=False)) is None


def test_the_warning_is_in_the_reply_and_its_summary(tmp_path: Path) -> None:
    workload = SimpleNamespace(model_path="/models/glm", framework="sglang", precision="fp8", tp=4, conc=8)
    plan = SimpleNamespace(workload=workload, source="shapes", aiter_root=tmp_path, notes={}, max_err_ratio=0.01)
    lease = SimpleNamespace(gpu_ids=(6,))

    reply = _output(plan, lease, [_table(tmp_path)], "t1", "a rebuild comes")

    assert reply.result["rebuild_warning"] == "a rebuild comes"
    assert reply.summary.endswith("WARNING: a rebuild comes")
    assert "rebuild_warning" not in _output(plan, lease, [_table(tmp_path)], "t1").result
