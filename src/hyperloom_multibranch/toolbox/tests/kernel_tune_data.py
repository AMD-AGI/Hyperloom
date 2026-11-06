# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ``kernel_tune`` tests read and the tool's readers are tested on: a server log, its launch line and hit log,
the tuned rows and profile lines the tuner writes, and the landed output of a test that carries them.

The server log lines are synthetic, in the words SGLang and aiter print, at the GEMM shapes of a generic fp8
blockscale decoder: lines that say a dense blockscale GEMM missed its tuned table, a per-token one that Forge's reader
does not take, and the lines between them. The tuned rows are rows of the image's own ``a8w8_blockscale_bpreshuffle``
table (``kernel_rows``). ``SERVER_ARGS`` is a launch line in SGLang's format, cut to the arguments the readers use (the
decode graph batch sizes among them), and ``HIT_LOG`` the lines aiter logs for lookups that resolve in a tuned row.
These need nothing above the tool itself; :mod:`.kernel_tune_rig` builds the local toolbox and the Forge stand-in on
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from hyperloom_multibranch.toolbox.tests.kernel_rows import ENV, HEADER, SHARED

TUNER = "a8w8_blockscale_bpreshuffle"
TABLE = "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
TEST_ID = "t1"
#: A stretch of a server log: a per-token a8w8 table that missed (Forge's reader takes no such line),
#: and the four (N, K) of the dense blockscale table at M=496, with aiter's import lines and a progress bar between.
SERVER_LOG = """\
[aiter] shape is M:512, N:7168, K:5120, q_dtype_w:torch.float8_e4m3fn, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] import [module_gemm_a8w8_bpreshuffle] under /sgl-workspace/aiter/aiter/jit/module_gemm_a8w8_bpreshuffle.so
[aiter] shape is M:512, N:34816, K:5120, q_dtype_w:torch.float8_e4m3fn, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] shape is M:512, N:5120, K:17408, q_dtype_w:torch.float8_e4m3fn, not found tuned config in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv, will use default config!
Capturing batches (bs=512 avail_mem=40.00 GB):   2%|▏         | 1/52 [00:02<01:42,  2.00s/it]Capturing batches (bs=496 avail_mem=39.50 GB):   2%|▏         | 1/52 [00:02<01:42,  2.00s/it][aiter] merge tuned file under model_configs/ and configs/ /sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
[aiter] shape is M:496, N:7168, K:5120, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] import [module_gemm_a8w8_blockscale_bpreshuffle] under /sgl-workspace/aiter/aiter/jit/module_gemm_a8w8_blockscale_bpreshuffle.so
[aiter] shape is M:496, N:5120, K:5120, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] shape is M:496, N:34816, K:5120, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
[aiter] shape is M:496, N:5120, K:17408, not found tuned config in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!
"""
#: The ``server_args`` line of that log's server, cut to the keys of the graph config. Its decode graphs were captured
#: for the sizes of ``GRAPH_SIZES``; the prefill list, a flag that is unset and the key a reader must not take for the
#: list are in it.
GRAPH_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 256, 512)
SERVER_ARGS = (
    "[2026-01-01 00:00:00] server_args={'model_path': '/models/m', 'tp_size': 1, 'cuda_graph_bs_decode': None, "
    "'cuda_graph_bs_prefill': None, 'cuda_graph_config': {'decode': {'backend': 'full', 'max_bs': 512, "
    f"'bs': {list(GRAPH_SIZES)}, 'tc_compiler': 'eager', 'max_context_size': None}}, "
    "'prefill': {'backend': 'disabled', 'max_bs': 16384, 'bs': [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]}}, "
    "'disable_cuda_graph': False}\n"
)
#: Lookups that resolved in a tuned row: M is the one asked for, ``padded_M`` the row found. Two of the blockscale table
#: (one at its merged view, one at the private copy of a table a test put), one of a per-token a8w8 table (its line does not
#: print ``q_dtype_w``, which keys it) and one of the dense bf16 table (which prints every column that keys it).
HIT_LOG = """\
[aiter] shape is M:12, N:7168, K:5120, found padded_M: 16, N:7168, K:5120 is tuned on cu_num = 256 in /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv , kernel name is a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256!
[aiter] shape is M:64, N:5120, K:17408, found padded_M: 64, N:5120, K:17408 is tuned on cu_num = 256 in /work/tables/AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE/0123456789abcdef/k-0123456789abcdef , kernel name is a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256!
[aiter] shape is M:512, N:7168, K:5120 found padded_M: 512, N:7168, K:5120 is tuned on cu_num = 256 in /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv , kernel name is a8w8_bpreshuffle_x!
[aiter] shape is M:4, N:6144, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False found padded_M: 4, N:6144, K:4096 is tuned on cu_num = 256 in /tmp/aiter_configs/bf16_tuned_gemm.csv, libtype is torch, kernel name is torch
"""
#: The shapes of ``HIT_LOG`` that become demand: (M of the row found, N, K) per table variable.
HIT_SHAPES = {
    ENV: {("16", "7168", "5120"), ("64", "5120", "17408")},
    "AITER_CONFIG_GEMM_BF16": {("4", "6144", "4096")},
}
#: The dense blockscale shapes of that log (at M=496): what Forge's demand holds for it, ranked by misses (all one).
LOG_SHAPES = {("496", "7168", "5120"), ("496", "5120", "5120"), ("496", "34816", "5120"), ("496", "5120", "17408")}


#: The header of a table a Forge that measures the relative-norm error of an asm splitK row writes: the image's columns and
#: Forge's own, blank on a row it did not measure.
REL_NORM_HEADER = f"{HEADER},relNormErr"


def table_text(header: str, rows: Iterable[str]) -> str:
    """A table of ``rows`` under ``header``, as the tuner writes one."""
    return "".join(f"{line}\n" for line in (header, *rows))


def tuned_rows(*, us: Sequence[float] = (8.0, 8.1), err: Sequence[float] = (0.003, 0.0)) -> tuple[str, ...]:
    """Rows of the image's table (the first two of ``SHARED``) with the microseconds and mismatch ratio given."""
    rows = []
    for line, microseconds, ratio in zip(SHARED, us, err, strict=False):
        cells = line.split(",")
        cells[HEADER.split(",").index("us")] = str(microseconds)
        cells[HEADER.split(",").index("errRatio")] = str(ratio)
        rows.append(",".join(cells))
    return tuple(rows)


def profile_row(
    m: int, n: int, k: int, *, lib: str, kernel_id: int, split_k: int, us: float, err: float, name: str
) -> str:
    """A line of the profile of every candidate the tuner measured: the columns of the table, whichever candidate won."""
    columns = HEADER.split(",")
    cells = dict.fromkeys(columns, "0") | {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "libtype": lib,
        "kernelId": str(kernel_id),
        "splitK": str(split_k),
        "us": str(us),
        "kernelName": name,
        "errRatio": str(err),
    }
    return ",".join(cells[column] for column in columns)


def forge_dropped(
    m: int, n: int, k: int, *, lib: str = "asm", split_k: int = 6, us: float = 20.0, err: float = 0.0125
) -> dict[str, Any]:
    """What Forge's report says of a row it took out of the table: its shape, libtype, splitK, microseconds and ratio."""
    return {
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "libtype": lib,
        "splitK": str(split_k),
        "us": str(us),
        "err_ratio": err,
    }


def landed_test(session: Path, model: Path, *, log: str | None = SERVER_LOG, concs: Sequence[int] = (32, 128)) -> Path:
    """The landed output of a test of the session: a point configuration per concurrency, and the server log."""
    if not (model / "config.json").exists():
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
    landed = session / "tests" / TEST_ID / "attempt-1"
    landed.mkdir(parents=True)
    for conc in concs:
        point = landed / f"variant_00_c{conc}"
        point.mkdir(parents=True)
        benchmark = {
            "framework": "sglang",
            "model": str(model),
            "precision": "fp8",
            "envs": {"TP": 1, "CONC": str(conc)},
        }
        (point / "config.yaml").write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    if log is not None:
        (landed / "server.log").write_text(log, encoding="utf-8")
    return landed


__all__ = [
    "ENV",
    "GRAPH_SIZES",
    "HIT_LOG",
    "HIT_SHAPES",
    "LOG_SHAPES",
    "REL_NORM_HEADER",
    "SERVER_ARGS",
    "SERVER_LOG",
    "TABLE",
    "TEST_ID",
    "TUNER",
    "forge_dropped",
    "landed_test",
    "profile_row",
    "table_text",
    "tuned_rows",
]
