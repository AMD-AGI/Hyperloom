# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real excerpts of one SGLang profile (equal input and output lengths, graphs replayed node by node) that the graded
mix is checked on.

``data/final_probe_*`` are cut from the windows TraceLens' splitter wrote for that profile's trace: four consecutive
decode steps of its ``decode_only`` window (each one graph replay), and two of the five prefill steps of its
``prefilldecode`` window, each with the launch calls, step annotations and module frames the readers use, and the steps
the splitter's ``execution_details.json`` lists for them. By the route, a wave of the profile's workload spends about a
tenth of its time in prefill, so paged attention's share of the wave is close to its share of the decode window and well
above its share of the ``mixed`` window, about half of whose time is its two prefill steps.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from hyperloom_multibranch.toolbox.tools.kernel_launchers import CAPTURE_MANIFEST

DATA = Path(__file__).parent / "data"
DECODE_WINDOW = DATA / "final_probe_decode4_window_launch_events.json.gz"
PREFILL_WINDOW = DATA / "final_probe_prefill2_window_launch_events.json.gz"
#: What the splitter lists for the two windows: ``(file name, steps)``, decode first.
DECODE_ENTRY, PREFILL_ENTRY = (
    (Path(entry["output_path"]).name, entry["steps"])
    for entry in json.loads((DATA / "final_probe_execution_details.json").read_text(encoding="utf-8"))
)
DECODE_CHUNK, PREFILL_CHUNK = DECODE_ENTRY[0], PREFILL_ENTRY[0]
#: The point's workload as the request carries it.
GRADED = {"split_conc": "64", "split_osl": "1024", "split_r": "1", "split_isl": "1024"}
PAGED_ATTENTION = "paged_attention_ll4mi_QKV_mfma16_kernel"
CK_GEMM = "ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle"


def install_capture(folder: Path) -> None:
    """The graph capture of batch size 64 of that server, in ``folder``, where the decode replays are named by module."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "nodewise_capture_execution_details.json", folder / CAPTURE_MANIFEST)
    shutil.copy(DATA / "nodewise_capture_bs64_launches.json.gz", folder / "DecodeCudaGraphRunner_bs_64_rank0.json.gz")


def starve_decode_window(target: Path) -> Path:
    """The decode excerpt at ``target`` without the kernels of its graph replays: what a profile that did not record them
    leaves (the host calls and the eager kernels stay, the window's kernels fill under 1% of its span)."""
    events = json.loads(gzip.decompress(DECODE_WINDOW.read_bytes()))["traceEvents"]
    replays = {e["args"]["correlation"] for e in events if e.get("name") == "hipGraphLaunch"}
    kept = [e for e in events if not (e.get("cat") == "kernel" and e["args"].get("correlation") in replays)]
    target.write_bytes(gzip.compress(json.dumps({"traceEvents": kept}).encode("utf-8")))
    return target
