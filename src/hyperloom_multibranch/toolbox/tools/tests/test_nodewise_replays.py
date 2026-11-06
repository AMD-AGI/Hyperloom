# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile taken with the graph replayed node by node has every decode replay's kernels, and the report reads them.

``data/nodewise_decode4_window_launch_events.json.gz`` is a synthetic window (``data/_make_traces.py``) shaped like four
consecutive decode steps (batch size 64) of an SGLang server profiled with ``DEBUG_CLR_GRAPH_PACKET_CAPTURE=0``: each
step's graph replay has its 80 kernels in the trace, in the order they were captured. With the runtime's default replay
of captured packets, a profile records the kernels of the last replay only (1 of 4 in such a window).
``nodewise_capture_bs64_launches.json.gz`` is the graph capture of batch size 64 those replays replay: the launches and
the ``nn.Module`` frames that made them.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools.kernel_launchers import CAPTURE_MANIFEST, read_launched_window, read_window
from hyperloom_multibranch.toolbox.tools.kernel_shares import kernel_window

DATA = Path(__file__).parent / "data"
WINDOW = DATA / "nodewise_decode4_window_launch_events.json.gz"
REPLAY_KERNELS = 80
CK_GEMM = "ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle"
PAGED_ATTENTION = "paged_attention_ll4mi_QKV_mfma16_kernel"
OUT_PROJECTION = "DecoderAttention.RowParallelLinear"


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    folder = tmp_path / "graph_capture_profile"
    folder.mkdir()
    shutil.copy(DATA / "nodewise_capture_execution_details.json", folder / CAPTURE_MANIFEST)
    shutil.copy(DATA / "nodewise_capture_bs64_launches.json.gz", folder / "DecodeCudaGraphRunner_bs_64_rank0.json.gz")
    return folder


def test_every_replay_of_the_window_has_its_kernels_in_the_trace() -> None:
    window = read_window(WINDOW)

    kernels = Counter(event.correlation for event in window.events)

    assert [kernels[replay.correlation] for replay in window.replays] == [REPLAY_KERNELS] * 4


def test_the_window_gives_kernel_shares_of_all_four_replays_with_the_gemms_split_by_launcher(capture: Path) -> None:
    launched = read_launched_window(WINDOW, capture)

    shares = kernel_window(launched.events, list).shares

    assert launched.notes == () and launched.renamed == 0, (
        "the nodewise replay records the names in the capture's order"
    )
    assert sum(shares.values()) == pytest.approx(100.0)
    assert round(shares[PAGED_ATTENTION], 2) == 20.25  # 4 x 40 us of a 790 us step
    gemms = {name: round(share, 2) for name, share in shares.items() if name.startswith(CK_GEMM)}
    assert gemms == {
        f"{CK_GEMM} [DecoderMLP.RowParallelLinear]": 20.25,
        f"{CK_GEMM} [DecoderMLP.MergedColumnParallelLinear]": 15.19,
        f"{CK_GEMM} [DecoderAttention.QKVParallelLinear]": 8.1,
        f"{CK_GEMM} [{OUT_PROJECTION}]": 7.09,
    }


def test_the_window_is_not_degraded_and_has_no_misreported_event(capture: Path) -> None:
    window = kernel_window(read_launched_window(WINDOW, capture).events, list)

    assert window.degraded_reason == "" and window.excluded_events == 0
    assert window.busy_percent > 90
